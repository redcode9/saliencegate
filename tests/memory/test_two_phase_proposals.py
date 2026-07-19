from __future__ import annotations

import json
from uuid import UUID

import pytest
from pydantic import ValidationError

from saliencegate.domain import (
    MAX_MEMORY_CONTENT_BYTES,
    MAX_MEMORY_DELTA_ITEMS,
    MAX_MEMORY_PROVENANCE_ITEMS,
    ClaimKind,
    EvidenceReference,
    EvidenceSource,
    InterventionAction,
)
from saliencegate.intervention import ProposedClaim
from saliencegate.memory.proposals import (
    INTERVENTION_OUTPUT_SCHEMA_VERSION,
    MAX_PROPOSAL_POINTER_SEGMENTS,
    MAX_PROPOSAL_POINTER_UTF8_BYTES,
    MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
    BankOperationsProposal,
    DeleteMemory,
    InterventionSelectionOutput,
    SaveKnowledge,
    SaveProcedural,
    UpdatePrivateStatus,
)

EVENT_ID = UUID("00000000-0000-4000-8000-00000000c001")
MEMORY_ID = UUID("00000000-0000-4000-8000-00000000c002")


def _event_evidence(index: int = 0) -> EvidenceReference:
    return EvidenceReference(
        source=EvidenceSource.EVENT,
        source_id=UUID(f"00000000-0000-4000-8000-{index + 1:012d}"),
        field_path="/payload/message",
    )


def _claim() -> ProposedClaim:
    return ProposedClaim(
        kind=ClaimKind.ENVIRONMENT_FACT,
        evidence=EvidenceReference(
            source=EvidenceSource.MEMORY,
            source_id=MEMORY_ID,
            revision=1,
            field_path="/content",
        ),
    )


def test_bank_operations_are_closed_ordered_and_model_owned_only() -> None:
    proposal = BankOperationsProposal(
        schema_version=MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
        operations=(
            UpdatePrivateStatus(
                operation="update_private_status",
                content="The migration is still open.",
                evidence=(_event_evidence(0),),
                confidence=0.9,
            ),
            SaveKnowledge(
                operation="save_knowledge",
                content="The repository requires Python 3.11 or newer.",
                evidence=(_event_evidence(1),),
                confidence=1.0,
            ),
            SaveProcedural(
                operation="save_procedural",
                content="The first migration attempt failed after schema validation.",
                evidence=(_event_evidence(2),),
                confidence=0.8,
            ),
            DeleteMemory(
                operation="delete_memory",
                memory_id=MEMORY_ID,
                expected_revision=2,
            ),
        ),
    )

    assert proposal.schema_version == MEMORY_EDIT_OUTPUT_SCHEMA_VERSION
    assert tuple(operation.operation for operation in proposal.operations) == (
        "update_private_status",
        "save_knowledge",
        "save_procedural",
        "delete_memory",
    )
    assert BankOperationsProposal.model_validate_json(proposal.model_dump_json()) == proposal
    assert (
        BankOperationsProposal(
            schema_version=MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
            operations=(),
        ).operations
        == ()
    )

    encoded_schema = json.dumps(BankOperationsProposal.model_json_schema(), sort_keys=True)
    for forbidden in (
        "created_at",
        "delta_id",
        "expires_at",
        "handle",
        "run_id",
        "trust_label",
        "updated_at",
    ):
        assert forbidden not in encoded_schema

    schema = BankOperationsProposal.model_json_schema()
    assert set(schema["required"]) == {"schema_version", "operations"}
    for operation_schema in (
        schema["$defs"]["UpdatePrivateStatus"],
        schema["$defs"]["SaveKnowledge"],
        schema["$defs"]["SaveProcedural"],
        schema["$defs"]["DeleteMemory"],
    ):
        assert "operation" in operation_schema["required"]


def test_schema_versions_are_pinned_and_cannot_cross_phases() -> None:
    with pytest.raises(ValidationError):
        BankOperationsProposal(  # type: ignore[arg-type]
            schema_version="memory-edit-output/v2",
            operations=(),
        )
    with pytest.raises(ValidationError):
        InterventionSelectionOutput(
            schema_version=MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,  # type: ignore[arg-type]
            action=InterventionAction.SILENCE,
            claims=(),
            confidence=1.0,
        )

    with pytest.raises(ValidationError):
        BankOperationsProposal(operations=())  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        SaveKnowledge(
            content="known",
            evidence=(_event_evidence(),),
            confidence=1.0,
        )  # type: ignore[call-arg]

    intervention_schema = InterventionSelectionOutput.model_json_schema()
    assert set(intervention_schema["required"]) == {
        "schema_version",
        "action",
        "claims",
        "confidence",
    }


def test_operation_limits_and_private_status_are_unambiguous() -> None:
    operation = SaveKnowledge(
        operation="save_knowledge",
        content="bounded",
        evidence=(_event_evidence(),),
        confidence=1.0,
    )
    assert (
        len(
            BankOperationsProposal(
                schema_version=MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
                operations=(operation,) * MAX_MEMORY_DELTA_ITEMS,
            ).operations
        )
        == MAX_MEMORY_DELTA_ITEMS
    )

    with pytest.raises(ValidationError, match="at most 64"):
        BankOperationsProposal(
            schema_version=MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
            operations=(operation,) * (MAX_MEMORY_DELTA_ITEMS + 1),
        )

    status = UpdatePrivateStatus(
        operation="update_private_status",
        content="open",
        evidence=(_event_evidence(),),
        confidence=1.0,
    )
    with pytest.raises(ValidationError, match="private-status"):
        BankOperationsProposal(
            schema_version=MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
            operations=(status, status),
        )


def test_write_content_is_bounded_by_utf8_bytes() -> None:
    evidence = (_event_evidence(),)
    accepted = "a" * MAX_MEMORY_CONTENT_BYTES
    assert (
        SaveKnowledge(
            operation="save_knowledge",
            content=accepted,
            evidence=evidence,
            confidence=1.0,
        ).content
        == accepted
    )

    for rejected in (
        "",
        "a" * (MAX_MEMORY_CONTENT_BYTES + 1),
        "é" * ((MAX_MEMORY_CONTENT_BYTES // 2) + 1),
        "\ud800",
    ):
        with pytest.raises(ValidationError, match="content"):
            SaveKnowledge(
                operation="save_knowledge",
                content=rejected,
                evidence=evidence,
                confidence=1.0,
            )

    with pytest.raises(ValidationError):
        SaveKnowledge(
            operation="save_knowledge",
            content=b"not-text",  # type: ignore[arg-type]
            evidence=evidence,
            confidence=1.0,
        )


def test_write_evidence_is_bounded_unique_and_recursively_validated() -> None:
    maximum = tuple(_event_evidence(index) for index in range(MAX_MEMORY_PROVENANCE_ITEMS))
    assert (
        len(
            SaveProcedural(
                operation="save_procedural",
                content="bounded",
                evidence=maximum,
                confidence=1.0,
            ).evidence
        )
        == 8
    )

    with pytest.raises(ValidationError, match="at most 8"):
        SaveProcedural(
            operation="save_procedural",
            content="too much evidence",
            evidence=(*maximum, _event_evidence(9)),
            confidence=1.0,
        )

    with pytest.raises(ValidationError, match="unique"):
        SaveProcedural(
            operation="save_procedural",
            content="duplicate evidence",
            evidence=(_event_evidence(), _event_evidence()),
            confidence=1.0,
        )

    forged = EvidenceReference.model_construct(
        source=EvidenceSource.EVENT,
        source_id=EVENT_ID,
        revision=1,
        field_path="/payload/message",
        span=None,
    )
    with pytest.raises(ValidationError):
        SaveProcedural(
            operation="save_procedural",
            content="forged evidence",
            evidence=(forged,),
            confidence=1.0,
        )


@pytest.mark.parametrize(
    "field_path",
    (
        "/a" * (MAX_PROPOSAL_POINTER_SEGMENTS + 1),
        "/a" + "é" * (MAX_PROPOSAL_POINTER_UTF8_BYTES // 2),
    ),
)
def test_proposal_evidence_enforces_grounding_pointer_limits(field_path: str) -> None:
    evidence = EvidenceReference(
        source=EvidenceSource.EVENT,
        source_id=EVENT_ID,
        field_path=field_path,
    )

    with pytest.raises(ValidationError, match="evidence failed validation"):
        SaveKnowledge(
            operation="save_knowledge",
            content="known",
            evidence=(evidence,),
            confidence=1.0,
        )
    with pytest.raises(ValidationError, match="claims failed validation"):
        InterventionSelectionOutput(
            schema_version=INTERVENTION_OUTPUT_SCHEMA_VERSION,
            action=InterventionAction.REMIND,
            claims=(
                ProposedClaim(
                    kind=ClaimKind.ENVIRONMENT_FACT,
                    evidence=evidence,
                ),
            ),
            confidence=1.0,
        )


def test_proposal_evidence_accepts_exact_grounding_pointer_limits() -> None:
    for field_path in (
        "/a" * MAX_PROPOSAL_POINTER_SEGMENTS,
        "/a" + "é" * ((MAX_PROPOSAL_POINTER_UTF8_BYTES - 2) // 2),
    ):
        evidence = EvidenceReference(
            source=EvidenceSource.EVENT,
            source_id=EVENT_ID,
            field_path=field_path,
        )
        assert SaveKnowledge(
            operation="save_knowledge",
            content="known",
            evidence=(evidence,),
            confidence=1.0,
        ).evidence == (evidence,)


def test_proposal_revisions_are_bounded_to_positive_signed_64() -> None:
    maximum_revision = (1 << 63) - 1
    maximum_reference = EvidenceReference(
        source=EvidenceSource.MEMORY,
        source_id=MEMORY_ID,
        revision=maximum_revision,
        field_path="/content",
    )
    assert (
        DeleteMemory(
            operation="delete_memory",
            memory_id=MEMORY_ID,
            expected_revision=maximum_revision,
        ).expected_revision
        == maximum_revision
    )
    assert SaveKnowledge(
        operation="save_knowledge",
        content="known",
        evidence=(maximum_reference,),
        confidence=1.0,
    ).evidence == (maximum_reference,)

    overflow_reference = EvidenceReference(
        source=EvidenceSource.MEMORY,
        source_id=MEMORY_ID,
        revision=maximum_revision + 1,
        field_path="/content",
    )
    with pytest.raises(ValidationError):
        DeleteMemory(
            operation="delete_memory",
            memory_id=MEMORY_ID,
            expected_revision=maximum_revision + 1,
        )
    with pytest.raises(ValidationError, match="evidence failed validation"):
        SaveKnowledge(
            operation="save_knowledge",
            content="known",
            evidence=(overflow_reference,),
            confidence=1.0,
        )
    with pytest.raises(ValidationError, match="claims failed validation"):
        InterventionSelectionOutput(
            schema_version=INTERVENTION_OUTPUT_SCHEMA_VERSION,
            action=InterventionAction.REMIND,
            claims=(
                ProposedClaim(
                    kind=ClaimKind.ENVIRONMENT_FACT,
                    evidence=overflow_reference,
                ),
            ),
            confidence=1.0,
        )


def test_delete_selects_only_an_existing_exact_revision() -> None:
    operation = DeleteMemory(
        operation="delete_memory",
        memory_id=MEMORY_ID,
        expected_revision=1,
    )
    assert operation.memory_id == MEMORY_ID
    assert operation.expected_revision == 1
    assert set(DeleteMemory.model_fields) == {
        "operation",
        "memory_id",
        "expected_revision",
    }

    for values in (
        {"memory_id": MEMORY_ID, "expected_revision": 0},
        {
            "memory_id": UUID("00000000-0000-1000-8000-00000000c002"),
            "expected_revision": 1,
        },
        {"memory_id": str(MEMORY_ID), "expected_revision": 1},
        {"memory_id": MEMORY_ID, "expected_revision": True},
    ):
        with pytest.raises(ValidationError):
            DeleteMemory(operation="delete_memory", **values)  # type: ignore[arg-type]


def test_duplicate_delete_targets_are_rejected_before_materialization() -> None:
    first = DeleteMemory(
        operation="delete_memory",
        memory_id=MEMORY_ID,
        expected_revision=1,
    )
    second = DeleteMemory(
        operation="delete_memory",
        memory_id=MEMORY_ID,
        expected_revision=2,
    )

    with pytest.raises(ValidationError, match="delete target"):
        BankOperationsProposal(
            schema_version=MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
            operations=(first, second),
        )


def test_operation_discriminator_rejects_unknown_or_cross_shaped_payloads() -> None:
    valid = BankOperationsProposal(
        schema_version=MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
        operations=(
            SaveKnowledge(
                operation="save_knowledge",
                content="known",
                evidence=(_event_evidence(),),
                confidence=1.0,
            ),
        ),
    ).model_dump(mode="json")

    secret_tag = "unknown-provider-secret"
    valid["operations"][0]["operation"] = secret_tag
    with pytest.raises(ValidationError) as captured:
        BankOperationsProposal.model_validate_json(json.dumps(valid))
    assert secret_tag not in str(captured.value)
    assert secret_tag not in repr(captured.value)

    crossed = {
        "schema_version": MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
        "operations": (
            {
                "operation": "delete_memory",
                "content": "not a delete",
                "evidence": (_event_evidence().model_dump(mode="json"),),
                "confidence": 1.0,
            },
        ),
    }
    with pytest.raises(ValidationError):
        BankOperationsProposal.model_validate_json(json.dumps(crossed))

    with pytest.raises(ValidationError):
        BankOperationsProposal(
            schema_version=MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
            operations="not-an-operation-list",  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError):
        BankOperationsProposal(
            schema_version=MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
            operations=(object(),),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "operation",
    ("update_private_status", "save_knowledge", "save_procedural"),
)
@pytest.mark.parametrize(
    "authoritative_field,value",
    (
        ("memory_id", str(MEMORY_ID)),
        ("run_id", "00000000-0000-4000-8000-000000000001"),
        ("revision", 1),
        ("trust_label", "trusted"),
        ("created_at", "2026-07-12T00:00:00Z"),
        ("handle", "model-owned-handle"),
    ),
)
def test_write_operations_reject_model_authored_authority(
    operation: str,
    authoritative_field: str,
    value: object,
) -> None:
    payload = {
        "operation": operation,
        "content": "known",
        "evidence": (_event_evidence().model_dump(mode="json"),),
        "confidence": 1.0,
        authoritative_field: value,
    }
    operation_type = {
        "update_private_status": UpdatePrivateStatus,
        "save_knowledge": SaveKnowledge,
        "save_procedural": SaveProcedural,
    }[operation]

    with pytest.raises(ValidationError):
        operation_type.model_validate(payload)


def test_intervention_selection_has_only_typed_citation_claims() -> None:
    silence = InterventionSelectionOutput(
        schema_version=INTERVENTION_OUTPUT_SCHEMA_VERSION,
        action=InterventionAction.SILENCE,
        claims=(),
        confidence=0.75,
    )
    reminder = InterventionSelectionOutput(
        schema_version=INTERVENTION_OUTPUT_SCHEMA_VERSION,
        action=InterventionAction.REMIND,
        claims=(_claim(),),
        confidence=1.0,
    )

    assert silence.schema_version == reminder.schema_version
    assert reminder.schema_version == INTERVENTION_OUTPUT_SCHEMA_VERSION
    assert reminder.to_grounding_proposal().model_free_text is None
    assert reminder.to_grounding_proposal().claims == reminder.claims
    assert set(InterventionSelectionOutput.model_fields) == {
        "schema_version",
        "action",
        "claims",
        "confidence",
    }

    for values in (
        {
            "schema_version": INTERVENTION_OUTPUT_SCHEMA_VERSION,
            "action": InterventionAction.REMIND,
            "claims": (),
            "confidence": 1.0,
        },
        {
            "schema_version": INTERVENTION_OUTPUT_SCHEMA_VERSION,
            "action": InterventionAction.SILENCE,
            "claims": (_claim(),),
            "confidence": 1.0,
        },
        {
            "schema_version": INTERVENTION_OUTPUT_SCHEMA_VERSION,
            "action": InterventionAction.REMIND,
            "claims": (_claim(), _claim(), _claim()),
            "confidence": 1.0,
        },
    ):
        with pytest.raises(ValidationError):
            InterventionSelectionOutput(**values)  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="claims must be unique"):
        InterventionSelectionOutput(
            schema_version=INTERVENTION_OUTPUT_SCHEMA_VERSION,
            action=InterventionAction.REMIND,
            claims=(_claim(), _claim()),
            confidence=1.0,
        )

    with pytest.raises(ValidationError):
        InterventionSelectionOutput.model_validate(
            {
                "schema_version": INTERVENTION_OUTPUT_SCHEMA_VERSION,
                "action": "remind",
                "claims": (_claim().model_dump(mode="json"),),
                "confidence": 1.0,
                "model_free_text": "must never reach delivery",
            }
        )


def test_proposal_models_are_strict_frozen_and_revalidate_nested_values() -> None:
    proposal = BankOperationsProposal(
        schema_version=MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
        operations=(
            SaveKnowledge(
                operation="save_knowledge",
                content="known",
                evidence=(_event_evidence(),),
                confidence=1.0,
            ),
        ),
    )

    with pytest.raises(ValidationError):
        BankOperationsProposal(
            schema_version=MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
            operations=list(proposal.operations),  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError):
        BankOperationsProposal.model_validate(
            {
                **proposal.model_dump(mode="json"),
                "run_id": "00000000-0000-4000-8000-000000000001",
            }
        )
    with pytest.raises(ValidationError):
        SaveKnowledge(
            operation="save_knowledge",
            content="known",
            evidence=(_event_evidence(),),
            confidence=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError):
        proposal.operations = ()  # type: ignore[misc]

    forged = SaveKnowledge.model_construct(
        operation="save_knowledge",
        content="known",
        evidence=(),
        confidence=1.0,
    )
    with pytest.raises(ValidationError):
        BankOperationsProposal(
            schema_version=MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
            operations=(forged,),
        )

    forged_claim = ProposedClaim.model_construct(
        kind=ClaimKind.ENVIRONMENT_FACT,
        evidence=EvidenceReference.model_construct(
            source=EvidenceSource.EVENT,
            source_id=EVENT_ID,
            revision=1,
            field_path="/payload/message",
            span=None,
        ),
    )
    with pytest.raises(ValidationError):
        InterventionSelectionOutput(
            schema_version=INTERVENTION_OUTPUT_SCHEMA_VERSION,
            action=InterventionAction.REMIND,
            claims=(forged_claim,),
            confidence=1.0,
        )


def test_sensitive_proposal_values_are_absent_from_repr_and_validation_errors() -> None:
    secret = "credential-like-memory-content"
    operation = SaveKnowledge(
        operation="save_knowledge",
        content=secret,
        evidence=(_event_evidence(),),
        confidence=1.0,
    )
    proposal = BankOperationsProposal(
        schema_version=MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
        operations=(operation,),
    )
    reminder = InterventionSelectionOutput(
        schema_version=INTERVENTION_OUTPUT_SCHEMA_VERSION,
        action=InterventionAction.REMIND,
        claims=(_claim(),),
        confidence=1.0,
    )

    assert secret not in repr(operation)
    assert secret not in repr(proposal)
    assert str(operation.evidence[0].source_id) not in repr(operation)
    assert str(MEMORY_ID) not in repr(reminder)

    with pytest.raises(ValidationError) as captured:
        SaveKnowledge(
            operation="save_knowledge",
            content=secret,
            evidence=(_event_evidence(),),
            confidence=2.0,
        )
    assert secret not in str(captured.value)
