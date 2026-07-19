from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from saliencegate.domain import (
    MAX_MEMORY_CONTENT_BYTES,
    MAX_MEMORY_DELTA_ITEMS,
    MAX_MEMORY_PROVENANCE_ITEMS,
    EvidenceReference,
    InterventionAction,
    canonical_json,
)
from saliencegate.domain.records import UUID4, PositiveSigned64Offset, UnitInterval
from saliencegate.intervention.claims import InterventionProposal, ProposedClaim

MEMORY_EDIT_OUTPUT_SCHEMA_VERSION: Literal["memory-edit-output/v1"] = "memory-edit-output/v1"
INTERVENTION_OUTPUT_SCHEMA_VERSION: Literal["intervention-output/v1"] = "intervention-output/v1"

MAX_PROPOSAL_POINTER_SEGMENTS = 32
MAX_PROPOSAL_POINTER_UTF8_BYTES = 1_024
_MAX_SIGNED_64 = (1 << 63) - 1


class _ProposalModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


def _bounded_content(value: object) -> object:
    if type(value) is not str:
        raise ValueError("memory proposal content must be exact text")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise ValueError("memory proposal content must be valid UTF-8") from None
    if not encoded or len(encoded) > MAX_MEMORY_CONTENT_BYTES:
        raise ValueError("memory proposal content exceeds its UTF-8 byte limit")
    return value


def _validated_evidence_reference(value: object) -> EvidenceReference:
    if type(value) is not EvidenceReference:
        raise ValueError
    validated = EvidenceReference.model_validate_json(value.model_dump_json(warnings=False))
    pointer = validated.field_path
    if (
        len(pointer.encode("utf-8", errors="strict")) > MAX_PROPOSAL_POINTER_UTF8_BYTES
        or pointer.count("/") > MAX_PROPOSAL_POINTER_SEGMENTS
        or (validated.revision is not None and validated.revision > _MAX_SIGNED_64)
    ):
        raise ValueError
    return validated


def _validated_evidence(
    value: tuple[EvidenceReference, ...],
) -> tuple[EvidenceReference, ...]:
    try:
        validated = [_validated_evidence_reference(item) for item in value]
    except Exception:
        raise ValueError("memory proposal evidence failed validation") from None
    copied = tuple(validated)
    if len({canonical_json(item) for item in copied}) != len(copied):
        raise ValueError("memory proposal evidence must be unique")
    return copied


class _MemoryWrite(_ProposalModel):
    content: Annotated[str, Field(min_length=1, repr=False)]
    evidence: Annotated[
        tuple[EvidenceReference, ...],
        Field(min_length=1, max_length=MAX_MEMORY_PROVENANCE_ITEMS, repr=False),
    ]
    confidence: UnitInterval

    @field_validator("content", mode="before")
    @classmethod
    def bound_content(cls, value: object) -> object:
        return _bounded_content(value)

    @field_validator("evidence")
    @classmethod
    def revalidate_evidence(
        cls,
        value: tuple[EvidenceReference, ...],
    ) -> tuple[EvidenceReference, ...]:
        return _validated_evidence(value)


class UpdatePrivateStatus(_MemoryWrite):
    operation: Literal["update_private_status"]


class SaveKnowledge(_MemoryWrite):
    operation: Literal["save_knowledge"]


class SaveProcedural(_MemoryWrite):
    operation: Literal["save_procedural"]


class DeleteMemory(_ProposalModel):
    operation: Literal["delete_memory"]
    memory_id: UUID4
    expected_revision: PositiveSigned64Offset


BankOperation = Annotated[
    UpdatePrivateStatus | SaveKnowledge | SaveProcedural | DeleteMemory,
    Field(discriminator="operation"),
]

_OPERATION_TYPES = (
    UpdatePrivateStatus,
    SaveKnowledge,
    SaveProcedural,
    DeleteMemory,
)
_OPERATION_TAGS = frozenset(
    ("update_private_status", "save_knowledge", "save_procedural", "delete_memory")
)


class BankOperationsProposal(_ProposalModel):
    schema_version: Literal["memory-edit-output/v1"]
    operations: Annotated[
        tuple[BankOperation, ...],
        Field(max_length=MAX_MEMORY_DELTA_ITEMS, repr=False),
    ]

    @field_validator("operations", mode="before")
    @classmethod
    def prevalidate_operation_tags(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> object:
        if type(value) not in (list, tuple):
            return value
        assert isinstance(value, (list, tuple))
        for operation in value:
            tag: object
            if type(operation) is dict:
                tag = operation.get("operation")
            elif type(operation) in _OPERATION_TYPES:
                tag = operation.operation
            else:
                continue
            if type(tag) is not str or tag not in _OPERATION_TAGS:
                raise ValueError("memory proposal operation tag failed validation") from None
        if info.mode != "json" or type(value) is not list:
            return value
        normalized: list[object] = []
        for operation in value:
            if type(operation) is not dict or type(operation.get("evidence")) is not list:
                normalized.append(operation)
                continue
            copied = dict(operation)
            copied["evidence"] = tuple(operation["evidence"])
            normalized.append(copied)
        return tuple(normalized)

    @field_validator("operations")
    @classmethod
    def revalidate_operations(
        cls,
        value: tuple[BankOperation, ...],
    ) -> tuple[BankOperation, ...]:
        validated: list[BankOperation] = []
        try:
            for operation in value:
                operation_type = type(operation)
                if operation_type not in _OPERATION_TYPES:
                    raise ValueError
                validated.append(
                    operation_type.model_validate_json(operation.model_dump_json(warnings=False))
                )
        except Exception:
            raise ValueError("memory proposal operation failed validation") from None
        return tuple(validated)

    @model_validator(mode="after")
    def operations_are_unambiguous(self) -> Self:
        private_status_count = sum(
            type(operation) is UpdatePrivateStatus for operation in self.operations
        )
        if private_status_count > 1:
            raise ValueError("memory proposal has more than one private-status operation")
        delete_targets = tuple(
            operation.memory_id for operation in self.operations if type(operation) is DeleteMemory
        )
        if len(set(delete_targets)) != len(delete_targets):
            raise ValueError("memory proposal delete target must be unique")
        return self


class InterventionSelectionOutput(_ProposalModel):
    schema_version: Literal["intervention-output/v1"]
    action: InterventionAction
    claims: Annotated[tuple[ProposedClaim, ...], Field(max_length=2, repr=False)]
    confidence: UnitInterval

    @field_validator("claims")
    @classmethod
    def revalidate_claims(
        cls,
        value: tuple[ProposedClaim, ...],
    ) -> tuple[ProposedClaim, ...]:
        validated: list[ProposedClaim] = []
        try:
            for claim in value:
                if type(claim) is not ProposedClaim:
                    raise ValueError
                copied_claim = ProposedClaim.model_validate_json(
                    claim.model_dump_json(warnings=False)
                )
                validated.append(
                    ProposedClaim(
                        kind=copied_claim.kind,
                        evidence=_validated_evidence_reference(copied_claim.evidence),
                    )
                )
        except Exception:
            raise ValueError("intervention selection claims failed validation") from None
        copied_claims = tuple(validated)
        if len({canonical_json(item) for item in copied_claims}) != len(copied_claims):
            raise ValueError("intervention selection claims must be unique")
        return copied_claims

    @model_validator(mode="after")
    def action_matches_claims(self) -> Self:
        if self.action is InterventionAction.REMIND and not self.claims:
            raise ValueError("a reminder selection requires at least one claim")
        if self.action is InterventionAction.SILENCE and self.claims:
            raise ValueError("a silence selection cannot carry claims")
        return self

    def to_grounding_proposal(self) -> InterventionProposal:
        return InterventionProposal(
            action=self.action,
            claims=self.claims,
            confidence=self.confidence,
            model_free_text=None,
        )


__all__ = [
    "INTERVENTION_OUTPUT_SCHEMA_VERSION",
    "MAX_PROPOSAL_POINTER_SEGMENTS",
    "MAX_PROPOSAL_POINTER_UTF8_BYTES",
    "MEMORY_EDIT_OUTPUT_SCHEMA_VERSION",
    "BankOperation",
    "BankOperationsProposal",
    "DeleteMemory",
    "InterventionSelectionOutput",
    "SaveKnowledge",
    "SaveProcedural",
    "UpdatePrivateStatus",
]
