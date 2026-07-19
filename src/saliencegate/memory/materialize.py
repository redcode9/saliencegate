from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, NoReturn, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.domain import (
    MAX_MEMORY_DELTA_ITEMS,
    CycleRecord,
    CycleState,
    EvidenceReference,
    EvidenceSource,
    MemoryCreate,
    MemoryDelta,
    MemoryIdAssignment,
    MemoryInvalidation,
    MemoryKind,
    MemoryRecord,
    PayloadDigest,
    PrivateStatusReplacement,
    ReasonCode,
    TraceEvent,
    TrustLabel,
    ValidityState,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.domain.records import (
    UUID4,
    PositiveSigned64Offset,
    Sha256Digest,
    UtcDatetime,
)
from saliencegate.memory.proposals import (
    BankOperationsProposal,
    DeleteMemory,
    SaveKnowledge,
    SaveProcedural,
    UpdatePrivateStatus,
)
from saliencegate.ports.repository import (
    CrossRunReferenceError,
    CycleReceipt,
    InvalidRecordError,
    InvalidRecordTypeError,
    LedgerEntry,
    MemoryDeltaPreview,
    MemorySnapshot,
    PreviewConflictError,
    PreviewMemoryDelta,
    ProjectionInvariantError,
    RepositoryError,
    RevisionConflictError,
    RunRepository,
    UnsafeRecordContentError,
)

MATERIALIZATION_REQUEST_SCHEMA_VERSION: Literal["operation-materialization-request/v1"] = (
    "operation-materialization-request/v1"
)
MATERIALIZATION_RESULT_SCHEMA_VERSION: Literal["operation-materialization-result/v1"] = (
    "operation-materialization-result/v1"
)
OPERATION_HANDLE_PREFIX: Literal["memory-edit-operation/v1/"] = "memory-edit-operation/v1/"

_OPERATIONS_DIGEST_DOMAIN = "saliencegate:memory:source-operations:v1"
_MATERIALIZATION_DIGEST_DOMAIN = "saliencegate:memory:materialization:v1"
_POINTER_INDEX = re.compile(r"^(?:0|[1-9][0-9]*)$")


class MaterializationFailureReason(StrEnum):
    INVALID_INPUT = "invalid_input"
    SOURCE_CONFLICT = "source_conflict"
    IDENTITY_CONFLICT = "identity_conflict"
    ASSIGNMENT_CONFLICT = "assignment_conflict"
    REFERENCE_MISSING = "reference_missing"
    REFERENCE_STALE = "reference_stale"
    REFERENCE_INACTIVE = "reference_inactive"
    REFERENCE_EXPIRED = "reference_expired"
    REFERENCE_FUTURE = "reference_future"
    REFERENCE_INVALID = "reference_invalid"
    OPERATION_CONFLICT = "operation_conflict"
    PREVIEW_REJECTED = "preview_rejected"
    REPOSITORY_UNAVAILABLE = "repository_unavailable"
    RESULT_MISMATCH = "result_mismatch"


class MemoryOperationMaterializationError(ValueError):
    """Value-free rejection of model-authored bank operations."""

    def __init__(self, reason: MaterializationFailureReason) -> None:
        self.reason = reason
        super().__init__("memory operations failed authoritative materialization")


class _MaterializationModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


def operation_handle(ordinal: int) -> str:
    """Return the versioned, one-based handle for a proposal operation."""

    if type(ordinal) is not int or not 1 <= ordinal <= MAX_MEMORY_DELTA_ITEMS:
        raise ValueError("operation ordinal must be within the memory delta bound")
    return f"{OPERATION_HANDLE_PREFIX}{ordinal:04d}"


def _write_count(proposal: BankOperationsProposal) -> int:
    return sum(
        type(operation) in (UpdatePrivateStatus, SaveKnowledge, SaveProcedural)
        for operation in proposal.operations
    )


class OperationMaterializationRequest(_MaterializationModel):
    """Runtime-owned inputs that bind one proposal to one running cycle."""

    schema_version: Literal["operation-materialization-request/v1"]
    cycle_receipt: CycleReceipt = Field(repr=False)
    proposal: BankOperationsProposal = Field(repr=False)
    delta_id: UUID4
    created_at: UtcDatetime
    assigned_memory_ids: Annotated[
        tuple[UUID4, ...],
        Field(max_length=MAX_MEMORY_DELTA_ITEMS, repr=False),
    ] = ()

    @model_validator(mode="after")
    def cycle_time_and_assignments_match(self) -> Self:
        if self.cycle_receipt.cycle.state is not CycleState.RUNNING:
            raise ValueError("operation materialization requires a running cycle")
        if self.created_at < self.cycle_receipt.cycle.updated_at:
            raise ValueError("operation materialization time precedes its running cycle")
        if len(self.assigned_memory_ids) != _write_count(self.proposal):
            raise ValueError("operation materialization assignment count does not match")
        if len(set(self.assigned_memory_ids)) != len(self.assigned_memory_ids):
            raise ValueError("operation materialization assignments must be unique")
        return self


def source_operations_digest(proposal: BankOperationsProposal) -> str:
    """Bind a sanitized operation proposal to its materialized result."""

    return length_prefixed_sha256(
        canonical_json(proposal),
        domain=_OPERATIONS_DIGEST_DOMAIN,
    )


def _materialization_digest(values: Mapping[str, object]) -> str:
    return length_prefixed_sha256(
        canonical_json(
            {
                "schema_version": values["schema_version"],
                "run_id": str(values["run_id"]),
                "source_cycle_id": values["source_cycle_id"],
                "source_last_event_sequence": values["source_last_event_sequence"],
                "source_ledger_position": values["source_ledger_position"],
                "source_ingestion_cursor": values["source_ingestion_cursor"],
                "source_memory_cursor": values["source_memory_cursor"],
                "source_record_tag": values["source_record_tag"],
                "source_chain_tag": values["source_chain_tag"],
                "source_projection_digest": values["source_projection_digest"],
                "source_operations_digest": values["source_operations_digest"],
                "delta": values["delta"],
                "memory_id_assignments": values["memory_id_assignments"],
                "active_bank": values["active_bank"],
                "preview_projection_digest": values["preview_projection_digest"],
            }
        ),
        domain=_MATERIALIZATION_DIGEST_DOMAIN,
    )


class MaterializedBankOperations(_MaterializationModel):
    """A content-bound compiled delta and its exact uncommitted candidate bank.

    The SHA-256 materialization digest detects accidental divergence; it is not an
    authentication tag. Use ``verified_materialized_bank_operations_for_request``
    when accepting a result from outside the trusted runtime component.
    """

    schema_version: Literal["operation-materialization-result/v1"]
    run_id: UUID4
    source_cycle_id: Sha256Digest
    source_last_event_sequence: PositiveSigned64Offset
    source_ledger_position: PositiveSigned64Offset
    source_ingestion_cursor: PositiveSigned64Offset
    source_memory_cursor: Annotated[int, Field(ge=0, le=(1 << 63) - 1)]
    source_record_tag: PayloadDigest
    source_chain_tag: PayloadDigest
    source_projection_digest: PayloadDigest
    source_operations_digest: Sha256Digest
    delta: MemoryDelta = Field(repr=False)
    memory_id_assignments: tuple[MemoryIdAssignment, ...] = Field(repr=False)
    active_bank: tuple[MemoryRecord, ...] = Field(repr=False)
    preview_projection_digest: PayloadDigest
    materialization_digest: Sha256Digest

    @model_validator(mode="after")
    def compiled_view_and_digest_match(self) -> Self:
        expected_handles = tuple(item.handle for item in self.delta.creates)
        replacement = self.delta.private_status_replacement
        if replacement is not None:
            expected_handles += (replacement.replacement.handle,)
        active_sort = tuple(
            sorted(
                self.active_bank,
                key=lambda record: (record.kind.value, str(record.memory_id)),
            )
        )
        digest_algorithms = {
            self.source_record_tag.algorithm,
            self.source_chain_tag.algorithm,
            self.source_projection_digest.algorithm,
            self.preview_projection_digest.algorithm,
        }
        if (
            self.delta.run_id != self.run_id
            or len(digest_algorithms) != 1
            or tuple(item.handle for item in self.memory_id_assignments) != expected_handles
            or len({item.memory_id for item in self.memory_id_assignments})
            != len(self.memory_id_assignments)
            or any(
                record.run_id != self.run_id
                or record.validity is not ValidityState.ACTIVE
                or record.updated_at > self.delta.created_at
                or (record.expires_at is not None and record.expires_at <= self.delta.created_at)
                for record in self.active_bank
            )
            or len({record.memory_id for record in self.active_bank}) != len(self.active_bank)
            or self.active_bank != active_sort
        ):
            raise ValueError("materialized bank operations are inconsistent")
        values = self.model_dump(mode="json", exclude={"materialization_digest"})
        if self.materialization_digest != _materialization_digest(values):
            raise ValueError("materialized bank operations digest does not match")
        return self


def _fail(reason: MaterializationFailureReason) -> NoReturn:
    raise MemoryOperationMaterializationError(reason) from None


def _validated_request(value: object) -> OperationMaterializationRequest:
    if type(value) is not OperationMaterializationRequest:
        _fail(MaterializationFailureReason.INVALID_INPUT)
    try:
        return OperationMaterializationRequest.model_validate_json(
            value.model_dump_json(warnings=False)
        )
    except Exception:
        _fail(MaterializationFailureReason.INVALID_INPUT)


def validated_operation_materialization_request(
    value: object,
) -> OperationMaterializationRequest:
    """Return an exact, recursively revalidated materialization request."""

    return _validated_request(value)


def validated_materialized_bank_operations(value: object) -> MaterializedBankOperations:
    """Return an exact result only when its internal digest and bank are valid."""

    if type(value) is not MaterializedBankOperations:
        _fail(MaterializationFailureReason.RESULT_MISMATCH)
    try:
        return MaterializedBankOperations.model_validate_json(value.model_dump_json(warnings=False))
    except Exception:
        _fail(MaterializationFailureReason.RESULT_MISMATCH)


def _proposal_effects(
    request: OperationMaterializationRequest,
) -> tuple[
    tuple[MemoryCreate, ...],
    tuple[MemoryInvalidation, ...],
    MemoryCreate | None,
    tuple[MemoryIdAssignment, ...],
]:
    creates: list[MemoryCreate] = []
    invalidations: list[MemoryInvalidation] = []
    status: MemoryCreate | None = None
    assignment_by_handle: dict[str, UUID] = {}
    assigned = iter(request.assigned_memory_ids)
    for ordinal, operation in enumerate(request.proposal.operations, start=1):
        if isinstance(operation, DeleteMemory):
            invalidations.append(
                MemoryInvalidation(
                    memory_id=operation.memory_id,
                    expected_revision=operation.expected_revision,
                    reason_code=ReasonCode.CONFLICT,
                )
            )
            continue
        handle = operation_handle(ordinal)
        assignment_by_handle[handle] = next(assigned)
        create = MemoryCreate(
            handle=handle,
            kind=(
                MemoryKind.PRIVATE_STATUS
                if isinstance(operation, UpdatePrivateStatus)
                else (
                    MemoryKind.KNOWLEDGE
                    if isinstance(operation, SaveKnowledge)
                    else MemoryKind.PROCEDURAL
                )
            ),
            content=operation.content,
            provenance=operation.evidence,
            confidence=operation.confidence,
            trust_label=TrustLabel.UNTRUSTED_MODEL_OUTPUT,
            expires_at=None,
        )
        if isinstance(operation, UpdatePrivateStatus):
            status = create
        else:
            creates.append(create)
    handles = tuple(item.handle for item in creates)
    if status is not None:
        handles += (status.handle,)
    assignments = tuple(
        MemoryIdAssignment(handle=handle, memory_id=assignment_by_handle[handle])
        for handle in handles
    )
    return tuple(creates), tuple(invalidations), status, assignments


def validated_materialized_bank_operations_for_request(
    request: object,
    result: object,
) -> MaterializedBankOperations:
    """Structurally bind a result to a request without re-reading repository state."""

    checked_request = _validated_request(request)
    checked_result = validated_materialized_bank_operations(result)
    cycle_receipt = checked_request.cycle_receipt
    cycle = cycle_receipt.cycle
    creates, invalidations, status, assignments = _proposal_effects(checked_request)
    replacement = checked_result.delta.private_status_replacement
    if (
        checked_result.run_id != cycle.run_id
        or checked_result.source_cycle_id != cycle.cycle_id
        or checked_result.source_last_event_sequence != cycle.last_event_sequence
        or checked_result.source_ledger_position != cycle_receipt.ledger_position
        or checked_result.source_ingestion_cursor != cycle.last_event_sequence
        or checked_result.source_memory_cursor != cycle.first_event_sequence - 1
        or checked_result.source_record_tag != cycle_receipt.record_tag
        or checked_result.source_chain_tag != cycle_receipt.chain_tag
        or checked_result.source_operations_digest
        != source_operations_digest(checked_request.proposal)
        or checked_result.delta.delta_id != checked_request.delta_id
        or checked_result.delta.run_id != cycle.run_id
        or checked_result.delta.created_at != checked_request.created_at
        or checked_result.delta.creates != creates
        or checked_result.delta.updates
        or checked_result.delta.invalidations != invalidations
        or checked_result.memory_id_assignments != assignments
        or (status is None) != (replacement is None)
        or (status is not None and (replacement is None or replacement.replacement != status))
    ):
        _fail(MaterializationFailureReason.RESULT_MISMATCH)
    return checked_result


def _validated_snapshot(value: object) -> MemorySnapshot:
    if type(value) is not MemorySnapshot:
        _fail(MaterializationFailureReason.REPOSITORY_UNAVAILABLE)
    try:
        return MemorySnapshot.model_validate_json(value.model_dump_json(warnings=False))
    except Exception:
        _fail(MaterializationFailureReason.REPOSITORY_UNAVAILABLE)


def _validated_preview(value: object) -> MemoryDeltaPreview:
    if type(value) is not MemoryDeltaPreview:
        _fail(MaterializationFailureReason.PREVIEW_REJECTED)
    try:
        return MemoryDeltaPreview.model_validate_json(value.model_dump_json(warnings=False))
    except Exception:
        _fail(MaterializationFailureReason.PREVIEW_REJECTED)


def _active_preview_bank(
    preview: MemoryDeltaPreview,
    *,
    created_at: datetime,
) -> tuple[MemoryRecord, ...]:
    return tuple(
        record
        for record in preview.records
        if record.validity is ValidityState.ACTIVE
        and record.updated_at <= created_at
        and (record.expires_at is None or record.expires_at > created_at)
    )


async def verified_materialized_bank_operations_for_request(
    request: object,
    result: object,
    *,
    repository: RunRepository,
) -> MaterializedBankOperations:
    """Re-preview a structural result through the trusted repository contract."""

    checked_request = _validated_request(request)
    checked_result = validated_materialized_bank_operations_for_request(
        checked_request,
        result,
    )
    command = PreviewMemoryDelta(
        schema_version="memory-delta-preview-command/v1",
        run_id=checked_result.run_id,
        expected_ledger_position=checked_result.source_ledger_position,
        expected_ingestion_cursor=checked_result.source_ingestion_cursor,
        expected_memory_cursor=checked_result.source_memory_cursor,
        expected_projection_digest=checked_result.source_projection_digest,
        last_event_sequence=checked_result.source_last_event_sequence,
        delta=checked_result.delta,
        memory_id_assignments=checked_result.memory_id_assignments,
    )
    try:
        preview = _validated_preview(await repository.preview_memory_delta(command))
    except PreviewConflictError:
        _fail(MaterializationFailureReason.SOURCE_CONFLICT)
    except (
        CrossRunReferenceError,
        InvalidRecordError,
        InvalidRecordTypeError,
        ProjectionInvariantError,
        RevisionConflictError,
        UnsafeRecordContentError,
    ):
        _fail(MaterializationFailureReason.PREVIEW_REJECTED)
    except RepositoryError:
        _fail(MaterializationFailureReason.REPOSITORY_UNAVAILABLE)
    except MemoryOperationMaterializationError:
        raise
    except Exception:
        _fail(MaterializationFailureReason.PREVIEW_REJECTED)
    if (
        preview.run_id != checked_result.run_id
        or preview.command_digest != command.command_digest
        or preview.source_ledger_position != checked_result.source_ledger_position
        or preview.source_ingestion_cursor != checked_result.source_ingestion_cursor
        or preview.source_memory_cursor != checked_result.source_memory_cursor
        or preview.source_projection_digest != checked_result.source_projection_digest
        or preview.preview_projection_digest != checked_result.preview_projection_digest
        or _active_preview_bank(preview, created_at=checked_result.delta.created_at)
        != checked_result.active_bank
    ):
        _fail(MaterializationFailureReason.RESULT_MISMATCH)
    return checked_result


def _authoritative_source(
    request: OperationMaterializationRequest,
    ledger: tuple[LedgerEntry, ...],
    snapshot: MemorySnapshot,
) -> None:
    receipt = request.cycle_receipt
    cycle = receipt.cycle
    if (
        cycle.state is not CycleState.RUNNING
        or cycle.run_id != snapshot.run_id
        or len(ledger) != receipt.ledger_position
        or not ledger
        or any(
            entry.run_id != cycle.run_id or entry.position != position
            for position, entry in enumerate(ledger, start=1)
        )
        or ledger[-1].position != receipt.ledger_position
        or ledger[-1].record != cycle
        or ledger[-1].record_tag != receipt.record_tag
        or ledger[-1].chain_tag != receipt.chain_tag
        or snapshot.ledger_position != receipt.ledger_position
        or snapshot.ingestion_cursor != cycle.last_event_sequence
        or snapshot.memory_cursor != cycle.first_event_sequence - 1
    ):
        _fail(MaterializationFailureReason.SOURCE_CONFLICT)


def _revision_boundaries(ledger: tuple[LedgerEntry, ...]) -> dict[tuple[UUID, int], int]:
    boundaries: dict[tuple[UUID, int], int] = {}
    try:
        for entry in ledger:
            record = entry.record
            if type(record) is not CycleRecord or record.state is not CycleState.COMMITTED:
                continue
            delta = record.validated_delta
            if delta is None:
                raise ValueError
            assignments = {item.handle: item.memory_id for item in record.memory_id_assignments}
            for create in delta.creates:
                key = (assignments[create.handle], 1)
                if key in boundaries:
                    raise ValueError
                boundaries[key] = record.last_event_sequence
            for update in delta.updates:
                key = (update.memory_id, update.expected_revision + 1)
                if key in boundaries:
                    raise ValueError
                boundaries[key] = record.last_event_sequence
            for invalidation in delta.invalidations:
                key = (invalidation.memory_id, invalidation.expected_revision + 1)
                if key in boundaries:
                    raise ValueError
                boundaries[key] = record.last_event_sequence
            replacement = delta.private_status_replacement
            if replacement is not None:
                if replacement.expected_memory_id is not None:
                    assert replacement.expected_revision is not None
                    key = (replacement.expected_memory_id, replacement.expected_revision + 1)
                    if key in boundaries:
                        raise ValueError
                    boundaries[key] = record.last_event_sequence
                key = (assignments[replacement.replacement.handle], 1)
                if key in boundaries:
                    raise ValueError
                boundaries[key] = record.last_event_sequence
    except Exception:
        _fail(MaterializationFailureReason.SOURCE_CONFLICT)
    return boundaries


def _source_indexes(
    ledger: tuple[LedgerEntry, ...],
    snapshot: MemorySnapshot,
) -> tuple[dict[UUID, TraceEvent], dict[UUID, MemoryRecord], dict[tuple[UUID, int], int]]:
    events: dict[UUID, TraceEvent] = {}
    try:
        for entry in ledger:
            if type(entry.record) is TraceEvent:
                event = entry.record
                if event.event_id in events or event.run_id != snapshot.run_id:
                    raise ValueError
                events[event.event_id] = event
        memories = {record.memory_id: record for record in snapshot.records}
        if len(memories) != len(snapshot.records) or any(
            record.run_id != snapshot.run_id for record in snapshot.records
        ):
            raise ValueError
        boundaries = _revision_boundaries(ledger)
        if any(
            (record.memory_id, record.revision) not in boundaries for record in snapshot.records
        ):
            raise ValueError
    except MemoryOperationMaterializationError:
        raise
    except Exception:
        _fail(MaterializationFailureReason.SOURCE_CONFLICT)
    return events, memories, boundaries


def _resolve_pointer(root: object, pointer: str) -> str | None:
    value = root
    try:
        for segment in pointer.split("/")[1:]:
            decoded = segment.replace("~1", "/").replace("~0", "~")
            if isinstance(value, Mapping):
                if decoded not in value:
                    return None
                value = value[decoded]
            elif isinstance(value, (list, tuple)):
                if _POINTER_INDEX.fullmatch(decoded) is None:
                    return None
                index = int(decoded)
                if index >= len(value):
                    return None
                value = value[index]
            else:
                return None
    except Exception:
        return None
    return value if type(value) is str else None


def _select_span(text: str, reference: EvidenceReference) -> str | None:
    try:
        encoded = text.encode("utf-8", errors="strict")
        if reference.span is None:
            selected = text
        elif reference.span.end_byte <= len(encoded):
            selected = encoded[reference.span.start_byte : reference.span.end_byte].decode(
                "utf-8", errors="strict"
            )
        else:
            return None
    except Exception:
        return None
    return selected if selected else None


def _validate_memory_reference(
    memory_id: UUID,
    revision: int,
    *,
    memories: Mapping[UUID, MemoryRecord],
    boundaries: Mapping[tuple[UUID, int], int],
    working_active: set[UUID],
    last_event_sequence: int,
    created_at: datetime,
) -> MemoryRecord:
    record = memories.get(memory_id)
    if record is None:
        _fail(MaterializationFailureReason.REFERENCE_MISSING)
    assert record is not None
    if record.revision != revision:
        _fail(MaterializationFailureReason.REFERENCE_STALE)
    boundary = boundaries.get((record.memory_id, record.revision))
    if boundary is None or boundary > last_event_sequence or record.updated_at > created_at:
        _fail(MaterializationFailureReason.REFERENCE_FUTURE)
    if record.validity is ValidityState.EXPIRED or (
        record.expires_at is not None and record.expires_at <= created_at
    ):
        _fail(MaterializationFailureReason.REFERENCE_EXPIRED)
    if record.validity is not ValidityState.ACTIVE or record.memory_id not in working_active:
        _fail(MaterializationFailureReason.REFERENCE_INACTIVE)
    return record


def _validate_evidence(
    evidence: tuple[EvidenceReference, ...],
    *,
    events: Mapping[UUID, TraceEvent],
    memories: Mapping[UUID, MemoryRecord],
    boundaries: Mapping[tuple[UUID, int], int],
    working_active: set[UUID],
    last_event_sequence: int,
    created_at: datetime,
) -> None:
    for reference in evidence:
        selected: str | None = None
        if reference.source is EvidenceSource.EVENT:
            event = events.get(reference.source_id)
            if event is None:
                _fail(MaterializationFailureReason.REFERENCE_MISSING)
            assert event is not None
            if event.sequence > last_event_sequence or event.timestamp > created_at:
                _fail(MaterializationFailureReason.REFERENCE_FUTURE)
            if reference.field_path.startswith("/payload/"):
                selected = _resolve_pointer(
                    {"payload": event.payload},
                    reference.field_path,
                )
        else:
            assert reference.revision is not None
            record = _validate_memory_reference(
                reference.source_id,
                reference.revision,
                memories=memories,
                boundaries=boundaries,
                working_active=working_active,
                last_event_sequence=last_event_sequence,
                created_at=created_at,
            )
            if reference.field_path == "/content":
                selected = record.content
        if selected is None or _select_span(selected, reference) is None:
            _fail(MaterializationFailureReason.REFERENCE_INVALID)


def _preflight_operations(
    request: OperationMaterializationRequest,
    memories: Mapping[UUID, MemoryRecord],
    current_private_id: UUID | None,
) -> dict[int, UUID]:
    write_ordinals = tuple(
        ordinal
        for ordinal, operation in enumerate(request.proposal.operations, start=1)
        if type(operation) in (UpdatePrivateStatus, SaveKnowledge, SaveProcedural)
    )
    assignments = dict(zip(write_ordinals, request.assigned_memory_ids, strict=True))
    assigned_ids = set(request.assigned_memory_ids)
    if assigned_ids.intersection(memories):
        _fail(MaterializationFailureReason.ASSIGNMENT_CONFLICT)

    delete_targets: set[UUID] = set()
    has_status_replacement = False
    for operation in request.proposal.operations:
        if isinstance(operation, DeleteMemory):
            delete_targets.add(operation.memory_id)
            if operation.memory_id in assigned_ids:
                _fail(MaterializationFailureReason.OPERATION_CONFLICT)
            continue
        if isinstance(operation, UpdatePrivateStatus):
            has_status_replacement = True
        for reference in operation.evidence:
            if reference.source is EvidenceSource.MEMORY and reference.source_id in assigned_ids:
                _fail(MaterializationFailureReason.OPERATION_CONFLICT)
    if has_status_replacement and current_private_id in delete_targets:
        _fail(MaterializationFailureReason.OPERATION_CONFLICT)
    return assignments


def _compile_delta(
    request: OperationMaterializationRequest,
    *,
    events: Mapping[UUID, TraceEvent],
    memories: Mapping[UUID, MemoryRecord],
    boundaries: Mapping[tuple[UUID, int], int],
    current_private: MemoryRecord | None,
    assignments_by_ordinal: Mapping[int, UUID],
) -> tuple[MemoryDelta, tuple[MemoryIdAssignment, ...]]:
    creates: list[MemoryCreate] = []
    invalidations: list[MemoryInvalidation] = []
    replacement: PrivateStatusReplacement | None = None
    assignment_by_handle: dict[str, UUID] = {}
    working_active = {
        record.memory_id for record in memories.values() if record.validity is ValidityState.ACTIVE
    }
    cycle = request.cycle_receipt.cycle

    for ordinal, operation in enumerate(request.proposal.operations, start=1):
        if isinstance(operation, DeleteMemory):
            _validate_memory_reference(
                operation.memory_id,
                operation.expected_revision,
                memories=memories,
                boundaries=boundaries,
                working_active=working_active,
                last_event_sequence=cycle.last_event_sequence,
                created_at=request.created_at,
            )
            invalidations.append(
                MemoryInvalidation(
                    memory_id=operation.memory_id,
                    expected_revision=operation.expected_revision,
                    reason_code=ReasonCode.CONFLICT,
                )
            )
            working_active.remove(operation.memory_id)
            continue

        _validate_evidence(
            operation.evidence,
            events=events,
            memories=memories,
            boundaries=boundaries,
            working_active=working_active,
            last_event_sequence=cycle.last_event_sequence,
            created_at=request.created_at,
        )
        handle = operation_handle(ordinal)
        assignment_by_handle[handle] = assignments_by_ordinal[ordinal]
        if isinstance(operation, UpdatePrivateStatus):
            create = MemoryCreate(
                handle=handle,
                kind=MemoryKind.PRIVATE_STATUS,
                content=operation.content,
                provenance=operation.evidence,
                confidence=operation.confidence,
                trust_label=TrustLabel.UNTRUSTED_MODEL_OUTPUT,
                expires_at=None,
            )
            replacement = PrivateStatusReplacement(
                expected_memory_id=(None if current_private is None else current_private.memory_id),
                expected_revision=(None if current_private is None else current_private.revision),
                replacement=create,
            )
            if current_private is not None:
                working_active.discard(current_private.memory_id)
        else:
            creates.append(
                MemoryCreate(
                    handle=handle,
                    kind=(
                        MemoryKind.KNOWLEDGE
                        if type(operation) is SaveKnowledge
                        else MemoryKind.PROCEDURAL
                    ),
                    content=operation.content,
                    provenance=operation.evidence,
                    confidence=operation.confidence,
                    trust_label=TrustLabel.UNTRUSTED_MODEL_OUTPUT,
                    expires_at=None,
                )
            )

    try:
        delta = MemoryDelta(
            delta_id=request.delta_id,
            run_id=cycle.run_id,
            creates=tuple(creates),
            updates=(),
            invalidations=tuple(invalidations),
            private_status_replacement=replacement,
            created_at=request.created_at,
        )
        handles = tuple(item.handle for item in delta.creates)
        if replacement is not None:
            handles += (replacement.replacement.handle,)
        normalized = tuple(
            MemoryIdAssignment(handle=handle, memory_id=assignment_by_handle[handle])
            for handle in handles
        )
    except Exception:
        _fail(MaterializationFailureReason.OPERATION_CONFLICT)
    return delta, normalized


async def materialize_bank_operations(
    request: object,
    *,
    repository: RunRepository,
) -> MaterializedBankOperations:
    """Compile one ordered proposal against a pinned, authoritative cycle.

    The repository is part of the trusted computing base: its reads must authenticate
    their ledger, as both bundled backends do.
    """

    validated = _validated_request(request)
    cycle = validated.cycle_receipt.cycle
    try:
        ledger_value = await repository.ledger(cycle.run_id)
        snapshot_value = await repository.snapshot(cycle.run_id)
    except Exception:
        _fail(MaterializationFailureReason.REPOSITORY_UNAVAILABLE)
    if type(ledger_value) is not tuple or any(
        type(item) is not LedgerEntry for item in ledger_value
    ):
        _fail(MaterializationFailureReason.REPOSITORY_UNAVAILABLE)
    ledger = ledger_value
    snapshot = _validated_snapshot(snapshot_value)
    _authoritative_source(validated, ledger, snapshot)
    if any(
        type(entry.record) is CycleRecord
        and entry.record.state is CycleState.COMMITTED
        and entry.record.validated_delta is not None
        and entry.record.validated_delta.delta_id == validated.delta_id
        for entry in ledger
    ):
        _fail(MaterializationFailureReason.IDENTITY_CONFLICT)
    events, memories, boundaries = _source_indexes(ledger, snapshot)
    private_records = tuple(
        record
        for record in memories.values()
        if record.kind is MemoryKind.PRIVATE_STATUS and record.validity is ValidityState.ACTIVE
    )
    if len(private_records) > 1:
        _fail(MaterializationFailureReason.SOURCE_CONFLICT)
    current_private = private_records[0] if private_records else None
    assignments_by_ordinal = _preflight_operations(
        validated,
        memories,
        None if current_private is None else current_private.memory_id,
    )
    delta, assignments = _compile_delta(
        validated,
        events=events,
        memories=memories,
        boundaries=boundaries,
        current_private=current_private,
        assignments_by_ordinal=assignments_by_ordinal,
    )

    try:
        preview_command = PreviewMemoryDelta(
            schema_version="memory-delta-preview-command/v1",
            run_id=cycle.run_id,
            expected_ledger_position=snapshot.ledger_position,
            expected_ingestion_cursor=snapshot.ingestion_cursor,
            expected_memory_cursor=snapshot.memory_cursor,
            expected_projection_digest=snapshot.projection_digest,
            last_event_sequence=cycle.last_event_sequence,
            delta=delta,
            memory_id_assignments=assignments,
        )
        preview_value = await repository.preview_memory_delta(preview_command)
    except PreviewConflictError:
        _fail(MaterializationFailureReason.SOURCE_CONFLICT)
    except (
        CrossRunReferenceError,
        InvalidRecordError,
        InvalidRecordTypeError,
        ProjectionInvariantError,
        RevisionConflictError,
        UnsafeRecordContentError,
    ):
        _fail(MaterializationFailureReason.PREVIEW_REJECTED)
    except RepositoryError:
        _fail(MaterializationFailureReason.REPOSITORY_UNAVAILABLE)
    except Exception:
        _fail(MaterializationFailureReason.PREVIEW_REJECTED)
    preview = _validated_preview(preview_value)
    if (
        preview.run_id != cycle.run_id
        or preview.command_digest != preview_command.command_digest
        or preview.source_ledger_position != snapshot.ledger_position
        or preview.source_ingestion_cursor != snapshot.ingestion_cursor
        or preview.source_memory_cursor != snapshot.memory_cursor
        or preview.source_projection_digest != snapshot.projection_digest
    ):
        _fail(MaterializationFailureReason.PREVIEW_REJECTED)

    active_bank = _active_preview_bank(preview, created_at=validated.created_at)
    operations_digest = source_operations_digest(validated.proposal)
    result_values: dict[str, object] = {
        "schema_version": MATERIALIZATION_RESULT_SCHEMA_VERSION,
        "run_id": cycle.run_id,
        "source_cycle_id": cycle.cycle_id,
        "source_last_event_sequence": cycle.last_event_sequence,
        "source_ledger_position": validated.cycle_receipt.ledger_position,
        "source_ingestion_cursor": snapshot.ingestion_cursor,
        "source_memory_cursor": snapshot.memory_cursor,
        "source_record_tag": validated.cycle_receipt.record_tag,
        "source_chain_tag": validated.cycle_receipt.chain_tag,
        "source_projection_digest": snapshot.projection_digest,
        "source_operations_digest": operations_digest,
        "delta": delta,
        "memory_id_assignments": assignments,
        "active_bank": active_bank,
        "preview_projection_digest": preview.preview_projection_digest,
    }
    try:
        result = MaterializedBankOperations(
            schema_version=MATERIALIZATION_RESULT_SCHEMA_VERSION,
            run_id=cycle.run_id,
            source_cycle_id=cycle.cycle_id,
            source_last_event_sequence=cycle.last_event_sequence,
            source_ledger_position=validated.cycle_receipt.ledger_position,
            source_ingestion_cursor=snapshot.ingestion_cursor,
            source_memory_cursor=snapshot.memory_cursor,
            source_record_tag=validated.cycle_receipt.record_tag,
            source_chain_tag=validated.cycle_receipt.chain_tag,
            source_projection_digest=snapshot.projection_digest,
            source_operations_digest=operations_digest,
            delta=delta,
            memory_id_assignments=assignments,
            active_bank=active_bank,
            preview_projection_digest=preview.preview_projection_digest,
            materialization_digest=_materialization_digest(result_values),
        )
    except Exception:
        _fail(MaterializationFailureReason.PREVIEW_REJECTED)
    return validated_materialized_bank_operations_for_request(validated, result)


__all__ = [
    "MATERIALIZATION_REQUEST_SCHEMA_VERSION",
    "MATERIALIZATION_RESULT_SCHEMA_VERSION",
    "OPERATION_HANDLE_PREFIX",
    "MaterializationFailureReason",
    "MaterializedBankOperations",
    "MemoryOperationMaterializationError",
    "OperationMaterializationRequest",
    "materialize_bank_operations",
    "operation_handle",
    "source_operations_digest",
    "validated_materialized_bank_operations",
    "validated_materialized_bank_operations_for_request",
    "validated_operation_materialization_request",
    "verified_materialized_bank_operations_for_request",
]
