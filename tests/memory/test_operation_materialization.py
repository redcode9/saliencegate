from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from datetime import timedelta
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError
from tests.repository.conformance import (
    CYCLE_NOW,
    CYCLE_RUN_ID,
    RUN_B,
    CycleContext,
    advance_cycle_to_running,
    begin_cycle_context,
    cycle_commit_command,
    cycle_grounding_config,
    event_draft,
)

import saliencegate.memory.materialize as materialize_module
from saliencegate.domain import (
    BudgetAmounts,
    DeliveryTarget,
    EvidenceReference,
    EvidenceSource,
    InvocationDecision,
    MemoryCreate,
    MemoryDelta,
    MemoryIdAssignment,
    MemoryInvalidation,
    MemoryKind,
    MemoryUpdate,
    PayloadDigestAlgorithm,
    PrivateStatusReplacement,
    ReasonCode,
    TextSpan,
    TrustLabel,
    ValidityState,
)
from saliencegate.intervention.grounding import resolve_grounding_configuration
from saliencegate.memory.materialize import (
    MATERIALIZATION_REQUEST_SCHEMA_VERSION,
    MaterializationFailureReason,
    MaterializedBankOperations,
    MemoryOperationMaterializationError,
    OperationMaterializationRequest,
    materialize_bank_operations,
    operation_handle,
    validated_materialized_bank_operations,
    validated_materialized_bank_operations_for_request,
    validated_operation_materialization_request,
    verified_materialized_bank_operations_for_request,
)
from saliencegate.memory.proposals import (
    MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
    BankOperationsProposal,
    DeleteMemory,
    SaveKnowledge,
    SaveProcedural,
    UpdatePrivateStatus,
)
from saliencegate.ports.repository import (
    BeginCycle,
    CycleReceipt,
    MemoryDeltaPreview,
    MemoryQuery,
    PreviewConflictError,
    ProjectionInvariantError,
    RepositoryError,
    ReserveCycle,
    RunRepository,
    StartCycle,
)
from saliencegate.repository.memory import MemoryRunRepository
from saliencegate.repository.sqlite import SQLiteRunRepository
from saliencegate.security import InstallationKey

KEY = InstallationKey(b"m" * 32)
SEED_KNOWLEDGE_ID = UUID("00000000-0000-4000-8000-000000000b01")
SEED_PROCEDURAL_ID = UUID("00000000-0000-4000-8000-000000000b02")
SEED_STATUS_ID = UUID("00000000-0000-4000-8000-000000000b03")
SEED_EXPIRING_ID = UUID("00000000-0000-4000-8000-000000000b04")
SEED_INACTIVE_ID = UUID("00000000-0000-4000-8000-000000000b05")
NEW_STATUS_ID = UUID("00000000-0000-4000-8000-000000000b11")
NEW_PROCEDURAL_ID = UUID("00000000-0000-4000-8000-000000000b12")
NEW_KNOWLEDGE_ID = UUID("00000000-0000-4000-8000-000000000b13")
DELTA_ID = UUID("00000000-0000-4000-8000-000000000b20")
MISSING_ID = UUID("00000000-0000-4000-8000-000000000bff")
CROSS_RUN_EVENT_ID = UUID("00000000-0000-4000-8000-000000000bfe")


def _id_factory() -> Callable[[], UUID]:
    identifiers = itertools.count(0xC00)
    return lambda: UUID(f"00000000-0000-4000-8000-{next(identifiers):012x}")


@pytest.fixture(params=("memory", "sqlite"))
def repository(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> Iterator[RunRepository]:
    if request.param == "memory":
        yield MemoryRunRepository(installation_key=KEY, id_factory=_id_factory())
        return
    sqlite = SQLiteRunRepository(
        tmp_path / "operation-materialization.sqlite3",
        installation_key=KEY,
        id_factory=_id_factory(),
    )
    try:
        yield sqlite
    finally:
        sqlite.close()


def _event_evidence(context: CycleContext) -> EvidenceReference:
    return EvidenceReference(
        source=EvidenceSource.EVENT,
        source_id=context.event.event_id,
        field_path="/payload/message",
    )


def _memory_evidence(memory_id: UUID, revision: int = 1) -> EvidenceReference:
    return EvidenceReference(
        source=EvidenceSource.MEMORY,
        source_id=memory_id,
        revision=revision,
        field_path="/content",
    )


async def _running_cycle_with_bank(
    repository: RunRepository,
) -> tuple[CycleContext, CycleReceipt]:
    first, _reserved, _running = await advance_cycle_to_running(repository)
    evidence = (_event_evidence(first),)
    seed_delta = MemoryDelta(
        delta_id=UUID("00000000-0000-4000-8000-000000000b00"),
        run_id=CYCLE_RUN_ID,
        creates=(
            MemoryCreate(
                handle="seed-knowledge",
                kind=MemoryKind.KNOWLEDGE,
                content="Python 3.11 is the minimum runtime.",
                provenance=evidence,
                confidence=1.0,
                trust_label=TrustLabel.UNTRUSTED_MODEL_OUTPUT,
            ),
            MemoryCreate(
                handle="seed-procedural",
                kind=MemoryKind.PROCEDURAL,
                content="Run the narrow test before the complete gate.",
                provenance=evidence,
                confidence=0.9,
                trust_label=TrustLabel.UNTRUSTED_MODEL_OUTPUT,
            ),
            MemoryCreate(
                handle="seed-expiring",
                kind=MemoryKind.KNOWLEDGE,
                content="This fact expires before the second materialization.",
                provenance=evidence,
                confidence=0.8,
                trust_label=TrustLabel.UNTRUSTED_MODEL_OUTPUT,
                expires_at=first.commit_time + timedelta(seconds=15),
            ),
            MemoryCreate(
                handle="seed-inactive",
                kind=MemoryKind.KNOWLEDGE,
                content="This record is invalidated before materialization.",
                provenance=evidence,
                confidence=0.7,
                trust_label=TrustLabel.UNTRUSTED_MODEL_OUTPUT,
            ),
        ),
        updates=(
            MemoryUpdate(
                memory_id=SEED_EXPIRING_ID,
                expected_revision=1,
                confidence=0.7,
            ),
        ),
        invalidations=(
            MemoryInvalidation(
                memory_id=SEED_INACTIVE_ID,
                expected_revision=1,
                reason_code=ReasonCode.CONFLICT,
            ),
        ),
        private_status_replacement=PrivateStatusReplacement(
            replacement=MemoryCreate(
                handle="seed-status",
                kind=MemoryKind.PRIVATE_STATUS,
                content="The implementation is in progress.",
                provenance=evidence,
                confidence=1.0,
                trust_label=TrustLabel.UNTRUSTED_MODEL_OUTPUT,
            )
        ),
        created_at=first.commit_time,
    )
    await repository.commit_cycle(
        cycle_commit_command(
            first,
            delta=seed_delta,
            assignments=(
                MemoryIdAssignment(handle="seed-knowledge", memory_id=SEED_KNOWLEDGE_ID),
                MemoryIdAssignment(handle="seed-procedural", memory_id=SEED_PROCEDURAL_ID),
                MemoryIdAssignment(handle="seed-expiring", memory_id=SEED_EXPIRING_ID),
                MemoryIdAssignment(handle="seed-inactive", memory_id=SEED_INACTIVE_ID),
                MemoryIdAssignment(handle="seed-status", memory_id=SEED_STATUS_ID),
            ),
        )
    )
    second = await begin_cycle_context(repository, ordinal=2)
    await repository.reserve_cycle(second.reserve)
    running = await repository.mark_cycle_running(second.start)
    return second, running


def _proposal(*operations: object) -> BankOperationsProposal:
    return BankOperationsProposal(
        schema_version=MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
        operations=operations,  # type: ignore[arg-type]
    )


def _request(
    context: CycleContext,
    running: CycleReceipt,
    proposal: BankOperationsProposal,
    assigned_memory_ids: tuple[UUID, ...],
    *,
    delta_id: UUID = DELTA_ID,
) -> OperationMaterializationRequest:
    return OperationMaterializationRequest(
        schema_version=MATERIALIZATION_REQUEST_SCHEMA_VERSION,
        cycle_receipt=running,
        proposal=proposal,
        delta_id=delta_id,
        created_at=context.commit_time,
        assigned_memory_ids=assigned_memory_ids,
    )


async def _repository_state(repository: RunRepository) -> tuple[object, ...]:
    return (
        await repository.ledger(CYCLE_RUN_ID),
        await repository.ledger_head(CYCLE_RUN_ID),
        await repository.snapshot(CYCLE_RUN_ID),
        await repository.budget_snapshot(CYCLE_RUN_ID),
        await repository.search(MemoryQuery(run_id=CYCLE_RUN_ID, limit=100)),
    )


async def test_materializes_every_write_kind_in_runtime_owned_channels(
    repository: RunRepository,
) -> None:
    context, running = await _running_cycle_with_bank(repository)
    proposal = _proposal(
        UpdatePrivateStatus(
            operation="update_private_status",
            content="The materializer is ready for review.",
            evidence=(_event_evidence(context),),
            confidence=0.95,
        ),
        SaveProcedural(
            operation="save_procedural",
            content="Preserve source order before compiling channels.",
            evidence=(_memory_evidence(SEED_KNOWLEDGE_ID),),
            confidence=0.9,
        ),
        DeleteMemory(
            operation="delete_memory",
            memory_id=SEED_PROCEDURAL_ID,
            expected_revision=1,
        ),
        SaveKnowledge(
            operation="save_knowledge",
            content="Preview and commit share one semantic oracle.",
            evidence=(_event_evidence(context),),
            confidence=1.0,
        ),
    )
    request = _request(
        context,
        running,
        proposal,
        (NEW_STATUS_ID, NEW_PROCEDURAL_ID, NEW_KNOWLEDGE_ID),
    )
    before = await _repository_state(repository)

    result = await materialize_bank_operations(request, repository=repository)
    repeated = await materialize_bank_operations(request, repository=repository)

    assert result == repeated
    assert await _repository_state(repository) == before
    assert tuple(item.handle for item in result.delta.creates) == (
        "memory-edit-operation/v1/0002",
        "memory-edit-operation/v1/0004",
    )
    assert result.delta.updates == ()
    assert tuple(item.memory_id for item in result.delta.invalidations) == (SEED_PROCEDURAL_ID,)
    assert result.delta.invalidations[0].reason_code is ReasonCode.CONFLICT
    replacement = result.delta.private_status_replacement
    assert replacement is not None
    assert replacement.expected_memory_id == SEED_STATUS_ID
    assert replacement.expected_revision == 1
    assert replacement.replacement.handle == "memory-edit-operation/v1/0001"
    assert tuple((item.handle, item.memory_id) for item in result.memory_id_assignments) == (
        ("memory-edit-operation/v1/0002", NEW_PROCEDURAL_ID),
        ("memory-edit-operation/v1/0004", NEW_KNOWLEDGE_ID),
        ("memory-edit-operation/v1/0001", NEW_STATUS_ID),
    )
    assert all(
        create.trust_label is TrustLabel.UNTRUSTED_MODEL_OUTPUT and create.expires_at is None
        for create in (*result.delta.creates, replacement.replacement)
    )
    active_ids = {item.memory_id for item in result.active_bank}
    assert active_ids == {
        SEED_KNOWLEDGE_ID,
        NEW_STATUS_ID,
        NEW_PROCEDURAL_ID,
        NEW_KNOWLEDGE_ID,
    }
    assert all(item.validity is ValidityState.ACTIVE for item in result.active_bank)
    assert len(result.materialization_digest) == 64
    assert result.source_cycle_id == running.cycle.cycle_id
    assert result.source_ledger_position == running.ledger_position
    assert (
        MaterializedBankOperations.model_validate_json(result.model_dump_json(warnings=False))
        == result
    )
    assert validated_materialized_bank_operations_for_request(request, result) == result
    assert (
        await verified_materialized_bank_operations_for_request(
            request,
            result,
            repository=repository,
        )
        == result
    )

    tampered_digest = result.model_copy(update={"materialization_digest": "0" * 64})
    with pytest.raises(MemoryOperationMaterializationError) as tampered:
        validated_materialized_bank_operations(tampered_digest)
    assert tampered.value.reason is MaterializationFailureReason.RESULT_MISMATCH

    mismatched_request = _request(
        context,
        running,
        _proposal(*reversed(proposal.operations)),
        (NEW_KNOWLEDGE_ID, NEW_PROCEDURAL_ID, NEW_STATUS_ID),
        delta_id=DELTA_ID,
    )
    with pytest.raises(MemoryOperationMaterializationError) as mismatched:
        validated_materialized_bank_operations_for_request(mismatched_request, result)
    assert mismatched.value.reason is MaterializationFailureReason.RESULT_MISMATCH


async def test_noop_returns_the_current_unexpired_bank_without_mutation(
    repository: RunRepository,
) -> None:
    context, running = await _running_cycle_with_bank(repository)
    request = _request(context, running, _proposal(), ())
    before = await _repository_state(repository)

    result = await materialize_bank_operations(request, repository=repository)

    assert result.delta.creates == ()
    assert result.delta.invalidations == ()
    assert result.delta.private_status_replacement is None
    assert result.memory_id_assignments == ()
    assert {item.memory_id for item in result.active_bank} == {
        SEED_KNOWLEDGE_ID,
        SEED_PROCEDURAL_ID,
        SEED_STATUS_ID,
    }
    assert result.preview_projection_digest == result.source_projection_digest
    assert await _repository_state(repository) == before


async def test_source_order_controls_evidence_validity(
    repository: RunRepository,
) -> None:
    context, running = await _running_cycle_with_bank(repository)
    save = SaveKnowledge(
        operation="save_knowledge",
        content="The cited source was active when this operation ran.",
        evidence=(_memory_evidence(SEED_KNOWLEDGE_ID),),
        confidence=1.0,
    )
    delete = DeleteMemory(
        operation="delete_memory",
        memory_id=SEED_KNOWLEDGE_ID,
        expected_revision=1,
    )

    accepted = await materialize_bank_operations(
        _request(context, running, _proposal(save, delete), (NEW_KNOWLEDGE_ID,)),
        repository=repository,
    )
    assert tuple(item.handle for item in accepted.delta.creates) == (
        "memory-edit-operation/v1/0001",
    )

    with pytest.raises(MemoryOperationMaterializationError) as raised:
        await materialize_bank_operations(
            _request(
                context,
                running,
                _proposal(delete, save),
                (NEW_KNOWLEDGE_ID,),
                delta_id=UUID("00000000-0000-4000-8000-000000000b21"),
            ),
            repository=repository,
        )
    assert raised.value.reason is MaterializationFailureReason.REFERENCE_INACTIVE


async def test_status_replacement_supersedes_old_status_in_the_working_view(
    repository: RunRepository,
) -> None:
    context, running = await _running_cycle_with_bank(repository)
    cite_old_status = SaveKnowledge(
        operation="save_knowledge",
        content="The old status was still current when cited.",
        evidence=(_memory_evidence(SEED_STATUS_ID),),
        confidence=1.0,
    )
    replace = UpdatePrivateStatus(
        operation="update_private_status",
        content="The new private status supersedes the old revision.",
        evidence=(_event_evidence(context),),
        confidence=1.0,
    )

    accepted = await materialize_bank_operations(
        _request(
            context,
            running,
            _proposal(cite_old_status, replace),
            (NEW_KNOWLEDGE_ID, NEW_STATUS_ID),
        ),
        repository=repository,
    )
    assert accepted.delta.private_status_replacement is not None

    with pytest.raises(MemoryOperationMaterializationError) as raised:
        await materialize_bank_operations(
            _request(
                context,
                running,
                _proposal(replace, cite_old_status),
                (NEW_STATUS_ID, NEW_KNOWLEDGE_ID),
                delta_id=UUID("00000000-0000-4000-8000-000000000b22"),
            ),
            repository=repository,
        )
    assert raised.value.reason is MaterializationFailureReason.REFERENCE_INACTIVE


@pytest.mark.parametrize(
    ("operation", "reason"),
    (
        (
            DeleteMemory(
                operation="delete_memory",
                memory_id=SEED_KNOWLEDGE_ID,
                expected_revision=2,
            ),
            MaterializationFailureReason.REFERENCE_STALE,
        ),
        (
            DeleteMemory(
                operation="delete_memory",
                memory_id=MISSING_ID,
                expected_revision=1,
            ),
            MaterializationFailureReason.REFERENCE_MISSING,
        ),
        (
            SaveKnowledge(
                operation="save_knowledge",
                content="Do not retain expired evidence.",
                evidence=(_memory_evidence(SEED_EXPIRING_ID, revision=2),),
                confidence=1.0,
            ),
            MaterializationFailureReason.REFERENCE_EXPIRED,
        ),
        (
            SaveKnowledge(
                operation="save_knowledge",
                content="Do not retain invalidated evidence.",
                evidence=(_memory_evidence(SEED_INACTIVE_ID, revision=2),),
                confidence=1.0,
            ),
            MaterializationFailureReason.REFERENCE_INACTIVE,
        ),
    ),
)
async def test_rejects_stale_missing_and_expired_references_atomically(
    repository: RunRepository,
    operation: object,
    reason: MaterializationFailureReason,
) -> None:
    context, running = await _running_cycle_with_bank(repository)
    assigned = (NEW_KNOWLEDGE_ID,) if isinstance(operation, SaveKnowledge) else ()
    request = _request(context, running, _proposal(operation), assigned)
    before = await _repository_state(repository)

    with pytest.raises(MemoryOperationMaterializationError) as raised:
        await materialize_bank_operations(request, repository=repository)

    assert raised.value.reason is reason
    assert await _repository_state(repository) == before


async def test_rejects_same_phase_targets_and_private_status_overlap(
    repository: RunRepository,
) -> None:
    context, running = await _running_cycle_with_bank(repository)
    same_phase = _proposal(
        SaveKnowledge(
            operation="save_knowledge",
            content="A same-phase ID is not authoritative evidence.",
            evidence=(_memory_evidence(NEW_KNOWLEDGE_ID),),
            confidence=1.0,
        )
    )
    overlap = _proposal(
        DeleteMemory(
            operation="delete_memory",
            memory_id=SEED_STATUS_ID,
            expected_revision=1,
        ),
        UpdatePrivateStatus(
            operation="update_private_status",
            content="This replacement conflicts with explicit deletion.",
            evidence=(_event_evidence(context),),
            confidence=1.0,
        ),
    )

    for request in (
        _request(context, running, same_phase, (NEW_KNOWLEDGE_ID,)),
        _request(context, running, overlap, (NEW_STATUS_ID,)),
    ):
        with pytest.raises(MemoryOperationMaterializationError) as raised:
            await materialize_bank_operations(request, repository=repository)
        assert raised.value.reason is MaterializationFailureReason.OPERATION_CONFLICT


async def test_creates_first_private_status_without_model_authored_cas(
    repository: RunRepository,
) -> None:
    context, _reserved, running = await advance_cycle_to_running(repository)
    request = _request(
        context,
        running,
        _proposal(
            UpdatePrivateStatus(
                operation="update_private_status",
                content="The first private status is runtime-owned.",
                evidence=(_event_evidence(context),),
                confidence=1.0,
            )
        ),
        (NEW_STATUS_ID,),
    )

    result = await materialize_bank_operations(request, repository=repository)

    replacement = result.delta.private_status_replacement
    assert replacement is not None
    assert replacement.expected_memory_id is None
    assert replacement.expected_revision is None
    assert tuple(item.memory_id for item in result.active_bank) == (NEW_STATUS_ID,)


async def test_assignment_may_share_an_event_uuid_but_not_an_existing_memory_uuid(
    repository: RunRepository,
) -> None:
    context, running = await _running_cycle_with_bank(repository)
    proposal = _proposal(
        SaveKnowledge(
            operation="save_knowledge",
            content="UUID namespaces remain semantically distinct.",
            evidence=(_event_evidence(context),),
            confidence=1.0,
        )
    )

    accepted = await materialize_bank_operations(
        _request(context, running, proposal, (context.event.event_id,)),
        repository=repository,
    )
    assert accepted.memory_id_assignments[0].memory_id == context.event.event_id

    with pytest.raises(MemoryOperationMaterializationError) as collision:
        await materialize_bank_operations(
            _request(context, running, proposal, (SEED_KNOWLEDGE_ID,)),
            repository=repository,
        )
    assert collision.value.reason is MaterializationFailureReason.ASSIGNMENT_CONFLICT


@pytest.mark.parametrize(
    "reference",
    (
        EvidenceReference(
            source=EvidenceSource.EVENT,
            source_id=CROSS_RUN_EVENT_ID,
            field_path="/payload/message",
        ),
        EvidenceReference(
            source=EvidenceSource.EVENT,
            source_id=MISSING_ID,
            field_path="/payload/message",
        ),
    ),
)
async def test_cross_run_and_missing_event_references_share_a_safe_rejection(
    repository: RunRepository,
    reference: EvidenceReference,
) -> None:
    context, running = await _running_cycle_with_bank(repository)
    await repository.append(
        event_draft(run_id=RUN_B, source_event_id="cross-run-materialization"),
        event_id=CROSS_RUN_EVENT_ID,
    )
    proposal = _proposal(
        SaveKnowledge(
            operation="save_knowledge",
            content="Never resolve evidence through another run.",
            evidence=(reference,),
            confidence=1.0,
        )
    )

    with pytest.raises(MemoryOperationMaterializationError) as raised:
        await materialize_bank_operations(
            _request(context, running, proposal, (NEW_KNOWLEDGE_ID,)),
            repository=repository,
        )
    assert raised.value.reason is MaterializationFailureReason.REFERENCE_MISSING


@pytest.mark.parametrize(
    "reference",
    (
        EvidenceReference(
            source=EvidenceSource.EVENT,
            source_id=UUID("00000000-0000-4000-8000-000000000bee"),
            field_path="/payload/missing",
        ),
        EvidenceReference(
            source=EvidenceSource.EVENT,
            source_id=UUID("00000000-0000-4000-8000-000000000bee"),
            field_path="/payload/message",
            span=TextSpan(start_byte=999, end_byte=1_000),
        ),
    ),
)
async def test_invalid_pointer_or_out_of_bounds_span_is_rejected(
    repository: RunRepository,
    reference: EvidenceReference,
) -> None:
    context, running = await _running_cycle_with_bank(repository)
    event_reference = reference.model_copy(update={"source_id": context.event.event_id})
    proposal = _proposal(
        SaveKnowledge(
            operation="save_knowledge",
            content="Evidence must resolve to exact UTF-8 text.",
            evidence=(event_reference,),
            confidence=1.0,
        )
    )

    with pytest.raises(MemoryOperationMaterializationError) as raised:
        await materialize_bank_operations(
            _request(context, running, proposal, (NEW_KNOWLEDGE_ID,)),
            repository=repository,
        )
    assert raised.value.reason is MaterializationFailureReason.REFERENCE_INVALID


async def test_stale_cycle_anchor_and_forged_request_fail_value_free(
    repository: RunRepository,
) -> None:
    context, running = await _running_cycle_with_bank(repository)
    secret = "never-leak-this-model-content"
    valid = _request(
        context,
        running,
        _proposal(
            SaveKnowledge(
                operation="save_knowledge",
                content=secret,
                evidence=(_event_evidence(context),),
                confidence=1.0,
            )
        ),
        (NEW_KNOWLEDGE_ID,),
    )
    forged = valid.model_copy(update={"assigned_memory_ids": ()})
    before = await _repository_state(repository)

    with pytest.raises(MemoryOperationMaterializationError) as invalid:
        await materialize_bank_operations(forged, repository=repository)
    assert invalid.value.reason is MaterializationFailureReason.INVALID_INPUT

    await repository.append(
        event_draft(
            run_id=CYCLE_RUN_ID,
            source_event_id="materialization-anchor-advanced",
            timestamp=context.commit_time + timedelta(seconds=1),
        )
    )
    advanced = await _repository_state(repository)
    with pytest.raises(MemoryOperationMaterializationError) as stale:
        await materialize_bank_operations(valid, repository=repository)
    assert stale.value.reason is MaterializationFailureReason.SOURCE_CONFLICT
    assert await _repository_state(repository) == advanced
    for error in (invalid.value, stale.value):
        rendered = f"{error!r} {error}"
        assert secret not in rendered
        assert str(NEW_KNOWLEDGE_ID) not in rendered
    assert before != advanced


async def test_repository_content_rejection_is_not_misclassified_as_an_outage(
    repository: RunRepository,
) -> None:
    context, running = await _running_cycle_with_bank(repository)
    request = _request(
        context,
        running,
        _proposal(
            SaveKnowledge(
                operation="save_knowledge",
                content="api_key=sk-proj-abcdefghijklmnop",
                evidence=(_event_evidence(context),),
                confidence=1.0,
            )
        ),
        (NEW_KNOWLEDGE_ID,),
    )
    before = await _repository_state(repository)

    with pytest.raises(MemoryOperationMaterializationError) as raised:
        await materialize_bank_operations(request, repository=repository)

    assert raised.value.reason is MaterializationFailureReason.PREVIEW_REJECTED
    assert await _repository_state(repository) == before


async def test_rejects_reused_delta_identity_before_preview(
    repository: RunRepository,
) -> None:
    context, running = await _running_cycle_with_bank(repository)
    reused = _request(
        context,
        running,
        _proposal(),
        (),
        delta_id=UUID("00000000-0000-4000-8000-000000000b00"),
    )

    with pytest.raises(MemoryOperationMaterializationError) as raised:
        await materialize_bank_operations(reused, repository=repository)

    assert raised.value.reason is MaterializationFailureReason.IDENTITY_CONFLICT


async def test_public_request_and_result_boundaries_are_closed(
    repository: RunRepository,
) -> None:
    context, running = await _running_cycle_with_bank(repository)
    request = _request(context, running, _proposal(), ())
    result = await materialize_bank_operations(request, repository=repository)

    assert validated_operation_materialization_request(request) == request
    for ordinal in (True, 0, 65):
        with pytest.raises(ValueError, match="ordinal"):
            operation_handle(ordinal)
    with pytest.raises(MemoryOperationMaterializationError) as wrong_request:
        validated_operation_materialization_request({"proposal": "untrusted"})
    assert wrong_request.value.reason is MaterializationFailureReason.INVALID_INPUT
    with pytest.raises(MemoryOperationMaterializationError) as wrong_result:
        validated_materialized_bank_operations({"active_bank": "untrusted"})
    assert wrong_result.value.reason is MaterializationFailureReason.RESULT_MISMATCH

    with pytest.raises(ValidationError, match="precedes"):
        OperationMaterializationRequest(
            schema_version=MATERIALIZATION_REQUEST_SCHEMA_VERSION,
            cycle_receipt=running,
            proposal=_proposal(),
            delta_id=DELTA_ID,
            created_at=running.cycle.updated_at - timedelta(microseconds=1),
            assigned_memory_ids=(),
        )
    two_saves = _proposal(
        SaveKnowledge(
            operation="save_knowledge",
            content="first",
            evidence=(_event_evidence(context),),
            confidence=1.0,
        ),
        SaveProcedural(
            operation="save_procedural",
            content="second",
            evidence=(_event_evidence(context),),
            confidence=1.0,
        ),
    )
    with pytest.raises(ValidationError, match="assignment count"):
        _request(context, running, two_saves, (NEW_KNOWLEDGE_ID,))
    with pytest.raises(ValidationError, match="assignments must be unique"):
        _request(
            context,
            running,
            two_saves,
            (NEW_KNOWLEDGE_ID, NEW_KNOWLEDGE_ID),
        )

    duplicate_bank = result.model_copy(
        update={"active_bank": (*result.active_bank, result.active_bank[0])}
    )
    with pytest.raises(MemoryOperationMaterializationError) as inconsistent:
        validated_materialized_bank_operations(duplicate_bank)
    assert inconsistent.value.reason is MaterializationFailureReason.RESULT_MISMATCH
    mismatched_algorithm = result.model_copy(
        update={
            "preview_projection_digest": result.preview_projection_digest.model_copy(
                update={"algorithm": PayloadDigestAlgorithm.SYNTHETIC_SHA256}
            )
        }
    )
    with pytest.raises(MemoryOperationMaterializationError) as algorithm:
        validated_materialized_bank_operations(mismatched_algorithm)
    assert algorithm.value.reason is MaterializationFailureReason.RESULT_MISMATCH

    committed = await repository.commit_cycle(
        cycle_commit_command(
            context,
            delta=result.delta,
            assignments=result.memory_id_assignments,
            intervention_id=UUID("00000000-0000-4000-8000-000000000b24"),
        )
    )
    with pytest.raises(ValidationError, match="running cycle"):
        OperationMaterializationRequest(
            schema_version=MATERIALIZATION_REQUEST_SCHEMA_VERSION,
            cycle_receipt=committed,
            proposal=_proposal(),
            delta_id=UUID("00000000-0000-4000-8000-000000000b23"),
            created_at=committed.cycle.updated_at,
            assigned_memory_ids=(),
        )


async def test_event_pointer_span_and_future_time_are_authoritatively_checked() -> None:
    repository = MemoryRunRepository(installation_key=KEY, id_factory=_id_factory())
    safe = (
        await repository.append(
            event_draft(
                run_id=CYCLE_RUN_ID,
                source_event_id="materialization-structured-payload",
                payload={"items": ["é", "alpha"], "empty": ""},
                timestamp=CYCLE_NOW,
            )
        )
    ).event
    future = (
        await repository.append(
            event_draft(
                run_id=CYCLE_RUN_ID,
                source_event_id="materialization-future-time",
                payload={"message": "future wall clock"},
                timestamp=CYCLE_NOW + timedelta(hours=1),
            )
        )
    ).event
    context, _reserved, running = await advance_cycle_to_running(repository)

    valid_reference = EvidenceReference(
        source=EvidenceSource.EVENT,
        source_id=safe.event_id,
        field_path="/payload/items/1",
        span=TextSpan(start_byte=0, end_byte=5),
    )
    result = await materialize_bank_operations(
        _request(
            context,
            running,
            _proposal(
                SaveKnowledge(
                    operation="save_knowledge",
                    content="Nested array evidence resolves deterministically.",
                    evidence=(valid_reference,),
                    confidence=1.0,
                )
            ),
            (NEW_KNOWLEDGE_ID,),
        ),
        repository=repository,
    )
    assert result.delta.creates[0].provenance == (valid_reference,)

    invalid_references = (
        EvidenceReference(
            source=EvidenceSource.EVENT,
            source_id=safe.event_id,
            field_path="/payload/items/nope",
        ),
        EvidenceReference(
            source=EvidenceSource.EVENT,
            source_id=safe.event_id,
            field_path="/payload/items/9",
        ),
        EvidenceReference(
            source=EvidenceSource.EVENT,
            source_id=safe.event_id,
            field_path="/payload/items/0/more",
        ),
        EvidenceReference(
            source=EvidenceSource.EVENT,
            source_id=safe.event_id,
            field_path="/payload/items",
        ),
        EvidenceReference(
            source=EvidenceSource.EVENT,
            source_id=safe.event_id,
            field_path="/payload/empty",
        ),
        EvidenceReference(
            source=EvidenceSource.EVENT,
            source_id=safe.event_id,
            field_path="/content",
        ),
        EvidenceReference(
            source=EvidenceSource.EVENT,
            source_id=safe.event_id,
            field_path="/payload/items/0",
            span=TextSpan(start_byte=0, end_byte=1),
        ),
    )
    for index, reference in enumerate(invalid_references, start=1):
        with pytest.raises(MemoryOperationMaterializationError) as invalid:
            await materialize_bank_operations(
                _request(
                    context,
                    running,
                    _proposal(
                        SaveKnowledge(
                            operation="save_knowledge",
                            content="Reject malformed evidence.",
                            evidence=(reference,),
                            confidence=1.0,
                        )
                    ),
                    (NEW_KNOWLEDGE_ID,),
                    delta_id=UUID(f"00000000-0000-4000-8000-{0xB30 + index:012x}"),
                ),
                repository=repository,
            )
        assert invalid.value.reason is MaterializationFailureReason.REFERENCE_INVALID

    with pytest.raises(MemoryOperationMaterializationError) as future_error:
        await materialize_bank_operations(
            _request(
                context,
                running,
                _proposal(
                    SaveKnowledge(
                        operation="save_knowledge",
                        content="Reject future event time.",
                        evidence=(
                            EvidenceReference(
                                source=EvidenceSource.EVENT,
                                source_id=future.event_id,
                                field_path="/payload/message",
                            ),
                        ),
                        confidence=1.0,
                    )
                ),
                (NEW_KNOWLEDGE_ID,),
                delta_id=UUID("00000000-0000-4000-8000-000000000b3f"),
            ),
            repository=repository,
        )
    assert future_error.value.reason is MaterializationFailureReason.REFERENCE_FUTURE


async def test_memory_evidence_must_resolve_the_content_field(
    repository: RunRepository,
) -> None:
    context, running = await _running_cycle_with_bank(repository)
    proposal = _proposal(
        SaveKnowledge(
            operation="save_knowledge",
            content="Memory metadata is not evidence text.",
            evidence=(
                EvidenceReference(
                    source=EvidenceSource.MEMORY,
                    source_id=SEED_KNOWLEDGE_ID,
                    revision=1,
                    field_path="/confidence",
                ),
            ),
            confidence=1.0,
        )
    )

    with pytest.raises(MemoryOperationMaterializationError) as raised:
        await materialize_bank_operations(
            _request(context, running, proposal, (NEW_KNOWLEDGE_ID,)),
            repository=repository,
        )
    assert raised.value.reason is MaterializationFailureReason.REFERENCE_INVALID


async def test_retrograde_cycle_excludes_wall_clock_future_memory() -> None:
    repository = MemoryRunRepository(installation_key=KEY, id_factory=_id_factory())
    first, _reserved, _running = await advance_cycle_to_running(repository)
    seed_delta = MemoryDelta(
        delta_id=UUID("00000000-0000-4000-8000-000000000b60"),
        run_id=CYCLE_RUN_ID,
        creates=(
            MemoryCreate(
                handle="future-by-clock",
                kind=MemoryKind.KNOWLEDGE,
                content="This memory precedes by sequence but follows by wall clock.",
                provenance=(_event_evidence(first),),
                confidence=1.0,
                trust_label=TrustLabel.UNTRUSTED_MODEL_OUTPUT,
            ),
        ),
        created_at=first.commit_time,
    )
    await repository.commit_cycle(
        cycle_commit_command(
            first,
            delta=seed_delta,
            assignments=(
                MemoryIdAssignment(
                    handle="future-by-clock",
                    memory_id=SEED_KNOWLEDGE_ID,
                ),
            ),
        )
    )

    retrograde = CYCLE_NOW - timedelta(hours=1)
    event = (
        await repository.append(
            event_draft(
                run_id=CYCLE_RUN_ID,
                source_event_id="retrograde-cycle-event",
                timestamp=retrograde,
            )
        )
    ).event
    decision = InvocationDecision(
        decision_id=UUID("00000000-0000-4000-8000-000000000b61"),
        run_id=CYCLE_RUN_ID,
        event_sequence=event.sequence,
        invoke=True,
        risk_score=0.8,
        reason_codes=(ReasonCode.SCRIPTED_INVOKE,),
        policy_version="cycle-fixture/1",
        configuration_digest="f" * 64,
        budget_snapshot=await repository.budget_snapshot(CYCLE_RUN_ID),
        cooldown_active=False,
        created_at=retrograde + timedelta(seconds=1),
    )
    await repository.record_invocation_decision(decision)
    resolved = resolve_grounding_configuration(cycle_grounding_config())
    pending = await repository.begin_cycle(
        BeginCycle(
            run_id=CYCLE_RUN_ID,
            invocation_decision_id=decision.decision_id,
            grounding_version=resolved.pipeline_version,
            grounding_configuration=resolved.configuration,
            grounding_configuration_digest=resolved.configuration_digest,
            requested_delivery_target=DeliveryTarget.NEXT_MODEL_CALL,
            created_at=retrograde + timedelta(seconds=2),
        )
    )
    await repository.reserve_cycle(
        ReserveCycle(
            run_id=CYCLE_RUN_ID,
            cycle_id=pending.cycle.cycle_id,
            reservation=BudgetAmounts(
                model_calls=1,
                input_tokens=100,
                output_tokens=50,
                canonical_token_equivalents=150,
                latency_us=10_000,
                interventions=1,
                schema_repairs=1,
            ),
            updated_at=retrograde + timedelta(seconds=3),
        )
    )
    running = await repository.mark_cycle_running(
        StartCycle(
            run_id=CYCLE_RUN_ID,
            cycle_id=pending.cycle.cycle_id,
            batch_digest="2" * 64,
            updated_at=retrograde + timedelta(seconds=4),
        )
    )
    created_at = retrograde + timedelta(seconds=5)
    noop = OperationMaterializationRequest(
        schema_version=MATERIALIZATION_REQUEST_SCHEMA_VERSION,
        cycle_receipt=running,
        proposal=_proposal(),
        delta_id=UUID("00000000-0000-4000-8000-000000000b62"),
        created_at=created_at,
        assigned_memory_ids=(),
    )

    result = await materialize_bank_operations(noop, repository=repository)

    assert result.active_bank == ()
    cited = OperationMaterializationRequest(
        schema_version=MATERIALIZATION_REQUEST_SCHEMA_VERSION,
        cycle_receipt=running,
        proposal=_proposal(
            SaveKnowledge(
                operation="save_knowledge",
                content="Do not cite wall-clock future memory.",
                evidence=(_memory_evidence(SEED_KNOWLEDGE_ID),),
                confidence=1.0,
            )
        ),
        delta_id=UUID("00000000-0000-4000-8000-000000000b63"),
        created_at=created_at,
        assigned_memory_ids=(NEW_KNOWLEDGE_ID,),
    )
    with pytest.raises(MemoryOperationMaterializationError) as raised:
        await materialize_bank_operations(cited, repository=repository)
    assert raised.value.reason is MaterializationFailureReason.REFERENCE_FUTURE


async def test_same_phase_delete_target_is_rejected_before_compilation(
    repository: RunRepository,
) -> None:
    context, running = await _running_cycle_with_bank(repository)
    proposal = _proposal(
        SaveKnowledge(
            operation="save_knowledge",
            content="new",
            evidence=(_event_evidence(context),),
            confidence=1.0,
        ),
        DeleteMemory(
            operation="delete_memory",
            memory_id=NEW_KNOWLEDGE_ID,
            expected_revision=1,
        ),
    )

    with pytest.raises(MemoryOperationMaterializationError) as raised:
        await materialize_bank_operations(
            _request(context, running, proposal, (NEW_KNOWLEDGE_ID,)),
            repository=repository,
        )

    assert raised.value.reason is MaterializationFailureReason.OPERATION_CONFLICT


async def test_repository_failures_are_classified_and_suppress_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = MemoryRunRepository(installation_key=KEY, id_factory=_id_factory())
    context, _reserved, running = await advance_cycle_to_running(repository)
    request = _request(context, running, _proposal(), ())
    real_preview = repository.preview_memory_delta
    captured: list[MemoryDeltaPreview] = []

    async def capture_preview(command: object) -> MemoryDeltaPreview:
        receipt = await real_preview(command)  # type: ignore[arg-type]
        captured.append(receipt)
        return receipt

    monkeypatch.setattr(repository, "preview_memory_delta", capture_preview)
    await materialize_bank_operations(request, repository=repository)
    baseline = captured[0]

    async def preview_conflict(_command: object) -> MemoryDeltaPreview:
        raise PreviewConflictError()

    async def semantic_rejection(_command: object) -> MemoryDeltaPreview:
        raise ProjectionInvariantError("semantic preview rejection")

    async def repository_outage(_command: object) -> MemoryDeltaPreview:
        raise RepositoryError("repository-secret")

    async def unexpected_failure(_command: object) -> MemoryDeltaPreview:
        raise RuntimeError("provider-secret-in-context")

    async def wrong_type(_command: object) -> object:
        return object()

    async def invalid_preview(_command: object) -> MemoryDeltaPreview:
        return baseline.model_copy(update={"source_ledger_position": -1})

    async def mismatched_preview(_command: object) -> MemoryDeltaPreview:
        return baseline.model_copy(update={"command_digest": "0" * 64})

    cases = (
        (preview_conflict, MaterializationFailureReason.SOURCE_CONFLICT),
        (semantic_rejection, MaterializationFailureReason.PREVIEW_REJECTED),
        (repository_outage, MaterializationFailureReason.REPOSITORY_UNAVAILABLE),
        (unexpected_failure, MaterializationFailureReason.PREVIEW_REJECTED),
        (wrong_type, MaterializationFailureReason.PREVIEW_REJECTED),
        (invalid_preview, MaterializationFailureReason.PREVIEW_REJECTED),
        (mismatched_preview, MaterializationFailureReason.PREVIEW_REJECTED),
    )
    for replacement, reason in cases:
        monkeypatch.setattr(repository, "preview_memory_delta", replacement)
        with pytest.raises(MemoryOperationMaterializationError) as raised:
            await materialize_bank_operations(request, repository=repository)
        assert raised.value.reason is reason
        assert raised.value.__suppress_context__ is True
        assert "secret" not in f"{raised.value!r} {raised.value}"


async def test_authoritative_result_verifier_rejects_rehashed_cas_and_bank_tampering(
    repository: RunRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, running = await _running_cycle_with_bank(repository)
    request = _request(
        context,
        running,
        _proposal(
            UpdatePrivateStatus(
                operation="update_private_status",
                content="A repository-owned CAS cannot be rewritten.",
                evidence=(_event_evidence(context),),
                confidence=1.0,
            )
        ),
        (NEW_STATUS_ID,),
    )
    result = await materialize_bank_operations(request, repository=repository)
    replacement = result.delta.private_status_replacement
    assert replacement is not None
    tampered_delta = result.delta.model_copy(
        update={
            "private_status_replacement": replacement.model_copy(
                update={"expected_memory_id": None, "expected_revision": None}
            )
        }
    )
    tampered_cas = result.model_copy(update={"delta": tampered_delta})
    tampered_cas = tampered_cas.model_copy(
        update={
            "materialization_digest": materialize_module._materialization_digest(
                tampered_cas.model_dump(mode="json", exclude={"materialization_digest"})
            )
        }
    )
    assert validated_materialized_bank_operations_for_request(request, tampered_cas) == tampered_cas
    with pytest.raises(MemoryOperationMaterializationError) as cas_error:
        await verified_materialized_bank_operations_for_request(
            request,
            tampered_cas,
            repository=repository,
        )
    assert cas_error.value.reason is MaterializationFailureReason.PREVIEW_REJECTED

    tampered_bank = result.model_copy(update={"active_bank": result.active_bank[1:]})
    tampered_bank = tampered_bank.model_copy(
        update={
            "materialization_digest": materialize_module._materialization_digest(
                tampered_bank.model_dump(mode="json", exclude={"materialization_digest"})
            )
        }
    )
    assert (
        validated_materialized_bank_operations_for_request(request, tampered_bank) == tampered_bank
    )
    with pytest.raises(MemoryOperationMaterializationError) as bank_error:
        await verified_materialized_bank_operations_for_request(
            request,
            tampered_bank,
            repository=repository,
        )
    assert bank_error.value.reason is MaterializationFailureReason.RESULT_MISMATCH

    async def stale_preview(_command: object) -> MemoryDeltaPreview:
        raise PreviewConflictError()

    async def unavailable_preview(_command: object) -> MemoryDeltaPreview:
        raise RepositoryError("verifier-repository-secret")

    async def unexpected_preview(_command: object) -> MemoryDeltaPreview:
        raise RuntimeError("verifier-provider-secret")

    async def invalid_preview_type(_command: object) -> object:
        return object()

    for replacement_preview, reason in (
        (stale_preview, MaterializationFailureReason.SOURCE_CONFLICT),
        (unavailable_preview, MaterializationFailureReason.REPOSITORY_UNAVAILABLE),
        (unexpected_preview, MaterializationFailureReason.PREVIEW_REJECTED),
        (invalid_preview_type, MaterializationFailureReason.PREVIEW_REJECTED),
    ):
        monkeypatch.setattr(repository, "preview_memory_delta", replacement_preview)
        with pytest.raises(MemoryOperationMaterializationError) as verifier_error:
            await verified_materialized_bank_operations_for_request(
                request,
                result,
                repository=repository,
            )
        assert verifier_error.value.reason is reason
        assert verifier_error.value.__suppress_context__ is True


async def test_authoritative_verifier_rejects_cross_run_empty_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = MemoryRunRepository(installation_key=KEY, id_factory=_id_factory())
    context, _reserved, running = await advance_cycle_to_running(repository)
    request = _request(context, running, _proposal(), ())
    result = await materialize_bank_operations(request, repository=repository)
    assert result.active_bank == ()
    real_preview = repository.preview_memory_delta

    async def cross_run_preview(command: object) -> MemoryDeltaPreview:
        receipt = await real_preview(command)  # type: ignore[arg-type]
        return receipt.model_copy(update={"run_id": RUN_B})

    monkeypatch.setattr(repository, "preview_memory_delta", cross_run_preview)
    with pytest.raises(MemoryOperationMaterializationError) as raised:
        await verified_materialized_bank_operations_for_request(
            request,
            result,
            repository=repository,
        )
    assert raised.value.reason is MaterializationFailureReason.RESULT_MISMATCH


async def test_corrupt_repository_views_fail_closed_before_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = MemoryRunRepository(installation_key=KEY, id_factory=_id_factory())
    context, running = await _running_cycle_with_bank(repository)
    request = _request(context, running, _proposal(), ())
    real_snapshot = repository.snapshot
    real_ledger = repository.ledger
    source = await real_snapshot(CYCLE_RUN_ID)

    duplicate_records = (source.records[0], source.records[0], *source.records[1:])
    cross_run_records = (
        source.records[0].model_copy(update={"run_id": RUN_B}),
        *source.records[1:],
    )
    missing_boundary_records = (
        source.records[0].model_copy(update={"memory_id": MISSING_ID}),
        *source.records[1:],
    )
    private_index = next(
        index for index, record in enumerate(source.records) if record.kind is MemoryKind.KNOWLEDGE
    )
    extra_private = list(source.records)
    extra_private[private_index] = extra_private[private_index].model_copy(
        update={"kind": MemoryKind.PRIVATE_STATUS}
    )

    for records in (
        duplicate_records,
        cross_run_records,
        missing_boundary_records,
        tuple(extra_private),
    ):

        async def forged_snapshot(_run_id: UUID, records: tuple = records) -> object:
            return source.model_copy(update={"records": records})

        monkeypatch.setattr(repository, "snapshot", forged_snapshot)
        with pytest.raises(MemoryOperationMaterializationError) as raised:
            await materialize_bank_operations(request, repository=repository)
        assert raised.value.reason is MaterializationFailureReason.SOURCE_CONFLICT

    monkeypatch.setattr(repository, "snapshot", real_snapshot)
    ledger = await real_ledger(CYCLE_RUN_ID)
    event_index = next(
        index for index, entry in enumerate(ledger) if entry.record.record_type == "trace_event"
    )
    cross_run_event = ledger[event_index].record.model_copy(update={"run_id": RUN_B})
    cross_run_entry = ledger[event_index].model_copy(update={"record": cross_run_event})
    cross_run_ledger = (*ledger[:event_index], cross_run_entry, *ledger[event_index + 1 :])

    async def forged_event_ledger(_run_id: UUID) -> tuple:
        return cross_run_ledger

    monkeypatch.setattr(repository, "ledger", forged_event_ledger)
    with pytest.raises(MemoryOperationMaterializationError) as event_error:
        await materialize_bank_operations(request, repository=repository)
    assert event_error.value.reason is MaterializationFailureReason.SOURCE_CONFLICT

    committed_index = next(
        index
        for index, entry in enumerate(ledger)
        if entry.record.record_type == "cycle_record"
        and getattr(entry.record, "state", None).value == "committed"
    )
    committed = ledger[committed_index].record.model_copy(update={"validated_delta": None})
    corrupt_entry = ledger[committed_index].model_copy(update={"record": committed})
    corrupt_ledger = (*ledger[:committed_index], corrupt_entry, *ledger[committed_index + 1 :])

    async def forged_cycle_ledger(_run_id: UUID) -> tuple:
        return corrupt_ledger

    monkeypatch.setattr(repository, "ledger", forged_cycle_ledger)
    with pytest.raises(MemoryOperationMaterializationError) as cycle_error:
        await materialize_bank_operations(request, repository=repository)
    assert cycle_error.value.reason is MaterializationFailureReason.SOURCE_CONFLICT


async def test_repository_read_shape_and_internal_compilation_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = MemoryRunRepository(installation_key=KEY, id_factory=_id_factory())
    context, _reserved, running = await advance_cycle_to_running(repository)
    request = _request(context, running, _proposal(), ())
    real_ledger = repository.ledger
    real_snapshot = repository.snapshot

    async def read_failure(_run_id: UUID) -> tuple[object, ...]:
        raise RuntimeError("read-secret")

    monkeypatch.setattr(repository, "ledger", read_failure)
    with pytest.raises(MemoryOperationMaterializationError) as read_error:
        await materialize_bank_operations(request, repository=repository)
    assert read_error.value.reason is MaterializationFailureReason.REPOSITORY_UNAVAILABLE
    assert read_error.value.__suppress_context__ is True

    async def wrong_ledger_shape(_run_id: UUID) -> list[object]:
        return list(await real_ledger(CYCLE_RUN_ID))

    monkeypatch.setattr(repository, "ledger", wrong_ledger_shape)
    with pytest.raises(MemoryOperationMaterializationError) as shape:
        await materialize_bank_operations(request, repository=repository)
    assert shape.value.reason is MaterializationFailureReason.REPOSITORY_UNAVAILABLE

    monkeypatch.setattr(repository, "ledger", real_ledger)

    async def wrong_snapshot_type(_run_id: UUID) -> object:
        return object()

    monkeypatch.setattr(repository, "snapshot", wrong_snapshot_type)
    with pytest.raises(MemoryOperationMaterializationError) as snapshot_type:
        await materialize_bank_operations(request, repository=repository)
    assert snapshot_type.value.reason is MaterializationFailureReason.REPOSITORY_UNAVAILABLE

    source = await real_snapshot(CYCLE_RUN_ID)

    async def invalid_snapshot(_run_id: UUID) -> object:
        return source.model_copy(update={"ledger_position": -1})

    monkeypatch.setattr(repository, "snapshot", invalid_snapshot)
    with pytest.raises(MemoryOperationMaterializationError) as snapshot_value:
        await materialize_bank_operations(request, repository=repository)
    assert snapshot_value.value.reason is MaterializationFailureReason.REPOSITORY_UNAVAILABLE

    monkeypatch.setattr(repository, "snapshot", real_snapshot)
    two_saves = _proposal(
        SaveKnowledge(
            operation="save_knowledge",
            content="one",
            evidence=(_event_evidence(context),),
            confidence=1.0,
        ),
        SaveProcedural(
            operation="save_procedural",
            content="two",
            evidence=(_event_evidence(context),),
            confidence=1.0,
        ),
    )
    duplicate_handles = _request(
        context,
        running,
        two_saves,
        (NEW_KNOWLEDGE_ID, NEW_PROCEDURAL_ID),
    )
    monkeypatch.setattr(materialize_module, "operation_handle", lambda _ordinal: "duplicate")
    with pytest.raises(MemoryOperationMaterializationError) as compilation:
        await materialize_bank_operations(duplicate_handles, repository=repository)
    assert compilation.value.reason is MaterializationFailureReason.OPERATION_CONFLICT

    monkeypatch.undo()
    repository = MemoryRunRepository(installation_key=KEY, id_factory=_id_factory())
    context, _reserved, running = await advance_cycle_to_running(repository)
    request = _request(context, running, _proposal(), ())
    monkeypatch.setattr(materialize_module, "_materialization_digest", lambda _values: "bad")
    with pytest.raises(MemoryOperationMaterializationError) as result_failure:
        await materialize_bank_operations(request, repository=repository)
    assert result_failure.value.reason is MaterializationFailureReason.PREVIEW_REJECTED


def test_request_schema_rejects_assignment_mismatch_before_repository_use() -> None:
    proposal = _proposal(
        SaveKnowledge(
            operation="save_knowledge",
            content="bounded",
            evidence=(
                EvidenceReference(
                    source=EvidenceSource.EVENT,
                    source_id=UUID("00000000-0000-4000-8000-000000000bee"),
                    field_path="/payload/message",
                ),
            ),
            confidence=1.0,
        )
    )
    with pytest.raises(ValidationError):
        OperationMaterializationRequest(
            schema_version=MATERIALIZATION_REQUEST_SCHEMA_VERSION,
            cycle_receipt=None,  # type: ignore[arg-type]
            proposal=proposal,
            delta_id=DELTA_ID,
            created_at=None,  # type: ignore[arg-type]
            assigned_memory_ids=(),
        )


async def test_results_are_identical_across_repository_backends(tmp_path: Path) -> None:
    memory = MemoryRunRepository(installation_key=KEY, id_factory=_id_factory())
    sqlite = SQLiteRunRepository(
        tmp_path / "deterministic-materialization.sqlite3",
        installation_key=KEY,
        id_factory=_id_factory(),
    )
    try:
        materialized = []
        for repository in (memory, sqlite):
            context, running = await _running_cycle_with_bank(repository)
            proposal = _proposal(
                SaveKnowledge(
                    operation="save_knowledge",
                    content="Both backends compile the same operation.",
                    evidence=(_event_evidence(context),),
                    confidence=1.0,
                )
            )
            materialized.append(
                await materialize_bank_operations(
                    _request(context, running, proposal, (NEW_KNOWLEDGE_ID,)),
                    repository=repository,
                )
            )
        assert materialized[0] == materialized[1]
    finally:
        sqlite.close()
