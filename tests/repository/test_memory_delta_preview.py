from __future__ import annotations

import itertools
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError
from tests.repository.conformance import (
    CYCLE_MEMORY_ID,
    CYCLE_RUN_ID,
    CycleContext,
    advance_cycle_to_running,
    cycle_commit_command,
    event_draft,
)

from saliencegate.domain import (
    EvidenceReference,
    EvidenceSource,
    MemoryCreate,
    MemoryDelta,
    MemoryIdAssignment,
    MemoryKind,
    PayloadDigestAlgorithm,
    TrustLabel,
)
from saliencegate.ports import MemoryDeltaPreview, PreviewMemoryDelta
from saliencegate.ports.repository import (
    LedgerEntry,
    LedgerHead,
    MemorySnapshot,
    PreviewConflictError,
    RunRepository,
)
from saliencegate.repository.memory import MemoryRunRepository
from saliencegate.repository.projector import (
    apply_entry,
    empty_projection,
)
from saliencegate.repository.projector import (
    preview_memory_delta as preview_projected_memory_delta,
)
from saliencegate.repository.sqlite import SQLiteRunRepository
from saliencegate.security import InstallationKey

DELTA_ID = UUID("00000000-0000-4000-8000-000000000901")
KEY = InstallationKey(b"v" * 32)


@pytest.fixture(params=("memory", "sqlite"))
def repository(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> Iterator[RunRepository]:
    identifiers = itertools.count(0xA00)

    def id_factory() -> UUID:
        return UUID(f"00000000-0000-4000-8000-{next(identifiers):012x}")

    if request.param == "memory":
        repository: RunRepository = MemoryRunRepository(
            installation_key=KEY,
            id_factory=id_factory,
        )
        yield repository
        return

    sqlite = SQLiteRunRepository(
        tmp_path / "memory-delta-preview.sqlite3",
        installation_key=KEY,
        id_factory=id_factory,
    )
    try:
        yield sqlite
    finally:
        sqlite.close()


def delta_and_assignments(
    context: CycleContext,
) -> tuple[MemoryDelta, tuple[MemoryIdAssignment, ...]]:
    delta = MemoryDelta(
        delta_id=DELTA_ID,
        run_id=CYCLE_RUN_ID,
        creates=(
            MemoryCreate(
                handle="preview-created-memory",
                kind=MemoryKind.KNOWLEDGE,
                content="Preview this memory without publishing it.",
                provenance=(
                    EvidenceReference(
                        source=EvidenceSource.EVENT,
                        source_id=context.event.event_id,
                        field_path="/payload/message",
                    ),
                ),
                confidence=1.0,
                trust_label=TrustLabel.UNTRUSTED_MODEL_OUTPUT,
            ),
        ),
        created_at=context.commit_time,
    )
    assignments = (
        MemoryIdAssignment(
            handle="preview-created-memory",
            memory_id=CYCLE_MEMORY_ID,
        ),
    )
    return delta, assignments


def preview_command(
    context: CycleContext,
    snapshot: MemorySnapshot,
    delta: MemoryDelta,
    assignments: tuple[MemoryIdAssignment, ...],
) -> PreviewMemoryDelta:
    return PreviewMemoryDelta(
        schema_version="memory-delta-preview-command/v1",
        run_id=CYCLE_RUN_ID,
        expected_ledger_position=snapshot.ledger_position,
        expected_ingestion_cursor=snapshot.ingestion_cursor,
        expected_memory_cursor=snapshot.memory_cursor,
        expected_projection_digest=snapshot.projection_digest,
        last_event_sequence=context.event.sequence,
        delta=delta,
        memory_id_assignments=assignments,
    )


async def repository_state(
    repository: RunRepository,
) -> tuple[tuple[LedgerEntry, ...], LedgerHead, MemorySnapshot]:
    return (
        await repository.ledger(CYCLE_RUN_ID),
        await repository.ledger_head(CYCLE_RUN_ID),
        await repository.snapshot(CYCLE_RUN_ID),
    )


async def test_preview_matches_projector_and_commit_without_mutation(
    repository: RunRepository,
) -> None:
    context, _reserved, _running = await advance_cycle_to_running(repository)
    delta, assignments = delta_and_assignments(context)
    before = await repository_state(repository)
    command = preview_command(context, before[2], delta, assignments)

    receipt = await repository.preview_memory_delta(command)

    assert isinstance(receipt, MemoryDeltaPreview)
    assert receipt.schema_version == "memory-delta-preview/v1"
    assert receipt.command_digest == command.command_digest
    assert await repository_state(repository) == before
    projection = empty_projection(CYCLE_RUN_ID)
    for entry in before[0]:
        projection = apply_entry(projection, entry)
    expected = preview_projected_memory_delta(
        projection,
        delta,
        assignments,
        last_event_sequence=context.event.sequence,
    )
    assert receipt.records == tuple(
        sorted(
            expected.memories.values(),
            key=lambda record: (record.kind.value, str(record.memory_id)),
        )
    )
    assert receipt.current_private_status_id == expected.current_private_status_id
    assert receipt.source_ledger_position == before[2].ledger_position
    assert receipt.source_ingestion_cursor == before[2].ingestion_cursor
    assert receipt.source_memory_cursor == before[2].memory_cursor
    assert receipt.source_projection_digest == before[2].projection_digest
    assert receipt.preview_projection_digest != receipt.source_projection_digest

    forged_command = command.model_copy(update={"command_digest": "0" * 64})
    with pytest.raises(ValidationError, match="command digest"):
        PreviewMemoryDelta.model_validate_json(forged_command.model_dump_json(warnings=False))
    with pytest.raises(ValidationError, match="at most 65"):
        PreviewMemoryDelta(
            **command.model_dump(
                mode="python", exclude={"command_digest", "memory_id_assignments"}
            ),
            memory_id_assignments=assignments * 66,
        )

    wrong_algorithm = receipt.model_copy(
        update={
            "preview_projection_digest": receipt.preview_projection_digest.model_copy(
                update={"algorithm": PayloadDigestAlgorithm.SYNTHETIC_SHA256}
            )
        }
    )
    with pytest.raises(ValidationError, match="algorithms must match"):
        MemoryDeltaPreview.model_validate_json(wrong_algorithm.model_dump_json(warnings=False))
    wrong_private = receipt.model_copy(update={"current_private_status_id": CYCLE_MEMORY_ID})
    with pytest.raises(ValidationError, match="private status is not active"):
        MemoryDeltaPreview.model_validate_json(wrong_private.model_dump_json(warnings=False))

    await repository.commit_cycle(
        cycle_commit_command(
            context,
            delta=delta,
            assignments=assignments,
        )
    )
    assert (await repository.snapshot(CYCLE_RUN_ID)).records == receipt.records


async def test_preview_rejects_a_stale_anchor_without_mutation(
    repository: RunRepository,
) -> None:
    context, _reserved, _running = await advance_cycle_to_running(repository)
    delta, assignments = delta_and_assignments(context)
    source = await repository.snapshot(CYCLE_RUN_ID)
    command = preview_command(context, source, delta, assignments)
    await repository.append(
        event_draft(
            run_id=CYCLE_RUN_ID,
            source_event_id="preview-anchor-advanced",
            timestamp=context.commit_time + timedelta(seconds=1),
        )
    )
    before = await repository_state(repository)

    with pytest.raises(PreviewConflictError, match="preview anchor is stale"):
        await repository.preview_memory_delta(command)

    assert await repository_state(repository) == before
