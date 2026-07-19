from __future__ import annotations

from uuid import UUID

import pytest
from tests.repository.conformance import (
    CYCLE_RUN_ID,
    RUN_A,
    commit_grounded_delivery,
    event_draft,
)

from saliencegate.ports.repository import (
    DigestVerificationError,
    LedgerHead,
    ProjectionInvariantError,
    RebuildError,
)
from saliencegate.repository.memory import MemoryRunRepository
from saliencegate.repository.projector import apply_entry, empty_projection
from saliencegate.security import InstallationKey

KEY = InstallationKey(b"r" * 32)


def repository(*, key: InstallationKey = KEY) -> MemoryRunRepository:
    identifiers = iter(
        UUID(f"00000000-0000-4000-8000-{value:012x}") for value in range(0x700, 0x800)
    )
    return MemoryRunRepository(
        installation_key=key,
        id_factory=lambda: next(identifiers),
    )


async def test_verified_state_and_restore_are_deeply_defensive() -> None:
    source = repository()
    await source.append(event_draft())
    source_snapshot = await source.snapshot(RUN_A)
    state = await source._verified_state(RUN_A)

    restored = repository()
    restored_state = await restored._restore_run(state.ledger, state.ledger_head)
    assert await restored.ledger(RUN_A) == await source.ledger(RUN_A)
    assert await restored.snapshot(RUN_A) == source_snapshot

    exported_event = state.ledger[0].record
    object.__setattr__(exported_event, "source_adapter", "tampered-export")
    object.__setattr__(
        state.projection,
        "events_by_id",
        {exported_event.event_id: exported_event},
    )
    object.__setattr__(restored_state.ledger_head, "entry_count", 999)
    object.__setattr__(
        restored_state.projection,
        "events_by_sequence",
        {},
    )

    assert await source.snapshot(RUN_A) == source_snapshot
    assert await restored.snapshot(RUN_A) == source_snapshot
    assert (await source.ledger(RUN_A))[0].record.source_adapter == "fixture-adapter"
    assert (await restored.ledger(RUN_A))[0].record.source_adapter == "fixture-adapter"


async def test_verified_state_replays_before_attesting_the_live_projection() -> None:
    source = repository()
    await source.append(event_draft())
    slot = source._slots[RUN_A]
    assert slot.projection is not None
    event = next(iter(slot.projection.events_by_id.values()))
    tampered = event.model_copy(update={"source_adapter": "tampered-live-projection"})
    object.__setattr__(
        slot.projection,
        "events_by_id",
        {tampered.event_id: tampered},
    )

    with pytest.raises(DigestVerificationError, match="live projection"):
        await source._verified_state(RUN_A)


async def test_restore_rejects_tampering_wrong_keys_and_existing_runs_atomically() -> None:
    source = repository()
    await source.append(event_draft())
    state = await source._verified_state(RUN_A)
    original = state.ledger[0]
    tampered_record = original.record.model_copy(update={"source_adapter": "tampered-ledger"})
    tampered = original.model_copy(update={"record": tampered_record})

    target = repository()
    with pytest.raises(DigestVerificationError):
        await target._restore_run((tampered,), state.ledger_head)
    assert target._slots == {}
    assert target._trusted_heads == {}
    assert target._trusted_projections == {}

    wrong_key = repository(key=InstallationKey(b"w" * 32))
    with pytest.raises(DigestVerificationError):
        await wrong_key._restore_run(state.ledger, state.ledger_head)
    assert wrong_key._slots == {}

    restored = repository()
    await restored._restore_run(state.ledger, state.ledger_head)
    before = await restored.snapshot(RUN_A)
    with pytest.raises(ProjectionInvariantError, match="already installed"):
        await restored._restore_run(state.ledger, state.ledger_head)
    assert await restored.snapshot(RUN_A) == before


async def test_restore_copies_inputs_before_installing_them() -> None:
    source = repository()
    await source.append(event_draft())
    state = await source._verified_state(RUN_A)
    entries = tuple(
        type(entry).model_validate_json(entry.model_dump_json(warnings=False))
        for entry in state.ledger
    )
    head = type(state.ledger_head).model_validate_json(
        state.ledger_head.model_dump_json(warnings=False)
    )
    target = repository()
    await target._restore_run(entries, head)
    before = await target.snapshot(RUN_A)

    object.__setattr__(entries[0].record, "source_adapter", "mutated-after-restore")
    object.__setattr__(head, "entry_count", 999)

    assert await target.snapshot(RUN_A) == before
    assert (await target.ledger(RUN_A))[0].record.source_adapter == "fixture-adapter"


async def test_authenticated_replay_rejects_committed_reminder_without_outbox() -> None:
    source = repository()
    await commit_grounded_delivery(source)
    slot = source._slots[CYCLE_RUN_ID]
    truncated = slot.ledger[:-1]
    assert truncated

    projected = empty_projection(CYCLE_RUN_ID)
    projection_tag = None
    for entry in truncated:
        projected = apply_entry(projected, entry)
        projection_tag = source._projection_checkpoint_tag(
            entry,
            projected,
            previous=projection_tag,
        )
    assert projection_tag is not None
    last = truncated[-1]
    head_tag = source._integrity.tag(
        source._head_value(
            CYCLE_RUN_ID,
            len(truncated),
            last.chain_tag,
            projection_tag,
        ),
        domain="saliencegate:ledger-head:v1",
    )
    incomplete_head = LedgerHead(
        run_id=CYCLE_RUN_ID,
        entry_count=len(truncated),
        chain_tag=last.chain_tag,
        projection_tag=projection_tag,
        head_tag=head_tag,
    )

    with pytest.raises(RebuildError):
        source._replay_run(truncated, incomplete_head)
