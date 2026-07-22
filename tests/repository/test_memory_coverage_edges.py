"""Coverage-oriented regressions for otherwise rare contract edges."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import timedelta
from typing import Any

import pytest
from pydantic import BaseModel
from tests.repository.conformance import (
    CONDITIONAL_EVENT_ID_A,
    RUN_A,
    RUN_B,
    event_draft,
)
from tests.repository.test_memory_repository import create_repository
from tests.repository.test_projector import RUN_ID as PROJECTOR_RUN_ID
from tests.repository.test_projector import first_reminder_projection

from saliencegate.domain import (
    DeduplicationGuarantee,
    DeliveryRecord,
    DeliveryState,
    PayloadDigest,
    canonical_digest,
)
from saliencegate.ports.repository import (
    ConditionalEventAppend,
    DeliveryNotFoundError,
    DigestVerificationError,
    InvalidRecordError,
    InvalidRecordTypeError,
    LedgerHeadConflictError,
    ProjectionInvariantError,
    RebuildError,
    RunNotFoundError,
)
from saliencegate.repository import memory as memory_module
from saliencegate.repository.memory import MemoryRunRepository, _RunSlot


async def _repository_with_two_events() -> MemoryRunRepository:
    repository = create_repository()
    await repository.append(event_draft(source_event_id="replay-one"))
    await repository.append(
        event_draft(
            source_event_id="replay-two",
            timestamp=event_draft().timestamp + timedelta(seconds=1),
        )
    )
    return repository


async def test_slot_rejects_missing_run_with_a_trusted_anchor() -> None:
    repository = create_repository()
    await repository.append(event_draft())
    slot = repository._slots.pop(RUN_A)
    assert slot.ledger_head is not None

    with pytest.raises(DigestVerificationError, match="trusted ledger anchor"):
        await repository._slot(RUN_A)


async def test_slot_rejects_empty_run_with_a_trusted_anchor() -> None:
    repository = create_repository()
    await repository.append(event_draft())
    repository._slots[RUN_A] = _RunSlot(run_id=RUN_A)

    with pytest.raises(DigestVerificationError, match="trusted ledger anchor"):
        await repository._slot(RUN_A)


async def test_empty_untrusted_slot_is_not_a_run() -> None:
    repository = create_repository()
    repository._slots[RUN_A] = _RunSlot(run_id=RUN_A)

    with pytest.raises(RunNotFoundError):
        await repository._slot(RUN_A)


async def test_release_append_slot_preserves_non_cancellation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = create_repository()

    async def fail(_slot: _RunSlot) -> None:
        raise RuntimeError("release failed")

    monkeypatch.setattr(repository, "_release_append_slot", fail)

    with pytest.raises(RuntimeError, match="release failed"):
        await repository._release_append_slot_safely(_RunSlot(run_id=RUN_A))


async def test_release_append_slot_restores_pending_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = create_repository()
    calls = 0

    async def fail(_slot: _RunSlot) -> None:
        raise RuntimeError("release failed")

    async def interrupt_once(task: asyncio.Task[None]) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise asyncio.CancelledError
        await task

    monkeypatch.setattr(repository, "_release_append_slot", fail)
    monkeypatch.setattr(memory_module.asyncio, "shield", interrupt_once)

    with pytest.raises(asyncio.CancelledError):
        await repository._release_append_slot_safely(_RunSlot(run_id=RUN_A))
    assert calls == 2


def test_record_key_rejects_an_unsupported_record() -> None:
    with pytest.raises(ProjectionInvariantError, match="unsupported ledger record key"):
        MemoryRunRepository._record_key(object())  # type: ignore[arg-type]


def test_trace_event_rejects_an_invalid_repository_sequence() -> None:
    repository = create_repository()
    draft = repository._validate_draft(event_draft())
    redacted = repository._redact_draft(draft)

    with pytest.raises(ProjectionInvariantError, match="trace event is invalid"):
        repository._trace_event(redacted, sequence=0)


async def test_expected_head_is_rejected_for_an_empty_slot() -> None:
    repository = create_repository()
    receipt = await repository.append(event_draft())
    expected = await repository.ledger_head(receipt.event.run_id)

    with pytest.raises(LedgerHeadConflictError):
        repository._require_expected_head(_RunSlot(run_id=RUN_B), expected)


def test_cycle_command_validation_normalizes_a_forged_model() -> None:
    class Command(BaseModel):
        count: int

    forged = Command(count=1).model_copy(update={"count": "invalid"})

    with pytest.raises(InvalidRecordError, match="fixture"):
        MemoryRunRepository._validate_cycle_command(forged, Command, "fixture")


def test_cycle_command_validation_rejects_the_wrong_exact_type() -> None:
    class Command(BaseModel):
        count: int

    with pytest.raises(InvalidRecordTypeError, match="fixture"):
        MemoryRunRepository._validate_cycle_command(object(), Command, "fixture")


def test_conditional_batch_rejects_an_unknown_operation() -> None:
    repository = create_repository()

    with pytest.raises(InvalidRecordTypeError, match="conditional_batch"):
        repository._prepare_conditional_batch((object(),))


def test_conditional_batch_normalizes_canonical_encoding_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = create_repository()
    operation = ConditionalEventAppend(
        event=event_draft(source_event_id="conditional-canonical"),
        event_id=CONDITIONAL_EVENT_ID_A,
    )

    def fail(_value: object) -> str:
        raise RuntimeError("canonical failure")

    monkeypatch.setattr(memory_module, "canonical_json", fail)

    with pytest.raises(InvalidRecordError, match="conditional_batch"):
        repository._prepare_conditional_batch((operation,))


def test_conditional_batch_receipt_normalizes_unexpected_validation_error() -> None:
    with pytest.raises(ProjectionInvariantError, match="receipt is invalid"):
        MemoryRunRepository._bounded_conditional_batch_receipt(
            initial_head=None,
            receipts=(),
            final_head=object(),  # type: ignore[arg-type]
        )


async def _conditional_candidate() -> tuple[MemoryRunRepository, _RunSlot, _RunSlot]:
    repository = create_repository()
    await repository.append(event_draft(source_event_id="conditional-prefix"))
    slot = repository._slots[RUN_A]
    shadow = repository._shadow_slot(slot)
    draft = repository._validate_draft(
        event_draft(
            source_event_id="conditional-suffix",
            timestamp=event_draft().timestamp + timedelta(seconds=1),
        )
    )
    repository._append_conditional_to_slot(
        shadow,
        draft,
        repository._redact_draft(draft),
        event_id=CONDITIONAL_EVENT_ID_A,
        publish=False,
    )
    return repository, slot, shadow


async def test_conditional_shadow_accepts_an_authenticated_suffix() -> None:
    repository, slot, shadow = await _conditional_candidate()

    repository._verify_conditional_shadow(slot, shadow)


async def test_conditional_shadow_requires_complete_state() -> None:
    repository, slot, shadow = await _conditional_candidate()
    shadow.projection = None

    with pytest.raises(ProjectionInvariantError, match="state is unavailable"):
        repository._verify_conditional_shadow(slot, shadow)


async def test_conditional_shadow_requires_a_strict_suffix() -> None:
    repository, slot, _shadow = await _conditional_candidate()

    with pytest.raises(DigestVerificationError, match="conditional batch candidate"):
        repository._verify_conditional_shadow(slot, repository._shadow_slot(slot))


async def test_conditional_shadow_rejects_cross_run_suffix_entry() -> None:
    repository, slot, shadow = await _conditional_candidate()
    last = shadow.ledger[-1].model_copy(update={"run_id": RUN_B})
    shadow.ledger = (*shadow.ledger[:-1], last)

    with pytest.raises(DigestVerificationError, match="conditional batch candidate"):
        repository._verify_conditional_shadow(slot, shadow)


async def test_conditional_shadow_rejects_mismatched_head() -> None:
    repository, slot, shadow = await _conditional_candidate()
    assert shadow.ledger_head is not None
    shadow.ledger_head = shadow.ledger_head.model_copy(
        update={"entry_count": shadow.ledger_head.entry_count + 1}
    )

    with pytest.raises(DigestVerificationError, match="conditional batch candidate"):
        repository._verify_conditional_shadow(slot, shadow)


async def test_conditional_shadow_rejects_unexpected_derived_state() -> None:
    repository, slot, shadow = await _conditional_candidate()
    shadow.collision_receipts = {"unexpected": object()}  # type: ignore[dict-item]

    with pytest.raises(DigestVerificationError, match="conditional batch candidate"):
        repository._verify_conditional_shadow(slot, shadow)


async def test_conditional_shadow_commit_restores_missing_trusted_maps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, slot, shadow = await _conditional_candidate()
    repository._trusted_heads.clear()
    repository._trusted_projections.clear()

    def fail(_slot: _RunSlot, _shadow: _RunSlot) -> None:
        raise RuntimeError("publication failed")

    monkeypatch.setattr(MemoryRunRepository, "_publish_shadow", staticmethod(fail))

    with pytest.raises(RuntimeError, match="publication failed"):
        repository._commit_conditional_shadow(slot, shadow)
    assert RUN_A not in repository._trusted_heads
    assert RUN_A not in repository._trusted_projections


def _pending_delivery() -> tuple[object, DeliveryRecord]:
    projection = first_reminder_projection()
    cycle = next(iter(projection.cycles.values()))
    intervention = next(iter(projection.interventions.values()))
    assert intervention.delivery_target is not None
    assert intervention.rendered_text is not None
    delivery = DeliveryRecord(
        delivery_id=CONDITIONAL_EVENT_ID_A,
        run_id=PROJECTOR_RUN_ID,
        revision=1,
        cycle_id=cycle.cycle_id,
        intervention_id=intervention.intervention_id,
        rendered_text_digest=canonical_digest(intervention.rendered_text),
        target_request_id="request-coverage",
        target=intervention.delivery_target,
        state=DeliveryState.PENDING,
        attempt_count=0,
        adapter_id="fixture",
        adapter_deduplicates=True,
        adapter_deduplication_guarantee=DeduplicationGuarantee.DURABLE_DELIVERY_ID,
        adapter_supports_pre_action=True,
        adapter_contract_version="adapter-contract/v1",
        adapter_capabilities_digest="8" * 64,
        created_at=cycle.updated_at,
        updated_at=cycle.updated_at,
    )
    return projection, delivery


def test_delivery_base_rejects_a_missing_delivery() -> None:
    with pytest.raises(DeliveryNotFoundError):
        MemoryRunRepository._delivery_base(
            _RunSlot(run_id=RUN_A),
            CONDITIONAL_EVENT_ID_A,
            1,
        )


def test_delivery_transition_normalizes_invalid_record() -> None:
    _projection, delivery = _pending_delivery()

    with pytest.raises(InvalidRecordError, match="transition"):
        MemoryRunRepository._transition_delivery(
            delivery,
            "transition",
            updated_at=None,
        )


def test_attempt_envelope_requires_an_attempting_delivery() -> None:
    projection, delivery = _pending_delivery()

    with pytest.raises(ProjectionInvariantError, match="no grounded reminder"):
        MemoryRunRepository._attempt_envelope(projection, delivery)  # type: ignore[arg-type]


@pytest.mark.parametrize("recovered_at", [None, event_draft().timestamp.replace(tzinfo=None)])
async def test_cycle_recovery_rejects_invalid_time(recovered_at: Any) -> None:
    repository = create_repository()

    with pytest.raises(Exception, match="recovery"):
        await repository.recover_cycles(RUN_A, recovered_at=recovered_at)


@pytest.mark.parametrize("delivery_id", [object(), RUN_A.int])
async def test_delivery_lookup_rejects_non_uuid_identifier(delivery_id: object) -> None:
    repository = create_repository()

    with pytest.raises(DeliveryNotFoundError):
        await repository.delivery(RUN_A, delivery_id)  # type: ignore[arg-type]


async def test_delivery_lookup_rejects_unknown_delivery() -> None:
    repository = create_repository()
    await repository.append(event_draft())

    with pytest.raises(DeliveryNotFoundError):
        await repository.delivery(RUN_A, CONDITIONAL_EVENT_ID_A)


@pytest.mark.parametrize("recovered_at", [None, event_draft().timestamp.replace(tzinfo=None)])
async def test_delivery_recovery_rejects_invalid_time(recovered_at: Any) -> None:
    repository = create_repository()

    with pytest.raises(Exception, match="recovery"):
        await repository.recover_deliveries(RUN_A, recovered_at=recovered_at)


@pytest.mark.parametrize("entries", [(), (object(),)])
async def test_replay_rejects_invalid_entry_container(entries: tuple[object, ...]) -> None:
    repository = await _repository_with_two_events()
    head = repository._slots[RUN_A].ledger_head
    assert head is not None

    with pytest.raises(RebuildError):
        repository._replay_run(entries, head)  # type: ignore[arg-type]


async def test_replay_normalizes_model_copy_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = await _repository_with_two_events()
    slot = repository._slots[RUN_A]
    assert slot.ledger_head is not None

    def fail(_value: object) -> object:
        raise AttributeError("copy failed")

    monkeypatch.setattr(memory_module, "_copy_model", fail)

    with pytest.raises(RebuildError):
        repository._replay_run(slot.ledger, slot.ledger_head)


async def test_replay_rejects_mismatched_head_shape() -> None:
    repository = await _repository_with_two_events()
    slot = repository._slots[RUN_A]
    assert slot.ledger_head is not None
    head = slot.ledger_head.model_copy(update={"entry_count": 99})

    with pytest.raises(DigestVerificationError, match="durable ledger head"):
        repository._replay_run(slot.ledger, head)


async def test_replay_rejects_cross_run_prefix_entry() -> None:
    repository = await _repository_with_two_events()
    slot = repository._slots[RUN_A]
    assert slot.ledger_head is not None
    first_record = slot.ledger[0].record.model_copy(update={"run_id": RUN_B})
    first = slot.ledger[0].model_copy(update={"run_id": RUN_B, "record": first_record})

    with pytest.raises(DigestVerificationError, match="durable ledger run"):
        repository._replay_run((first, *slot.ledger[1:]), slot.ledger_head)


async def test_replay_rejects_projection_checkpoint_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = await _repository_with_two_events()
    slot = repository._slots[RUN_A]
    assert slot.ledger_head is not None
    wrong = PayloadDigest(
        algorithm=slot.ledger_head.projection_tag.algorithm,
        value="0" * 64,
    )

    monkeypatch.setattr(
        MemoryRunRepository,
        "_projection_checkpoint_tag",
        lambda *_args, **_kwargs: wrong,
    )

    with pytest.raises(DigestVerificationError, match="durable projection head"):
        repository._replay_run(slot.ledger, slot.ledger_head)


async def test_verified_state_normalizes_live_projection_digest_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = await _repository_with_two_events()
    calls = 0
    original = memory_module.projection_digests

    def fail(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original(*_args, **_kwargs)
        raise ValueError("projection failure")

    monkeypatch.setattr(memory_module, "projection_digests", fail)

    with pytest.raises(DigestVerificationError, match="live projection"):
        await repository._verified_state(RUN_A)


async def _collision_projection() -> tuple[MemoryRunRepository, object, object]:
    repository = create_repository()
    await repository.append(event_draft(source_event_id="collision-source"))
    collision = await repository.append(
        event_draft(
            source_event_id="collision-source",
            payload={"message": "different"},
        )
    )
    assert collision.collision_event is not None
    projection = repository._slots[RUN_A].projection
    assert projection is not None
    return repository, projection, collision.collision_event


def _projection_with_collision(projection: Any, event: object) -> Any:
    return replace(
        projection,
        events_by_sequence={
            **projection.events_by_sequence,
            projection.ingestion_cursor: event,
        },
    )


async def test_collision_receipts_reject_wrong_audit_identity() -> None:
    repository, projection, collision = await _collision_projection()
    changed = collision.model_copy(update={"source_adapter": "wrong"})

    with pytest.raises(ProjectionInvariantError, match="invalid collision audit event"):
        repository._collision_receipts(_projection_with_collision(projection, changed))


async def test_collision_receipts_require_mapping_fingerprint() -> None:
    repository, projection, collision = await _collision_projection()
    payload = dict(collision.payload)
    payload["collision_fingerprint"] = "wrong"
    changed = collision.model_copy(update={"payload": payload})

    with pytest.raises(ProjectionInvariantError, match="invalid collision audit event"):
        repository._collision_receipts(_projection_with_collision(projection, changed))


async def test_collision_receipts_reject_invalid_fingerprint_content() -> None:
    repository, projection, collision = await _collision_projection()
    payload = dict(collision.payload)
    payload["collision_fingerprint"] = {
        "algorithm": collision.payload_digest.algorithm.value,
        "value": "short",
    }
    changed = collision.model_copy(update={"payload": payload})

    with pytest.raises(ProjectionInvariantError, match="invalid collision audit event"):
        repository._collision_receipts(_projection_with_collision(projection, changed))


async def test_collision_receipts_reject_source_id_fingerprint_mismatch() -> None:
    repository, projection, collision = await _collision_projection()
    changed = collision.model_copy(update={"source_event_id": f"saliencegate:collision:{'0' * 64}"})

    with pytest.raises(ProjectionInvariantError, match="invalid collision audit event"):
        repository._collision_receipts(_projection_with_collision(projection, changed))


async def test_collision_receipts_require_existing_parent() -> None:
    repository, projection, collision = await _collision_projection()
    changed = collision.model_copy(update={"parent_ids": (RUN_B,)})

    with pytest.raises(ProjectionInvariantError, match="invalid collision audit event"):
        repository._collision_receipts(_projection_with_collision(projection, changed))
