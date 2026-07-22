"""Coverage-oriented regressions for otherwise rare contract edges."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError
from tests.repository.test_contract import OTHER_RUN_ID, hmac_tag
from tests.repository.test_projector import RUN_ID, tag, trace_event
from tests.runtime.test_cycles import (
    MODEL_DIGEST,
    NOW,
    CycleCommandFactory,
    cycle,
    intervention,
    memory_delta,
    reservation,
)
from tests.runtime.test_fixed_step_runtime import (
    _DeliveryAdapter,
    _make_repository,
    _run,
    _run_start,
)

import saliencegate.ports.repository as repository_module
from saliencegate.domain import (
    CycleState,
    DeduplicationGuarantee,
    DeliveryRecord,
    DeliveryTarget,
    PayloadDigestAlgorithm,
)
from saliencegate.ports.repository import (
    AppendDisposition,
    AppendReceipt,
    ConditionalBatchReceipt,
    CycleReceipt,
    DeliveryNotFoundError,
    EnqueueDelivery,
    InvalidRecoveryTimeError,
    LedgerHead,
    MemoryDeltaPreview,
    PreviewMemoryDelta,
)


def _head(*, run_id: UUID = RUN_ID) -> LedgerHead:
    return LedgerHead(
        run_id=run_id,
        entry_count=1,
        chain_tag=tag("a"),
        projection_tag=tag("b"),
        head_tag=tag("c"),
    )


def _batch() -> ConditionalBatchReceipt:
    event = trace_event(1, UUID("00000000-0000-4000-8000-00000000b201"))
    duplicate = AppendReceipt(
        disposition=AppendDisposition.DUPLICATE,
        event=event,
        ledger_position=1,
        ingestion_cursor=1,
    )
    head = _head()
    return ConditionalBatchReceipt(
        initial_head=head,
        receipts=(duplicate,),
        final_head=head,
    )


@pytest.fixture(scope="module")
def reminder_receipt_and_preview() -> tuple[CycleReceipt, MemoryDeltaPreview]:
    async def build() -> tuple[CycleReceipt, MemoryDeltaPreview]:
        repository = _make_repository("memory", Path("unused-port-contract.sqlite3"))
        result, _client = await _run(
            repository,
            (_run_start(target_request_id="port-contract-coverage-request"),),
            mode="reminder",
            cycle_capacity=1,
            delivery_adapter=_DeliveryAdapter("deliver"),
            requested_delivery_target=DeliveryTarget.NEXT_MODEL_CALL,
        )
        snapshot = await repository.snapshot(result.run_id)
        pending_delivery = next(
            entry.record
            for entry in await repository.ledger(result.run_id)
            if type(entry.record) is DeliveryRecord and entry.record.revision == 1
        )
        receipt = CycleReceipt(
            appended=True,
            cycle=result.cycles[0],
            record_tag=tag("d"),
            ledger_position=1,
            chain_tag=tag("e"),
            budget_snapshot=await repository.budget_snapshot(result.run_id),
            delivery=pending_delivery,
        )
        preview = MemoryDeltaPreview(
            schema_version="memory-delta-preview/v1",
            run_id=result.run_id,
            command_digest="a" * 64,
            source_ledger_position=snapshot.ledger_position,
            source_ingestion_cursor=snapshot.ingestion_cursor,
            source_memory_cursor=snapshot.memory_cursor,
            source_projection_digest=snapshot.projection_digest,
            records=snapshot.records,
            preview_projection_digest=snapshot.projection_digest,
        )
        return receipt, preview

    return asyncio.run(build())


def test_value_free_repository_errors_cover_delivery_and_recovery_boundaries() -> None:
    assert str(DeliveryNotFoundError()) == "delivery was not found in the authoritative ledger"
    assert str(InvalidRecoveryTimeError()) == "cycle recovery time must be an exact UTC datetime"


def test_repository_model_revalidation_rejects_every_input_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="LedgerHead failed validation"):
        repository_module._validated_repository_model({}, LedgerHead)
    with pytest.raises(ValueError, match="must be an object"):
        repository_module._validated_repository_model(object(), LedgerHead, json_mode=True)
    with pytest.raises(ValueError, match="must be exactly"):
        repository_module._validated_repository_model(object(), LedgerHead)

    head = _head()
    monkeypatch.setattr(LedgerHead, "model_dump_json", staticmethod(lambda *_a, **_k: 1 / 0))
    with pytest.raises(ValueError, match="LedgerHead failed validation"):
        repository_module._validated_repository_model(head, LedgerHead)


@pytest.mark.parametrize("unsupported", ({}, {"disposition": "duplicate", "appended": False}))
def test_conditional_receipt_rejects_ambiguous_mapping_shapes(
    unsupported: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="unsupported shape"):
        ConditionalBatchReceipt(
            initial_head=_head(),
            receipts=(unsupported,),
            final_head=_head(),
        )


def test_conditional_receipt_rejects_an_untyped_receipt() -> None:
    with pytest.raises(ValidationError, match="unsupported type"):
        ConditionalBatchReceipt(
            initial_head=_head(),
            receipts=(object(),),
            final_head=_head(),
        )


def test_conditional_receipt_rejects_head_identity_and_algorithm_drift() -> None:
    receipt = _batch()
    other_run = receipt.initial_head.model_copy(update={"run_id": OTHER_RUN_ID})
    with pytest.raises(ValueError, match="different runs"):
        receipt.model_copy(update={"initial_head": other_run}).batch_receipt_is_consistent()

    other_algorithm = receipt.initial_head.model_copy(update={"head_tag": hmac_tag()})
    with pytest.raises(ValueError, match="head algorithms"):
        receipt.model_copy(update={"initial_head": other_algorithm}).batch_receipt_is_consistent()


def test_conditional_receipt_rejects_event_and_position_drift() -> None:
    batch = _batch()
    source = batch.receipts[0]
    assert isinstance(source, AppendReceipt)

    wrong_run_event = source.event.model_copy(update={"run_id": OTHER_RUN_ID})
    wrong_run = source.model_copy(update={"event": wrong_run_event})
    with pytest.raises(ValueError, match="different run"):
        batch.model_copy(update={"receipts": (wrong_run,)}).batch_receipt_is_consistent()

    wrong_digest = source.event.payload_digest.model_copy(
        update={"algorithm": PayloadDigestAlgorithm.HMAC_SHA256}
    )
    wrong_event = source.event.model_copy(update={"payload_digest": wrong_digest})
    wrong_algorithm = source.model_copy(update={"event": wrong_event})
    with pytest.raises(ValueError, match="integrity algorithm"):
        batch.model_copy(update={"receipts": (wrong_algorithm,)}).batch_receipt_is_consistent()

    future = source.model_copy(update={"ledger_position": 2})
    with pytest.raises(ValueError, match="future ledger position"):
        batch.model_copy(update={"receipts": (future,)}).batch_receipt_is_consistent()


def test_commit_cycle_requires_delivery_exactly_for_a_reminder() -> None:
    command = CycleCommandFactory().commit(
        cycle(CycleState.RUNNING),
        settlement=reservation(),
        validated_delta=memory_delta(),
        memory_id_assignments=(),
        intervention=intervention(),
        model_call_digests=(MODEL_DIGEST,),
        model_call_latencies_us=(reservation().latency_us,),
        updated_at=NOW + timedelta(seconds=4),
    )
    delivery = EnqueueDelivery(
        target_request_id="port-contract-target-request",
        adapter_id="port-contract-adapter/v1",
        adapter_deduplicates=True,
        adapter_deduplication_guarantee=DeduplicationGuarantee.DURABLE_DELIVERY_ID,
        adapter_supports_pre_action=False,
        adapter_contract_version="port-contract-adapter-contract/v1",
        adapter_capabilities_digest="f" * 64,
    )

    with pytest.raises(ValueError, match="only a reminder"):
        command.model_copy(update={"delivery": delivery}).reconcile_model_call_receipts()


def test_preview_command_rejects_run_and_cursor_drift() -> None:
    valid = PreviewMemoryDelta(
        schema_version="memory-delta-preview-command/v1",
        run_id=RUN_ID,
        expected_ledger_position=1,
        expected_ingestion_cursor=1,
        expected_memory_cursor=1,
        expected_projection_digest=tag("a"),
        last_event_sequence=1,
        delta=memory_delta(run_id=RUN_ID),
    )

    with pytest.raises(ValueError, match="different run"):
        valid.model_copy(
            update={"delta": memory_delta(run_id=OTHER_RUN_ID)}
        ).delta_and_anchor_match_the_run()
    with pytest.raises(ValueError, match="cursor anchor"):
        valid.model_copy(update={"expected_memory_cursor": 2}).delta_and_anchor_match_the_run()
    with pytest.raises(ValueError, match="event range"):
        valid.model_copy(update={"last_event_sequence": 2}).delta_and_anchor_match_the_run()


def test_cycle_receipt_rejects_deep_delivery_drift(
    reminder_receipt_and_preview: tuple[CycleReceipt, MemoryDeltaPreview],
) -> None:
    receipt, _preview = reminder_receipt_and_preview
    assert receipt.delivery is not None
    delivery = receipt.delivery.model_copy(update={"run_id": OTHER_RUN_ID})

    with pytest.raises(ValueError, match="does not match"):
        receipt.model_copy(update={"delivery": delivery}).delivery_matches_committed_cycle()


def test_memory_preview_rejects_record_identity_and_anchor_drift(
    reminder_receipt_and_preview: tuple[CycleReceipt, MemoryDeltaPreview],
) -> None:
    _receipt, preview = reminder_receipt_and_preview
    assert len(preview.records) == 1
    record = preview.records[0]

    cross_run = record.model_copy(update={"run_id": OTHER_RUN_ID})
    with pytest.raises(ValueError, match="different run"):
        preview.model_copy(update={"records": (cross_run,)}).records_belong_to_the_preview_run()
    with pytest.raises(ValueError, match="unique IDs"):
        preview.model_copy(update={"records": (record, record)}).records_belong_to_the_preview_run()
    with pytest.raises(ValueError, match="cursor anchor"):
        preview.model_copy(
            update={"source_memory_cursor": preview.source_ingestion_cursor + 1}
        ).records_belong_to_the_preview_run()
    with pytest.raises(ValueError, match="private status is unavailable"):
        preview.model_copy(
            update={"current_private_status_id": UUID("00000000-0000-4000-8000-00000000b202")}
        ).records_belong_to_the_preview_run()
