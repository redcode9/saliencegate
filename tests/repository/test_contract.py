from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import TypeAdapter, ValidationError
from tests.repository.test_projector import RUN_ID, tag, trace_event

import saliencegate.ports as public_ports
import saliencegate.ports.repository as repository_contract
import saliencegate.repository.memory as memory_module
from saliencegate.domain import PayloadDigest, PayloadDigestAlgorithm, canonical_json
from saliencegate.ports.repository import (
    MAX_CONDITIONAL_BATCH_EVENTS,
    MAX_CONDITIONAL_BATCH_OPERATIONS,
    MAX_CONDITIONAL_BATCH_RECEIPT_BYTES,
    MAX_CONDITIONAL_BATCH_REQUEST_BYTES,
    MAX_CONDITIONAL_BATCH_SIGNALS,
    AppendDisposition,
    AppendReceipt,
    ConditionalAppendOperation,
    ConditionalBatchReceipt,
    ConditionalEventAppend,
    ConditionalSignalAppend,
    LedgerEntry,
    LedgerHead,
    LedgerHeadConflictError,
    LedgerReceipt,
    ProjectionDigests,
)
from saliencegate.repository.integrity import IntegrityContext
from saliencegate.repository.memory import MemoryRunRepository
from saliencegate.security import (
    AmbiguousDigestModeError,
    InstallationKey,
    MissingInstallationKeyError,
)

OTHER_RUN_ID = UUID("00000000-0000-4000-8000-000000000299")


def hmac_tag() -> PayloadDigest:
    return PayloadDigest(
        algorithm=PayloadDigestAlgorithm.HMAC_SHA256,
        value="f" * 64,
    )


def ledger_values() -> dict[str, object]:
    event = trace_event(1, UUID("00000000-0000-4000-8000-000000000298"))
    return {
        "run_id": RUN_ID,
        "position": 1,
        "record_key": f"trace_event:{event.event_id}",
        "record_tag": tag("a"),
        "chain_tag": tag("b"),
        "record": event,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"run_id": OTHER_RUN_ID},
        {"position": 2},
        {"previous_chain_tag": tag("c")},
        {"chain_tag": hmac_tag()},
    ],
)
def test_ledger_entry_rejects_inconsistent_envelopes(change: dict[str, object]) -> None:
    values = ledger_values()
    values.update(change)

    with pytest.raises(ValidationError):
        LedgerEntry.model_validate(values)


def test_ledger_entry_requires_previous_tag_after_genesis() -> None:
    values = ledger_values()
    values.update(position=2, previous_chain_tag=tag("c"))

    entry = LedgerEntry.model_validate(values)

    assert entry.position == 2
    assert entry.previous_chain_tag == tag("c")


def test_ledger_head_requires_one_integrity_algorithm() -> None:
    with pytest.raises(ValidationError, match="algorithms"):
        LedgerHead(
            run_id=RUN_ID,
            entry_count=1,
            chain_tag=tag("a"),
            projection_tag=tag("b"),
            head_tag=hmac_tag(),
        )


def test_ledger_head_conflict_error_is_value_free() -> None:
    error = LedgerHeadConflictError()

    assert str(error) == "ledger head does not match the conditional write precondition"
    assert not vars(error)
    assert public_ports.LedgerHeadConflictError is LedgerHeadConflictError
    assert "LedgerHeadConflictError" in public_ports.__all__


@pytest.mark.parametrize(
    "disposition",
    [AppendDisposition.APPENDED, AppendDisposition.DUPLICATE],
)
def test_non_collision_receipt_cannot_carry_an_audit_event(
    disposition: AppendDisposition,
) -> None:
    event = trace_event(1, UUID("00000000-0000-4000-8000-000000000291"))
    collision = trace_event(2, UUID("00000000-0000-4000-8000-000000000292"))

    with pytest.raises(ValidationError):
        AppendReceipt(
            disposition=disposition,
            event=event,
            ledger_position=1,
            ingestion_cursor=2,
            collision_event=collision,
        )


def test_collision_receipt_validates_run_and_cursors() -> None:
    event = trace_event(2, UUID("00000000-0000-4000-8000-000000000293"))
    collision = trace_event(3, UUID("00000000-0000-4000-8000-000000000294"))

    with pytest.raises(ValidationError):
        AppendReceipt(
            disposition=AppendDisposition.APPENDED,
            event=event,
            ledger_position=1,
            ingestion_cursor=1,
        )
    with pytest.raises(ValidationError):
        AppendReceipt(
            disposition=AppendDisposition.COLLISION,
            event=event,
            ledger_position=1,
            ingestion_cursor=2,
            collision_event=collision,
        )
    with pytest.raises(ValidationError):
        AppendReceipt(
            disposition=AppendDisposition.COLLISION,
            event=event,
            ledger_position=1,
            ingestion_cursor=3,
            collision_event=collision.model_copy(update={"run_id": OTHER_RUN_ID}),
        )


def test_conditional_batch_contract_is_public_and_bounded() -> None:
    expected = {
        "MAX_CONDITIONAL_BATCH_EVENTS": MAX_CONDITIONAL_BATCH_EVENTS,
        "MAX_CONDITIONAL_BATCH_OPERATIONS": MAX_CONDITIONAL_BATCH_OPERATIONS,
        "MAX_CONDITIONAL_BATCH_RECEIPT_BYTES": MAX_CONDITIONAL_BATCH_RECEIPT_BYTES,
        "MAX_CONDITIONAL_BATCH_REQUEST_BYTES": MAX_CONDITIONAL_BATCH_REQUEST_BYTES,
        "MAX_CONDITIONAL_BATCH_SIGNALS": MAX_CONDITIONAL_BATCH_SIGNALS,
        "ConditionalBatchReceipt": ConditionalBatchReceipt,
        "ConditionalAppendOperation": ConditionalAppendOperation,
        "ConditionalEventAppend": ConditionalEventAppend,
        "ConditionalSignalAppend": ConditionalSignalAppend,
    }

    assert MAX_CONDITIONAL_BATCH_OPERATIONS == 5_000
    assert MAX_CONDITIONAL_BATCH_EVENTS == 1_000
    assert MAX_CONDITIONAL_BATCH_SIGNALS == 4_000
    assert MAX_CONDITIONAL_BATCH_REQUEST_BYTES == 128 * 1024 * 1024
    assert MAX_CONDITIONAL_BATCH_RECEIPT_BYTES == 256 * 1024 * 1024
    for name, value in expected.items():
        assert getattr(public_ports, name) is value
        assert name in public_ports.__all__


def test_conditional_event_operation_detaches_input_and_hides_payload() -> None:
    from tests.repository.conformance import CONDITIONAL_EVENT_ID_A, event_draft

    draft = event_draft(payload={"message": "private marker"})
    operation = ConditionalEventAppend(event=draft, event_id=CONDITIONAL_EVENT_ID_A)
    object.__setattr__(draft, "source_event_id", "mutated")

    assert operation.operation == "append_event"
    assert operation.event.source_event_id == "source-1"
    assert "private marker" not in repr(operation)
    assert ConditionalEventAppend.model_validate_json(operation.model_dump_json()) == operation
    assert ConditionalEventAppend.model_validate(operation.model_dump(mode="python")) == operation
    detached = ConditionalEventAppend.model_validate(operation)
    assert type(detached) is ConditionalEventAppend
    assert detached is not operation
    adapter = TypeAdapter(ConditionalAppendOperation)
    assert adapter.validate_json(operation.model_dump_json()) == operation
    with pytest.raises(ValidationError):
        ConditionalEventAppend.model_validate(
            {"operation": "record_signal", "event": operation.event, "event_id": operation.event_id}
        )


def test_conditional_signal_operation_detaches_input_and_hides_payload() -> None:
    from tests.repository.conformance import conditional_signal

    event = trace_event(1, UUID("00000000-0000-4000-8000-000000000295"))
    signal = conditional_signal(event)
    operation = ConditionalSignalAppend(signal=signal)
    object.__setattr__(signal, "strength", 0.1)

    assert operation.operation == "record_signal"
    assert operation.signal.strength == 0.8
    assert "strength" not in repr(operation)
    assert ConditionalSignalAppend.model_validate_json(operation.model_dump_json()) == operation
    assert ConditionalSignalAppend.model_validate(operation.model_dump(mode="python")) == operation
    assert ConditionalSignalAppend.model_validate(operation) is not operation


def test_conditional_batch_receipt_validates_head_progress_and_detaches_values() -> None:
    event = trace_event(1, UUID("00000000-0000-4000-8000-000000000296"))
    event_receipt = AppendReceipt(
        disposition=AppendDisposition.APPENDED,
        event=event,
        ledger_position=1,
        ingestion_cursor=1,
    )
    final_head = LedgerHead(
        run_id=RUN_ID,
        entry_count=1,
        chain_tag=tag("a"),
        projection_tag=tag("b"),
        head_tag=tag("c"),
    )

    receipt = ConditionalBatchReceipt(
        initial_head=None,
        receipts=(event_receipt,),
        final_head=final_head,
    )
    object.__setattr__(event_receipt, "ledger_position", 99)
    object.__setattr__(final_head, "entry_count", 99)

    assert receipt.receipts[0].ledger_position == 1
    assert receipt.final_head.entry_count == 1
    assert "receipts=" not in repr(receipt)
    assert ConditionalBatchReceipt.model_validate_json(receipt.model_dump_json()) == receipt
    assert ConditionalBatchReceipt.model_validate(receipt.model_dump(mode="python")) == receipt
    assert ConditionalBatchReceipt.model_validate(receipt) is not receipt

    with pytest.raises(ValidationError, match="appended receipt count"):
        ConditionalBatchReceipt(
            initial_head=receipt.final_head,
            receipts=(
                LedgerReceipt(
                    appended=False,
                    record_id=UUID("00000000-0000-4000-8000-000000000297"),
                    record_tag=tag("a"),
                    ledger_position=1,
                    chain_tag=tag("b"),
                ),
            ),
            final_head=receipt.final_head.model_copy(update={"entry_count": 2}),
        )


def test_conditional_batch_receipt_rejects_collision_and_non_tuple_receipts() -> None:
    event = trace_event(1, UUID("00000000-0000-4000-8000-000000000288"))
    collision = trace_event(2, UUID("00000000-0000-4000-8000-000000000289"))
    collision_receipt = AppendReceipt(
        disposition=AppendDisposition.COLLISION,
        event=event,
        ledger_position=1,
        ingestion_cursor=2,
        collision_event=collision,
    )
    final_head = LedgerHead(
        run_id=RUN_ID,
        entry_count=2,
        chain_tag=tag("a"),
        projection_tag=tag("b"),
        head_tag=tag("c"),
    )

    with pytest.raises(ValidationError, match="cannot return collisions"):
        ConditionalBatchReceipt(
            initial_head=None,
            receipts=(collision_receipt,),
            final_head=final_head,
        )
    with pytest.raises(ValidationError, match="exact tuple"):
        ConditionalBatchReceipt(
            initial_head=None,
            receipts=[collision_receipt],
            final_head=final_head,
        )


def test_conditional_batch_receipt_requires_ordered_positions_and_one_algorithm() -> None:
    first = trace_event(1, UUID("00000000-0000-4000-8000-000000000286"))
    second = trace_event(2, UUID("00000000-0000-4000-8000-000000000287"))
    receipts = tuple(
        AppendReceipt(
            disposition=AppendDisposition.APPENDED,
            event=event,
            ledger_position=1,
            ingestion_cursor=index,
        )
        for index, event in enumerate((first, second), start=1)
    )
    final_head = LedgerHead(
        run_id=RUN_ID,
        entry_count=2,
        chain_tag=tag("a"),
        projection_tag=tag("b"),
        head_tag=tag("c"),
    )

    with pytest.raises(ValidationError, match="contiguous and ordered"):
        ConditionalBatchReceipt(
            initial_head=None,
            receipts=receipts,
            final_head=final_head,
        )

    with pytest.raises(ValidationError, match="algorithm"):
        ConditionalBatchReceipt(
            initial_head=final_head.model_copy(update={"entry_count": 1}),
            receipts=(
                LedgerReceipt(
                    appended=True,
                    record_id=UUID("00000000-0000-4000-8000-000000000285"),
                    record_tag=hmac_tag(),
                    ledger_position=2,
                    chain_tag=hmac_tag(),
                ),
            ),
            final_head=final_head,
        )

    with pytest.raises(ValidationError, match="genesis batch"):
        ConditionalBatchReceipt(
            initial_head=None,
            receipts=(
                LedgerReceipt(
                    appended=True,
                    record_id=UUID("00000000-0000-4000-8000-000000000283"),
                    record_tag=tag("a"),
                    ledger_position=1,
                    chain_tag=tag("b"),
                ),
            ),
            final_head=final_head.model_copy(update={"entry_count": 1}),
        )


def test_conditional_batch_receipt_noop_preserves_head_and_enforces_exact_byte_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = trace_event(1, UUID("00000000-0000-4000-8000-000000000284")).model_copy(
        update={"payload": {"message": "multibyte π🙂"}}
    )
    duplicate = AppendReceipt(
        disposition=AppendDisposition.DUPLICATE,
        event=event,
        ledger_position=1,
        ingestion_cursor=1,
    )
    initial_head = LedgerHead(
        run_id=RUN_ID,
        entry_count=1,
        chain_tag=tag("a"),
        projection_tag=tag("b"),
        head_tag=tag("c"),
    )
    receipt = ConditionalBatchReceipt(
        initial_head=initial_head,
        receipts=(duplicate,),
        final_head=initial_head,
    )

    with pytest.raises(ValidationError, match="preserve the exact"):
        ConditionalBatchReceipt(
            initial_head=initial_head,
            receipts=(duplicate,),
            final_head=initial_head.model_copy(update={"head_tag": tag("d")}),
        )

    monkeypatch.setattr(repository_contract, "MAX_CONDITIONAL_BATCH_EVENTS", 1)
    with pytest.raises(ValidationError, match="operation-kind limit"):
        ConditionalBatchReceipt(
            initial_head=initial_head,
            receipts=(duplicate, duplicate),
            final_head=initial_head,
        )

    encoded_size = len(canonical_json(receipt))
    assert encoded_size > len(canonical_json(receipt).decode("utf-8"))
    monkeypatch.setattr(
        repository_contract,
        "MAX_CONDITIONAL_BATCH_RECEIPT_BYTES",
        encoded_size,
    )
    assert ConditionalBatchReceipt.model_validate_json(receipt.model_dump_json()) == receipt
    monkeypatch.setattr(
        repository_contract,
        "MAX_CONDITIONAL_BATCH_RECEIPT_BYTES",
        encoded_size - 1,
    )
    with pytest.raises(ValidationError, match="canonical byte limit"):
        ConditionalBatchReceipt.model_validate_json(receipt.model_dump_json())


def test_projection_digests_require_one_integrity_algorithm() -> None:
    values = {field_name: tag("a") for field_name in ProjectionDigests.model_fields}
    values["overall"] = hmac_tag()

    with pytest.raises(ValidationError, match="algorithms"):
        ProjectionDigests.model_validate(values)


def test_integrity_context_requires_exactly_one_mode_and_verifies_tags() -> None:
    key = InstallationKey(b"k" * 32)
    with pytest.raises(AmbiguousDigestModeError):
        IntegrityContext(key=key, synthetic_benchmark=True)
    with pytest.raises(MissingInstallationKeyError):
        IntegrityContext(key=None, synthetic_benchmark=False)

    keyed = IntegrityContext(key=key, synthetic_benchmark=False)
    first = keyed.tag({"value": "safe"}, domain="fixture:first")
    second = keyed.tag({"value": "safe"}, domain="fixture:second")
    assert first.algorithm is PayloadDigestAlgorithm.HMAC_SHA256
    assert first != second
    assert keyed.verify({"value": "safe"}, first, domain="fixture:first")
    assert not keyed.verify({"value": "changed"}, first, domain="fixture:first")
    assert "kkkk" not in repr(keyed)

    synthetic = IntegrityContext(key=None, synthetic_benchmark=True)
    assert (
        synthetic.tag(
            {"value": "safe"},
            domain="fixture:first",
        ).algorithm
        is PayloadDigestAlgorithm.SYNTHETIC_SHA256
    )
    assert "synthetic" in repr(synthetic)

    class InstallationKeySubclass(InstallationKey):
        pass

    with pytest.raises(TypeError, match="exactly InstallationKey"):
        IntegrityContext(
            key=InstallationKeySubclass(b"k" * 32),
            synthetic_benchmark=False,
        )


def test_repository_defaults_load_an_installation_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        memory_module,
        "load_or_create_installation_key",
        lambda: InstallationKey(b"k" * 32),
    )

    repository = MemoryRunRepository()

    assert repr(repository) == "MemoryRunRepository(runs=0)"
