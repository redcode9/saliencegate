"""Coverage-oriented regressions for otherwise rare contract edges."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from tests.shadow.conftest import NOW, OTHER_RUN_ID, RUN_ID, TraceEventFactory
from tests.shadow.test_trace import build_trace

import saliencegate.shadow.session as session_module
from saliencegate.domain import TrustLabel
from saliencegate.ports.repository import (
    AppendDisposition,
    AppendReceipt,
    LedgerEntry,
    LedgerHead,
)
from saliencegate.repository import MemoryRunRepository, SQLiteRunRepository
from saliencegate.security import InstallationKey, RedactionPolicy
from saliencegate.security.files import inspect_private_file_location
from saliencegate.shadow import ShadowSession
from saliencegate.shadow.errors import (
    ShadowConfigurationError,
    ShadowInputError,
    ShadowInvariantError,
    ShadowStateError,
)
from saliencegate.shadow.inputs import (
    SHADOW_PROJECTION_MATRIX,
    ShadowFinishInput,
    ShadowInputKind,
    ShadowStartInput,
    derive_shadow_event_id,
)

_KEY = InstallationKey(b"m" * 32)


def _memory_session() -> ShadowSession:
    return ShadowSession.in_memory(run_id=RUN_ID, installation_key=_KEY)


@pytest.fixture
def seeded_ledger() -> tuple[ShadowSession, LedgerEntry, LedgerHead]:
    session = _memory_session()

    async def seed() -> tuple[LedgerEntry, LedgerHead]:
        await session.start(source_event_id="seed-start", occurred_at=NOW)
        assert isinstance(session._repository, MemoryRunRepository)
        entries = await session._repository.ledger(RUN_ID)
        head = await session._repository.ledger_head(RUN_ID)
        return entries[0], head

    entry, head = asyncio.run(seed())
    return session, entry, head


def _head_with_count(template: LedgerHead, count: int) -> LedgerHead:
    return template.model_copy(update={"entry_count": count})


def _entries_for_events(
    template: LedgerEntry,
    events: tuple[Any, ...],
) -> tuple[LedgerEntry, ...]:
    return tuple(
        template.model_copy(
            update={
                "position": position,
                "record_key": f"tail-event:{position}",
                "previous_chain_tag": None if position == 1 else template.chain_tag,
                "record": event,
            }
        )
        for position, event in enumerate(events, start=1)
    )


def _event_for_kind(
    trace_event_factory: TraceEventFactory,
    sequence: int,
    kind: ShadowInputKind,
    *,
    parent_ids: tuple[UUID, ...] = (),
    trust_label: TrustLabel | None = None,
) -> Any:
    projection = SHADOW_PROJECTION_MATRIX[kind]
    return trace_event_factory(
        sequence,
        event_type=projection.event_type,
        phase=projection.phase,
        payload={projection.payload_namespace: {}},
        parent_ids=parent_ids,
        trust_label=projection.trust_label if trust_label is None else trust_label,
    )


def test_redaction_policy_copy_rejects_constructor_equality_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = RedactionPolicy()
    monkeypatch.setattr(RedactionPolicy, "__eq__", lambda _self, _other: False)

    with pytest.raises(ValueError, match="redaction policy is invalid"):
        session_module._copy_redaction_policy(policy)


def test_memory_factory_preserves_state_failure_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_bind(_cls: type[ShadowSession], *_args: object, **_kwargs: object) -> ShadowSession:
        raise ShadowStateError()

    monkeypatch.setattr(ShadowSession, "_bind", classmethod(fail_bind))

    with pytest.raises(ShadowStateError):
        ShadowSession.in_memory(run_id=RUN_ID, installation_key=_KEY)


def test_sqlite_authorization_binding_rejects_coercible_authorization() -> None:
    session = _memory_session()

    with pytest.raises(ShadowConfigurationError):
        ShadowSession._bind_sqlite_authorization(object(), session._options)  # type: ignore[arg-type]


def test_sqlite_authorization_binding_closes_before_propagating_interrupt(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _memory_session()
    authorization = inspect_private_file_location(tmp_path / "tail.sqlite3")
    closed: list[bool] = []

    class FakeRepository:
        def close(self) -> None:
            closed.append(True)

    repository = FakeRepository()
    monkeypatch.setattr(
        SQLiteRunRepository,
        "_from_file_authorization",
        classmethod(lambda _cls, *_args, **_kwargs: repository),
    )

    def interrupt_bind(
        _cls: type[ShadowSession],
        *_args: object,
        **_kwargs: object,
    ) -> ShadowSession:
        raise KeyboardInterrupt()

    monkeypatch.setattr(ShadowSession, "_bind", classmethod(interrupt_bind))

    with pytest.raises(KeyboardInterrupt):
        ShadowSession._bind_sqlite_authorization(authorization, session._options)
    assert closed == [True]


def test_sqlite_authorization_binding_propagates_constructor_interrupt(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _memory_session()
    authorization = inspect_private_file_location(tmp_path / "constructor-interrupt.sqlite3")

    def interrupt_open(_cls: type[SQLiteRunRepository], *_args: object, **_kwargs: object) -> Any:
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        SQLiteRunRepository,
        "_from_file_authorization",
        classmethod(interrupt_open),
    )

    with pytest.raises(KeyboardInterrupt):
        ShadowSession._bind_sqlite_authorization(authorization, session._options)


def test_repository_lookup_rejects_closed_unmaterialized_session() -> None:
    session = _memory_session()
    session._repository = None
    session._closed = True

    with pytest.raises(ShadowStateError):
        session._repository_for_operation()


@pytest.mark.asyncio
async def test_trace_batch_enters_the_open_bound_path(monkeypatch: pytest.MonkeyPatch) -> None:
    trace = build_trace()
    session = ShadowSession.in_memory_for_trace(
        run_id=trace.run_id,
        trace_binding=trace.binding,
        installation_key=_KEY,
    )
    marker = object()

    async def accepted(
        _session: ShadowSession,
        _operations: object,
        *,
        expected_head: object,
    ) -> object:
        assert expected_head is None
        return marker

    monkeypatch.setattr(ShadowSession, "_append_trace_batch_locked", accepted)

    assert await session._append_trace_batch((), expected_head=None) is marker


def test_public_input_rejects_model_returning_a_non_input() -> None:
    class WrongModel:
        @staticmethod
        def model_validate(_values: object) -> object:
            return object()

    with pytest.raises(ShadowInputError):
        ShadowSession._public_input(WrongModel, source_event_id="tail")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_submit_maps_internal_state_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _memory_session()
    value = ShadowStartInput(source_event_id="tail-start", occurred_at=NOW)

    async def fail_state(
        _session: ShadowSession,
        _value: object,
        *,
        cli_input_ordinal: int | None,
    ) -> Any:
        assert cli_input_ordinal is None
        raise ShadowStateError()

    monkeypatch.setattr(ShadowSession, "_submit_locked", fail_state)

    with pytest.raises(ShadowStateError):
        await session._submit(value, cli_input_ordinal=None)


@pytest.mark.asyncio
async def test_submit_maps_unknown_failures_to_invariant_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _memory_session()
    value = ShadowStartInput(source_event_id="tail-unknown", occurred_at=NOW)

    async def fail_unknown(
        _session: ShadowSession,
        _value: object,
        *,
        cli_input_ordinal: int | None,
    ) -> Any:
        assert cli_input_ordinal is None
        raise ValueError("unknown")

    monkeypatch.setattr(ShadowSession, "_submit_locked", fail_unknown)

    with pytest.raises(ShadowInvariantError):
        await session._submit(value, cli_input_ordinal=None)


@pytest.mark.asyncio
async def test_submit_locked_rejects_invalid_cli_ordinal() -> None:
    session = _memory_session()
    value = ShadowStartInput(source_event_id="tail-start", occurred_at=NOW)

    with pytest.raises(ShadowInputError):
        await session._submit_locked(value, cli_input_ordinal=0)


@pytest.mark.asyncio
async def test_start_marker_must_be_redaction_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _memory_session()
    value = ShadowStartInput(source_event_id="tail-start", occurred_at=NOW)
    monkeypatch.setattr(session_module, "_marker_is_redaction_identity", lambda *_args: False)

    with pytest.raises(ShadowConfigurationError):
        await session._submit_locked(value, cli_input_ordinal=None)


@pytest.mark.asyncio
async def test_finish_marker_must_be_redaction_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _memory_session()
    await session.start(source_event_id="tail-start", occurred_at=NOW)
    value = ShadowFinishInput(source_event_id="tail-finish", occurred_at=NOW + timedelta(seconds=1))
    monkeypatch.setattr(session_module, "_marker_is_redaction_identity", lambda *_args: False)

    with pytest.raises(ShadowConfigurationError):
        await session._submit_locked(value, cli_input_ordinal=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ("collision", "different_event"))
async def test_submit_locked_rejects_invalid_append_receipts(
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
    session = _memory_session()
    value = ShadowStartInput(source_event_id=f"tail-{scenario}", occurred_at=NOW)

    async def no_state(_session: ShadowSession) -> None:
        return None

    async def invalid_receipt(
        _repository: MemoryRunRepository,
        draft: Any,
        *,
        event_id: UUID,
        expected_head: LedgerHead | None,
    ) -> AppendReceipt:
        assert expected_head is None
        candidate = session._preflight_event(
            draft,
            kind=ShadowInputKind.START,
            event_id=event_id,
            sequence=1,
        )
        if scenario == "collision":
            return AppendReceipt(
                disposition=AppendDisposition.COLLISION,
                event=candidate,
                collision_event=candidate,
                ledger_position=1,
                ingestion_cursor=1,
            )
        different = candidate.model_copy(update={"sequence": 2})
        return AppendReceipt(
            disposition=AppendDisposition.APPENDED,
            event=different,
            ledger_position=1,
            ingestion_cursor=2,
        )

    monkeypatch.setattr(ShadowSession, "_load_state", no_state)
    monkeypatch.setattr(MemoryRunRepository, "append_event_if_head", invalid_receipt)

    expected = ShadowInputError if scenario == "collision" else ShadowStateError
    with pytest.raises(expected):
        await session._submit_locked(value, cli_input_ordinal=None)


async def _patch_appended_start(
    session: ShadowSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_state(_session: ShadowSession) -> None:
        return None

    async def appended(
        _repository: MemoryRunRepository,
        draft: Any,
        *,
        event_id: UUID,
        expected_head: LedgerHead | None,
    ) -> AppendReceipt:
        candidate = session._preflight_event(
            draft,
            kind=ShadowInputKind.START,
            event_id=event_id,
            sequence=1,
        )
        return AppendReceipt(
            disposition=AppendDisposition.APPENDED,
            event=candidate,
            ledger_position=1,
            ingestion_cursor=1,
        )

    monkeypatch.setattr(ShadowSession, "_load_state", no_state)
    monkeypatch.setattr(MemoryRunRepository, "append_event_if_head", appended)


@pytest.mark.asyncio
async def test_submit_locked_rejects_prefix_drift_after_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _memory_session()
    await _patch_appended_start(session, monkeypatch)
    value = ShadowStartInput(source_event_id="tail-prefix", occurred_at=NOW)
    state = SimpleNamespace(signals=())

    async def loaded(_session: ShadowSession) -> Any:
        return state

    calls = 0

    def drifting_prefix(_state: object, event: Any) -> tuple[Any, ...]:
        nonlocal calls
        calls += 1
        return (event,) if calls == 1 else ()

    monkeypatch.setattr(ShadowSession, "_load_state_required", loaded)
    monkeypatch.setattr(ShadowSession, "_prefix_for_receipt", staticmethod(drifting_prefix))

    with pytest.raises(ShadowStateError):
        await session._submit_locked(value, cli_input_ordinal=None)


@pytest.mark.asyncio
async def test_submit_locked_rechecks_persisted_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _memory_session()
    await _patch_appended_start(session, monkeypatch)
    value = ShadowStartInput(source_event_id="tail-signals", occurred_at=NOW)
    state = SimpleNamespace(signals=())
    signal = SimpleNamespace(signal_id=uuid4())

    async def loaded(_session: ShadowSession) -> Any:
        return state

    async def ignore_signal(
        _session: ShadowSession,
        _signal: object,
        *,
        expected_prefix: object,
    ) -> None:
        assert expected_prefix

    monkeypatch.setattr(ShadowSession, "_load_state_required", loaded)
    monkeypatch.setattr(
        ShadowSession,
        "_prefix_for_receipt",
        staticmethod(lambda _state, event: (event,)),
    )
    monkeypatch.setattr(
        ShadowSession,
        "_extract_report",
        lambda _session, _context: SimpleNamespace(signals=(signal,)),
    )
    monkeypatch.setattr(ShadowSession, "_persist_signal", ignore_signal)

    with pytest.raises(ShadowStateError):
        await session._submit_locked(value, cli_input_ordinal=None)


@pytest.mark.asyncio
async def test_submit_locked_fails_closed_when_result_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _memory_session()
    await _patch_appended_start(session, monkeypatch)
    value = ShadowStartInput(source_event_id="tail-result", occurred_at=NOW)
    state = SimpleNamespace(signals=())

    async def loaded(_session: ShadowSession) -> Any:
        return state

    def fail_result(**_kwargs: object) -> Any:
        raise ValueError("drift")

    monkeypatch.setattr(ShadowSession, "_load_state_required", loaded)
    monkeypatch.setattr(
        ShadowSession,
        "_prefix_for_receipt",
        staticmethod(lambda _state, event: (event,)),
    )
    monkeypatch.setattr(
        session_module,
        "derive_shadow_feature_snapshot_digest",
        lambda **_kwargs: "0" * 64,
    )
    monkeypatch.setattr(
        session_module,
        "evaluate_shadow_heuristic",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        session_module,
        "_build_shadow_observation_from_selection",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(session_module, "ShadowEventResult", fail_result)

    with pytest.raises(ShadowInvariantError):
        await session._submit_locked(value, cli_input_ordinal=None)


def test_preflight_event_rejects_metadata_redaction_drift() -> None:
    session = _memory_session()
    session._redactor = SimpleNamespace(
        redact_payload=lambda _payload: SimpleNamespace(
            payload=SimpleNamespace(root={"changed": True})
        )
    )
    draft = SimpleNamespace(source_event_id="tail", source_adapter="tail-adapter")

    with pytest.raises(ShadowInputError):
        session._preflight_event(
            draft,  # type: ignore[arg-type]
            kind=ShadowInputKind.START,
            event_id=uuid4(),
            sequence=1,
        )


def test_authorization_rejects_a_second_start() -> None:
    session = _memory_session()
    value = ShadowStartInput(source_event_id="tail-start", occurred_at=NOW)
    state = SimpleNamespace(events_by_source={}, finish=None, events=(object(),))

    with pytest.raises(ShadowInputError):
        session._authorize_input(value, ShadowInputKind.START, state)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_required_state_bounds_repeated_snapshot_races(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _memory_session()

    async def race(_session: ShadowSession) -> Any:
        raise session_module._RetryableSnapshotRaceError()

    monkeypatch.setattr(ShadowSession, "_load_state", race)

    with pytest.raises(ShadowStateError):
        await session._load_state_required()


@pytest.mark.asyncio
async def test_required_state_rejects_absent_run(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _memory_session()

    async def absent(_session: ShadowSession) -> None:
        return None

    monkeypatch.setattr(ShadowSession, "_load_state", absent)

    with pytest.raises(ShadowStateError):
        await session._load_state_required()


@pytest.mark.asyncio
async def test_state_load_rejects_non_tuple_ledger_snapshot() -> None:
    session = _memory_session()

    class Repository:
        async def ledger(self, _run_id: UUID) -> list[object]:
            return []

        async def ledger_head(self, _run_id: UUID) -> object:
            return object()

    session._repository = Repository()  # type: ignore[assignment]

    with pytest.raises(session_module._RetryableSnapshotRaceError):
        await session._load_state()


def test_run_state_rejects_head_identity_mismatch(
    seeded_ledger: tuple[ShadowSession, LedgerEntry, LedgerHead],
) -> None:
    session, entry, head = seeded_ledger
    damaged = head.model_copy(update={"run_id": OTHER_RUN_ID})

    with pytest.raises(ShadowStateError):
        session._validate_run_state((entry,), damaged)


def test_run_state_rejects_non_entry_records(
    seeded_ledger: tuple[ShadowSession, LedgerEntry, LedgerHead],
) -> None:
    session, _entry, head = seeded_ledger

    with pytest.raises(ShadowStateError):
        session._validate_run_state((object(),), head)  # type: ignore[arg-type]


def test_run_state_requires_at_least_one_trace_event(
    seeded_ledger: tuple[ShadowSession, LedgerEntry, LedgerHead],
) -> None:
    session, _entry, head = seeded_ledger

    with pytest.raises(ShadowStateError):
        session._validate_run_state((), _head_with_count(head, 0))


def test_run_state_enforces_event_count_bound(
    seeded_ledger: tuple[ShadowSession, LedgerEntry, LedgerHead],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, entry, head = seeded_ledger
    monkeypatch.setattr(session_module, "_MAX_SHADOW_EVENTS", 0)

    with pytest.raises(ShadowStateError):
        session._validate_run_state((entry,), head)


def test_run_state_rejects_duplicate_record_positions(
    seeded_ledger: tuple[ShadowSession, LedgerEntry, LedgerHead],
) -> None:
    session, entry, head = seeded_ledger
    duplicate = entry.model_copy(update={"position": 2, "previous_chain_tag": entry.chain_tag})

    with pytest.raises(ShadowStateError):
        session._validate_run_state((entry, duplicate), _head_with_count(head, 2))


def test_run_state_rejects_noncontiguous_event_sequences(
    seeded_ledger: tuple[ShadowSession, LedgerEntry, LedgerHead],
) -> None:
    session, entry, head = seeded_ledger
    first = entry.record
    source_id = "tail-sequence"
    second = first.model_copy(
        update={
            "event_id": derive_shadow_event_id(RUN_ID, source_id),
            "source_event_id": source_id,
            "sequence": 3,
            "timestamp": first.timestamp + timedelta(seconds=1),
        }
    )
    entries = _entries_for_events(entry, (first, second))

    with pytest.raises(ShadowStateError):
        session._validate_run_state(entries, _head_with_count(head, 2))


def test_run_state_rejects_duplicate_source_identities(
    seeded_ledger: tuple[ShadowSession, LedgerEntry, LedgerHead],
) -> None:
    session, entry, head = seeded_ledger
    first = entry.record
    second = first.model_copy(update={"event_id": uuid4(), "sequence": 2})
    entries = _entries_for_events(entry, (first, second))

    with pytest.raises(ShadowStateError):
        session._validate_run_state(entries, _head_with_count(head, 2))


def test_run_state_rejects_event_binding_drift(
    seeded_ledger: tuple[ShadowSession, LedgerEntry, LedgerHead],
) -> None:
    session, entry, head = seeded_ledger
    damaged = entry.record.model_copy(update={"source_adapter": "tail-drift"})

    with pytest.raises(ShadowStateError):
        session._validate_run_state(
            _entries_for_events(entry, (damaged,)),
            head,
        )


def test_run_state_rejects_nonmonotonic_timestamps(
    seeded_ledger: tuple[ShadowSession, LedgerEntry, LedgerHead],
) -> None:
    session, entry, head = seeded_ledger
    first = entry.record
    source_id = "tail-time"
    second = first.model_copy(
        update={
            "event_id": derive_shadow_event_id(RUN_ID, source_id),
            "source_event_id": source_id,
            "sequence": 2,
            "timestamp": first.timestamp - timedelta(seconds=1),
        }
    )

    with pytest.raises(ShadowStateError):
        session._validate_run_state(
            _entries_for_events(entry, (first, second)),
            _head_with_count(head, 2),
        )


def test_run_state_requires_a_recognized_start_kind(
    seeded_ledger: tuple[ShadowSession, LedgerEntry, LedgerHead],
) -> None:
    session, entry, head = seeded_ledger
    damaged = entry.record.model_copy(update={"payload": {"unknown": {}}})

    with pytest.raises(ShadowStateError):
        session._validate_run_state(_entries_for_events(entry, (damaged,)), head)


def test_run_state_rejects_multiple_start_markers(
    seeded_ledger: tuple[ShadowSession, LedgerEntry, LedgerHead],
    trace_event_factory: TraceEventFactory,
) -> None:
    session, entry, head = seeded_ledger
    second = _event_for_kind(trace_event_factory, 2, ShadowInputKind.START)

    with pytest.raises(ShadowStateError):
        session._validate_run_state(
            _entries_for_events(entry, (entry.record, second)),
            _head_with_count(head, 2),
        )


def test_run_state_requires_finish_to_be_unique_and_final(
    seeded_ledger: tuple[ShadowSession, LedgerEntry, LedgerHead],
    trace_event_factory: TraceEventFactory,
) -> None:
    session, entry, head = seeded_ledger
    finish = _event_for_kind(trace_event_factory, 2, ShadowInputKind.FINISH)
    action = _event_for_kind(trace_event_factory, 3, ShadowInputKind.ACTION)

    with pytest.raises(ShadowStateError):
        session._validate_run_state(
            _entries_for_events(entry, (entry.record, finish, action)),
            _head_with_count(head, 3),
        )


def test_run_state_rechecks_start_marker_payload(
    seeded_ledger: tuple[ShadowSession, LedgerEntry, LedgerHead],
) -> None:
    session, entry, head = seeded_ledger
    damaged = entry.record.model_copy(update={"payload": {"shadow_run": {"drift": True}}})

    with pytest.raises(ShadowStateError):
        session._validate_run_state(_entries_for_events(entry, (damaged,)), head)


def test_run_state_rechecks_finish_marker_payload(
    seeded_ledger: tuple[ShadowSession, LedgerEntry, LedgerHead],
    trace_event_factory: TraceEventFactory,
) -> None:
    session, entry, head = seeded_ledger
    finish = _event_for_kind(trace_event_factory, 2, ShadowInputKind.FINISH)

    with pytest.raises(ShadowStateError):
        session._validate_run_state(
            _entries_for_events(entry, (entry.record, finish)),
            _head_with_count(head, 2),
        )


def test_run_state_rechecks_projected_payloads(
    seeded_ledger: tuple[ShadowSession, LedgerEntry, LedgerHead],
    trace_event_factory: TraceEventFactory,
) -> None:
    session, entry, head = seeded_ledger
    action = _event_for_kind(trace_event_factory, 2, ShadowInputKind.ACTION)

    with pytest.raises(ShadowStateError):
        session._validate_run_state(
            _entries_for_events(entry, (entry.record, action)),
            _head_with_count(head, 2),
        )


@pytest.mark.parametrize("scenario", ("parent_count", "invalid_parent", "unexpected_parent"))
def test_run_state_rechecks_parent_topology(
    seeded_ledger: tuple[ShadowSession, LedgerEntry, LedgerHead],
    trace_event_factory: TraceEventFactory,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
    session, entry, head = seeded_ledger
    if scenario == "parent_count":
        event = _event_for_kind(trace_event_factory, 2, ShadowInputKind.TOOL_RESULT)
    elif scenario == "invalid_parent":
        event = _event_for_kind(
            trace_event_factory,
            2,
            ShadowInputKind.TOOL_RESULT,
            parent_ids=(uuid4(),),
        )
    else:
        event = _event_for_kind(
            trace_event_factory,
            2,
            ShadowInputKind.ACTION,
            parent_ids=(entry.record.event_id,),
        )
    monkeypatch.setattr(ShadowSession, "_payload_is_valid", lambda *_args: True)

    with pytest.raises(ShadowStateError):
        session._validate_run_state(
            _entries_for_events(entry, (entry.record, event)),
            _head_with_count(head, 2),
        )


def test_event_kind_falls_through_for_unrecognized_projection(
    seeded_ledger: tuple[ShadowSession, LedgerEntry, LedgerHead],
) -> None:
    session, entry, _head = seeded_ledger
    event = entry.record.model_copy(update={"payload": {"unknown": {}}})

    assert session._event_kind(event) is None


def test_event_kind_rejects_invalid_observation_trust(
    seeded_ledger: tuple[ShadowSession, LedgerEntry, LedgerHead],
    trace_event_factory: TraceEventFactory,
) -> None:
    session, _entry, _head = seeded_ledger
    event = _event_for_kind(
        trace_event_factory,
        1,
        ShadowInputKind.OBSERVATION,
        trust_label=TrustLabel.SYNTHETIC_FIXTURE,
    )

    assert session._event_kind(event) is None


def test_event_kind_rejects_invalid_fixed_trust(
    seeded_ledger: tuple[ShadowSession, LedgerEntry, LedgerHead],
) -> None:
    session, entry, _head = seeded_ledger
    event = entry.record.model_copy(update={"trust_label": TrustLabel.SYNTHETIC_FIXTURE})

    assert session._event_kind(event) is None


def test_test_result_payload_requires_exact_keys(
    seeded_ledger: tuple[ShadowSession, LedgerEntry, LedgerHead],
) -> None:
    session, _entry, _head = seeded_ledger
    event = SimpleNamespace(payload={"test_report": {}})

    assert session._payload_is_valid(event, ShadowInputKind.TEST_RESULT) is False  # type: ignore[arg-type]


def test_test_result_payload_requires_tuple_failures(
    seeded_ledger: tuple[ShadowSession, LedgerEntry, LedgerHead],
) -> None:
    session, _entry, _head = seeded_ledger
    event = SimpleNamespace(
        payload={
            "test_report": {
                "schema_version": "1.0",
                "framework": "pytest",
                "status": "failed",
                "failures": [],
            }
        }
    )

    assert session._payload_is_valid(event, ShadowInputKind.TEST_RESULT) is False  # type: ignore[arg-type]


def test_payload_validation_rejects_equal_but_non_enum_kind(
    seeded_ledger: tuple[ShadowSession, LedgerEntry, LedgerHead],
) -> None:
    session, _entry, _head = seeded_ledger
    kind = str(ShadowInputKind.START.value)
    event = SimpleNamespace(payload={"shadow_run": {}})

    assert session._payload_is_valid(event, kind) is False  # type: ignore[arg-type]


def test_existing_signals_require_run_and_evidence_identity() -> None:
    session = _memory_session()
    signal = SimpleNamespace(run_id=OTHER_RUN_ID, evidence_event_ids=())
    state = SimpleNamespace(events_by_id={}, signals=(signal,))

    with pytest.raises(ShadowStateError):
        session._validate_existing_signals(state)  # type: ignore[arg-type]


def test_existing_signals_require_present_evidence() -> None:
    session = _memory_session()
    signal = SimpleNamespace(run_id=RUN_ID, evidence_event_ids=(uuid4(),))
    state = SimpleNamespace(events_by_id={}, signals=(signal,))

    with pytest.raises(ShadowStateError):
        session._validate_existing_signals(state)  # type: ignore[arg-type]


def test_existing_signals_must_follow_their_evidence(
    seeded_ledger: tuple[ShadowSession, LedgerEntry, LedgerHead],
) -> None:
    session, entry, _head = seeded_ledger
    signal_id = uuid4()
    signal = SimpleNamespace(
        signal_id=signal_id,
        run_id=RUN_ID,
        evidence_event_ids=(entry.record.event_id,),
    )
    state = SimpleNamespace(
        events_by_id={entry.record.event_id: entry.record},
        signals=(signal,),
        signal_positions={signal_id: 1},
        event_positions={entry.record.event_id: 1},
    )

    with pytest.raises(ShadowStateError):
        session._validate_existing_signals(state)  # type: ignore[arg-type]


def test_prefix_for_receipt_rejects_event_identity_drift(
    seeded_ledger: tuple[ShadowSession, LedgerEntry, LedgerHead],
) -> None:
    _session, entry, _head = seeded_ledger
    state = SimpleNamespace(start=entry.record, events=(entry.record,))
    damaged = entry.record.model_copy(update={"run_id": OTHER_RUN_ID})

    with pytest.raises(ShadowStateError):
        ShadowSession._prefix_for_receipt(state, damaged)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_signal_persistence_bounds_repeated_snapshot_races(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _memory_session()

    async def race(_session: ShadowSession) -> Any:
        raise session_module._RetryableSnapshotRaceError()

    monkeypatch.setattr(ShadowSession, "_load_state", race)

    with pytest.raises(ShadowStateError):
        await session._persist_signal(object(), expected_prefix=())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_signal_persistence_requires_present_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _memory_session()

    async def absent(_session: ShadowSession) -> None:
        return None

    monkeypatch.setattr(ShadowSession, "_load_state", absent)

    with pytest.raises(ShadowStateError):
        await session._persist_signal(object(), expected_prefix=())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_signal_persistence_rechecks_extraction_membership(
    seeded_ledger: tuple[ShadowSession, LedgerEntry, LedgerHead],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, entry, head = seeded_ledger
    state = SimpleNamespace(events=(entry.record,), signals=(), head=head)

    async def loaded(_session: ShadowSession) -> Any:
        return state

    monkeypatch.setattr(ShadowSession, "_load_state", loaded)
    monkeypatch.setattr(session_module, "select_detection_context", lambda _prefix: object())
    monkeypatch.setattr(
        ShadowSession,
        "_extract_report",
        lambda _session, _context: SimpleNamespace(signals=()),
    )

    with pytest.raises(ShadowStateError):
        await session._persist_signal(object(), expected_prefix=(entry.record,))  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_signal_persistence_rejects_same_id_different_signal(
    seeded_ledger: tuple[ShadowSession, LedgerEntry, LedgerHead],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, entry, head = seeded_ledger
    signal_id = uuid4()
    signal = SimpleNamespace(signal_id=signal_id)
    different = SimpleNamespace(signal_id=signal_id, drift=True)
    state = SimpleNamespace(events=(entry.record,), signals=(different,), head=head)

    async def loaded(_session: ShadowSession) -> Any:
        return state

    monkeypatch.setattr(ShadowSession, "_load_state", loaded)
    monkeypatch.setattr(session_module, "select_detection_context", lambda _prefix: object())
    monkeypatch.setattr(
        ShadowSession,
        "_extract_report",
        lambda _session, _context: SimpleNamespace(signals=(signal,)),
    )

    with pytest.raises(ShadowStateError):
        await session._persist_signal(signal, expected_prefix=(entry.record,))  # type: ignore[arg-type]
