from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests.shadow.conftest import OTHER_RUN_ID
from tests.shadow.test_trace import (
    ENVIRONMENT_DIGEST,
    build_trace,
    complete_records,
    identity_records,
)

import saliencegate
import saliencegate.shadow as public_shadow
import saliencegate.shadow.session as session_module
from saliencegate.domain import Signal
from saliencegate.ports.repository import (
    ConditionalAppendOperation,
    ConditionalBatchReceipt,
    ConditionalEventAppend,
    ConditionalSignalAppend,
    LedgerHead,
)
from saliencegate.repository.memory import MemoryRunRepository
from saliencegate.security import InstallationKey
from saliencegate.shadow import (
    ShadowAnalyzer,
    ShadowConfigurationError,
    ShadowEventResult,
    ShadowInputError,
    ShadowSession,
    ShadowStateError,
    ShadowTraceReport,
    decode_shadow_trace_report,
    encode_shadow_trace_report,
)
from saliencegate.shadow.analyzer import (
    _analyze_prepared,
    _prepare_analysis,
    _preview_prepared,
)
from saliencegate.shadow.inputs import (
    ShadowInputKind,
    ShadowToolResultInput,
    derive_shadow_event_id,
    project_shadow_input,
)
from saliencegate.shadow.trace import ShadowTrace

_KEY = InstallationKey(b"a" * 32)
_STARTED_AT = datetime(2026, 7, 17, 9, 0, tzinfo=UTC)
_ACTION_AT = datetime(2026, 7, 17, 9, 0, 1, tzinfo=UTC)
_TOOL_AT = _ACTION_AT
_TASK_SCOPE_DIGEST = "1" * 64
_LINEAGE_SCOPE_DIGEST = "2" * 64
_CAPTURE_MANIFEST_DIGEST = "3" * 64


def _memory_session(
    trace: ShadowTrace,
    *,
    installation_key: InstallationKey = _KEY,
) -> ShadowSession:
    return ShadowSession.in_memory_for_trace(
        run_id=trace.run_id,
        trace_binding=trace.binding,
        installation_key=installation_key,
    )


async def _seed_start(session: ShadowSession) -> None:
    await session.start(
        source_event_id="start-1",
        occurred_at=_STARTED_AT,
    )


async def _seed_action(
    session: ShadowSession,
    *,
    command: str = "pytest -q",
) -> ShadowEventResult:
    return await session.action(
        source_event_id="action-1",
        occurred_at=_ACTION_AT,
        command=command,
        working_directory="/private/project",
        environment_digest=ENVIRONMENT_DIGEST,
    )


def _sqlite_slots(path: Path) -> tuple[Path, ...]:
    return tuple(Path(f"{path}{suffix}") for suffix in ("", "-wal", "-shm", "-journal"))


def _count_memory_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[ConditionalAppendOperation, ...]]:
    original = MemoryRunRepository.append_records_if_head
    calls: list[tuple[ConditionalAppendOperation, ...]] = []

    async def counted(
        repository: MemoryRunRepository,
        operations: tuple[ConditionalAppendOperation, ...],
        *,
        expected_head: LedgerHead | None,
    ) -> ConditionalBatchReceipt:
        calls.append(operations)
        return await original(
            repository,
            operations,
            expected_head=expected_head,
        )

    monkeypatch.setattr(MemoryRunRepository, "append_records_if_head", counted)
    return calls


def test_shadow_analyzer_is_additive_public_api() -> None:
    assert public_shadow.ShadowAnalyzer is ShadowAnalyzer
    assert "ShadowAnalyzer" in public_shadow.__all__
    assert not hasattr(saliencegate, "ShadowAnalyzer")


@pytest.mark.asyncio
async def test_in_memory_for_trace_uses_an_ephemeral_key_without_key_file_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()

    def forbidden_key_lookup(*_args: object, **_kwargs: object) -> InstallationKey:
        raise AssertionError("trace-specific in-memory sessions must not load a key file")

    monkeypatch.setattr(
        session_module,
        "load_or_create_installation_key",
        forbidden_key_lookup,
    )

    session = ShadowSession.in_memory_for_trace(
        run_id=trace.run_id,
        trace_binding=trace.binding,
    )
    assert session._trace_binding == trace.binding
    assert session._trace_binding is not trace.binding
    async with session:
        report = await ShadowAnalyzer(session).analyze(trace)

    assert type(report) is ShadowTraceReport


@pytest.mark.asyncio
async def test_identity_trace_analyzes_and_encodes_through_the_report_boundary() -> None:
    trace = build_trace(identity_records())
    session = _memory_session(trace)

    async with session:
        report = await ShadowAnalyzer(session).analyze(trace)

    assert tuple(row.input_kind for row in report.shadow_report.rows) == (
        ShadowInputKind.START,
        ShadowInputKind.ACTION_IDENTITY,
        ShadowInputKind.TOOL_RESULT,
        ShadowInputKind.TEST_RESULT,
        ShadowInputKind.ACTION_IDENTITY,
        ShadowInputKind.ACTION_IDENTITY,
        ShadowInputKind.FINISH,
    )
    encoded = encode_shadow_trace_report(report)
    assert decode_shadow_trace_report(encoded) == report
    assert report.diagnostics == trace.diagnostics


@pytest.mark.asyncio
async def test_sqlite_for_trace_is_unused_close_lazy_and_requires_an_explicit_key(
    tmp_path: Path,
) -> None:
    trace = build_trace()
    path = tmp_path / "lazy-shadow.sqlite3"

    with pytest.raises(TypeError):
        ShadowSession.sqlite_for_trace(  # type: ignore[call-arg]
            path,
            run_id=trace.run_id,
            trace_binding=trace.binding,
        )

    session = ShadowSession.sqlite_for_trace(
        path,
        run_id=trace.run_id,
        trace_binding=trace.binding,
        installation_key=_KEY,
    )

    assert all(not slot.exists() for slot in _sqlite_slots(path))
    assert str(path) not in repr(session)
    async with session:
        assert all(not slot.exists() for slot in _sqlite_slots(path))
    await session.aclose()
    assert all(not slot.exists() for slot in _sqlite_slots(path))


@pytest.mark.asyncio
async def test_invalid_trace_fails_before_lazy_sqlite_is_materialized(tmp_path: Path) -> None:
    trace = build_trace()
    damaged = build_trace()
    object.__setattr__(damaged, "mapped_record_digest", "0" * 64)
    path = tmp_path / "invalid-trace.sqlite3"
    session = ShadowSession.sqlite_for_trace(
        path,
        run_id=trace.run_id,
        trace_binding=trace.binding,
        installation_key=_KEY,
    )

    async with session:
        with pytest.raises(ShadowInputError) as captured:
            await ShadowAnalyzer(session).analyze(damaged)

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert all(not slot.exists() for slot in _sqlite_slots(path))


@pytest.mark.asyncio
async def test_run_or_binding_mismatch_fails_before_repository_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    changed_traces = (
        build_trace(
            adapter_descriptor={
                "schema_version": "example-shadow-adapter/v1",
                "mapping": {"mode": "changed"},
            }
        ),
        build_trace(run_id=OTHER_RUN_ID),
    )
    session = _memory_session(trace)

    async def forbidden_ledger(
        _repository: MemoryRunRepository,
        _run_id: object,
    ) -> tuple[object, ...]:
        raise AssertionError("binding mismatch reached the repository")

    monkeypatch.setattr(MemoryRunRepository, "ledger", forbidden_ledger)

    async with session:
        for changed in changed_traces:
            with pytest.raises(ShadowConfigurationError):
                await ShadowAnalyzer(session).analyze(changed)


@pytest.mark.asyncio
async def test_empty_repository_uses_one_batch_and_reanalysis_uses_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace(
        task_scope_digest=_TASK_SCOPE_DIGEST,
        lineage_scope_digest=_LINEAGE_SCOPE_DIGEST,
        capture_manifest_digest=_CAPTURE_MANIFEST_DIGEST,
    )
    batch_calls = _count_memory_batches(monkeypatch)
    session = _memory_session(trace)
    analyzer = ShadowAnalyzer(session)

    async with session:
        first = await analyzer.analyze(trace)
        first_ledger = await session._repository.ledger(trace.run_id)
        second = await analyzer.analyze(trace)
        second_ledger = await session._repository.ledger(trace.run_id)

    assert len(batch_calls) == 1
    assert any(type(operation) is ConditionalEventAppend for operation in batch_calls[0])
    assert first_ledger == second_ledger
    assert first.run_id == trace.run_id
    assert first.binding == trace.binding
    assert first.binding is not trace.binding
    assert first.binding_digest == trace.binding.binding_digest
    assert first.diagnostics == trace.diagnostics
    assert first.mapped_record_digest == trace.mapped_record_digest
    assert first.shadow_report.input_byte_digest == trace.binding.source_byte_digest
    assert first.shadow_report.capture_scope == trace.binding.capture_scope
    assert first.shadow_report.task_scope_digest == _TASK_SCOPE_DIGEST
    assert first.shadow_report.lineage_scope_digest == _LINEAGE_SCOPE_DIGEST
    assert first.shadow_report.capture_manifest_digest == _CAPTURE_MANIFEST_DIGEST
    assert first.shadow_report.initial_ledger_entry_count == 0
    assert first.shadow_report.appended_event_count == len(trace.records)
    assert first.shadow_report.preexisting_event_count == 0
    assert second.shadow_report.appended_event_count == 0
    assert second.shadow_report.preexisting_event_count == len(trace.records)
    assert first.shadow_report.observations == second.shadow_report.observations


@pytest.mark.asyncio
async def test_exact_event_prefix_resumes_with_report_dispositions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    session = _memory_session(trace)
    async with session:
        await _seed_start(session)
        await _seed_action(session)
        initial_head = await session._repository.ledger_head(trace.run_id)
        batch_calls = _count_memory_batches(monkeypatch)

        report = await ShadowAnalyzer(session).analyze(trace)

    nested = report.shadow_report
    assert len(batch_calls) == 1
    assert nested.initial_ledger_entry_count == initial_head.entry_count == 2
    assert nested.preexisting_event_count == 2
    assert nested.appended_event_count == len(trace.records) - 2
    assert tuple(row.persistence_disposition for row in nested.rows[:2]) == (
        "preexisting",
        "preexisting",
    )
    assert all(row.persistence_disposition == "appended" for row in nested.rows[2:])


@pytest.mark.asyncio
async def test_prepared_preview_is_exact_and_rejects_a_changed_initial_snapshot() -> None:
    trace = build_trace()
    session = _memory_session(trace)

    async with session:
        prepared = _prepare_analysis(session, trace)
        preview = await _preview_prepared(session, prepared, assume_empty=True)
        report = await _analyze_prepared(
            session,
            prepared,
            expected_initial_state=preview.initial_state,
        )
        before = await session._repository.ledger(trace.run_id)

        with pytest.raises(ShadowStateError):
            await _analyze_prepared(
                session,
                prepared,
                expected_initial_state=preview.initial_state,
            )

        after = await session._repository.ledger(trace.run_id)

    assert report.shadow_report == preview.shadow_report
    assert after == before


@pytest.mark.asyncio
async def test_divergent_prefix_fails_without_mutation_or_source_disclosure() -> None:
    trace = build_trace()
    secret = "fixture-secret-divergent-command"
    session = _memory_session(trace)
    async with session:
        await _seed_start(session)
        await _seed_action(session, command=secret)
        before = await session._repository.ledger(trace.run_id)

        with pytest.raises(ShadowStateError) as captured:
            await ShadowAnalyzer(session).analyze(trace)

        after = await session._repository.ledger(trace.run_id)

    assert after == before
    assert secret not in str(captured.value)
    assert secret not in repr(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.asyncio
async def test_missing_signal_is_repaired_before_remaining_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    session = _memory_session(trace)
    async with session:
        await _seed_start(session)
        action = await _seed_action(session)
        tool_input = ShadowToolResultInput(
            source_event_id="tool-1",
            occurred_at=_TOOL_AT,
            action=action.ref,
            status="failed",
            exit_status=1,
            error_code="TEST_FAILURE",
        )
        tool_draft = project_shadow_input(
            tool_input,
            run_id=trace.run_id,
            source_adapter=trace.binding.source_adapter,
        )
        head = await session._repository.ledger_head(trace.run_id)
        await session._repository.append_event_if_head(
            tool_draft,
            event_id=derive_shadow_event_id(trace.run_id, "tool-1"),
            expected_head=head,
        )
        initial_head = await session._repository.ledger_head(trace.run_id)
        batch_calls = _count_memory_batches(monkeypatch)

        report = await ShadowAnalyzer(session).analyze(trace)
        entries = await session._repository.ledger(trace.run_id)

    assert len(batch_calls) == 1
    assert type(batch_calls[0][0]) is ConditionalSignalAppend
    assert any(type(operation) is ConditionalEventAppend for operation in batch_calls[0][1:])
    assert type(entries[3].record) is Signal
    assert report.shadow_report.initial_ledger_entry_count == initial_head.entry_count == 3
    assert report.shadow_report.preexisting_event_count == 3
    assert report.shadow_report.appended_event_count == len(trace.records) - 3


@pytest.mark.asyncio
async def test_retry_rows_keep_legacy_ordinals_without_duplicate_event_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = complete_records()
    records.insert(2, dict(records[1]))
    trace = build_trace(records)
    batch_calls = _count_memory_batches(monkeypatch)
    session = _memory_session(trace)

    async with session:
        report = await ShadowAnalyzer(session).analyze(trace)

    nested = report.shadow_report
    event_operations = tuple(
        operation for operation in batch_calls[0] if type(operation) is ConditionalEventAppend
    )
    first_action = nested.rows[1]
    retry_action = nested.rows[2]

    assert len(batch_calls) == 1
    assert len(event_operations) == len(trace.records) - 1
    assert nested.input_row_count == len(trace.records)
    assert nested.unique_input_event_count == len(trace.records) - 1
    assert nested.retry_row_count == 1
    assert first_action.input_ordinal == first_action.first_occurrence_ordinal == 2
    assert first_action.retry_target_ordinal is None
    assert retry_action.input_ordinal == 3
    assert retry_action.first_occurrence_ordinal is None
    assert retry_action.retry_target_ordinal == 2
    assert retry_action.persistence_disposition == "preexisting"
    assert retry_action.observation_digest == first_action.observation_digest


@pytest.mark.asyncio
async def test_memory_and_sqlite_emit_identical_fresh_report_bytes(tmp_path: Path) -> None:
    trace = build_trace()
    sqlite_path = tmp_path / "parity.sqlite3"
    memory = _memory_session(trace)
    sqlite = ShadowSession.sqlite_for_trace(
        sqlite_path,
        run_id=trace.run_id,
        trace_binding=trace.binding,
        installation_key=_KEY,
    )

    async with memory:
        memory_report = await ShadowAnalyzer(memory).analyze(trace)
    assert not sqlite_path.exists()
    async with sqlite:
        sqlite_report = await ShadowAnalyzer(sqlite).analyze(trace)

    assert sqlite_path.exists()
    assert encode_shadow_trace_report(memory_report) == encode_shadow_trace_report(sqlite_report)


@pytest.mark.asyncio
async def test_public_report_and_reprs_do_not_retain_selected_source_content() -> None:
    command = "fixture-secret-command --token fixture-secret-token"
    working_directory = "/fixture-secret/private-project"
    source = b"fixture-secret-native-source"
    records = complete_records()
    records[1]["command"] = command
    records[1]["working_directory"] = working_directory
    trace = build_trace(
        records,
        source_bytes=source,
        source_format="example",
        source_schema_version="example/v1",
    )
    session = _memory_session(trace)
    analyzer = ShadowAnalyzer(session)

    async with session:
        report = await analyzer.analyze(trace)

    encoded = encode_shadow_trace_report(report)
    forbidden = (
        command.encode(),
        working_directory.encode(),
        source,
    )
    assert all(value not in encoded for value in forbidden)
    rendered = " ".join((repr(trace), repr(session), repr(analyzer), repr(report)))
    assert "fixture-secret" not in rendered
