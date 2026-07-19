from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from tests.shadow.test_analyzer import _memory_session
from tests.shadow.test_trace import build_trace

import saliencegate.shadow.session as session_module
from saliencegate.repository import SQLiteRunRepository
from saliencegate.security import InstallationKey
from saliencegate.security.files import (
    StableFileAuthorization,
    inspect_private_file_location,
)
from saliencegate.shadow import (
    ShadowConfigurationError,
    ShadowInputError,
    ShadowSession,
    ShadowStateError,
)
from saliencegate.shadow.trace import ShadowTraceBinding

_KEY = InstallationKey(b"s" * 32)


def test_session_constructor_and_trace_binding_copy_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ShadowConfigurationError):
        ShadowSession()

    trace = build_trace()
    replacement = build_trace(
        adapter_descriptor={
            "schema_version": "example-shadow-adapter/v1",
            "mapping": {"mode": "copy-drift"},
        }
    ).binding

    def drifted_copy(_cls: type[ShadowTraceBinding], _value: bytes) -> ShadowTraceBinding:
        return replacement

    monkeypatch.setattr(
        ShadowTraceBinding,
        "model_validate_json",
        classmethod(drifted_copy),
    )
    with pytest.raises(ValueError, match="trace binding is invalid"):
        ShadowSession._copy_trace_binding(trace.binding)


def test_trace_factories_preserve_process_interrupts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()

    def interrupt_copy(_value: object) -> ShadowTraceBinding:
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        ShadowSession,
        "_copy_trace_binding",
        staticmethod(interrupt_copy),
    )
    with pytest.raises(KeyboardInterrupt):
        ShadowSession.in_memory_for_trace(
            run_id=trace.run_id,
            trace_binding=trace.binding,
            installation_key=_KEY,
        )

    def interrupt_inspection(_path: object) -> StableFileAuthorization:
        raise SystemExit(23)

    monkeypatch.setattr(
        session_module,
        "inspect_private_file_location",
        interrupt_inspection,
    )
    with pytest.raises(SystemExit, match="23"):
        ShadowSession.sqlite_for_trace(
            tmp_path / "interrupted.sqlite3",
            run_id=trace.run_id,
            trace_binding=trace.binding,
            installation_key=_KEY,
        )


def test_sqlite_trace_factory_sanitizes_inspection_and_authorization_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()

    def fail_inspection(_path: object) -> StableFileAuthorization:
        raise RuntimeError("fixture-secret-path-detail")

    monkeypatch.setattr(
        session_module,
        "inspect_private_file_location",
        fail_inspection,
    )
    with pytest.raises(ShadowConfigurationError) as captured:
        ShadowSession.sqlite_for_trace(
            tmp_path / "invalid.sqlite3",
            run_id=trace.run_id,
            trace_binding=trace.binding,
            installation_key=_KEY,
        )
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None

    with pytest.raises(ShadowConfigurationError):
        ShadowSession._from_sqlite_authorization_for_trace(  # type: ignore[arg-type]
            object(),
            run_id=trace.run_id,
            trace_binding=trace.binding,
            installation_key=_KEY,
        )


def test_authorized_sqlite_trace_factory_preserves_copy_interrupts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    locations = tuple(
        inspect_private_file_location(tmp_path / f"slot-{index}") for index in range(4)
    )

    def interrupt_copy(_value: object) -> ShadowTraceBinding:
        raise SystemExit(29)

    monkeypatch.setattr(
        ShadowSession,
        "_copy_trace_binding",
        staticmethod(interrupt_copy),
    )
    with pytest.raises(SystemExit, match="29"):
        ShadowSession._from_sqlite_authorization_for_trace(
            locations[0],
            sidecar_authorizations=locations[1:],
            run_id=trace.run_id,
            trace_binding=trace.binding,
            installation_key=_KEY,
        )


@pytest.mark.asyncio
async def test_closed_session_rejects_reentry_trace_snapshot_and_batch() -> None:
    trace = build_trace()
    session = _memory_session(trace)
    await session.aclose()

    with pytest.raises(ShadowInputError):
        await session.__aenter__()
    with pytest.raises(ShadowInputError):
        await session._snapshot_for_trace()
    with pytest.raises(ShadowInputError):
        await session._append_trace_batch((), expected_head=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", (asyncio.CancelledError, RuntimeError))
async def test_sqlite_close_propagates_cancellation_and_sanitizes_other_failures(
    monkeypatch: pytest.MonkeyPatch,
    failure: type[BaseException],
) -> None:
    trace = build_trace()
    session = _memory_session(trace)
    repository = object.__new__(SQLiteRunRepository)
    session._repository = repository

    async def fail_close(_repository: SQLiteRunRepository) -> None:
        raise failure("fixture-secret-close-detail")

    monkeypatch.setattr(SQLiteRunRepository, "aclose", fail_close)
    expected = asyncio.CancelledError if failure is asyncio.CancelledError else ShadowStateError
    with pytest.raises(expected) as captured:
        await session.aclose()
    if failure is RuntimeError:
        assert captured.value.__cause__ is None
        assert captured.value.__context__ is None
    assert not session._closed


def test_lazy_sqlite_claim_preserves_process_interrupts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    session = ShadowSession.sqlite_for_trace(
        tmp_path / "claim-interrupt.sqlite3",
        run_id=trace.run_id,
        trace_binding=trace.binding,
        installation_key=_KEY,
    )

    def interrupt_claim(
        _location: StableFileAuthorization,
        *,
        sidecar_locations: tuple[StableFileAuthorization, ...],
    ) -> StableFileAuthorization:
        assert len(sidecar_locations) == 3
        raise SystemExit(31)

    monkeypatch.setattr(
        session_module,
        "_claim_private_sqlite_location",
        interrupt_claim,
    )
    with pytest.raises(SystemExit, match="31"):
        session._repository_for_operation()


def test_lazy_sqlite_constructor_interrupt_cleans_claimed_placeholders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    path = tmp_path / "constructor-interrupt.sqlite3"
    session = ShadowSession.sqlite_for_trace(
        path,
        run_id=trace.run_id,
        trace_binding=trace.binding,
        installation_key=_KEY,
    )

    def interrupt_open(
        _repository_type: type[SQLiteRunRepository],
        _authorization: StableFileAuthorization,
        **_kwargs: object,
    ) -> SQLiteRunRepository:
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        SQLiteRunRepository,
        "_from_file_authorization",
        classmethod(interrupt_open),
    )
    with pytest.raises(KeyboardInterrupt):
        session._repository_for_operation()
    assert path.exists()
    assert all(not Path(f"{path}{suffix}").exists() for suffix in ("-wal", "-shm", "-journal"))


@pytest.mark.asyncio
async def test_snapshot_helpers_bound_repeated_repository_races(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    trace_session = _memory_session(trace)
    legacy_session = ShadowSession.in_memory(
        run_id=trace.run_id,
        installation_key=_KEY,
    )
    calls = 0

    async def always_racing(_session: ShadowSession) -> None:
        nonlocal calls
        calls += 1
        raise session_module._RetryableSnapshotRaceError()

    monkeypatch.setattr(ShadowSession, "_load_state", always_racing)
    with pytest.raises(ShadowStateError):
        await trace_session._snapshot_for_trace()
    with pytest.raises(ShadowStateError):
        await legacy_session._snapshot_for_cli()
    assert calls == 2 * session_module._MAX_CAS_ATTEMPTS
