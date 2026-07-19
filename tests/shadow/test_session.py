from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Literal

import pytest

import saliencegate.shadow.observation as observation_module
import saliencegate.shadow.session as session_module
from saliencegate.domain import EventPhase, EventType, Signal, TraceEvent, TrustLabel
from saliencegate.security import InstallationKey, RedactionPolicy
from saliencegate.shadow.errors import ShadowInputError
from saliencegate.shadow.inputs import ShadowObservationSource
from saliencegate.shadow.session import ShadowSession
from saliencegate.signals import DetectionStatus
from saliencegate.signals import TestFailureEvidence as FailureEvidence

from .conftest import NOW, RUN_ID

Backend = Literal["memory", "sqlite"]


@pytest.fixture(params=("memory", "sqlite"))
def backend(request: pytest.FixtureRequest) -> Backend:
    return request.param


@asynccontextmanager
async def opened_session(
    backend: Backend,
    tmp_path: Path,
) -> AsyncIterator[ShadowSession]:
    kwargs = {
        "run_id": RUN_ID,
        "installation_key": InstallationKey(b"s" * 32),
        "redaction_policy": RedactionPolicy(literal_secrets=("top-secret",)),
        "capture_scope": "bounded_window",
        "task_scope_digest": "1" * 64,
        "lineage_scope_digest": "2" * 64,
        "capture_manifest_digest": "3" * 64,
        "source_adapter": "shadow-test/v1",
    }
    session = (
        ShadowSession.in_memory(**kwargs)
        if backend == "memory"
        else ShadowSession.sqlite(tmp_path / "shadow.sqlite3", **kwargs)
    )
    async with session:
        yield session


async def _start(session: ShadowSession):
    return await session.start(source_event_id="start", occurred_at=NOW)


async def _action(session: ShadowSession, *, source_event_id: str = "action-1"):
    return await session.action(
        source_event_id=source_event_id,
        occurred_at=NOW + timedelta(seconds=1),
        argv=("pytest", "-q"),
        working_directory="/project",
        environment_digest="a" * 64,
    )


@pytest.mark.asyncio
async def test_vertical_slice_persists_redacted_events_and_flagged_signal(
    backend: Backend,
    tmp_path: Path,
) -> None:
    async with opened_session(backend, tmp_path) as session:
        started = await _start(session)
        action = await _action(session)
        result = await session.tool_result(
            source_event_id="tool-1",
            occurred_at=NOW + timedelta(seconds=2),
            action=action.ref,
            status="failed",
            exit_status=1,
            exception_type="AssertionError top-secret",
        )

        assert started.ref.sequence == 1
        assert action.ref.sequence == 2
        assert result.ref.sequence == 3
        assert result.observation.heuristic_evaluations[0].disposition == "flagged"
        evaluations = {
            item.signal_type.value: item.outcome.status
            for item in result.observation.detector_evaluations
        }
        assert evaluations["tool_error"] is DetectionStatus.DETECTED
        entries = await session._repository.ledger(RUN_ID)
        assert tuple(type(entry.record) for entry in entries) == (
            TraceEvent,
            TraceEvent,
            TraceEvent,
            Signal,
        )
        events = tuple(entry.record for entry in entries if not isinstance(entry.record, Signal))
        assert len(events) == 3
        assert events[-1].payload["tool_outcome"]["exception_type"] == ("AssertionError [REDACTED]")
        signals = tuple(entry.record for entry in entries if isinstance(entry.record, Signal))
        assert signals == result.observation.detected_signals


@pytest.mark.asyncio
async def test_every_method_uses_the_normative_projection(
    backend: Backend,
    tmp_path: Path,
) -> None:
    async with opened_session(backend, tmp_path) as session:
        start = await _start(session)
        action = await _action(session)
        tool = await session.tool_result(
            source_event_id="tool-1",
            occurred_at=NOW + timedelta(seconds=2),
            action=action.ref,
            status="succeeded",
            exit_status=0,
        )
        test = await session.test_result(
            source_event_id="test-1",
            occurred_at=NOW + timedelta(seconds=3),
            action=action.ref,
            framework="pytest",
            status="failed",
            failures=(
                FailureEvidence(
                    schema_version="1.0",
                    test_id="tests/test_example.py::test_one",
                    failure_type="AssertionError",
                    signature="assertion",
                ),
            ),
        )
        observation = await session.observation(
            source_event_id="observation-1",
            occurred_at=NOW + timedelta(seconds=4),
            source=ShadowObservationSource.MODEL_OUTPUT,
            payload={"message": "bounded"},
        )
        error = await session.controller_error(
            source_event_id="error-1",
            occurred_at=NOW + timedelta(seconds=5),
            error_code="adapter_timeout",
        )
        finish = await session.finish(
            source_event_id="finish",
            occurred_at=NOW + timedelta(seconds=6),
        )

        entries = await session._repository.ledger(RUN_ID)
        events = tuple(entry.record for entry in entries if not isinstance(entry.record, Signal))
        assert tuple(event.event_type for event in events) == (
            EventType.RUN_START,
            EventType.ACTION_PROPOSAL,
            EventType.TOOL_COMPLETION,
            EventType.OBSERVATION,
            EventType.OBSERVATION,
            EventType.CONTROLLER_ERROR,
            EventType.RUN_END,
        )
        assert tuple(event.phase for event in events) == (
            EventPhase.INITIALIZATION,
            EventPhase.PRE_ACTION,
            EventPhase.POST_ACTION,
            EventPhase.POST_ACTION,
            EventPhase.POST_ACTION,
            EventPhase.INTERNAL,
            EventPhase.TERMINAL,
        )
        assert events[4].trust_label is TrustLabel.UNTRUSTED_MODEL_OUTPUT
        assert events[2].parent_ids == events[3].parent_ids == (action.ref.event_id,)
        assert tuple(
            item.ref for item in (start, action, tool, test, observation, error, finish)
        ) == tuple(
            type(start.ref)(run_id=RUN_ID, event_id=event.event_id, sequence=event.sequence)
            for event in events
        )


@pytest.mark.asyncio
async def test_duplicate_retry_is_identical_before_and_after_finish(
    backend: Backend,
    tmp_path: Path,
) -> None:
    async with opened_session(backend, tmp_path) as session:
        first_start = await _start(session)
        assert await _start(session) == first_start
        action = await _action(session)
        first = await session.tool_result(
            source_event_id="tool-1",
            occurred_at=NOW + timedelta(seconds=2),
            action=action.ref,
            status="failed",
            exit_status=1,
        )
        assert (
            await session.tool_result(
                source_event_id="tool-1",
                occurred_at=NOW + timedelta(seconds=2),
                action=action.ref,
                status="failed",
                exit_status=1,
            )
            == first
        )
        await session.finish(
            source_event_id="finish",
            occurred_at=NOW + timedelta(seconds=3),
        )
        assert (
            await session.tool_result(
                source_event_id="tool-1",
                occurred_at=NOW + timedelta(seconds=2),
                action=action.ref,
                status="failed",
                exit_status=1,
            )
            == first
        )


@pytest.mark.asyncio
async def test_lifecycle_parent_and_timestamp_failures_are_value_free(
    backend: Backend,
    tmp_path: Path,
) -> None:
    async with opened_session(backend, tmp_path) as session:
        invalid_calls: list[Callable[[], object]] = [
            lambda: session.action(
                source_event_id="before-start",
                occurred_at=NOW,
                command="pwd",
                working_directory="/project",
                environment_digest="a" * 64,
            )
        ]
        for call in invalid_calls:
            with pytest.raises(ShadowInputError, match=r"^shadow input is invalid$"):
                await call()  # type: ignore[misc]

        await _start(session)
        action = await _action(session)
        with pytest.raises(ShadowInputError, match=r"^shadow input is invalid$"):
            await session.tool_result(
                source_event_id="future-parent",
                occurred_at=NOW + timedelta(seconds=2),
                action=action.ref.model_copy(update={"sequence": action.ref.sequence + 10}),
                status="failed",
            )
        with pytest.raises(ShadowInputError, match=r"^shadow input is invalid$"):
            await session.observation(
                source_event_id="decreasing-time",
                occurred_at=NOW,
                source=ShadowObservationSource.TASK_INPUT,
                payload={"value": 1},
            )
        await session.finish(
            source_event_id="finish",
            occurred_at=NOW + timedelta(seconds=3),
        )
        with pytest.raises(ShadowInputError, match=r"^shadow input is invalid$"):
            await session.observation(
                source_event_id="after-finish",
                occurred_at=NOW + timedelta(seconds=4),
                source=ShadowObservationSource.TASK_INPUT,
                payload={"value": 2},
            )


@pytest.mark.asyncio
async def test_sqlite_reopen_reconstructs_marker_and_duplicate_observation(tmp_path: Path) -> None:
    path = tmp_path / "reopen.sqlite3"
    kwargs = {
        "run_id": RUN_ID,
        "installation_key": InstallationKey(b"r" * 32),
        "capture_scope": "unknown",
    }
    async with ShadowSession.sqlite(path, **kwargs) as first:
        await _start(first)
        action = await _action(first)
        original = await first.tool_result(
            source_event_id="tool-1",
            occurred_at=NOW + timedelta(seconds=2),
            action=action.ref,
            status="failed",
            exit_status=1,
        )

    async with ShadowSession.sqlite(path, **kwargs) as reopened:
        retry = await reopened.tool_result(
            source_event_id="tool-1",
            occurred_at=NOW + timedelta(seconds=2),
            action=action.ref,
            status="failed",
            exit_status=1,
        )
        assert retry == original


@pytest.mark.asyncio
async def test_context_manager_close_is_idempotent(tmp_path: Path) -> None:
    session = ShadowSession.sqlite(
        tmp_path / "close.sqlite3",
        run_id=RUN_ID,
        installation_key=InstallationKey(b"c" * 32),
    )
    async with session:
        await _start(session)
    await session.aclose()
    with pytest.raises(ShadowInputError, match=r"^shadow input is invalid$"):
        await _start(session)


@pytest.mark.asyncio
async def test_each_submit_reuses_its_single_prevalidated_context_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = session_module._select_detection_context
    selection_calls = 0

    def counted_selection(prefix: tuple[TraceEvent, ...]):
        nonlocal selection_calls
        selection_calls += 1
        return original(prefix)

    def forbidden_reselection(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the observation builder must reuse the selected context")

    monkeypatch.setattr(session_module, "_select_detection_context", counted_selection)
    monkeypatch.setattr(
        observation_module,
        "select_detection_context",
        forbidden_reselection,
    )
    async with ShadowSession.in_memory(
        run_id=RUN_ID,
        installation_key=InstallationKey(b"o" * 32),
    ) as session:
        await _start(session)
        await session.observation(
            source_event_id="one-selection",
            occurred_at=NOW + timedelta(seconds=1),
            source=ShadowObservationSource.TASK_INPUT,
            payload={"bounded": True},
        )

    assert selection_calls == 2
