from __future__ import annotations

import multiprocessing
import os
import sqlite3
import time
from itertools import pairwise
from pathlib import Path
from queue import Empty
from typing import Any

import pytest
from tests.capture.store_support import (
    CONNECTION_ID,
    INSTALLATION_KEY,
    INSTALLATION_KEY_MATERIAL,
    authenticated_intake,
    capture_context,
    register_connection,
)

from saliencegate.capture.migrations import initialize_capture_store
from saliencegate.capture.schema import CaptureIntake
from saliencegate.capture.store import (
    CaptureAppendDisposition,
    CaptureStore,
    CaptureStoreBusyError,
    CaptureStoreMode,
)

_PROCESS_COUNT = 16
_EVENTS_PER_PROCESS = 100
_EVENTS_PER_PROCESS_PER_SESSION = _EVENTS_PER_PROCESS // 2
_EVENTS_PER_SESSION = _PROCESS_COUNT * _EVENTS_PER_PROCESS_PER_SESSION
_TOTAL_EVENTS = _PROCESS_COUNT * _EVENTS_PER_PROCESS
_SESSION_NATIVE_VALUES = (
    b"concurrency-session-secret-a",
    b"concurrency-session-secret-b",
)
_WORKER_DEADLINE_SECONDS = 300.0
_FAULT_WORKER_TIMEOUT_SECONDS = 15.0
_FAULT_EXIT_CODE = 73
_FAULT_WORKER_ERROR_EXIT_CODE = 74
_FAULT_NOT_REACHED_EXIT_CODE = 75
_PRE_COMMIT_FAULT_STAGES = (
    "before_begin",
    "after_begin",
    "after_session_insert_or_load",
    "after_event_insert",
    "after_head_write",
    "after_session_health_write",
    "before_commit",
)
_FAULT_SESSION_NATIVE = b"fault-session"


def _producer_indices(worker_index: int, session_index: int) -> range:
    session_offset = session_index * _EVENTS_PER_SESSION
    worker_offset = worker_index * _EVENTS_PER_PROCESS_PER_SESSION
    first = session_offset + worker_offset + 1
    return range(first, first + _EVENTS_PER_PROCESS_PER_SESSION)


def _append_worker(
    path: str,
    worker_index: int,
    open_lock: Any,
    open_barrier: Any,
    contention_barrier: Any,
    result_queue: Any,
) -> None:
    pid = os.getpid()
    stage = "construct_intakes"
    try:
        context = capture_context()
        intakes: list[list[CaptureIntake]] = []
        for session_index, session_native in enumerate(_SESSION_NATIVE_VALUES):
            session_intakes = []
            for position, producer_index in enumerate(
                _producer_indices(worker_index, session_index)
            ):
                kind = "session_started" if worker_index == 0 and position == 0 else "turn_finished"
                session_intakes.append(
                    authenticated_intake(
                        kind,
                        session_native=session_native,
                        producer_index=producer_index,
                        context=context,
                    )
                )
            intakes.append(session_intakes)

        admitted = 0
        representations_are_redacted = True
        stage = "open_store"
        with open_lock:
            store = CaptureStore.open(
                path,
                installation_key=INSTALLATION_KEY,
                busy_timeout_ms=60_000,
                mode=CaptureStoreMode.HOOK,
            )
        with store:
            stage = "open_barrier"
            open_barrier.wait(timeout=30.0)

            stage = "create_sessions"
            if worker_index == 0:
                for session_intakes in intakes:
                    receipt = store.append(session_intakes[0])
                    admitted += receipt.disposition is CaptureAppendDisposition.ADMITTED
                    representations_are_redacted &= repr(receipt) == (
                        "CaptureAppendReceipt(<redacted>)"
                    )

            stage = "contention_barrier"
            contention_barrier.wait(timeout=30.0)

            stage = "append_events"
            first_position = 1 if worker_index == 0 else 0
            for position in range(first_position, _EVENTS_PER_PROCESS_PER_SESSION):
                for session_intakes in intakes:
                    intake = session_intakes[position]
                    receipt = store.append(intake)
                    admitted += receipt.disposition is CaptureAppendDisposition.ADMITTED
                    representation = repr(receipt)
                    representations_are_redacted &= representation == (
                        "CaptureAppendReceipt(<redacted>)"
                    )
                    representations_are_redacted &= all(
                        secret not in representation
                        for secret in (
                            CONNECTION_ID,
                            intake.session_id,
                            intake.producer_event_digest,
                            intake.intake_tag,
                            INSTALLATION_KEY_MATERIAL.decode(),
                        )
                    )
        result_queue.put(("ok", worker_index, pid, admitted, representations_are_redacted))
    except BaseException as error:
        result_queue.put(
            (
                "error",
                worker_index,
                pid,
                stage,
                type(error).__name__,
                str(error),
                repr(error),
            )
        )


def _run_workers(path: Path) -> list[tuple[object, ...]]:
    context = multiprocessing.get_context("spawn")
    open_lock = context.Lock()
    open_barrier = context.Barrier(_PROCESS_COUNT)
    contention_barrier = context.Barrier(_PROCESS_COUNT)
    result_queue = context.Queue()
    processes = tuple(
        context.Process(
            target=_append_worker,
            args=(
                os.fspath(path),
                worker_index,
                open_lock,
                open_barrier,
                contention_barrier,
                result_queue,
            ),
            name=f"capture-store-writer-{worker_index}",
        )
        for worker_index in range(_PROCESS_COUNT)
    )
    for process in processes:
        process.start()

    results: list[tuple[object, ...]] = []
    deadline = time.monotonic() + _WORKER_DEADLINE_SECONDS
    try:
        while len(results) < _PROCESS_COUNT:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                result = result_queue.get(timeout=min(1.0, remaining))
            except Empty:
                if all(process.exitcode is not None for process in processes):
                    break
                continue
            assert type(result) is tuple
            results.append(result)
    finally:
        for process in processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        result_queue.close()
        result_queue.join_thread()

    assert len(results) == _PROCESS_COUNT, (
        results,
        tuple((process.name, process.pid, process.exitcode) for process in processes),
    )
    assert all(process.exitcode == 0 for process in processes), tuple(
        (process.name, process.pid, process.exitcode) for process in processes
    )
    return results


def _hard_exit_append_worker(path: str, fault_stage: str) -> None:
    def hard_exit_at_stage(stage: str) -> None:
        if stage == fault_stage:
            os._exit(_FAULT_EXIT_CODE)

    try:
        with CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            busy_timeout_ms=5_000,
            mode=CaptureStoreMode.HOOK,
            _fault_injector=hard_exit_at_stage,
        ) as store:
            store.append(
                authenticated_intake(
                    "session_started",
                    session_native=_FAULT_SESSION_NATIVE,
                )
            )
    except BaseException:
        os._exit(_FAULT_WORKER_ERROR_EXIT_CODE)
    os._exit(_FAULT_NOT_REACHED_EXIT_CODE)


def _run_hard_exit_append(path: Path, fault_stage: str) -> None:
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_hard_exit_append_worker,
        args=(os.fspath(path), fault_stage),
        name="capture-store-fault-worker",
    )
    process.start()
    timed_out = False
    try:
        process.join(timeout=_FAULT_WORKER_TIMEOUT_SECONDS)
        if process.is_alive():
            timed_out = True
            process.terminate()
            process.join(timeout=5.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=5.0)
        exit_code = process.exitcode
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=5.0)
        process.close()

    assert timed_out is False
    assert exit_code == _FAULT_EXIT_CODE


def _capture_transaction_row_counts(path: Path) -> tuple[int, int, int]:
    connection = sqlite3.connect(path)
    try:
        counts = tuple(
            int(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
            for table in ("capture_sessions", "capture_events", "capture_heads")
        )
        return counts[0], counts[1], counts[2]
    finally:
        connection.close()


def _initialize_enabled_fault_store(path: Path) -> None:
    initialize_capture_store(path)
    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=5_000,
        mode=CaptureStoreMode.HOOK,
    ) as store:
        register_connection(store)


def test_sixteen_processes_append_1600_events_with_contiguous_authenticated_chains(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sixteen-process-writers.sqlite3"
    initialize_capture_store(path)
    expected_session_ids = tuple(
        capture_context().session_id(value) for value in _SESSION_NATIVE_VALUES
    )

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=60_000,
        mode=CaptureStoreMode.HOOK,
    ) as coordinator:
        register_connection(coordinator)
        results = _run_workers(path)

        assert {result[0] for result in results} == {"ok"}, results
        assert {result[1] for result in results} == set(range(_PROCESS_COUNT))
        assert len({result[2] for result in results}) == _PROCESS_COUNT
        assert all(result[3] == _EVENTS_PER_PROCESS for result in results)
        assert all(result[4] is True for result in results)

        verifications = tuple(
            coordinator.verify_session(CONNECTION_ID, session_id)
            for session_id in expected_session_ids
        )
        assert all(item.event_count == _EVENTS_PER_SESSION for item in verifications)
        assert all(item.last_receipt_ordinal == _EVENTS_PER_SESSION for item in verifications)

    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            """
            SELECT session_id, receipt_ordinal, previous_event_tag, event_tag, event_kind
            FROM capture_events
            ORDER BY session_id, receipt_ordinal
            """
        ).fetchall()
    finally:
        connection.close()

    assert len(rows) == _TOTAL_EVENTS
    assert {row[0] for row in rows} == set(expected_session_ids)
    for session_id in expected_session_ids:
        session_rows = [row for row in rows if row[0] == session_id]
        assert [row[1] for row in session_rows] == list(range(1, _EVENTS_PER_SESSION + 1))
        assert session_rows[0][2] is None
        assert session_rows[0][4] == "session_started"
        assert all(row[4] == "turn_finished" for row in session_rows[1:])
        assert all(current[2] == previous[3] for previous, current in pairwise(session_rows))


@pytest.mark.parametrize("fault_stage", _PRE_COMMIT_FAULT_STAGES)
def test_hard_exit_before_commit_rolls_back_the_entire_capture_transaction(
    tmp_path: Path,
    fault_stage: str,
) -> None:
    path = tmp_path / "pre-commit.sqlite3"
    _initialize_enabled_fault_store(path)

    _run_hard_exit_append(path, fault_stage)

    assert _capture_transaction_row_counts(path) == (0, 0, 0)


def test_hard_exit_after_commit_preserves_one_verifiable_event_and_replays_as_noop(
    tmp_path: Path,
) -> None:
    path = tmp_path / "post-commit.sqlite3"
    _initialize_enabled_fault_store(path)
    intake = authenticated_intake(
        "session_started",
        session_native=_FAULT_SESSION_NATIVE,
    )

    _run_hard_exit_append(path, "after_commit")

    assert _capture_transaction_row_counts(path) == (1, 1, 1)
    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=5_000,
        mode=CaptureStoreMode.HOOK,
    ) as store:
        before_replay = store.verify_session(CONNECTION_ID, intake.session_id)
        replay = store.append(intake)
        after_replay = store.verify_session(CONNECTION_ID, intake.session_id)

    assert before_replay.event_count == 1
    assert before_replay.last_receipt_ordinal == 1
    assert replay.disposition is CaptureAppendDisposition.REPLAYED
    assert replay.receipt_ordinal == 1
    assert after_replay == before_replay
    assert _capture_transaction_row_counts(path) == (1, 1, 1)


def test_primary_sqlite_busy_is_bounded_classified_and_leaves_no_partial_event(
    tmp_path: Path,
) -> None:
    path = tmp_path / "busy-secret.sqlite3"
    initialize_capture_store(path)
    intake = authenticated_intake(
        "session_started",
        session_native=b"busy-session-secret",
    )

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=25,
        mode=CaptureStoreMode.HOOK,
    ) as store:
        register_connection(store)
        blocker = sqlite3.connect(path, isolation_level=None)
        try:
            blocker.execute("BEGIN IMMEDIATE")
            started = time.monotonic()
            with pytest.raises(CaptureStoreBusyError) as raised:
                store.append(intake)
            elapsed = time.monotonic() - started

            assert elapsed < 1.0
            assert str(raised.value) == "capture store is busy"
            for secret in (
                "busy-secret",
                "busy-session-secret",
                intake.session_id,
                intake.producer_event_digest,
                intake.intake_tag,
                INSTALLATION_KEY_MATERIAL.decode(),
            ):
                assert secret not in str(raised.value)
                assert secret not in repr(raised.value)
            assert raised.value.__cause__ is None
            assert blocker.execute("SELECT count(*) FROM capture_events").fetchone() == (0,)
        finally:
            blocker.rollback()
            blocker.close()

        admitted = store.append(intake)
        assert admitted.disposition is CaptureAppendDisposition.ADMITTED
        assert admitted.receipt_ordinal == 1
