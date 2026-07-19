from __future__ import annotations

import os
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests.capture.store_support import (
    CONNECTION_ID,
    INSTALLATION_KEY,
    WRONG_INSTALLATION_KEY,
    all_authenticated_intakes,
    authenticated_intake,
    initialized_store,
    register_connection,
)

import saliencegate.capture.spool as spool_module
from saliencegate.capture.locations import resolve_capture_store_locations
from saliencegate.capture.schema import (
    CaptureIntake,
    canonical_capture_intake,
    load_capture_intake,
)
from saliencegate.capture.spool import (
    MAX_CAPTURE_SPOOL_BYTES,
    MAX_CAPTURE_SPOOL_EVENTS,
    CaptureSpool,
    CaptureSpoolError,
    CaptureSpoolIntegrityError,
    admit_capture_intake,
)
from saliencegate.capture.store import (
    CaptureStoreBusyError,
    CaptureStoreIntegrityError,
    CaptureStoreStateError,
)

_SPOOL_ENTRY_NAME = re.compile(r"^[0-9a-f]{64}\.capture-intake$")


def _locations(tmp_path: Path, name: str = "state"):
    return resolve_capture_store_locations(
        environ={"XDG_STATE_HOME": str(tmp_path / name)},
        home=tmp_path / "home",
        platform="posix",
    )


def _entries(spool_directory: Path) -> tuple[Path, ...]:
    return tuple(sorted(spool_directory.glob("*.capture-intake")))


def _framed_intake(path: Path) -> bytes:
    header, tag, intake = path.read_bytes().split(b"\n", maxsplit=2)
    assert header == b"capture-spool/v1"
    assert re.fullmatch(rb"[0-9a-f]{64}", tag)
    return intake


class _RecordingStore:
    def __init__(self) -> None:
        self.received: list[CaptureIntake] = []

    def append(self, intake: CaptureIntake) -> object:
        self.received.append(intake)
        return object()


class _AppendOutcomeStore:
    def __init__(self, outcome: object, *, raises: bool = False) -> None:
        self.outcome = outcome
        self.raises = raises
        self.calls = 0

    def append(self, _intake: CaptureIntake) -> object:
        self.calls += 1
        if self.raises:
            assert isinstance(self.outcome, BaseException)
            raise self.outcome
        return self.outcome


class _DelayedFailureStore:
    def __init__(self, error: BaseException, *, fail_on_call: int) -> None:
        self.error = error
        self.fail_on_call = fail_on_call
        self.calls = 0
        self.received: list[CaptureIntake] = []

    def append(self, intake: CaptureIntake) -> object:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise self.error
        self.received.append(intake)
        return object()


def _queued_intakes(spool_directory: Path) -> tuple[CaptureIntake, ...]:
    return tuple(
        sorted(
            (load_capture_intake(_framed_intake(path)) for path in _entries(spool_directory)),
            key=lambda intake: intake.producer_sequence or 0,
        )
    )


def _assert_content_free_spool_error(
    error: CaptureSpoolError,
    *,
    message: str,
    secret: str,
) -> None:
    assert str(error) == message
    assert secret not in str(error)
    assert secret not in repr(error)
    assert error.__cause__ is None
    assert error.__context__ is None


def test_spool_v1_limits_are_fixed_per_installation() -> None:
    assert MAX_CAPTURE_SPOOL_BYTES == 32 * 1_024 * 1_024
    assert MAX_CAPTURE_SPOOL_EVENTS == 10_000


def test_enqueue_persists_one_exact_canonical_intake_without_admission_metadata(
    tmp_path: Path,
) -> None:
    locations = _locations(tmp_path)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    intake = authenticated_intake("action_started", producer_index=7)

    result = spool.enqueue(intake)

    entries = _entries(locations.spool_directory)
    assert result.disposition == "queued"
    assert len(entries) == 1
    assert _SPOOL_ENTRY_NAME.fullmatch(entries[0].name)
    assert _framed_intake(entries[0]) == canonical_capture_intake(intake)
    assert load_capture_intake(_framed_intake(entries[0])) == intake
    persisted = entries[0].read_bytes()
    assert b'"receipt_ordinal"' not in persisted
    assert b'"previous_event_tag"' not in persisted
    assert b'"event_tag"' not in persisted

    health = spool.health()
    assert health.queued_events == 1
    assert health.queued_bytes == entries[0].stat().st_size
    assert health.dropped_events == 0
    assert health.coverage_degraded is False


def test_enqueue_is_a_deterministic_no_clobber_replay(tmp_path: Path) -> None:
    intake = authenticated_intake("session_started", producer_index=1)
    first_locations = _locations(tmp_path, "first")
    second_locations = _locations(tmp_path, "second")
    first = CaptureSpool.open(first_locations, INSTALLATION_KEY)
    second = CaptureSpool.open(second_locations, INSTALLATION_KEY)

    queued = first.enqueue(intake)
    replayed = first.enqueue(intake)
    second.enqueue(intake)

    first_entries = _entries(first_locations.spool_directory)
    second_entries = _entries(second_locations.spool_directory)
    assert queued.disposition == "queued"
    assert replayed.disposition == "already_queued"
    assert len(first_entries) == 1
    assert [(path.name, path.read_bytes()) for path in first_entries] == [
        (path.name, path.read_bytes()) for path in second_entries
    ]
    assert first.health().queued_events == 1
    assert first.health().dropped_events == 0


def test_drain_uses_deterministic_entry_order_and_assigns_no_ordinal_itself(
    tmp_path: Path,
) -> None:
    locations = _locations(tmp_path)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    intakes = tuple(
        authenticated_intake("action_started", producer_index=index) for index in (9, 2, 7, 1)
    )
    for intake in intakes:
        spool.enqueue(intake)
    expected = tuple(sorted(intakes, key=lambda intake: intake.producer_sequence or 0))
    store = _RecordingStore()

    result = spool.drain(store)

    assert tuple(store.received) == expected
    assert all(not hasattr(intake, "receipt_ordinal") for intake in store.received)
    assert result.admitted_events == len(intakes)
    assert result.remaining_events == 0
    assert _entries(locations.spool_directory) == ()
    assert spool.health().queued_events == 0


def test_drain_deterministically_orders_mixed_sequence_timestamp_and_unavailable_events(
    tmp_path: Path,
) -> None:
    locations = _locations(tmp_path)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    sequence_early = authenticated_intake(
        "action_started",
        producer_index=21,
        changes={"producer_sequence": 2},
    )
    sequence_late = authenticated_intake(
        "action_started",
        producer_index=22,
        changes={"producer_sequence": 8},
    )
    timestamp_early = authenticated_intake(
        "action_started",
        producer_index=31,
        changes={
            "occurred_at": datetime(2026, 7, 19, 9, tzinfo=UTC),
            "timestamp_authority": "local_observation",
            "producer_sequence": None,
            "sequence_authority": "unavailable",
        },
    )
    timestamp_late = authenticated_intake(
        "action_started",
        producer_index=32,
        changes={
            "occurred_at": datetime(2026, 7, 19, 10, tzinfo=UTC),
            "timestamp_authority": "local_observation",
            "producer_sequence": None,
            "sequence_authority": "unavailable",
        },
    )
    unavailable = (
        authenticated_intake(
            "action_started",
            producer_index=41,
            changes={
                "producer_sequence": None,
                "sequence_authority": "unavailable",
            },
        ),
        authenticated_intake(
            "action_started",
            producer_index=42,
            changes={
                "producer_sequence": None,
                "sequence_authority": "unavailable",
            },
        ),
    )
    intakes = (
        unavailable[1],
        timestamp_late,
        sequence_late,
        unavailable[0],
        timestamp_early,
        sequence_early,
    )
    for intake in intakes:
        spool.enqueue(intake)
    persisted_by_name = tuple(
        (path.name, load_capture_intake(_framed_intake(path)))
        for path in _entries(locations.spool_directory)
    )
    unavailable_by_name = tuple(
        intake
        for _name, intake in persisted_by_name
        if intake.producer_sequence is None and intake.occurred_at is None
    )
    expected = (
        sequence_early,
        sequence_late,
        timestamp_early,
        timestamp_late,
        *unavailable_by_name,
    )
    store = _RecordingStore()

    result = spool.drain(store)

    assert tuple(store.received) == expected
    assert result.admitted_events == len(intakes)
    assert result.remaining_events == 0
    assert _entries(locations.spool_directory) == ()


def test_drain_orders_a_complete_session_and_store_assigns_ordinals_transactionally(
    tmp_path: Path,
) -> None:
    locations = _locations(tmp_path)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    intakes = all_authenticated_intakes()
    for intake in reversed(intakes):
        spool.enqueue(intake)
    database_path = tmp_path / "capture.sqlite3"

    with initialized_store(database_path) as store:
        register_connection(store)
        result = spool.drain(store)
        verification = store.verify_session(CONNECTION_ID, intakes[0].session_id)

    assert result.admitted_events == len(intakes)
    assert result.remaining_events == 0
    assert verification.event_count == len(intakes)
    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute(
            """
            SELECT receipt_ordinal, event_kind, admission_source
            FROM capture_events ORDER BY receipt_ordinal
            """
        ).fetchall()
    finally:
        connection.close()
    assert rows == [
        (index, intake.kind, "spool_drain") for index, intake in enumerate(intakes, start=1)
    ]
    assert _entries(locations.spool_directory) == ()


@pytest.mark.parametrize("quota", ("events", "bytes"))
def test_spool_quota_drops_new_intakes_and_persists_degraded_health(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quota: str,
) -> None:
    if quota == "events":
        monkeypatch.setattr(spool_module, "MAX_CAPTURE_SPOOL_EVENTS", 1)
    else:
        monkeypatch.setattr(spool_module, "MAX_CAPTURE_SPOOL_BYTES", 1)
    locations = _locations(tmp_path)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    first = authenticated_intake("session_started", producer_index=1)
    second = authenticated_intake("session_finished", producer_index=2)

    first_result = spool.enqueue(first)
    dropped = spool.enqueue(second)
    if quota == "events":
        assert first_result.disposition == "queued"
        expected_queued = 1
    else:
        assert first_result.disposition == "dropped_quota"
        expected_queued = 0

    health = spool.health()
    assert dropped.disposition == "dropped_quota"
    assert health.queued_events == expected_queued
    assert health.dropped_events == 1 + (quota == "bytes")
    assert health.coverage_degraded is True
    assert health.last_drop_reason == "spool_quota"

    reopened = CaptureSpool.open(locations, INSTALLATION_KEY)
    assert reopened.health() == health


def test_tampered_spool_record_fails_content_free_and_is_never_removed(
    tmp_path: Path,
) -> None:
    locations = _locations(tmp_path)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    spool.enqueue(authenticated_intake("session_started", producer_index=1))
    entry = _entries(locations.spool_directory)[0]
    tampered = entry.read_bytes()[:-1] + b"x"
    entry.write_bytes(tampered)
    store = _RecordingStore()

    with pytest.raises(CaptureSpoolIntegrityError) as captured:
        spool.drain(store)

    assert store.received == []
    assert entry.read_bytes() == tampered
    assert str(captured.value) == "capture spool integrity check failed"
    assert "synthetic" not in repr(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.parametrize("mutation", ("bad_header", "nonhex_tag", "different_name"))
def test_corrupt_spool_entry_format_is_rejected_without_removal(
    tmp_path: Path,
    mutation: str,
) -> None:
    locations = _locations(tmp_path)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    spool.enqueue(authenticated_intake("session_started", producer_index=1))
    entry = _entries(locations.spool_directory)[0]
    original = entry.read_bytes()
    corrupted_path = entry
    secret = "provider-native-entry-format-secret"
    if mutation == "bad_header":
        _header, tag, intake = original.split(b"\n", maxsplit=2)
        corrupted_path.write_bytes(b"\n".join((secret.encode(), tag, intake)))
    elif mutation == "nonhex_tag":
        header, _tag, intake = original.split(b"\n", maxsplit=2)
        corrupted_path.write_bytes(b"\n".join((header, b"g" * 64, intake)))
    else:
        alternate = "0" * 64 if entry.name != f"{'0' * 64}.capture-intake" else "1" * 64
        corrupted_path = entry.with_name(f"{alternate}.capture-intake")
        entry.rename(corrupted_path)
    corrupted = corrupted_path.read_bytes()
    store = _RecordingStore()

    with pytest.raises(CaptureSpoolIntegrityError) as health_error:
        spool.health()
    with pytest.raises(CaptureSpoolIntegrityError) as drain_error:
        spool.drain(store)

    assert store.received == []
    assert corrupted_path.read_bytes() == corrupted
    _assert_content_free_spool_error(
        health_error.value,
        message="capture spool integrity check failed",
        secret=secret,
    )
    _assert_content_free_spool_error(
        drain_error.value,
        message="capture spool integrity check failed",
        secret=secret,
    )


@pytest.mark.skipif(os.name != "posix", reason="private health files require POSIX")
@pytest.mark.parametrize(
    "mutation",
    ("bad_header", "nonhex_tag", "tampered_payload", "mode", "hardlink", "symlink"),
)
def test_corrupt_or_unsafe_health_marker_is_rejected_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    monkeypatch.setattr(spool_module, "MAX_CAPTURE_SPOOL_EVENTS", 0)
    locations = _locations(tmp_path)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    dropped = spool.enqueue(authenticated_intake("session_started", producer_index=1))
    marker = locations.spool_directory / ".capture-spool-health"
    original = marker.read_bytes()
    preserved = marker
    secret = "provider-native-health-secret"
    if mutation == "bad_header":
        _header, tag, payload = original.split(b"\n", maxsplit=2)
        marker.write_bytes(b"\n".join((secret.encode(), tag, payload)))
    elif mutation == "nonhex_tag":
        header, _tag, payload = original.split(b"\n", maxsplit=2)
        marker.write_bytes(b"\n".join((header, b"g" * 64, payload)))
    elif mutation == "tampered_payload":
        header, tag, payload = original.split(b"\n", maxsplit=2)
        marker.write_bytes(
            b"\n".join((header, tag, payload.replace(b'"dropped_events":1', b'"dropped_events":2')))
        )
    elif mutation == "mode":
        marker.chmod(0o666)
    elif mutation == "hardlink":
        preserved = tmp_path / secret
        preserved.hardlink_to(marker)
    else:
        preserved = tmp_path / secret
        marker.rename(preserved)
        marker.symlink_to(preserved)
    corrupted = preserved.read_bytes()

    with pytest.raises(CaptureSpoolIntegrityError) as captured:
        spool.health()

    assert dropped.disposition == "dropped_quota"
    assert preserved.read_bytes() == corrupted
    if mutation == "symlink":
        assert marker.is_symlink()
    _assert_content_free_spool_error(
        captured.value,
        message="capture spool integrity check failed",
        secret=secret,
    )


def test_renamed_spool_record_cannot_disappear_from_health_or_drain(
    tmp_path: Path,
) -> None:
    locations = _locations(tmp_path)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    spool.enqueue(authenticated_intake("session_started", producer_index=1))
    entry = _entries(locations.spool_directory)[0]
    concealed = entry.with_name("concealed-intake")
    entry.rename(concealed)
    store = _RecordingStore()

    with pytest.raises(CaptureSpoolIntegrityError):
        spool.health()
    with pytest.raises(CaptureSpoolIntegrityError):
        spool.drain(store)

    assert store.received == []
    assert concealed.exists()


def test_wrong_installation_key_cannot_read_or_remove_spooled_intakes(
    tmp_path: Path,
) -> None:
    locations = _locations(tmp_path)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    spool.enqueue(authenticated_intake("session_started", producer_index=1))
    before = tuple((path.name, path.read_bytes()) for path in _entries(locations.spool_directory))
    store = _RecordingStore()

    with pytest.raises(CaptureSpoolIntegrityError):
        wrong_key_spool = CaptureSpool.open(locations, WRONG_INSTALLATION_KEY)
        wrong_key_spool.drain(store)

    assert store.received == []
    after = tuple((path.name, path.read_bytes()) for path in _entries(locations.spool_directory))
    assert after == before


def test_spool_requires_an_exact_installation_key(tmp_path: Path) -> None:
    locations = _locations(tmp_path)

    for invalid in (b"k" * 32, object(), None):
        with pytest.raises(CaptureSpoolError):
            CaptureSpool.open(locations, invalid)  # type: ignore[arg-type]

    assert not locations.spool_directory.exists()


def test_admission_returns_direct_receipt_and_spools_only_store_busy(
    tmp_path: Path,
) -> None:
    locations = _locations(tmp_path)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    intake = authenticated_intake("session_started", producer_index=1)
    receipt = object()
    direct_store = _AppendOutcomeStore(receipt)

    assert admit_capture_intake(direct_store, spool, intake) is receipt
    assert direct_store.calls == 1
    assert spool.health().queued_events == 0

    busy_store = _AppendOutcomeStore(CaptureStoreBusyError(), raises=True)
    fallback = admit_capture_intake(busy_store, spool, intake)
    assert fallback.disposition == "queued"
    assert busy_store.calls == 1
    assert spool.health().queued_events == 1


@pytest.mark.parametrize(
    "error",
    (
        CaptureStoreIntegrityError(),
        CaptureStoreStateError(),
        RuntimeError("provider-native-sentinel"),
    ),
)
def test_admission_never_spools_non_busy_failures(
    tmp_path: Path,
    error: BaseException,
) -> None:
    locations = _locations(tmp_path)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    intake = authenticated_intake("session_started", producer_index=1)
    store = _AppendOutcomeStore(error, raises=True)

    with pytest.raises(type(error)) as captured:
        admit_capture_intake(store, spool, intake)

    assert captured.value is error
    assert store.calls == 1
    assert spool.health().queued_events == 0
    assert _entries(locations.spool_directory) == ()


def test_drain_stops_on_busy_and_retains_the_current_and_later_entries(
    tmp_path: Path,
) -> None:
    locations = _locations(tmp_path)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    for index in range(1, 4):
        spool.enqueue(authenticated_intake("action_started", producer_index=index))
    before = tuple(path.name for path in _entries(locations.spool_directory))
    store = _AppendOutcomeStore(CaptureStoreBusyError(), raises=True)

    result = spool.drain(store)

    assert result.admitted_events == 0
    assert result.remaining_events == 3
    assert store.calls == 1
    assert tuple(path.name for path in _entries(locations.spool_directory)) == before


def test_drain_busy_after_partial_progress_retains_only_current_and_later_entries(
    tmp_path: Path,
) -> None:
    locations = _locations(tmp_path)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    expected = tuple(
        authenticated_intake("action_started", producer_index=index) for index in range(1, 6)
    )
    for intake in reversed(expected):
        spool.enqueue(intake)
    store = _DelayedFailureStore(CaptureStoreBusyError(), fail_on_call=3)

    result = spool.drain(store)

    assert result.admitted_events == 2
    assert result.remaining_events == 3
    assert store.calls == 3
    assert tuple(store.received) == expected[:2]
    assert _queued_intakes(locations.spool_directory) == expected[2:]
    assert spool.health().queued_events == 3


def test_drain_non_busy_error_after_partial_progress_propagates_and_retains_tail(
    tmp_path: Path,
) -> None:
    locations = _locations(tmp_path)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    expected = tuple(
        authenticated_intake("action_started", producer_index=index) for index in range(1, 6)
    )
    for intake in reversed(expected):
        spool.enqueue(intake)
    error = CaptureStoreIntegrityError()
    store = _DelayedFailureStore(error, fail_on_call=3)

    with pytest.raises(CaptureStoreIntegrityError) as captured:
        spool.drain(store)

    assert captured.value is error
    assert store.calls == 3
    assert tuple(store.received) == expected[:2]
    assert _queued_intakes(locations.spool_directory) == expected[2:]
    assert spool.health().queued_events == 3


@pytest.mark.skipif(os.name != "posix", reason="private lock files require POSIX")
@pytest.mark.parametrize("mutation", ("mode", "hardlink", "symlink"))
def test_spool_rejects_unsafe_lock_files_without_following_or_mutating_them(
    tmp_path: Path,
    mutation: str,
) -> None:
    locations = _locations(tmp_path)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    lock = locations.spool_directory / ".capture-spool-lock"
    secret = "provider-native-lock-secret"
    preserved = lock
    if mutation == "symlink":
        preserved = tmp_path / "lock-target"
        preserved.write_bytes(secret.encode())
        preserved.chmod(0o600)
        lock.symlink_to(preserved)
    else:
        lock.write_bytes(secret.encode())
        lock.chmod(0o600)
        if mutation == "mode":
            lock.chmod(0o644)
        else:
            preserved = tmp_path / "lock-alias"
            preserved.hardlink_to(lock)

    with pytest.raises(CaptureSpoolError) as captured:
        spool.health()

    assert preserved.read_bytes() == secret.encode()
    if mutation == "symlink":
        assert lock.is_symlink()
    _assert_content_free_spool_error(
        captured.value,
        message="capture spool operation failed",
        secret=secret,
    )


@pytest.mark.skipif(os.name != "posix", reason="authorized deletion requires POSIX")
def test_drain_never_unlinks_a_replacement_installed_after_the_authenticated_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations = _locations(tmp_path)
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    intake = authenticated_intake("session_started", producer_index=1)
    spool.enqueue(intake)
    entry = _entries(locations.spool_directory)[0]
    displaced = locations.spool_directory / "authenticated-displaced"
    real_delete = spool_module.security_files._delete_authorized_private_file_at_descriptor

    def replace_then_delete(
        directory: object,
        directory_fd: int,
        authorization: object,
    ) -> None:
        entry.rename(displaced)
        entry.write_bytes(b"unexpected-replacement")
        entry.chmod(0o600)
        real_delete(directory, directory_fd, authorization)  # type: ignore[arg-type]

    monkeypatch.setattr(
        spool_module.security_files,
        "_delete_authorized_private_file_at_descriptor",
        replace_then_delete,
    )
    store = _RecordingStore()

    with pytest.raises(CaptureSpoolError) as captured:
        spool.drain(store)

    assert store.received == [intake]
    assert displaced.exists()
    assert entry.read_bytes() == b"unexpected-replacement"
    assert str(captured.value) == "capture spool operation failed"
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_spool_representations_do_not_expose_paths_keys_or_intakes(tmp_path: Path) -> None:
    locations = _locations(tmp_path / "provider-native-secret")
    spool = CaptureSpool.open(locations, INSTALLATION_KEY)
    result = spool.enqueue(authenticated_intake("session_started", producer_index=1))

    rendered = f"{spool!r}\n{result!r}\n{spool.health()!r}"
    assert "provider-native-secret" not in rendered
    assert "capture-intake" not in rendered
    assert "redacted" in rendered.casefold()
