"""Coverage-oriented regressions for otherwise rare contract edges."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from tests.capture.store_support import INSTALLATION_KEY, authenticated_intake

import saliencegate.capture.spool as spool_module
from saliencegate.capture.locations import resolve_capture_store_locations
from saliencegate.capture.schema import CaptureIntake
from saliencegate.capture.spool import (
    CaptureSpool,
    CaptureSpoolError,
    CaptureSpoolIntegrityError,
    CaptureSpoolObservationError,
    CaptureSpoolUnavailableError,
)
from saliencegate.capture.store import CaptureStoreBusyError
from saliencegate.domain import canonical_json


def _spool(tmp_path: Path) -> CaptureSpool:
    locations = resolve_capture_store_locations(
        environ={"XDG_STATE_HOME": str(tmp_path / "state")},
        home=tmp_path / "home",
        platform="posix",
    )
    return CaptureSpool.open(locations, INSTALLATION_KEY)


def _health_frame(spool: CaptureSpool, value: object, *, version: int = 2) -> bytes:
    payload = canonical_json(value)
    tag = spool._key._hmac_sha256(payload, domain=spool_module._HEALTH_DOMAIN)
    header = spool_module._HEALTH_HEADER if version == 2 else spool_module._HEALTH_HEADER_V1
    return b"\n".join((header, tag.encode("ascii"), payload))


class _TransportStore:
    def __init__(self, outcome: object = None, *, failure: BaseException | None = None) -> None:
        self.outcome = outcome
        self.failure = failure
        self.calls = 0

    def append_transport_chunk(self, _chunk: object, _intakes: object) -> object:
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        return self.outcome


@pytest.mark.parametrize(
    ("value", "version"),
    (
        ([], 2),
        (
            {
                "schema_version": "capture-spool-health/v1",
                "dropped_events": 1,
                "last_drop_reason": "spool_incomplete",
                "acknowledged_marker": None,
            },
            2,
        ),
        (
            {
                "schema_version": "capture-spool-health/v2",
                "dropped_events": 1,
                "last_drop_reason": "spool_incomplete",
                "acknowledged_marker": None,
                "bridge_barrier_marker": None,
            },
            1,
        ),
        ({"schema_version": "unknown"}, 2),
        (
            {
                "schema_version": "capture-spool-health/v2",
                "dropped_events": 0,
                "last_drop_reason": "spool_quota",
                "acknowledged_marker": None,
                "bridge_barrier_marker": None,
            },
            2,
        ),
        (
            {
                "schema_version": "capture-spool-health/v2",
                "dropped_events": 1,
                "last_drop_reason": "spool_quota",
                "acknowledged_marker": "bad",
                "bridge_barrier_marker": None,
            },
            2,
        ),
        (
            {
                "schema_version": "capture-spool-health/v2",
                "dropped_events": 1,
                "last_drop_reason": "spool_quota",
                "acknowledged_marker": None,
                "bridge_barrier_marker": "bad",
            },
            2,
        ),
    ),
)
def test_drop_state_decoder_rejects_each_authenticated_grammar_boundary(
    tmp_path: Path,
    value: object,
    version: int,
) -> None:
    spool = _spool(tmp_path)
    assert spool._decode_drop_state(_health_frame(spool, value, version=version)) is None


def test_drop_state_decoder_rejects_signed_noncanonical_and_non_json_payloads(
    tmp_path: Path,
) -> None:
    spool = _spool(tmp_path)
    for payload in (b"not-json", b'{"schema_version": "capture-spool-health/v2"}'):
        tag = spool._key._hmac_sha256(payload, domain=spool_module._HEALTH_DOMAIN)
        frame = b"\n".join((spool_module._HEALTH_HEADER, tag.encode("ascii"), payload))
        assert spool._decode_drop_state(frame) is None


@pytest.mark.parametrize(
    ("intakes", "fallback", "store", "expected"),
    (
        ([], (), _TransportStore(), CaptureSpoolError),
        ((), (), _TransportStore(), CaptureSpoolError),
        ((), (object(),), _TransportStore(), CaptureSpoolError),
        (
            (),
            (authenticated_intake("session_started"),),
            object(),
            CaptureSpoolError,
        ),
    ),
)
def test_transport_admission_rejects_invalid_container_and_store_boundaries(
    tmp_path: Path,
    intakes: object,
    fallback: object,
    store: object,
    expected: type[Exception],
) -> None:
    with pytest.raises(expected):
        _spool(tmp_path).admit_transport(
            store,
            object(),
            intakes,  # type: ignore[arg-type]
            fallback,  # type: ignore[arg-type]
        )


def test_transport_admission_never_normalizes_nonbusy_store_failures(
    tmp_path: Path,
) -> None:
    spool = _spool(tmp_path)
    fallback = (authenticated_intake("session_started"),)
    store = _TransportStore(failure=RuntimeError("store secret"))
    with pytest.raises(RuntimeError, match="store secret"):
        spool.admit_transport(store, object(), (), fallback)


def test_transport_admission_classifies_failures_after_lock_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spool = _spool(tmp_path)
    fallback = (authenticated_intake("session_started"),)
    monkeypatch.setattr(
        CaptureSpool,
        "_drop_state",
        lambda *_args: (_ for _ in ()).throw(ValueError()),
    )
    with pytest.raises(CaptureSpoolError):
        spool.admit_transport(_TransportStore(), object(), (), fallback)


def test_transport_admission_classifies_failures_before_lock_entry_as_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spool = _spool(tmp_path)
    fallback = (authenticated_intake("session_started"),)

    @contextmanager
    def unavailable(*_args: object, **_kwargs: object) -> Iterator[int | None]:
        raise ValueError("lock unavailable")
        yield None

    monkeypatch.setattr(CaptureSpool, "_locked", unavailable)
    with pytest.raises(CaptureSpoolUnavailableError):
        spool.admit_transport(_TransportStore(), object(), (), fallback)


def test_direct_admission_classifies_pre_and_post_lock_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intake = authenticated_intake("session_started")

    @contextmanager
    def unavailable(*_args: object, **_kwargs: object) -> Iterator[int | None]:
        raise ValueError("lock unavailable")
        yield None

    with monkeypatch.context() as patch:
        patch.setattr(CaptureSpool, "_locked", unavailable)
        with pytest.raises(CaptureSpoolUnavailableError):
            _spool(tmp_path / "before").admit(object(), intake)

    with monkeypatch.context() as patch:
        patch.setattr(
            CaptureSpool,
            "_admit_locked",
            lambda *_args: (_ for _ in ()).throw(ValueError("inside lock")),
        )
        with pytest.raises(CaptureSpoolError):
            _spool(tmp_path / "after").admit(object(), intake)


def _marker(
    spool: CaptureSpool,
    key: tuple[str, str],
    tag: str,
    state: str,
) -> tuple[Path, Any, object]:
    path = spool._locations.spool_directory / f"{tag}{spool_module._SESSION_MARKER_SUFFIX}"
    return path, cast(Any, state), object()


def test_orphan_reconciliation_covers_barrier_and_acknowledgement_faults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spool = _spool(tmp_path)
    key = ("connection-one", "a" * 64)
    other = ("connection-two", "b" * 64)
    tag = "c" * 64

    scenarios = (
        ((0, None, None, tag), {key}, {key: _marker(spool, key, tag, "invalid")}),
        ((0, None, None, tag), {key}, {key: _marker(spool, key, "d" * 64, "pending")}),
        ((1, "spool_incomplete", tag, None), {key}, {other: _marker(spool, other, tag, "pending")}),
        ((0, None, None, None), {key}, {key: _marker(spool, key, tag, "acknowledged")}),
    )
    for drop_state, orphans, markers in scenarios:
        with monkeypatch.context() as patch:
            patch.setattr(CaptureSpool, "_drop_state", lambda *_args, state=drop_state: state)
            with pytest.raises(CaptureSpoolIntegrityError):
                spool._persist_orphan_degradation_locked(
                    set(orphans),
                    dict(markers),
                    None,
                    remove=False,
                )


def test_orphan_reconciliation_acknowledges_and_removes_each_marker_class(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spool = _spool(tmp_path)
    first = ("connection-one", "a" * 64)
    second = ("connection-two", "b" * 64)
    first_tag = "c" * 64
    second_tag = "d" * 64
    markers = {
        first: _marker(spool, first, first_tag, "pending"),
        second: _marker(spool, second, second_tag, "pending"),
    }
    deleted: list[object] = []

    def set_state(
        _self: CaptureSpool,
        key: tuple[str, str],
        state: str,
        _directory_fd: int | None,
    ) -> tuple[Path, Any, object]:
        path, _old_state, stable = markers[key]
        return path, cast(Any, state), stable

    with monkeypatch.context() as patch:
        patch.setattr(
            CaptureSpool,
            "_drop_state",
            lambda *_args: (1, "spool_incomplete", first_tag, second_tag),
        )
        patch.setattr(CaptureSpool, "_set_session_marker_state_locked", set_state)
        patch.setattr(
            CaptureSpool,
            "_delete_stable_file_locked",
            lambda _self, stable, _fd: deleted.append(stable),
        )
        spool._persist_orphan_degradation_locked(
            {first, second},
            markers,
            None,
            remove=True,
            clear_bridge_barrier=True,
        )
    assert len(deleted) == 2
    assert markers == {}


def test_orphan_reconciliation_persists_and_acknowledges_a_new_orphan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spool = _spool(tmp_path)
    key = ("connection-one", "a" * 64)
    tag = "c" * 64
    markers = {key: _marker(spool, key, tag, "pending")}
    writes: list[tuple[object, ...]] = []

    with monkeypatch.context() as patch:
        patch.setattr(CaptureSpool, "_drop_state", lambda *_args: (0, None, None, None))
        patch.setattr(
            CaptureSpool,
            "_write_drop_state",
            lambda _self, *args, **kwargs: writes.append((*args, kwargs)),
        )
        patch.setattr(
            CaptureSpool,
            "_set_session_marker_state_locked",
            lambda _self, _key, state, _fd: (markers[key][0], state, markers[key][2]),
        )
        spool._persist_orphan_degradation_locked(
            {key},
            markers,
            None,
            remove=False,
        )
    assert writes
    assert markers[key][1] == "acknowledged"


@pytest.mark.parametrize(
    ("intake", "error"),
    (
        (object(), CaptureSpoolError),
        (authenticated_intake("session_started"), CaptureStoreBusyError),
    ),
)
def test_spool_admit_frame_and_store_failure_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    intake: object,
    error: type[Exception],
) -> None:
    spool = _spool(tmp_path)
    if error is CaptureStoreBusyError:

        class BusyStore:
            def append(self, _intake: CaptureIntake) -> object:
                raise CaptureStoreBusyError()

        receipt = spool.admit(BusyStore(), cast(CaptureIntake, intake))
        assert receipt.disposition == "queued"
    else:
        with pytest.raises(error):
            spool.admit(object(), cast(CaptureIntake, intake))


def test_lock_rejects_nonboolean_mode_and_invalid_spool_identity(
    tmp_path: Path,
) -> None:
    spool = _spool(tmp_path)
    with pytest.raises(CaptureSpoolError), spool._locked(blocking=1):  # type: ignore[arg-type]
        pass
    original = spool._spool_identity
    spool._spool_identity = object()  # type: ignore[assignment]
    try:
        with pytest.raises(spool_module.SecureFileError), spool._locked():
            pass
    finally:
        spool._spool_identity = original


def test_marker_publication_rejects_an_unreadable_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spool = _spool(tmp_path)
    directory = cast(
        spool_module.security_files._PrivateDirectoryAuthorization,
        spool._spool_identity,
    )
    descriptor = spool_module.security_files._open_authorized_private_directory(directory)
    try:
        monkeypatch.setattr(
            spool_module.security_files,
            "_publish_private_file_at_descriptor",
            lambda *_args, **_kwargs: object(),
        )
        monkeypatch.setattr(CaptureSpool, "_read_session_marker", lambda *_args: None)
        with pytest.raises(CaptureSpoolIntegrityError):
            spool._ensure_session_marker_locked("connection-one", "a" * 64, descriptor)
        with pytest.raises(CaptureSpoolIntegrityError):
            spool._set_session_marker_state_locked(
                ("connection-one", "a" * 64),
                "acknowledged",
                descriptor,
            )
    finally:
        os.close(descriptor)


def test_spool_drain_receipt_repr_is_redacted() -> None:
    receipt = spool_module.CaptureSpoolDrainReceipt(admitted_events=1, remaining_events=2)
    assert repr(receipt) == "CaptureSpoolDrainReceipt(<redacted>)"


def test_observation_sealing_rejects_inexact_health_and_key_types() -> None:
    with pytest.raises(CaptureSpoolObservationError):
        spool_module._seal_spool_observation(
            object(),  # type: ignore[arg-type]
            snapshot_digest="0" * 64,
            spool_boundary_digest="1" * 64,
            installation_key=INSTALLATION_KEY,
        )
    with pytest.raises(CaptureSpoolObservationError):
        spool_module._seal_spool_observation(
            spool_module.CaptureSpoolHealth(
                queued_events=0,
                queued_bytes=0,
                dropped_events=0,
                coverage_degraded=False,
                last_drop_reason=None,
            ),
            snapshot_digest="0" * 64,
            spool_boundary_digest="1" * 64,
            installation_key=object(),  # type: ignore[arg-type]
        )


def test_observation_verification_never_normalizes_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupt(_value: object) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(spool_module, "_validated_spool_observation", interrupt)
    with pytest.raises(KeyboardInterrupt):
        spool_module.verify_capture_spool_observation(
            object(),
            expected_snapshot_digest="0" * 64,
            expected_spool_boundary_digest="1" * 64,
            installation_key=INSTALLATION_KEY,
        )


def test_inactive_maintenance_token_rejects_all_operations(tmp_path: Path) -> None:
    token = spool_module.CaptureSpoolMaintenance(_spool(tmp_path), None)
    token._close()
    with pytest.raises(CaptureSpoolError):
        token.drain(object())  # type: ignore[arg-type]
    with pytest.raises(CaptureSpoolError):
        token.clear_drop_health_if_empty()


def test_capture_spool_cannot_be_constructed_directly() -> None:
    with pytest.raises(CaptureSpoolError):
        CaptureSpool()


def test_spool_open_rejects_a_platform_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations = resolve_capture_store_locations(
        environ={"XDG_STATE_HOME": str(tmp_path / "state")},
        home=tmp_path / "home",
        platform="posix",
    )
    monkeypatch.setattr(spool_module.os, "name", "nt")
    with pytest.raises(CaptureSpoolError):
        CaptureSpool._open(locations, INSTALLATION_KEY, create=True)


def test_spool_read_only_audit_normalizes_unexpected_open_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spool = _spool(tmp_path)

    def fail(*_args: object, **_kwargs: object) -> CaptureSpool:
        raise ValueError("audit secret")

    monkeypatch.setattr(CaptureSpool, "_open", fail)
    with pytest.raises(CaptureSpoolError):
        CaptureSpool.audit_read_only(
            spool._locations,
            installation_key=INSTALLATION_KEY,
        )


def test_lock_rechecks_the_posix_spool_identity_after_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spool = _spool(tmp_path)
    spool._spool_identity = object()  # type: ignore[assignment]
    monkeypatch.setattr(CaptureSpool, "_revalidate", lambda _self: None)
    with pytest.raises(spool_module.SecureFileError), spool._locked():
        pass


@pytest.mark.parametrize("failure_call", (2, 4))
def test_lock_rejects_metadata_replacement_before_and_after_the_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_call: int,
) -> None:
    spool = _spool(tmp_path)
    calls = 0

    def safe(_value: object) -> bool:
        nonlocal calls
        calls += 1
        return calls != failure_call

    monkeypatch.setattr(spool_module, "_safe_lock_metadata", safe)
    with pytest.raises(spool_module.SecureFileError), spool._locked():
        pass


def test_session_marker_helpers_reject_invalid_identity_and_state(tmp_path: Path) -> None:
    spool = _spool(tmp_path)
    with pytest.raises(CaptureSpoolIntegrityError):
        spool._session_marker_name_tag("", "a" * 64)
    with pytest.raises(CaptureSpoolIntegrityError):
        spool._session_marker_frame("connection-one", "a" * 64, cast(Any, "invalid"))


def test_posix_spool_file_helpers_require_a_directory_descriptor(tmp_path: Path) -> None:
    spool = _spool(tmp_path)
    with pytest.raises(spool_module.SecureFileError):
        spool._read_spool_file(Path("entry"), None, maximum_bytes=1)
    with pytest.raises(spool_module.SecureFileError):
        spool._named_spool_file_exists("entry", None)
    with pytest.raises(spool_module.SecureFileError):
        spool._audit_lock_boundary(None)
    with pytest.raises(CaptureSpoolIntegrityError):
        spool._drop_state_record(None)
    with pytest.raises(spool_module.SecureFileError):
        spool._write_drop_state(1, "spool_quota", None)


def test_session_marker_reader_rejects_a_mismatched_authenticated_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spool = _spool(tmp_path)
    monkeypatch.setattr(CaptureSpool, "_named_spool_file_exists", lambda *_args: True)
    monkeypatch.setattr(
        CaptureSpool,
        "_decode_session_marker",
        lambda *_args: (("other", "b" * 64), "pending", object()),
    )
    with pytest.raises(CaptureSpoolIntegrityError):
        spool._read_session_marker("connection-one", "a" * 64, 7)


@pytest.mark.parametrize("case", ("header", "tag", "grammar", "name"))
def test_session_marker_decoder_rejects_each_frame_and_name_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    spool = _spool(tmp_path)
    tag, valid = spool._session_marker_frame("connection-one", "a" * 64)
    if case == "header":
        framed = b"bad"
        path = tmp_path / f"{tag}{spool_module._SESSION_MARKER_SUFFIX}"
    elif case == "tag":
        framed = b"\n".join((spool_module._SESSION_MARKER_HEADER, b"bad", b"{}"))
        path = tmp_path / f"{tag}{spool_module._SESSION_MARKER_SUFFIX}"
    elif case == "grammar":
        payload = canonical_json({"schema_version": "capture-spool-session/v1"})
        content_tag = spool._key._hmac_sha256(
            payload,
            domain=spool_module._SESSION_MARKER_DOMAIN,
        )
        framed = b"\n".join(
            (spool_module._SESSION_MARKER_HEADER, content_tag.encode("ascii"), payload)
        )
        path = tmp_path / f"{tag}{spool_module._SESSION_MARKER_SUFFIX}"
    else:
        framed = valid
        path = tmp_path / f"{'f' * 64}{spool_module._SESSION_MARKER_SUFFIX}"
    monkeypatch.setattr(
        CaptureSpool,
        "_read_spool_file",
        lambda *_args, **_kwargs: SimpleNamespace(data=framed),
    )
    assert spool._decode_session_marker(path, 7) is None


def test_audit_lock_boundary_rechecks_post_read_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spool = _spool(tmp_path)
    checks = iter((True, False))
    monkeypatch.setattr(spool_module.os, "stat", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(spool_module, "_safe_lock_metadata", lambda _value: next(checks))
    monkeypatch.setattr(
        spool_module.security_files,
        "_read_private_file_at_descriptor",
        lambda *_args, **_kwargs: object(),
    )
    with pytest.raises(spool_module.SecureFileError):
        spool._audit_lock_boundary(7)


def test_audit_rejects_an_invalid_posix_spool_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spool = _spool(tmp_path)
    spool._spool_identity = object()  # type: ignore[assignment]
    monkeypatch.setattr(CaptureSpool, "_revalidate", lambda _self: None)
    with pytest.raises(CaptureSpoolError):
        spool._audit_read_only()


def test_spool_child_scan_checks_identity_before_and_after_listing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spool = _spool(tmp_path)
    with pytest.raises(spool_module.SecureFileError):
        spool._spool_child_paths(None)

    original = spool._spool_identity

    def invalidate(*_args: object) -> None:
        spool._spool_identity = object()  # type: ignore[assignment]

    monkeypatch.setattr(
        spool_module.security_files,
        "_require_authorized_private_directory_descriptor",
        invalidate,
    )
    monkeypatch.setattr(spool_module.os, "listdir", lambda _descriptor: [])
    try:
        with pytest.raises(spool_module.SecureFileError):
            spool._spool_child_paths(7)
    finally:
        spool._spool_identity = original


def test_backlog_scan_rejects_unknown_names_and_missing_descriptor(
    tmp_path: Path,
) -> None:
    spool = _spool(tmp_path)
    with pytest.raises(spool_module.SecureFileError):
        spool._bridge_fallback_backlog_exceeds_limit_locked(None)
    with spool._locked() as directory_fd:
        assert directory_fd is not None
        (spool._locations.spool_directory / "rogue").write_bytes(b"")
        with pytest.raises(CaptureSpoolIntegrityError):
            spool._bridge_fallback_backlog_exceeds_limit_locked(directory_fd)


def test_entry_decoder_rejects_invalid_identity_canonicalization_and_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spool = _spool(tmp_path)
    original = spool._spool_identity
    spool._spool_identity = object()  # type: ignore[assignment]
    try:
        assert spool._decode_entry(tmp_path / "entry", 7) is None
    finally:
        spool._spool_identity = original

    intake = authenticated_intake("session_started")
    tag, framed = spool._frame(intake)
    path = tmp_path / f"{tag}{spool_module._ENTRY_SUFFIX}"
    stable = SimpleNamespace(
        data=framed,
        authorization=SimpleNamespace(revalidate=lambda: None),
    )
    monkeypatch.setattr(
        spool_module.security_files,
        "_read_private_file_at_descriptor",
        lambda *_args, **_kwargs: stable,
    )
    monkeypatch.setattr(spool_module, "canonical_capture_intake", lambda _value: b"different")
    assert spool._decode_entry(path, 7) is None

    def fail(*_args: object, **_kwargs: object) -> object:
        raise OSError("read secret")

    monkeypatch.setattr(spool_module.security_files, "_read_private_file_at_descriptor", fail)
    assert spool._decode_entry(path, 7) is None


def test_session_marker_inventory_rejects_duplicate_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spool = _spool(tmp_path)
    paths = (tmp_path / "first", tmp_path / "second")
    decoded = (("connection-one", "a" * 64), "pending", object())
    monkeypatch.setattr(CaptureSpool, "_session_marker_paths", lambda *_args: paths)
    monkeypatch.setattr(CaptureSpool, "_decode_session_marker", lambda *_args: decoded)
    with pytest.raises(CaptureSpoolIntegrityError):
        spool._session_markers(7)


def test_health_state_rejects_marker_acknowledgement_inconsistencies(
    tmp_path: Path,
) -> None:
    path = tmp_path / f"{'a' * 64}{spool_module._SESSION_MARKER_SUFFIX}"
    key = ("connection-one", "b" * 64)
    marker = (path, cast(Any, "acknowledged"), object())
    with pytest.raises(CaptureSpoolIntegrityError):
        CaptureSpool._health_from_state(
            (),
            dropped_events=0,
            last_drop_reason=None,
            acknowledged_marker=None,
            bridge_barrier_marker=None,
            markers={key: marker},
            orphan_markers=set(),
        )

    pending = (path, cast(Any, "pending"), object())
    with pytest.raises(CaptureSpoolIntegrityError):
        CaptureSpool._health_from_state(
            (),
            dropped_events=1,
            last_drop_reason="spool_incomplete",
            acknowledged_marker="a" * 64,
            bridge_barrier_marker=None,
            markers={key: pending},
            orphan_markers=set(),
        )
    with pytest.raises(CaptureSpoolIntegrityError):
        CaptureSpool._health_from_state(
            (),
            dropped_events=1,
            last_drop_reason="spool_incomplete",
            acknowledged_marker=None,
            bridge_barrier_marker="c" * 64,
            markers={key: pending},
            orphan_markers=set(),
        )


def test_stable_delete_rejects_an_inexact_read_type(tmp_path: Path) -> None:
    spool = _spool(tmp_path)
    stable = SimpleNamespace(authorization=SimpleNamespace(revalidate=lambda: None))
    with pytest.raises(CaptureSpoolIntegrityError):
        spool._delete_stable_file_locked(stable, 7)  # type: ignore[arg-type]


def test_orphan_reconciliation_acknowledges_a_prebound_pending_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spool = _spool(tmp_path)
    key = ("connection-one", "a" * 64)
    tag = "b" * 64
    markers = {key: _marker(spool, key, tag, "pending")}
    monkeypatch.setattr(
        CaptureSpool,
        "_drop_state",
        lambda *_args: (1, "spool_incomplete", tag, None),
    )
    monkeypatch.setattr(
        CaptureSpool,
        "_set_session_marker_state_locked",
        lambda _self, marker_key, state, _fd: (
            markers[marker_key][0],
            state,
            markers[marker_key][2],
        ),
    )
    spool._persist_orphan_degradation_locked({key}, markers, None, remove=False)
    assert markers[key][1] == "acknowledged"


def test_marker_writers_require_a_posix_directory_descriptor(tmp_path: Path) -> None:
    spool = _spool(tmp_path)
    with pytest.raises(spool_module.SecureFileError):
        spool._set_session_marker_state_locked(
            ("connection-one", "a" * 64),
            "pending",
            None,
        )
    with pytest.raises(spool_module.SecureFileError):
        spool._ensure_session_marker_locked("connection-one", "a" * 64, None)


def test_health_and_drain_normalize_failures_inside_the_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spool = _spool(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(CaptureSpool, "_health_locked", lambda *_args: 1 / 0)
        with pytest.raises(CaptureSpoolError):
            spool.health()
    with monkeypatch.context() as patch:
        patch.setattr(CaptureSpool, "_drain_locked", lambda *_args: 1 / 0)
        with pytest.raises(CaptureSpoolError):
            spool.drain(object())  # type: ignore[arg-type]


def test_enqueue_rejects_existing_content_mismatch_and_missing_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spool = _spool(tmp_path)
    intake = authenticated_intake("session_started")
    tag = "a" * 64
    framed = b"frame"
    path = spool._locations.spool_directory / f"{tag}{spool_module._ENTRY_SUFFIX}"
    entries = ((path, intake, SimpleNamespace(data=b"different")),)
    monkeypatch.setattr(CaptureSpool, "_entries", lambda *_args: entries)
    monkeypatch.setattr(CaptureSpool, "_session_markers", lambda *_args: {})
    monkeypatch.setattr(CaptureSpool, "_reconcile_session_markers", lambda *_args: set())
    monkeypatch.setattr(
        CaptureSpool,
        "_persist_orphan_degradation_locked",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(CaptureSpoolIntegrityError):
        spool._enqueue_locked(intake, tag, framed, None)

    monkeypatch.setattr(CaptureSpool, "_entries", lambda *_args: ())
    monkeypatch.setattr(CaptureSpool, "_ensure_session_marker_locked", lambda *_args: object())
    with pytest.raises(spool_module.SecureFileError):
        spool._enqueue_locked(intake, tag, framed, None)


def test_bridge_fallback_and_barrier_cover_missing_marker_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spool = _spool(tmp_path)
    key = ("connection-one", "a" * 64)
    marker = _marker(spool, key, "b" * 64, "pending")
    monkeypatch.setattr(CaptureSpool, "_drop_state", lambda *_args: (0, None, None, None))
    monkeypatch.setattr(CaptureSpool, "_read_session_marker", lambda *_args: None)
    monkeypatch.setattr(CaptureSpool, "_ensure_session_marker_locked", lambda *_args: marker)
    monkeypatch.setattr(CaptureSpool, "_write_drop_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(CaptureSpool, "_set_session_marker_state_locked", lambda *_args: marker)
    receipts = spool._drop_bridge_fallback_locked(key, 2, None)
    assert len(receipts) == 2

    monkeypatch.setattr(CaptureSpool, "_decode_session_marker", lambda *_args: None)
    with pytest.raises(CaptureSpoolIntegrityError):
        spool._require_bridge_barrier_locked("b" * 64, None)
    monkeypatch.setattr(
        CaptureSpool,
        "_decode_session_marker",
        lambda *_args: (key, "invalid", object()),
    )
    with pytest.raises(CaptureSpoolIntegrityError):
        spool._require_bridge_barrier_locked("b" * 64, None)


def test_enqueue_normalizes_framing_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spool = _spool(tmp_path)
    monkeypatch.setattr(CaptureSpool, "_frame", lambda *_args: 1 / 0)
    with pytest.raises(CaptureSpoolError):
        spool.enqueue(authenticated_intake("session_started"))


def test_transport_admission_rejects_cross_session_fallback(tmp_path: Path) -> None:
    fallback = (
        authenticated_intake("session_started", session_native=b"first"),
        authenticated_intake("session_started", session_native=b"second"),
    )
    with pytest.raises(CaptureSpoolError):
        _spool(tmp_path).admit_transport(_TransportStore(), object(), (), fallback)


def test_read_only_audit_and_child_scan_cover_the_valid_posix_path(tmp_path: Path) -> None:
    spool = _spool(tmp_path)
    assert spool._audit_read_only().queued_events == 0
    with spool._locked() as directory_fd:
        assert directory_fd is not None
        assert spool._spool_child_paths(directory_fd)


def test_drop_state_writer_rejects_zero_drops_before_publication(tmp_path: Path) -> None:
    with pytest.raises(CaptureSpoolIntegrityError):
        _spool(tmp_path)._write_drop_state(0, "spool_quota", None)


def test_orphan_reconciliation_skips_a_nonorphan_bridge_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spool = _spool(tmp_path)
    barrier_key = ("barrier", "a" * 64)
    orphan_key = ("orphan", "b" * 64)
    barrier_tag = "c" * 64
    markers = {
        barrier_key: _marker(spool, barrier_key, barrier_tag, "acknowledged"),
        orphan_key: _marker(spool, orphan_key, "d" * 64, "acknowledged"),
    }
    monkeypatch.setattr(
        CaptureSpool,
        "_drop_state",
        lambda *_args: (1, "spool_incomplete", None, barrier_tag),
    )
    spool._persist_orphan_degradation_locked(
        {orphan_key},
        markers,
        None,
        remove=False,
    )


def test_marker_creation_reaches_the_descriptor_guard_after_an_absent_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spool = _spool(tmp_path)
    monkeypatch.setattr(CaptureSpool, "_read_session_marker", lambda *_args: None)
    with pytest.raises(spool_module.SecureFileError):
        spool._ensure_session_marker_locked("connection-one", "a" * 64, None)


def test_bridge_fallback_keeps_an_already_acknowledged_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spool = _spool(tmp_path)
    key = ("connection-one", "a" * 64)
    marker = _marker(spool, key, "b" * 64, "acknowledged")
    monkeypatch.setattr(CaptureSpool, "_drop_state", lambda *_args: (0, None, None, None))
    monkeypatch.setattr(CaptureSpool, "_read_session_marker", lambda *_args: marker)
    monkeypatch.setattr(CaptureSpool, "_write_drop_state", lambda *_args, **_kwargs: None)
    assert len(spool._drop_bridge_fallback_locked(key, 1, None)) == 1
