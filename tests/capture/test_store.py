from __future__ import annotations

import hmac
import sqlite3
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from threading import Event, Thread

import pytest
from pydantic import ValidationError
from tests.capture.store_support import (
    CAPABILITY_MANIFEST_DIGEST,
    CONNECTION_ID,
    HOST_VERSION,
    INSTALLATION_KEY,
    INSTALLATION_KEY_MATERIAL,
    OTHER_CONNECTION_ID,
    OTHER_PROJECT_DIGEST,
    PROFILE_ID,
    PROJECT_DIGEST,
    WRONG_INSTALLATION_KEY,
    ZERO_TAG,
    all_authenticated_intakes,
    authenticated_intake,
    capture_context,
    initialized_store,
    register_connection,
    unauthenticated_intake,
)

import saliencegate.capture.store as store_module
from saliencegate.capture.capabilities import (
    CaptureProfile,
    capture_capability_digest,
    capture_profile,
)
from saliencegate.capture.health import CaptureHealthCode
from saliencegate.capture.migrations import initialize_capture_store
from saliencegate.capture.publication import (
    CaptureIntakeAuthenticationError,
    authenticate_capture_intake,
    verify_capture_intake_authentication,
)
from saliencegate.capture.schema import CaptureIntake, canonical_capture_intake
from saliencegate.capture.store import (
    CaptureAppendDisposition,
    CaptureAppendReceipt,
    CaptureConnectionRegistration,
    CaptureConnectionState,
    CaptureConnectionTransition,
    CaptureSessionState,
    CaptureSessionVerification,
    CaptureStore,
    CaptureStoreBusyError,
    CaptureStoreClosedError,
    CaptureStoreError,
    CaptureStoreIntegrityError,
    CaptureStoreMode,
    CaptureStoreStateError,
)
from saliencegate.domain import canonical_json

_AUTHENTICATION_DOMAIN = b"saliencegate:capture:integrity-tag:v1"
_KNOWN_INTAKE_TAG = "d16b16f7c8fec78a4203fa29e77df71ba3be5613e75152ad45e1904702e0f7be"


def _expected_hmac(material: bytes, *, domain: bytes, value: bytes) -> str:
    framed = (
        len(domain).to_bytes(8, byteorder="big", signed=False)
        + domain
        + len(value).to_bytes(8, byteorder="big", signed=False)
        + value
    )
    return hmac.new(material, framed, sha256).hexdigest()


def _authentication_preimage(intake: CaptureIntake) -> bytes:
    return canonical_json(
        {
            "schema_version": "capture-intake-integrity/v1",
            "intake": intake.model_dump(
                mode="json",
                exclude={"intake_tag"},
                warnings="error",
            ),
        }
    )


def _registration(
    store: CaptureStore,
    connection_id: str = CONNECTION_ID,
) -> CaptureConnectionRegistration:
    return store.register_connection(
        connection_id=connection_id,
        project_digest=(PROJECT_DIGEST if connection_id == CONNECTION_ID else OTHER_PROJECT_DIGEST),
        profile_id=PROFILE_ID,
        capability_manifest_digest=CAPABILITY_MANIFEST_DIGEST,
        host_version=HOST_VERSION,
    )


def _enable(
    store: CaptureStore,
    connection_id: str = CONNECTION_ID,
) -> CaptureConnectionTransition:
    return store.transition_connection(
        connection_id=connection_id,
        expected_state=CaptureConnectionState.PENDING,
        target_state=CaptureConnectionState.ENABLED,
    )


def _row_count(path: Path, table: str) -> int:
    assert table in {"connections", "capture_sessions", "capture_events", "capture_heads"}
    connection = sqlite3.connect(path)
    try:
        row = connection.execute(f"SELECT count(*) FROM {table}").fetchone()
        assert row is not None
        return int(row[0])
    finally:
        connection.close()


def test_intake_authentication_has_a_known_domain_separated_hmac_vector() -> None:
    context = capture_context()
    unsigned = unauthenticated_intake("session_started", context=context)

    expected = _expected_hmac(
        INSTALLATION_KEY_MATERIAL,
        domain=_AUTHENTICATION_DOMAIN,
        value=_authentication_preimage(unsigned),
    )
    authenticated = authenticate_capture_intake(unsigned, context=context)

    assert expected == _KNOWN_INTAKE_TAG
    assert authenticated.intake_tag == expected
    assert authenticated is not unsigned
    assert unsigned.intake_tag == ZERO_TAG
    assert canonical_capture_intake(
        authenticate_capture_intake(
            unsigned.model_copy(update={"intake_tag": "f" * 64}),
            context=context,
        )
    ) == canonical_capture_intake(authenticated)
    verified = verify_capture_intake_authentication(authenticated, context=context)
    assert verified == authenticated
    assert verified is not authenticated


def test_all_nine_intake_kinds_authenticate_and_tampering_or_wrong_key_fails_closed() -> None:
    intakes = all_authenticated_intakes()

    assert tuple(intake.kind for intake in intakes) == (
        "session_started",
        "action_started",
        "action_finished",
        "permission_denied",
        "subagent_started",
        "subagent_finished",
        "turn_finished",
        "controller_failed",
        "session_finished",
    )
    assert all(
        verify_capture_intake_authentication(intake, context=capture_context()) == intake
        for intake in intakes
    )

    marker = "provider-native-secret-marker"
    tampered = intakes[0].model_copy(
        update={"producer_event_digest": "f" * 64, "intake_tag": marker}
    )
    for value, context in (
        (tampered, capture_context()),
        (intakes[0], capture_context(b"wrong-store-test-key-material!!!")),
    ):
        with pytest.raises(CaptureIntakeAuthenticationError) as raised:
            verify_capture_intake_authentication(value, context=context)
        assert str(raised.value) == "capture intake authentication failed"
        assert marker not in str(raised.value)
        assert marker not in repr(raised.value)
        assert raised.value.__cause__ is None


def test_store_enums_are_closed_and_stable() -> None:
    assert tuple(CaptureStoreMode) == (
        CaptureStoreMode.HOOK,
        CaptureStoreMode.MAINTENANCE,
    )
    assert tuple(item.value for item in CaptureStoreMode) == ("hook", "maintenance")
    assert tuple(item.value for item in CaptureConnectionState) == (
        "pending",
        "enabled",
        "draining",
        "disabled",
        "deleting",
    )
    assert tuple(item.value for item in CaptureSessionState) == (
        "open",
        "closed",
        "quarantined",
        "deleting",
    )
    assert tuple(item.value for item in CaptureAppendDisposition) == (
        "admitted",
        "replayed",
        "quarantined",
        "overflow",
    )


def test_store_public_guards_fail_closed_without_partial_state(tmp_path: Path) -> None:
    path = tmp_path / "public-guards.sqlite3"
    initialize_capture_store(path)

    invalid_open_calls = (
        lambda: CaptureStore(),
        lambda: CaptureStore.open(  # type: ignore[arg-type]
            path,
            installation_key=b"k" * 32,
        ),
        lambda: CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            busy_timeout_ms=0,
        ),
        lambda: CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            busy_timeout_ms=True,
        ),
        lambda: CaptureStore.open(  # type: ignore[arg-type]
            path,
            installation_key=INSTALLATION_KEY,
            mode="hook",
        ),
        lambda: CaptureStore.open(  # type: ignore[arg-type]
            b"provider-native-path",
            installation_key=INSTALLATION_KEY,
        ),
        lambda: CaptureStore.open("", installation_key=INSTALLATION_KEY),
    )
    for invalid_open in invalid_open_calls:
        with pytest.raises(CaptureStoreError):
            invalid_open()

    with (
        CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            busy_timeout_ms=250,
            mode=CaptureStoreMode.HOOK,
        ) as store,
        pytest.raises(CaptureStoreStateError),
    ):
        _registration(store)
    assert _row_count(path, "connections") == 0

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        with pytest.raises(CaptureStoreStateError):
            store.register_connection(
                connection_id=CONNECTION_ID,
                project_digest=PROJECT_DIGEST,
                profile_id=PROFILE_ID,
                capability_manifest_digest=CAPABILITY_MANIFEST_DIGEST,
                host_version="01.2",
            )
        assert _row_count(path, "connections") == 0

        register_connection(store)
        start = authenticated_intake("session_started", producer_index=1)
        with pytest.raises(CaptureStoreError):
            store.append(start, source="direct")  # type: ignore[arg-type]
        assert _row_count(path, "capture_sessions") == 0

        with pytest.raises(CaptureStoreError):
            store.verify_session(1, start.session_id)  # type: ignore[arg-type]
        with pytest.raises(CaptureStoreError):
            store.verify_session(CONNECTION_ID, None)  # type: ignore[arg-type]
        with pytest.raises(CaptureStoreStateError):
            store.verify_session("missing-connection", start.session_id)
        with pytest.raises(CaptureStoreStateError):
            store.verify_session(CONNECTION_ID, "f" * 64)

    with (
        CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            busy_timeout_ms=250,
            mode=CaptureStoreMode.HOOK,
        ) as hook,
        pytest.raises(CaptureStoreStateError),
    ):
        hook.transition_connection(
            CONNECTION_ID,
            expected_state=CaptureConnectionState.ENABLED,
            target_state=CaptureConnectionState.DRAINING,
        )

    assert _row_count(path, "capture_sessions") == 0


def test_hook_open_defers_the_full_audit_but_maintenance_open_performs_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "mode-audit.sqlite3"
    initialize_capture_store(path)
    calls: list[bool] = []
    original = CaptureStore._verify_all_state

    def recording_verify_all_state(
        self: CaptureStore,
        *,
        immutable: bool = False,
    ) -> None:
        calls.append(immutable)
        original(self, immutable=immutable)

    monkeypatch.setattr(CaptureStore, "_verify_all_state", recording_verify_all_state)
    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.HOOK,
    ):
        pass
    assert calls == []

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.MAINTENANCE,
    ):
        pass
    assert calls == [False]


def test_hook_connection_lookup_authenticates_only_the_selected_row(tmp_path: Path) -> None:
    path = tmp_path / "hook-connection.sqlite3"
    initialize_capture_store(path)
    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as maintenance:
        register_connection(maintenance)
        register_connection(
            maintenance,
            connection_id=OTHER_CONNECTION_ID,
            project_digest=OTHER_PROJECT_DIGEST,
        )

    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE connections SET row_tag = ? WHERE connection_id = ?",
        ("0" * 64, OTHER_CONNECTION_ID),
    )
    connection.commit()
    connection.close()

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.HOOK,
    ) as hook:
        narrow = hook._get_hook_connection(CONNECTION_ID)
        selected = hook.get_connection(CONNECTION_ID)
        assert type(narrow) is store_module._CaptureHookConnection
        assert (
            narrow.connection_id,
            narrow.project_digest,
            narrow.profile_id,
            narrow.capability_manifest_digest,
            narrow.host_version,
            narrow.state,
        ) == (
            selected.connection_id,
            selected.project_digest,
            selected.profile_id,
            selected.capability_manifest_digest,
            selected.host_version,
            selected.state,
        )
        assert repr(narrow) == "_CaptureHookConnection(<redacted>)"
        assert selected.connection_id == CONNECTION_ID
        assert selected.state is CaptureConnectionState.ENABLED
        with pytest.raises(CaptureStoreIntegrityError):
            hook._get_hook_connection(OTHER_CONNECTION_ID)
        with pytest.raises(CaptureStoreIntegrityError):
            hook.get_connection(OTHER_CONNECTION_ID)


def test_cached_hook_append_verifies_only_the_authenticated_session_tip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "append-tip.sqlite3"
    initialize_capture_store(path)
    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as maintenance:
        register_connection(maintenance)
    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.HOOK,
    ) as store:
        first = authenticated_intake("session_started", producer_index=1)
        assert store.append(first).receipt_ordinal == 1

        def reject_full_chain(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("the hook append path performed a full-chain audit")

        monkeypatch.setattr(CaptureStore, "_verify_chain", reject_full_chain)
        second = authenticated_intake("turn_finished", producer_index=2)
        receipt = store.append(second)

    assert receipt.receipt_ordinal == 2
    assert receipt.previous_event_tag is not None


def test_fresh_hook_append_authenticates_the_full_long_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "bounded-cold-hook-append.sqlite3"
    with initialized_store(path) as maintenance:
        register_connection(maintenance)
        for producer_index in range(1, 129):
            maintenance.append(
                authenticated_intake(
                    "session_started" if producer_index == 1 else "turn_finished",
                    producer_index=producer_index,
                )
            )

    verified_ordinals: list[int] = []
    original = CaptureStore._load_verified_append_event

    def counted(self: CaptureStore, row: sqlite3.Row):
        verified_ordinals.append(row["receipt_ordinal"])
        return original(self, row)

    monkeypatch.setattr(CaptureStore, "_load_verified_append_event", counted)
    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.HOOK,
    ) as hook:
        receipt = hook.append(authenticated_intake("turn_finished", producer_index=129))

    assert receipt.receipt_ordinal == 129
    assert verified_ordinals == list(range(1, 129))


def test_append_rejects_an_intake_bound_to_a_different_profile_without_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profile-binding.sqlite3"
    other_profile = CaptureProfile.CLAUDE_CODE_HOOKS_V1
    intake = authenticated_intake(
        "session_started",
        changes={
            "adapter_profile": other_profile.value,
            "capability_manifest_digest": capture_capability_digest(capture_profile(other_profile)),
        },
    )

    with initialized_store(path) as store:
        register_connection(store)

        with pytest.raises(CaptureStoreStateError):
            store.append(intake)

        for table in ("capture_sessions", "capture_events", "capture_heads"):
            assert _row_count(path, table) == 0


def test_connection_registration_is_idempotent_and_transitions_are_linear(
    tmp_path: Path,
) -> None:
    path = tmp_path / "connection-state.sqlite3"
    with initialized_store(path) as store:
        first = _registration(store)
        repeated = _registration(store)

        assert type(first) is CaptureConnectionRegistration
        assert repeated == first
        assert first.connection_id == CONNECTION_ID
        assert first.project_digest == PROJECT_DIGEST
        assert first.profile_id is PROFILE_ID
        assert first.capability_manifest_digest == CAPABILITY_MANIFEST_DIGEST
        assert first.host_version == HOST_VERSION
        assert first.state is CaptureConnectionState.PENDING

        enabled = _enable(store)
        assert type(enabled) is CaptureConnectionTransition
        assert enabled.previous_state is CaptureConnectionState.PENDING
        assert enabled.state is CaptureConnectionState.ENABLED

        transitions = (
            (CaptureConnectionState.ENABLED, CaptureConnectionState.DRAINING),
            (CaptureConnectionState.DRAINING, CaptureConnectionState.DISABLED),
            (CaptureConnectionState.DISABLED, CaptureConnectionState.DELETING),
        )
        for previous, target in transitions:
            receipt = store.transition_connection(
                connection_id=CONNECTION_ID,
                expected_state=previous,
                target_state=target,
            )
            assert receipt.previous_state is previous
            assert receipt.state is target

        with pytest.raises(CaptureStoreStateError):
            store.transition_connection(
                connection_id=CONNECTION_ID,
                expected_state=CaptureConnectionState.PENDING,
                target_state=CaptureConnectionState.ENABLED,
            )

    assert _row_count(path, "connections") == 1


def test_registration_collision_and_illegal_transition_do_not_mutate_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "connection-conflict.sqlite3"
    with initialized_store(path) as store:
        original = _registration(store)

        with pytest.raises(CaptureStoreStateError):
            store.register_connection(
                connection_id=CONNECTION_ID,
                project_digest=OTHER_PROJECT_DIGEST,
                profile_id=PROFILE_ID,
                capability_manifest_digest=CAPABILITY_MANIFEST_DIGEST,
                host_version=HOST_VERSION,
            )
        with pytest.raises(CaptureStoreStateError):
            store.transition_connection(
                connection_id=CONNECTION_ID,
                expected_state=CaptureConnectionState.PENDING,
                target_state=CaptureConnectionState.DISABLED,
            )

        assert _registration(store) == original
        assert _row_count(path, "connections") == 1


def test_all_event_kinds_append_with_contiguous_authenticated_chain_and_close_session(
    tmp_path: Path,
) -> None:
    path = tmp_path / "all-kinds.sqlite3"
    with initialized_store(path) as store:
        register_connection(store)
        receipts = tuple(store.append(intake) for intake in all_authenticated_intakes())

        assert all(type(receipt) is CaptureAppendReceipt for receipt in receipts)
        assert (
            tuple(receipt.disposition for receipt in receipts)
            == (CaptureAppendDisposition.ADMITTED,) * 9
        )
        assert tuple(receipt.receipt_ordinal for receipt in receipts) == tuple(range(1, 10))
        assert receipts[0].previous_event_tag is None
        assert all(
            current.previous_event_tag == previous.event_tag
            for previous, current in pairwise(receipts)
        )
        assert len({receipt.event_tag for receipt in receipts}) == 9
        assert receipts[-1].session_state is CaptureSessionState.CLOSED
        assert receipts[-1].event_count == 9

        session_id = receipts[0].session_id
        verified = store.verify_session(CONNECTION_ID, session_id)
        assert type(verified) is CaptureSessionVerification
        assert verified.connection_id == CONNECTION_ID
        assert verified.session_id == session_id
        assert verified.state is CaptureSessionState.CLOSED
        assert verified.event_count == 9
        assert verified.last_receipt_ordinal == 9
        assert verified.head_event_tag == receipts[-1].event_tag
        assert len(verified.head_tag) == 64

        with pytest.raises(CaptureStoreStateError):
            store.append(authenticated_intake("turn_finished", producer_index=10))

    assert _row_count(path, "capture_events") == 9
    assert _row_count(path, "capture_sessions") == 1
    assert _row_count(path, "capture_heads") == 1


def test_identical_replay_is_a_noop_and_digest_collision_quarantines_atomically(
    tmp_path: Path,
) -> None:
    path = tmp_path / "replay-collision.sqlite3"
    with initialized_store(path) as store:
        register_connection(store)
        original_intake = authenticated_intake("session_started")
        admitted = store.append(original_intake)
        before = store.verify_session(CONNECTION_ID, original_intake.session_id)

        replayed = store.append(original_intake)
        after_replay = store.verify_session(CONNECTION_ID, original_intake.session_id)
        assert replayed.disposition is CaptureAppendDisposition.REPLAYED
        assert replayed.receipt_ordinal == admitted.receipt_ordinal
        assert replayed.previous_event_tag == admitted.previous_event_tag
        assert replayed.event_tag == admitted.event_tag
        assert after_replay == before

        collision = authenticated_intake(
            "turn_finished",
            producer_index=2,
            changes={"producer_event_digest": original_intake.producer_event_digest},
        )
        quarantined = store.append(collision)
        verified = store.verify_session(CONNECTION_ID, original_intake.session_id)

        assert quarantined.disposition is CaptureAppendDisposition.QUARANTINED
        assert quarantined.receipt_ordinal is None
        assert quarantined.previous_event_tag is None
        assert quarantined.event_tag is None
        assert quarantined.session_state is CaptureSessionState.QUARANTINED
        assert quarantined.event_count == 1
        assert verified.state is CaptureSessionState.QUARANTINED
        assert verified.event_count == 1
        assert verified.head_event_tag == admitted.event_tag
        assert _row_count(path, "capture_events") == 1


def test_non_enabled_direct_collision_cannot_mutate_while_replay_remains_a_noop(
    tmp_path: Path,
) -> None:
    path = tmp_path / "disabled-collision.sqlite3"
    with initialized_store(path) as store:
        register_connection(store)
        original = authenticated_intake("session_started", producer_index=1)
        store.append(original)
        store.transition_connection(
            CONNECTION_ID,
            expected_state=CaptureConnectionState.ENABLED,
            target_state=CaptureConnectionState.DRAINING,
        )
        before = store.verify_session(CONNECTION_ID, original.session_id)

        replay = store.append(original)
        collision = authenticated_intake(
            "turn_finished",
            producer_index=2,
            changes={"producer_event_digest": original.producer_event_digest},
        )
        with pytest.raises(CaptureStoreStateError):
            store.append(collision)

        assert replay.disposition is CaptureAppendDisposition.REPLAYED
        assert store.verify_session(CONNECTION_ID, original.session_id) == before
        assert _row_count(path, "capture_events") == 1
        connection = sqlite3.connect(path)
        try:
            assert connection.execute("SELECT count(*) FROM capture_health").fetchone() == (0,)
        finally:
            connection.close()


def test_cross_session_collision_durably_quarantines_the_unseen_incoming_session(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cross-session-collision.sqlite3"
    with initialized_store(path) as store:
        register_connection(store)
        original = authenticated_intake("session_started", producer_index=1)
        store.append(original)
        collision = authenticated_intake(
            "turn_finished",
            session_native=b"colliding-session-two",
            producer_index=2,
            changes={"producer_event_digest": original.producer_event_digest},
        )

        receipt = store.append(collision)
        original_session = store.verify_session(CONNECTION_ID, original.session_id)
        incoming_session = store.verify_session(CONNECTION_ID, collision.session_id)

        assert receipt.disposition is CaptureAppendDisposition.QUARANTINED
        assert receipt.event_count == 0
        assert original_session.state is CaptureSessionState.QUARANTINED
        assert original_session.event_count == 1
        assert incoming_session.state is CaptureSessionState.QUARANTINED
        assert incoming_session.event_count == 0
        assert incoming_session.head_event_tag is None
        later_start = authenticated_intake(
            "session_started",
            session_native=b"colliding-session-two",
            producer_index=3,
        )
        with pytest.raises(CaptureStoreStateError):
            store.append(later_start)

    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            "SELECT session_id, code, count FROM capture_health ORDER BY session_id"
        ).fetchall()
    finally:
        connection.close()
    assert rows == sorted(
        (
            (original.session_id, CaptureHealthCode.PRODUCER_COLLISION.value, 1),
            (collision.session_id, CaptureHealthCode.PRODUCER_COLLISION.value, 1),
        )
    )


def test_cross_session_collision_verifies_and_quarantines_an_existing_incoming_session(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cross-session-existing.sqlite3"
    with initialized_store(path) as store:
        register_connection(store)
        original = authenticated_intake(
            "session_started",
            session_native=b"collision-existing-one",
            producer_index=1,
        )
        incoming_start = authenticated_intake(
            "session_started",
            session_native=b"collision-existing-two",
            producer_index=2,
        )
        original_receipt = store.append(original)
        incoming_receipt = store.append(incoming_start)
        collision = authenticated_intake(
            "turn_finished",
            session_native=b"collision-existing-two",
            producer_index=3,
            changes={"producer_event_digest": original.producer_event_digest},
        )

        receipt = store.append(collision)
        original_session = store.verify_session(CONNECTION_ID, original.session_id)
        incoming_session = store.verify_session(CONNECTION_ID, incoming_start.session_id)

        assert receipt.disposition is CaptureAppendDisposition.QUARANTINED
        assert receipt.session_id == incoming_start.session_id
        assert receipt.session_state is CaptureSessionState.QUARANTINED
        assert receipt.event_count == 1
        assert original_session.state is CaptureSessionState.QUARANTINED
        assert original_session.event_count == 1
        assert original_session.head_event_tag == original_receipt.event_tag
        assert incoming_session.state is CaptureSessionState.QUARANTINED
        assert incoming_session.event_count == 1
        assert incoming_session.head_event_tag == incoming_receipt.event_tag

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT count(*) FROM capture_events").fetchone() == (2,)
        health = connection.execute(
            "SELECT session_id, code, count FROM capture_health ORDER BY session_id"
        ).fetchall()
    finally:
        connection.close()
    assert health == sorted(
        (
            (original.session_id, CaptureHealthCode.PRODUCER_COLLISION.value, 1),
            (incoming_start.session_id, CaptureHealthCode.PRODUCER_COLLISION.value, 1),
        )
    )


def test_health_counters_are_idempotently_identified_authenticated_and_never_repaired(
    tmp_path: Path,
) -> None:
    path = tmp_path / "health-integrity.sqlite3"
    with initialized_store(path) as store:
        register_connection(store)
        original = authenticated_intake("session_started")
        store.append(original)
        collision = authenticated_intake(
            "turn_finished",
            producer_index=2,
            changes={"producer_event_digest": original.producer_event_digest},
        )

        assert store.append(collision).disposition is CaptureAppendDisposition.QUARANTINED
        assert store.append(collision).disposition is CaptureAppendDisposition.QUARANTINED

    connection = sqlite3.connect(path)
    try:
        row = connection.execute(
            "SELECT code, count, lower_bound, row_tag FROM capture_health"
        ).fetchone()
        assert row is not None
        assert row[:3] == (CaptureHealthCode.PRODUCER_COLLISION.value, 2, 0)
        assert type(row[3]) is str and len(row[3]) == 64
        connection.execute("UPDATE capture_health SET count = 3")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(CaptureStoreIntegrityError):
        CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            busy_timeout_ms=250,
            mode=CaptureStoreMode.MAINTENANCE,
        )
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT count FROM capture_health").fetchone() == (3,)
    finally:
        connection.close()


def test_valid_health_reopens_and_live_tampering_rolls_back_repeated_collision(
    tmp_path: Path,
) -> None:
    path = tmp_path / "live-health-integrity.sqlite3"
    original = authenticated_intake("session_started", producer_index=1)
    collision = authenticated_intake(
        "turn_finished",
        producer_index=2,
        changes={"producer_event_digest": original.producer_event_digest},
    )
    with initialized_store(path) as store:
        register_connection(store)
        store.append(original)
        assert store.append(collision).disposition is CaptureAppendDisposition.QUARANTINED

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        assert store.verify_session(CONNECTION_ID, original.session_id).event_count == 1
        connection = sqlite3.connect(path)
        try:
            result = connection.execute("UPDATE capture_health SET count = 2")
            assert result.rowcount == 1
            connection.commit()
        finally:
            connection.close()
        connection = sqlite3.connect(path)
        try:
            tampered_sessions = tuple(
                connection.execute(
                    "SELECT * FROM capture_sessions ORDER BY connection_id, session_id"
                ).fetchall()
            )
            tampered_health = tuple(
                connection.execute("SELECT * FROM capture_health ORDER BY marker_id").fetchall()
            )
        finally:
            connection.close()

        with pytest.raises(CaptureStoreIntegrityError):
            store.append(collision)

        connection = sqlite3.connect(path)
        try:
            assert (
                tuple(
                    connection.execute(
                        "SELECT * FROM capture_sessions ORDER BY connection_id, session_id"
                    ).fetchall()
                )
                == tampered_sessions
            )
            assert (
                tuple(
                    connection.execute("SELECT * FROM capture_health ORDER BY marker_id").fetchall()
                )
                == tampered_health
            )
        finally:
            connection.close()

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT count FROM capture_health").fetchone() == (2,)
    finally:
        connection.close()


def test_repeated_health_updates_keep_set_count_stable_and_change_authenticated_digest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "repeated-health-set.sqlite3"
    original = authenticated_intake("session_started", producer_index=1)
    collision = authenticated_intake(
        "turn_finished",
        producer_index=2,
        changes={"producer_event_digest": original.producer_event_digest},
    )
    with initialized_store(path) as store:
        register_connection(store)
        store.append(original)
        assert store.append(collision).disposition is CaptureAppendDisposition.QUARANTINED

        connection = sqlite3.connect(path)
        try:
            first_session = connection.execute(
                """
                SELECT health_marker_count, health_set_digest, row_tag
                FROM capture_sessions
                WHERE connection_id = ? AND session_id = ?
                """,
                (CONNECTION_ID, original.session_id),
            ).fetchone()
            first_health = connection.execute(
                """
                SELECT marker_id, count, row_tag FROM capture_health
                WHERE connection_id = ? AND session_id = ?
                """,
                (CONNECTION_ID, original.session_id),
            ).fetchone()
        finally:
            connection.close()
        assert first_session is not None
        assert first_session[0] == 1
        assert type(first_session[1]) is str and len(first_session[1]) == 64
        assert first_health is not None
        assert first_health[1] == 1

        assert store.append(collision).disposition is CaptureAppendDisposition.QUARANTINED
        connection = sqlite3.connect(path)
        try:
            second_session = connection.execute(
                """
                SELECT health_marker_count, health_set_digest, row_tag
                FROM capture_sessions
                WHERE connection_id = ? AND session_id = ?
                """,
                (CONNECTION_ID, original.session_id),
            ).fetchone()
            second_health = connection.execute(
                """
                SELECT marker_id, count, row_tag FROM capture_health
                WHERE connection_id = ? AND session_id = ?
                """,
                (CONNECTION_ID, original.session_id),
            ).fetchone()
        finally:
            connection.close()
        assert second_session is not None
        assert second_session[0] == first_session[0] == 1
        assert second_session[1] != first_session[1]
        assert second_session[2] != first_session[2]
        assert second_health is not None
        assert second_health[0] == first_health[0]
        assert second_health[1] == 2
        assert second_health[2] != first_health[2]

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as reopened:
        assert reopened.verify_session(CONNECTION_ID, original.session_id).event_count == 1


def test_session_and_producer_identities_are_scoped_by_connection(tmp_path: Path) -> None:
    path = tmp_path / "connection-scope.sqlite3"
    with initialized_store(path) as store:
        for connection_id in (CONNECTION_ID, OTHER_CONNECTION_ID):
            _registration(store, connection_id)
            _enable(store, connection_id)

        first = authenticated_intake("session_started", connection_id=CONNECTION_ID)
        second = authenticated_intake("session_started", connection_id=OTHER_CONNECTION_ID)
        assert first.session_id == second.session_id
        assert first.producer_event_digest == second.producer_event_digest
        assert first.intake_tag != second.intake_tag

        first_receipt = store.append(first)
        second_receipt = store.append(second)
        assert first_receipt.disposition is CaptureAppendDisposition.ADMITTED
        assert second_receipt.disposition is CaptureAppendDisposition.ADMITTED
        assert first_receipt.receipt_ordinal == second_receipt.receipt_ordinal == 1
        assert first_receipt.event_tag != second_receipt.event_tag
        assert store.verify_session(CONNECTION_ID, first.session_id).event_count == 1
        assert store.verify_session(OTHER_CONNECTION_ID, second.session_id).event_count == 1

    assert _row_count(path, "capture_sessions") == 2
    assert _row_count(path, "capture_events") == 2


def test_human_session_id_extends_only_when_the_short_prefix_collides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "human-id-prefix.sqlite3"
    first = authenticated_intake(
        "session_started",
        session_native=b"human-prefix-one",
        producer_index=1,
    )
    second = authenticated_intake(
        "session_started",
        session_native=b"human-prefix-two",
        producer_index=2,
    )
    forced_human_digests = {
        first.session_id: "0" * 64,
        second.session_id: "0" * 15 + "1" + "0" * 48,
    }
    real_tag = store_module._CaptureStoreIntegrity.tag

    def controlled_human_tag(
        integrity: store_module._CaptureStoreIntegrity,
        purpose: str,
        value: object,
    ) -> str:
        if purpose == "human_id" and isinstance(value, dict):
            session_id = value.get("session_id")
            if isinstance(session_id, str) and session_id in forced_human_digests:
                return forced_human_digests[session_id]
        return real_tag(integrity, purpose, value)

    with initialized_store(path) as store:
        register_connection(store)
        monkeypatch.setattr(
            store_module._CaptureStoreIntegrity,
            "tag",
            controlled_human_tag,
        )
        store.append(first)
        store.append(second)

        connection = sqlite3.connect(path)
        try:
            rows = dict(
                connection.execute("SELECT session_id, human_id FROM capture_sessions").fetchall()
            )
        finally:
            connection.close()

    assert len(rows[first.session_id]) == 12
    assert len(rows[second.session_id]) == 13
    assert rows[second.session_id].startswith(rows[first.session_id])
    assert len(set(rows.values())) == 2


def test_pending_draining_closed_and_missing_sessions_fail_without_partial_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid-state.sqlite3"
    initialize_capture_store(path)
    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        _registration(store)
        start = authenticated_intake("session_started")
        with pytest.raises(CaptureStoreStateError):
            store.append(start)
        assert _row_count(path, "capture_sessions") == 0

        _enable(store)
        with pytest.raises(CaptureStoreStateError):
            store.append(authenticated_intake("turn_finished", producer_index=2))
        assert _row_count(path, "capture_sessions") == 0

        store.append(start)
        with pytest.raises(CaptureStoreStateError):
            store.append(authenticated_intake("session_started", producer_index=2))
        assert store.verify_session(CONNECTION_ID, start.session_id).event_count == 1
        store.transition_connection(
            connection_id=CONNECTION_ID,
            expected_state=CaptureConnectionState.ENABLED,
            target_state=CaptureConnectionState.DRAINING,
        )
        with pytest.raises(CaptureStoreStateError):
            store.append(authenticated_intake("turn_finished", producer_index=3))
        assert store.verify_session(CONNECTION_ID, start.session_id).event_count == 1


def test_wrong_authentication_key_and_database_tampering_fail_without_repair(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tamper.sqlite3"
    initialize_capture_store(path)
    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        register_connection(store)
        unsigned = unauthenticated_intake("session_started")
        wrong_key = authenticated_intake(
            "session_started",
            context=capture_context(b"wrong-store-test-key-material!!!"),
        )
        for invalid in (unsigned, wrong_key):
            with pytest.raises(CaptureStoreIntegrityError):
                store.append(invalid)
        assert _row_count(path, "capture_sessions") == 0

        admitted = store.append(authenticated_intake("session_started"))
        assert admitted.receipt_ordinal == 1

    connection = sqlite3.connect(path)
    try:
        row_before_wrong_key = connection.execute(
            "SELECT event_json, event_tag FROM capture_events"
        ).fetchone()
    finally:
        connection.close()
    assert row_before_wrong_key is not None
    with pytest.raises(CaptureStoreIntegrityError):
        CaptureStore.open(
            path,
            installation_key=WRONG_INSTALLATION_KEY,
            busy_timeout_ms=250,
            mode=CaptureStoreMode.MAINTENANCE,
        )
    connection = sqlite3.connect(path)
    try:
        assert (
            connection.execute("SELECT event_json, event_tag FROM capture_events").fetchone()
            == row_before_wrong_key
        )
        event_json = row_before_wrong_key[0]
        assert type(event_json) is bytes
        tampered_json = event_json.replace(b'"captured"', b'"degraded"')
        assert len(tampered_json) == len(event_json)
        assert tampered_json != event_json
        connection.execute(
            "UPDATE capture_events SET event_json = ? WHERE receipt_ordinal = 1",
            (tampered_json,),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(CaptureStoreIntegrityError):
        CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            busy_timeout_ms=250,
            mode=CaptureStoreMode.MAINTENANCE,
        )
    connection = sqlite3.connect(path)
    try:
        assert connection.execute(
            "SELECT event_json FROM capture_events WHERE receipt_ordinal = 1"
        ).fetchone() == (tampered_json,)
    finally:
        connection.close()


def test_the_1001st_session_event_is_not_stored_and_quarantines_coverage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "overflow.sqlite3"
    with initialized_store(path) as store:
        register_connection(store)
        start = authenticated_intake("session_started", producer_index=1)
        first = store.append(start)
        assert first.disposition is CaptureAppendDisposition.ADMITTED

        latest = first
        for producer_index in range(2, 1_001):
            latest = store.append(
                authenticated_intake("turn_finished", producer_index=producer_index)
            )
        assert latest.receipt_ordinal == 1_000
        assert latest.event_count == 1_000

        overflow = store.append(authenticated_intake("turn_finished", producer_index=1_001))
        verified = store.verify_session(CONNECTION_ID, start.session_id)
        assert overflow.disposition is CaptureAppendDisposition.OVERFLOW
        assert overflow.receipt_ordinal is None
        assert overflow.previous_event_tag is None
        assert overflow.event_tag is None
        assert overflow.session_state is CaptureSessionState.QUARANTINED
        assert overflow.event_count == 1_000
        assert verified.state is CaptureSessionState.QUARANTINED
        assert verified.event_count == 1_000
        assert verified.last_receipt_ordinal == 1_000
        assert verified.head_event_tag == latest.event_tag
        assert _row_count(path, "capture_events") == 1_000


def test_store_contract_models_errors_and_representations_are_content_free(
    tmp_path: Path,
) -> None:
    path = tmp_path / "redacted-secret-path.sqlite3"
    with initialized_store(path) as store:
        registration = _registration(store)
        transition = _enable(store)
        intake = authenticated_intake("session_started")
        receipt = store.append(intake)
        verification = store.verify_session(CONNECTION_ID, intake.session_id)

        assert "redacted-secret-path" not in repr(store)
        assert INSTALLATION_KEY_MATERIAL.decode() not in repr(store)
        for model in (registration, transition, receipt, verification):
            assert "<redacted>" in repr(model)
            for secret in (
                CONNECTION_ID,
                PROJECT_DIGEST,
                intake.session_id,
                intake.producer_event_digest,
                intake.intake_tag,
            ):
                assert secret not in repr(model)
            with pytest.raises(ValidationError):
                model.state = CaptureSessionState.DELETING  # type: ignore[misc,union-attr]

        invalid = receipt.model_dump(mode="python")
        invalid["event_count"] = "1"
        with pytest.raises(ValidationError):
            CaptureAppendReceipt.model_validate(invalid)

    store.close()
    with pytest.raises(CaptureStoreClosedError):
        store.verify_session(CONNECTION_ID, verification.session_id)

    expected_errors = {
        CaptureStoreError(): "capture store operation failed",
        CaptureStoreBusyError(): "capture store is busy",
        CaptureStoreClosedError(): "capture store is closed",
        CaptureStoreIntegrityError(): "capture store integrity failed",
        CaptureStoreStateError(): "capture store state transition failed",
    }
    for error, message in expected_errors.items():
        assert str(error) == message
        assert not vars(error)
        assert "redacted-secret-path" not in repr(error)


def test_concurrent_close_and_verify_session_never_exposes_raw_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "close-verify-race.sqlite3"
    initialize_capture_store(path)
    intake = authenticated_intake("session_started")
    store = CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=CaptureStoreMode.MAINTENANCE,
    )
    register_connection(store)
    store.append(intake)
    precheck_complete = Event()
    allow_verification = Event()
    original_ensure_open = CaptureStore._ensure_open
    gate_used = False
    outcomes: list[CaptureSessionVerification | BaseException] = []

    def gated_ensure_open(candidate: CaptureStore) -> None:
        nonlocal gate_used
        original_ensure_open(candidate)
        if candidate is store and not gate_used:
            gate_used = True
            precheck_complete.set()
            if not allow_verification.wait(timeout=5.0):
                raise AssertionError("verification gate timed out")

    def verify() -> None:
        try:
            outcomes.append(store.verify_session(CONNECTION_ID, intake.session_id))
        except BaseException as error:
            outcomes.append(error)

    monkeypatch.setattr(CaptureStore, "_ensure_open", gated_ensure_open)
    worker = Thread(target=verify, name="capture-store-verifier")
    try:
        worker.start()
        assert precheck_complete.wait(timeout=5.0)
        store.close()
        allow_verification.set()
        worker.join(timeout=5.0)
    finally:
        allow_verification.set()
        worker.join(timeout=5.0)
        store.close()

    assert worker.is_alive() is False
    assert len(outcomes) == 1
    outcome = outcomes[0]
    if isinstance(outcome, CaptureSessionVerification):
        assert outcome.event_count == 1
    else:
        assert type(outcome) is CaptureStoreClosedError
        assert not isinstance(outcome, sqlite3.Error)
        assert str(outcome) == "capture store is closed"
        assert outcome.__cause__ is None
        assert outcome.__context__ is None
