from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from tests.capture.store_support import (
    CONNECTION_ID,
    INSTALLATION_KEY,
    OTHER_CONNECTION_ID,
    OTHER_PROJECT_DIGEST,
    PROFILE_ID,
    PROJECT_DIGEST,
    authenticated_intake,
)

from saliencegate.capture.capabilities import (
    CaptureProfile,
    capture_capability_digest,
    capture_profile,
)
from saliencegate.capture.connections import (
    CaptureConnectionSummary,
    CaptureSessionSummary,
)
from saliencegate.capture.migrations import initialize_capture_store
from saliencegate.capture.store import (
    CaptureConnectionState,
    CaptureSessionState,
    CaptureStore,
    CaptureStoreIntegrityError,
    CaptureStoreMode,
    CaptureStoreStateError,
)


def _register(
    store: CaptureStore,
    connection_id: str,
    project_digest: str,
    *,
    profile_id: CaptureProfile = PROFILE_ID,
) -> tuple[str, str]:
    profile = capture_profile(profile_id)
    digest = capture_capability_digest(profile)
    store.register_connection(
        connection_id=connection_id,
        project_digest=project_digest,
        profile_id=profile_id,
        capability_manifest_digest=digest,
        host_version=profile.host_version,
    )
    store.transition_connection(
        connection_id,
        expected_state=CaptureConnectionState.PENDING,
        target_state=CaptureConnectionState.ENABLED,
    )
    return profile_id.value, digest


def _append_session(
    store: CaptureStore,
    *,
    connection_id: str,
    session_native: bytes,
    producer_index: int,
    profile_value: str,
    capability_digest: str,
    close: bool,
) -> None:
    changes = {
        "adapter_profile": profile_value,
        "capability_manifest_digest": capability_digest,
    }
    store.append(
        authenticated_intake(
            "session_started",
            connection_id=connection_id,
            session_native=session_native,
            producer_index=producer_index,
            changes=changes,
        )
    )
    if close:
        store.append(
            authenticated_intake(
                "session_finished",
                connection_id=connection_id,
                session_native=session_native,
                producer_index=producer_index + 1,
                changes=changes,
            )
        )


def _open(path: Path, *, mode: CaptureStoreMode) -> CaptureStore:
    return CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=mode,
    )


def test_maintenance_queries_authenticate_filter_and_order_connections_and_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "capture.sqlite3"
    initialize_capture_store(path)
    monkeypatch.setattr(
        "saliencegate.capture.store._now",
        lambda: "2026-07-20T12:00:00+00:00",
    )

    with _open(path, mode=CaptureStoreMode.MAINTENANCE) as store:
        codex_profile, codex_digest = _register(store, CONNECTION_ID, PROJECT_DIGEST)
        claude_id = "connection-claude"
        claude_profile, claude_digest = _register(
            store,
            claude_id,
            PROJECT_DIGEST,
            profile_id=CaptureProfile.CLAUDE_CODE_HOOKS_V1,
        )
        other_profile, other_digest = _register(
            store,
            OTHER_CONNECTION_ID,
            OTHER_PROJECT_DIGEST,
        )
        _append_session(
            store,
            connection_id=CONNECTION_ID,
            session_native=b"project-one-open",
            producer_index=1,
            profile_value=codex_profile,
            capability_digest=codex_digest,
            close=False,
        )
        _append_session(
            store,
            connection_id=claude_id,
            session_native=b"project-one-closed",
            producer_index=10,
            profile_value=claude_profile,
            capability_digest=claude_digest,
            close=True,
        )
        _append_session(
            store,
            connection_id=OTHER_CONNECTION_ID,
            session_native=b"project-two-open",
            producer_index=20,
            profile_value=other_profile,
            capability_digest=other_digest,
            close=False,
        )

        connections = store.list_connections()
        assert all(type(item) is CaptureConnectionSummary for item in connections)
        assert tuple(item.connection_id for item in connections) == tuple(
            sorted((CONNECTION_ID, claude_id, OTHER_CONNECTION_ID))
        )
        assert {item.project_digest for item in connections} == {
            PROJECT_DIGEST,
            OTHER_PROJECT_DIGEST,
        }
        assert tuple(
            item.connection_id
            for item in store.list_connections(
                project_digest=PROJECT_DIGEST,
                profile_id=CaptureProfile.CLAUDE_CODE_HOOKS_V1,
            )
        ) == (claude_id,)

        project_sessions = store.list_sessions(project_digest=PROJECT_DIGEST)
        assert len(project_sessions) == 2
        assert all(type(item) is CaptureSessionSummary for item in project_sessions)
        assert tuple(item.human_id for item in project_sessions) == tuple(
            sorted(item.human_id for item in project_sessions)
        )
        assert tuple(
            item.connection_id
            for item in store.list_sessions(
                project_digest=PROJECT_DIGEST,
                profile_id=CaptureProfile.CLAUDE_CODE_HOOKS_V1,
                state=CaptureSessionState.CLOSED,
                limit=1,
            )
        ) == (claude_id,)

        selected = store.session_by_human_id(project_sessions[0].human_id)
        assert selected == project_sessions[0]
        latest = store.latest_session(project_digest=PROJECT_DIGEST)
        assert latest == project_sessions[0]
        assert latest.project_digest == PROJECT_DIGEST
        assert latest != store.latest_session(project_digest=OTHER_PROJECT_DIGEST)

        assert repr(connections[0]) == "CaptureConnectionSummary(<redacted>)"
        assert repr(project_sessions[0]) == "CaptureSessionSummary(<redacted>)"
        for secret in (
            connections[0].connection_id,
            connections[0].project_digest,
            project_sessions[0].session_id,
            project_sessions[0].human_id,
        ):
            assert secret not in repr(connections[0]) + repr(project_sessions[0])

        with pytest.raises(ValidationError):
            CaptureSessionSummary.model_validate(
                {**project_sessions[0].model_dump(), "event_count": True}
            )


def test_query_methods_require_maintenance_mode_and_reject_invalid_filters(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capture.sqlite3"
    initialize_capture_store(path)
    with _open(path, mode=CaptureStoreMode.HOOK) as store:
        for operation in (
            lambda: store.list_connections(),
            lambda: store.list_sessions(),
            lambda: store.session_by_human_id("abcdefghijkl"),
            lambda: store.latest_session(project_digest=PROJECT_DIGEST),
        ):
            with pytest.raises(CaptureStoreStateError):
                operation()

    with _open(path, mode=CaptureStoreMode.MAINTENANCE) as store:
        invalid = (
            lambda: store.list_connections(project_digest="not-a-digest"),
            lambda: store.list_connections(profile_id="codex-hooks/v1"),
            lambda: store.list_sessions(limit=True),
            lambda: store.list_sessions(limit=0),
            lambda: store.list_sessions(state="open"),
            lambda: store.session_by_human_id("provider-native-secret"),
            lambda: store.latest_session(project_digest="not-a-digest"),
        )
        for operation in invalid:
            with pytest.raises(CaptureStoreStateError) as captured:
                operation()  # type: ignore[misc]
            assert "provider-native-secret" not in str(captured.value)


def test_authenticated_listing_detects_connection_and_session_row_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capture.sqlite3"
    initialize_capture_store(path)
    with _open(path, mode=CaptureStoreMode.MAINTENANCE) as store:
        profile_value, digest = _register(store, CONNECTION_ID, PROJECT_DIGEST)
        _append_session(
            store,
            connection_id=CONNECTION_ID,
            session_native=b"tamper-session",
            producer_index=1,
            profile_value=profile_value,
            capability_digest=digest,
            close=False,
        )
        store._connection.execute(
            "UPDATE capture_sessions SET event_count = 7 WHERE connection_id = ?",
            (CONNECTION_ID,),
        )
        with pytest.raises(CaptureStoreIntegrityError):
            store.list_sessions()
        store._connection.execute(
            "UPDATE connections SET state = 'disabled' WHERE connection_id = ?",
            (CONNECTION_ID,),
        )
        with pytest.raises(CaptureStoreIntegrityError):
            store.list_connections()


def test_empty_queries_are_deterministic_and_latest_requires_a_project_match(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capture.sqlite3"
    initialize_capture_store(path)
    with _open(path, mode=CaptureStoreMode.MAINTENANCE) as store:
        assert store.list_connections() == ()
        assert store.list_sessions() == ()
        with pytest.raises(CaptureStoreStateError):
            store.latest_session(project_digest=PROJECT_DIGEST)
        with pytest.raises(CaptureStoreStateError):
            store.session_by_human_id("abcdefghijkl")


def test_disabled_connection_can_reconnect_without_weakening_delete_transition(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capture.sqlite3"
    initialize_capture_store(path)
    with _open(path, mode=CaptureStoreMode.MAINTENANCE) as store:
        profile_value, digest = _register(store, CONNECTION_ID, PROJECT_DIGEST)
        store.transition_connection(
            CONNECTION_ID,
            expected_state=CaptureConnectionState.ENABLED,
            target_state=CaptureConnectionState.DRAINING,
        )
        store.transition_connection(
            CONNECTION_ID,
            expected_state=CaptureConnectionState.DRAINING,
            target_state=CaptureConnectionState.DISABLED,
        )

        reconnected = store.transition_connection(
            CONNECTION_ID,
            expected_state=CaptureConnectionState.DISABLED,
            target_state=CaptureConnectionState.ENABLED,
        )

        assert reconnected.previous_state is CaptureConnectionState.DISABLED
        assert reconnected.state is CaptureConnectionState.ENABLED
        assert store.list_connections()[0].state is CaptureConnectionState.ENABLED
        _append_session(
            store,
            connection_id=CONNECTION_ID,
            session_native=b"reconnected-session",
            producer_index=1,
            profile_value=profile_value,
            capability_digest=digest,
            close=False,
        )
        assert len(store.list_sessions(project_digest=PROJECT_DIGEST)) == 1


def test_root_exports_verified_query_contracts() -> None:
    import saliencegate.capture as capture

    assert capture.CaptureConnectionSummary is CaptureConnectionSummary
    assert capture.CaptureSessionSummary is CaptureSessionSummary
    assert {"CaptureConnectionSummary", "CaptureSessionSummary"} <= set(capture.__all__)
