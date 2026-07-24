from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError
from tests.capture.store_support import (
    CAPABILITY_MANIFEST_DIGEST,
    HOST_VERSION,
    INSTALLATION_KEY,
    PROFILE_ID,
    register_connection,
)

import saliencegate.capture.store as store_module
from saliencegate.capture.capabilities import CaptureProfile
from saliencegate.capture.migrations import initialize_capture_store
from saliencegate.capture.scopes import (
    MAX_GLOBAL_EXCLUSIONS_PER_PARENT,
    MAX_GLOBAL_HEALTH_COUNT,
    CaptureConnectionScope,
    CaptureGlobalHealthCode,
    CaptureGlobalParentRegistration,
    CaptureGlobalParentState,
    CaptureGlobalProvider,
    CaptureGlobalScopeError,
    capture_global_provider_profile,
    derive_global_child_identity,
    derive_global_config_root_digest,
    derive_global_parent_id,
)
from saliencegate.capture.store import (
    CaptureConnectionState,
    CaptureStore,
    CaptureStoreError,
    CaptureStoreIntegrityError,
    CaptureStoreMode,
    CaptureStoreStateError,
)

_CONFIG_ROOT = b"/synthetic/.codex"
_PROJECT = b"/synthetic/project"
_EXCLUDED_PROJECT = b"/synthetic/private-project"


def _register_parent(store: CaptureStore, *, generation: int = 1) -> str:
    config_root_digest = derive_global_config_root_digest(
        _CONFIG_ROOT,
        INSTALLATION_KEY,
    )
    registration = store.register_global_parent(
        provider_id=CaptureGlobalProvider.CODEX,
        config_root_digest=config_root_digest,
        profile_id=PROFILE_ID,
        capability_manifest_digest=CAPABILITY_MANIFEST_DIGEST,
        host_version=HOST_VERSION,
        generation=generation,
    )
    return registration.global_parent_id


def _enable_parent(store: CaptureStore, global_parent_id: str) -> None:
    store.transition_global_parent(
        global_parent_id,
        expected_state=CaptureGlobalParentState.PENDING,
        target_state=CaptureGlobalParentState.ENABLED,
    )


def test_global_scope_enums_profiles_and_models_are_closed_and_redacted() -> None:
    assert tuple(item.value for item in CaptureConnectionScope) == (
        "project",
        "user_global",
    )
    assert tuple(item.value for item in CaptureGlobalProvider) == (
        "codex",
        "claude-code",
        "opencode",
        "pi",
    )
    assert tuple(item.value for item in CaptureGlobalParentState) == (
        "pending",
        "enabled",
        "draining",
        "disabled",
        "deleting",
    )
    assert tuple(item.value for item in CaptureGlobalHealthCode) == (
        "unknown_child_event",
        "project_identity_unavailable",
        "enrollment_rejected",
        "child_limit_reached",
        "project_excluded",
    )
    assert capture_global_provider_profile(CaptureGlobalProvider.CODEX) is PROFILE_ID
    with pytest.raises(CaptureGlobalScopeError):
        capture_global_provider_profile("codex")  # type: ignore[arg-type]

    config_root_digest = derive_global_config_root_digest(
        _CONFIG_ROOT,
        INSTALLATION_KEY,
    )
    global_parent_id = derive_global_parent_id(
        provider_id=CaptureGlobalProvider.CODEX,
        config_root_digest=config_root_digest,
        generation=1,
        installation_key=INSTALLATION_KEY,
    )
    registration = CaptureGlobalParentRegistration(
        global_parent_id=global_parent_id,
        provider_id=CaptureGlobalProvider.CODEX,
        config_root_digest=config_root_digest,
        profile_id=PROFILE_ID,
        capability_manifest_digest=CAPABILITY_MANIFEST_DIGEST,
        host_version=HOST_VERSION,
        generation=1,
    )
    assert repr(registration) == "CaptureGlobalParentRegistration(<redacted>)"
    assert global_parent_id not in repr(registration)
    with pytest.raises(ValidationError):
        CaptureGlobalParentRegistration(
            global_parent_id=global_parent_id,
            provider_id=CaptureGlobalProvider.CODEX,
            config_root_digest=config_root_digest,
            profile_id=CaptureProfile.PI_EXTENSION_V1,
            capability_manifest_digest=CAPABILITY_MANIFEST_DIGEST,
            host_version=HOST_VERSION,
            generation=1,
        )


def test_global_parent_and_child_identity_vectors_are_stable_and_path_free() -> None:
    config_root_digest = derive_global_config_root_digest(
        _CONFIG_ROOT,
        INSTALLATION_KEY,
    )
    global_parent_id = derive_global_parent_id(
        provider_id=CaptureGlobalProvider.CODEX,
        config_root_digest=config_root_digest,
        generation=7,
        installation_key=INSTALLATION_KEY,
    )
    child = derive_global_child_identity(
        global_parent_id=global_parent_id,
        provider_id=CaptureGlobalProvider.CODEX,
        generation=7,
        canonical_project_identity=_PROJECT,
        installation_key=INSTALLATION_KEY,
    )

    assert config_root_digest == (
        "a3f2d7b5fbd783d12cda08685294b530c43654bc76aa7520ec805865c71b06a2"
    )
    assert global_parent_id == "sgg-ba05019c43975a1b79385abada2e098985305a5019c6730c"
    assert child.connection_id == "sgc-caae0250ad50642974082ff6dc6f7136cc5d85f4ebde9ab3"
    assert child.project_digest == (
        "80a04d967b69ffda6904dcbad93658e011afd2d63d227b88e67236a19d7019ac"
    )
    assert _CONFIG_ROOT not in config_root_digest.encode()
    assert _PROJECT not in child.connection_id.encode()
    assert (
        derive_global_child_identity(
            global_parent_id=global_parent_id,
            provider_id=CaptureGlobalProvider.CODEX,
            generation=7,
            canonical_project_identity=_PROJECT,
            installation_key=INSTALLATION_KEY,
        )
        == child
    )
    with pytest.raises(CaptureGlobalScopeError) as captured:
        derive_global_config_root_digest(b"", INSTALLATION_KEY)
    assert str(captured.value) == "capture global scope is invalid"
    assert captured.value.__cause__ is None


def test_global_lifecycle_hook_enrollment_exclusions_health_and_project_queries(
    tmp_path: Path,
) -> None:
    path = tmp_path / "global.sqlite3"
    initialize_capture_store(path)
    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        register_connection(store)
        global_parent_id = _register_parent(store)
        assert _register_parent(store) == global_parent_id
        pending = store.get_global_parent(global_parent_id)
        assert pending.state is CaptureGlobalParentState.PENDING
        assert pending.health_marker_count == 0
        assert pending.exclusion_count == 0
        assert store.list_global_parents(provider_id=CaptureGlobalProvider.CODEX) == (pending,)
        _enable_parent(store, global_parent_id)
        exclusions = store.replace_global_exclusions(
            global_parent_id,
            (_EXCLUDED_PROJECT,),
        )
        assert len(exclusions) == 1
        assert _EXCLUDED_PROJECT not in exclusions[0].project_digest.encode()
        assert (
            store.replace_global_exclusions(
                global_parent_id,
                (_EXCLUDED_PROJECT,),
            )
            == exclusions
        )

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.HOOK,
    ) as store:
        assert store.global_child_is_excluded(
            global_parent_id,
            _EXCLUDED_PROJECT,
        )
        assert not store.global_child_is_excluded(global_parent_id, _PROJECT)
        with pytest.raises(CaptureStoreStateError):
            store.enroll_global_child(global_parent_id, _EXCLUDED_PROJECT)
        with pytest.raises(CaptureStoreStateError):
            store.resolve_global_child(global_parent_id, _PROJECT)
        child = store.enroll_global_child(global_parent_id, _PROJECT)
        assert store.enroll_global_child(global_parent_id, _PROJECT) == child
        assert store.resolve_global_child(global_parent_id, _PROJECT) == child
        first = store.mark_global_parent_health(
            global_parent_id,
            CaptureGlobalHealthCode.UNKNOWN_CHILD_EVENT,
        )
        second = store.mark_global_parent_health(
            global_parent_id,
            CaptureGlobalHealthCode.UNKNOWN_CHILD_EVENT,
        )
        assert (first.count, first.saturated) == (1, False)
        assert (second.count, second.saturated) == (2, False)
        with pytest.raises(CaptureStoreStateError):
            _register_parent(store)

    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        assert store.list_global_children(global_parent_id) == (child,)
        assert store.list_global_exclusions(global_parent_id) == exclusions
        assert store.list_global_parent_health(global_parent_id) == (second,)
        summary = store.get_global_parent(global_parent_id)
        assert (summary.health_marker_count, summary.exclusion_count) == (1, 1)
        connections = store.list_connections()
        assert len(connections) == 2
        global_connection = store.get_connection(child.connection_id)
        assert global_connection.project_digest == child.project_digest
        assert global_connection.state is CaptureConnectionState.ENABLED
        store.transition_global_parent(
            global_parent_id,
            expected_state=CaptureGlobalParentState.ENABLED,
            target_state=CaptureGlobalParentState.DRAINING,
        )
        store.transition_global_parent(
            global_parent_id,
            expected_state=CaptureGlobalParentState.DRAINING,
            target_state=CaptureGlobalParentState.DISABLED,
        )

    with (
        CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            mode=CaptureStoreMode.HOOK,
        ) as store,
        pytest.raises(CaptureStoreStateError),
    ):
        store.resolve_global_child(global_parent_id, _PROJECT)


def test_global_exclusion_replacement_is_bounded_and_exact_tuple_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "global-exclusion-bounds.sqlite3"
    initialize_capture_store(path)
    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        global_parent_id = _register_parent(store)
        with pytest.raises(CaptureStoreStateError):
            store.replace_global_exclusions(
                global_parent_id,
                tuple(b"x" for _ in range(MAX_GLOBAL_EXCLUSIONS_PER_PARENT + 1)),
            )
        with pytest.raises(CaptureStoreStateError):
            store.replace_global_exclusions(
                global_parent_id,
                [_PROJECT],  # type: ignore[arg-type]
            )
        with pytest.raises(CaptureStoreError):
            store.replace_global_exclusions(
                global_parent_id,
                (b"",),
            )
        assert store.list_global_exclusions(global_parent_id) == ()


@pytest.mark.parametrize(
    ("table", "column", "value"),
    (
        ("capture_global_parents", "host_version", "9.9.9"),
        (
            "capture_global_children",
            "created_at",
            "2026-07-25T12:34:56+00:00",
        ),
        ("capture_global_health", "count", 2),
        (
            "capture_global_exclusions",
            "created_at",
            "2026-07-25T12:34:56+00:00",
        ),
    ),
)
def test_global_authenticated_rows_reject_tampering(
    tmp_path: Path,
    table: str,
    column: str,
    value: object,
) -> None:
    path = tmp_path / f"global-tamper-{table}.sqlite3"
    initialize_capture_store(path)
    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        global_parent_id = _register_parent(store)
        _enable_parent(store, global_parent_id)
        store.replace_global_exclusions(global_parent_id, (_EXCLUDED_PROJECT,))
        store.enroll_global_child(global_parent_id, _PROJECT)
        store.mark_global_parent_health(
            global_parent_id,
            CaptureGlobalHealthCode.UNKNOWN_CHILD_EVENT,
        )
    assert table in {
        "capture_global_parents",
        "capture_global_children",
        "capture_global_health",
        "capture_global_exclusions",
    }
    assert column in {"host_version", "created_at", "count"}
    with sqlite3.connect(path) as connection:
        connection.execute(f"UPDATE {table} SET {column} = ?", (value,))
        connection.commit()

    with pytest.raises(CaptureStoreIntegrityError):
        CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            mode=CaptureStoreMode.MAINTENANCE,
        )


@pytest.mark.parametrize(
    "table",
    ("capture_global_health", "capture_global_exclusions"),
)
def test_global_parent_set_commitments_detect_deleted_rows(
    tmp_path: Path,
    table: str,
) -> None:
    path = tmp_path / f"global-deleted-{table}.sqlite3"
    initialize_capture_store(path)
    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        global_parent_id = _register_parent(store)
        _enable_parent(store, global_parent_id)
        store.replace_global_exclusions(global_parent_id, (_EXCLUDED_PROJECT,))
        store.mark_global_parent_health(
            global_parent_id,
            CaptureGlobalHealthCode.UNKNOWN_CHILD_EVENT,
        )
    with sqlite3.connect(path) as connection:
        connection.execute(f"DELETE FROM {table}")
        connection.commit()

    with pytest.raises(CaptureStoreIntegrityError):
        CaptureStore.open(
            path,
            installation_key=INSTALLATION_KEY,
            mode=CaptureStoreMode.MAINTENANCE,
        )


def test_global_health_counter_saturates_without_overflow(tmp_path: Path) -> None:
    path = tmp_path / "global-health-saturation.sqlite3"
    initialize_capture_store(path)
    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        global_parent_id = _register_parent(store)
        _enable_parent(store, global_parent_id)
        store.mark_global_parent_health(
            global_parent_id,
            CaptureGlobalHealthCode.UNKNOWN_CHILD_EVENT,
        )

        parent = store._global_parent_row(global_parent_id)
        health = store._connection.execute(
            "SELECT * FROM capture_global_health WHERE global_parent_id = ?",
            (global_parent_id,),
        ).fetchone()
        assert health is not None
        health_material = dict(health)
        health_material["count"] = MAX_GLOBAL_HEALTH_COUNT - 1
        health_material["saturated"] = 0
        health_tag = store._integrity.tag(
            "global_health",
            store_module._global_health_material(health_material),
        )
        store._connection.execute(
            """
            UPDATE capture_global_health
            SET count = ?, saturated = 0, row_tag = ?
            WHERE marker_id = ?
            """,
            (
                MAX_GLOBAL_HEALTH_COUNT - 1,
                health_tag,
                health["marker_id"],
            ),
        )
        rows = store._load_verified_global_health_rows(global_parent_id)
        parent_material = dict(parent)
        parent_material["health_set_digest"] = store._global_health_set_digest(rows)
        parent_tag = store._integrity.tag(
            "global_parent",
            store_module._global_parent_material(parent_material),
        )
        store._connection.execute(
            """
            UPDATE capture_global_parents
            SET health_set_digest = ?, row_tag = ?
            WHERE global_parent_id = ?
            """,
            (
                parent_material["health_set_digest"],
                parent_tag,
                global_parent_id,
            ),
        )
        store._connection.commit()

        saturated = store.mark_global_parent_health(
            global_parent_id,
            CaptureGlobalHealthCode.UNKNOWN_CHILD_EVENT,
        )
        repeated = store.mark_global_parent_health(
            global_parent_id,
            CaptureGlobalHealthCode.UNKNOWN_CHILD_EVENT,
        )
        assert (saturated.count, saturated.saturated) == (
            MAX_GLOBAL_HEALTH_COUNT,
            True,
        )
        assert repeated == saturated
