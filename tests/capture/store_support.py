from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from saliencegate.capture.capabilities import (
    CaptureProfile,
    capture_capability_digest,
    capture_profile,
)
from saliencegate.capture.identities import CaptureDigestContext
from saliencegate.capture.migrations import initialize_capture_store
from saliencegate.capture.publication import authenticate_capture_intake
from saliencegate.capture.schema import CaptureIntake, validate_capture_intake
from saliencegate.capture.store import (
    CaptureConnectionState,
    CaptureStore,
    CaptureStoreMode,
)
from saliencegate.security import InstallationKey

INSTALLATION_KEY_MATERIAL = b"capture-store-test-key-material!"
WRONG_KEY_MATERIAL = b"wrong-store-test-key-material!!!"
INSTALLATION_KEY = InstallationKey(INSTALLATION_KEY_MATERIAL)
WRONG_INSTALLATION_KEY = InstallationKey(WRONG_KEY_MATERIAL)

PROFILE_ID = CaptureProfile.CODEX_HOOKS_V1
PROFILE_VALUE = PROFILE_ID.value
HOST_VERSION = "0.144.6"
CAPABILITY_MANIFEST_DIGEST = capture_capability_digest(capture_profile(PROFILE_ID))

CONNECTION_ID = "connection-one"
OTHER_CONNECTION_ID = "connection-two"
PROJECT_DIGEST = "8" * 64
OTHER_PROJECT_DIGEST = "9" * 64
ZERO_TAG = "0" * 64


def capture_context(
    material: bytes = INSTALLATION_KEY_MATERIAL,
) -> CaptureDigestContext:
    return CaptureDigestContext(InstallationKey(material))


def _common_intake_values(
    kind: str,
    *,
    connection_id: str,
    session_native: bytes,
    producer_index: int,
    context: CaptureDigestContext,
) -> dict[str, object]:
    producer_native = f"synthetic-event-{producer_index}".encode()
    return {
        "schema_version": "capture-intake/v1",
        "kind": kind,
        "adapter_profile": PROFILE_VALUE,
        "capability_manifest_digest": CAPABILITY_MANIFEST_DIGEST,
        "connection_id": connection_id,
        "session_id": context.session_id(session_native),
        "producer_event_digest": context.producer_event(producer_native),
        "intake_tag": ZERO_TAG,
        "occurred_at": None,
        "timestamp_authority": "unavailable",
        "producer_sequence": producer_index,
        "sequence_authority": "producer_exact",
        "capture_disposition": "captured",
    }


def unauthenticated_intake(
    kind: str,
    *,
    connection_id: str = CONNECTION_ID,
    session_native: bytes = b"synthetic-session-one",
    producer_index: int = 1,
    context: CaptureDigestContext | None = None,
    changes: Mapping[str, object] | None = None,
) -> CaptureIntake:
    digest_context = capture_context() if context is None else context
    values = _common_intake_values(
        kind,
        connection_id=connection_id,
        session_native=session_native,
        producer_index=producer_index,
        context=digest_context,
    )
    call_native = f"synthetic-call-{producer_index}".encode()
    if kind == "action_started":
        values.update(
            call_ref=digest_context.call_ref(call_native),
            action_digest=digest_context.action_identity(
                f"synthetic-action-{producer_index}".encode()
            ),
            workspace_digest=digest_context.workspace_identity(b"synthetic-workspace"),
            environment_digest=digest_context.environment_identity(b"synthetic-environment"),
            tool_class="shell",
            identity_authority="exact",
        )
    elif kind == "action_finished":
        values.update(
            call_ref=digest_context.call_ref(call_native),
            outcome_status="succeeded",
            outcome_authority="producer_claimed_structured",
            exit_status=0,
            error_code=None,
            failure_signature=None,
        )
    elif kind == "permission_denied":
        values["call_ref"] = digest_context.call_ref(call_native)
    elif kind in {"subagent_started", "subagent_finished"}:
        values["subagent_id"] = digest_context.subagent_id(
            f"synthetic-subagent-{producer_index}".encode()
        )
    elif kind == "turn_finished":
        values["turn_id"] = digest_context.turn_id(f"synthetic-turn-{producer_index}".encode())
    elif kind == "controller_failed":
        values.update(
            error_code="provider_callback_failed",
            failure_signature=digest_context.failure_signature(
                f"synthetic-failure-{producer_index}".encode()
            ),
        )
    elif kind not in {"session_started", "session_finished"}:
        raise AssertionError(f"unsupported synthetic intake kind: {kind}")
    if changes is not None:
        values.update(changes)
    return validate_capture_intake(values)


def authenticated_intake(
    kind: str,
    *,
    connection_id: str = CONNECTION_ID,
    session_native: bytes = b"synthetic-session-one",
    producer_index: int = 1,
    context: CaptureDigestContext | None = None,
    changes: Mapping[str, object] | None = None,
) -> CaptureIntake:
    digest_context = capture_context() if context is None else context
    intake = unauthenticated_intake(
        kind,
        connection_id=connection_id,
        session_native=session_native,
        producer_index=producer_index,
        context=digest_context,
        changes=changes,
    )
    return authenticate_capture_intake(intake, context=digest_context)


def all_authenticated_intakes() -> tuple[CaptureIntake, ...]:
    kinds = (
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
    return tuple(
        authenticated_intake(kind, producer_index=index)
        for index, kind in enumerate(kinds, start=1)
    )


def register_connection(
    store: CaptureStore,
    *,
    connection_id: str = CONNECTION_ID,
    project_digest: str = PROJECT_DIGEST,
) -> None:
    registration = store.register_connection(
        connection_id=connection_id,
        project_digest=project_digest,
        profile_id=PROFILE_ID,
        capability_manifest_digest=CAPABILITY_MANIFEST_DIGEST,
        host_version=HOST_VERSION,
    )
    assert registration.state is CaptureConnectionState.PENDING
    store.transition_connection(
        connection_id=connection_id,
        expected_state=CaptureConnectionState.PENDING,
        target_state=CaptureConnectionState.ENABLED,
    )


@contextmanager
def initialized_store(
    path: Path,
    *,
    mode: CaptureStoreMode = CaptureStoreMode.MAINTENANCE,
) -> Iterator[CaptureStore]:
    initialize_capture_store(path)
    with CaptureStore.open(
        path,
        installation_key=INSTALLATION_KEY,
        busy_timeout_ms=250,
        mode=mode,
    ) as store:
        yield store
