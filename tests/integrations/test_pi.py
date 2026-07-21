from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from saliencegate.capture.capabilities import (
    CaptureProfile,
    capture_capability_digest,
    capture_profile,
)
from saliencegate.capture.identities import CaptureDigestContext
from saliencegate.capture.locations import resolve_capture_store_locations
from saliencegate.capture.migrations import initialize_capture_store
from saliencegate.capture.publication import verify_capture_intake_authentication
from saliencegate.capture.schema import (
    CaptureActionFinishedIntake,
    CaptureActionStartedIntake,
    CaptureControllerFailedIntake,
    CaptureIntake,
    canonical_capture_intake,
)
from saliencegate.capture.store import (
    CaptureConnectionState,
    CaptureSessionState,
    CaptureStore,
    CaptureStoreMode,
)
from saliencegate.capture.transport import CaptureTransportDisposition
from saliencegate.commands.capture.connect import run_connect
from saliencegate.commands.capture.disconnect import run_disconnect
from saliencegate.commands.capture.status import CaptureOperationalStatus, run_status
from saliencegate.domain import canonical_json
from saliencegate.integrations.bootstrap import (
    IntegrationBootstrap,
    inspect_integration_bootstrap,
)
from saliencegate.integrations.hook import run_capture_hook
from saliencegate.integrations.installation import derive_installation_identity
from saliencegate.integrations.pi import (
    PI_HOST_VERSION,
    PI_PROFILE,
    PiCaptureAdapter,
    PiIntegrationError,
    provider_installation_spec,
)
from saliencegate.integrations.registry import ProviderInstallationKind
from saliencegate.security import InstallationKey, load_installation_key

KEY = InstallationKey(b"p" * 32)
CONTEXT = CaptureDigestContext(KEY)
CONNECTION_ID = "sg-" + "6" * 48
NATIVE_SESSION_ID = "synthetic-pi-session"
WINDOW_DISCRIMINATOR = "7" * 64
BATCH_ID = "8" * 64
CROSS_LANGUAGE_BATCH = Path(__file__).parents[1] / "fixtures" / "pi-cross-language-batch.json"
CROSS_LANGUAGE_OVERSIZE_BATCH = (
    Path(__file__).parents[1] / "fixtures" / "pi-cross-language-oversize-batch.json"
)


def _bootstrap() -> IntegrationBootstrap:
    return IntegrationBootstrap(
        profile=CaptureProfile.PI_EXTENSION_V1,
        connection_id=CONNECTION_ID,
        launcher_path=Path("/private/tmp/saliencegate-pi-hook"),
        capability_digest=capture_capability_digest(
            capture_profile(CaptureProfile.PI_EXTENSION_V1)
        ),
        bundle_digest="9" * 64,
        receipt_mac="a" * 64,
    )


def _adapter() -> PiCaptureAdapter:
    return PiCaptureAdapter(
        connection_id=CONNECTION_ID,
        bootstrap=_bootstrap(),
        project_root=Path("/synthetic/pi/project"),
    )


def _record(kind: str, **values: object) -> dict[str, object]:
    return {
        "kind": kind,
        "session_id": NATIVE_SESSION_ID,
        "window_discriminator": WINDOW_DISCRIMINATOR,
        **values,
    }


def _batch(
    events: list[dict[str, object]],
    *,
    session_id: str = NATIVE_SESSION_ID,
    window_discriminator: str = WINDOW_DISCRIMINATOR,
    batch_id: str = BATCH_ID,
    chunk_index: int = 0,
    chunk_count: int = 1,
    bootstrap: dict[str, object] | None = None,
) -> bytes:
    sidecar = _bootstrap().model_dump(mode="json", warnings="error")
    return canonical_json(
        {
            "schema_version": "capture-batch/v1",
            "bootstrap": sidecar if bootstrap is None else bootstrap,
            "batch_id": batch_id,
            "session_id": session_id,
            "window_discriminator": window_discriminator,
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "events": events,
        }
    )


def _adapt(events: list[dict[str, object]]) -> tuple[CaptureIntake, ...]:
    return _adapter().adapt_bytes(_batch(events), context=CONTEXT)


def test_capabilities_and_project_local_installation_are_exact(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    environment = {
        "HOME": str(tmp_path / "home"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }

    declaration = _adapter().capabilities()
    spec = provider_installation_spec(project, environ=environment)

    assert declaration.profile_id is PI_PROFILE
    assert declaration.host_version == PI_HOST_VERSION == "0.80.10"
    assert declaration.capability_digest == capture_capability_digest(capture_profile(PI_PROFILE))
    assert spec.installation_kind is ProviderInstallationKind.BRIDGE
    assert spec.config_path is None
    assert spec.config is None
    assert spec.bundle_path == project / ".pi" / "extensions" / "saliencegate.ts"
    assert spec.bootstrap_path == (project / ".pi" / "extensions" / "saliencegate.bootstrap.json")
    assert spec.project_local_paths == (spec.bundle_path, spec.bootstrap_path)
    assert spec.bundle_bytes is not None
    assert spec.bundle_bytes.startswith(
        b"export const saliencegateBootstrap = "
        b'new URL("./saliencegate.bootstrap.json", import.meta.url);\n'
    )
    assert spec.receipt_path.parent == spec.launcher_path.parent
    assert spec.receipt_path.parent.is_relative_to(
        Path(environment["XDG_STATE_HOME"]) / "saliencegate" / "integrations"
    )
    assert not spec.receipt_path.is_relative_to(project)


def test_parallel_tools_keep_exact_call_parentage_with_only_coarse_identity() -> None:
    intakes = _adapt(
        [
            _record(
                "tool_started",
                event_id="1",
                call_id="parallel-b",
                tool="bash",
                identity_authority="coarse",
            ),
            _record(
                "tool_finished",
                event_id="2",
                call_id="parallel-b",
                outcome="succeeded",
            ),
            _record(
                "tool_started",
                event_id="3",
                call_id="parallel-a",
                tool="read",
                identity_authority="coarse",
            ),
            _record(
                "tool_finished",
                event_id="4",
                call_id="parallel-a",
                outcome="succeeded",
            ),
            _record("turn_finished", event_id="5"),
        ]
    )

    assert tuple(item.kind for item in intakes) == (
        "session_started",
        "action_started",
        "action_finished",
        "action_started",
        "action_finished",
        "turn_finished",
    )
    first, finished_first = intakes[1], intakes[2]
    second, finished_second = intakes[3], intakes[4]
    assert isinstance(first, CaptureActionStartedIntake)
    assert isinstance(second, CaptureActionStartedIntake)
    assert isinstance(finished_second, CaptureActionFinishedIntake)
    assert isinstance(finished_first, CaptureActionFinishedIntake)
    assert first.identity_authority == second.identity_authority == "coarse"
    assert first.call_ref == finished_first.call_ref
    assert second.call_ref == finished_second.call_ref
    assert first.call_ref != second.call_ref
    assert first.action_digest != second.action_digest
    assert finished_second.outcome_status == "succeeded"
    assert finished_second.error_code is None
    assert finished_first.outcome_status == "succeeded"
    assert finished_first.error_code is None
    assert all(
        verify_capture_intake_authentication(item, context=CONTEXT) == item for item in intakes
    )


def test_node_reducer_batch_is_accepted_without_cross_language_schema_drift() -> None:
    intakes = _adapter().adapt_bytes(CROSS_LANGUAGE_BATCH.read_bytes(), context=CONTEXT)

    assert tuple(item.kind for item in intakes) == (
        "session_started",
        "action_started",
        "action_finished",
        "controller_failed",
        "controller_failed",
        "turn_finished",
        "turn_finished",
        "turn_finished",
        "session_finished",
    )
    assert isinstance(intakes[1], CaptureActionStartedIntake)
    assert isinstance(intakes[2], CaptureActionFinishedIntake)
    assert intakes[1].call_ref == intakes[2].call_ref
    assert intakes[2].outcome_status == "succeeded"
    assert all(
        isinstance(item, CaptureControllerFailedIntake) and item.capture_disposition == "degraded"
        for item in intakes[3:5]
    )
    assert intakes[5].capture_disposition == "coverage_boundary"
    assert intakes[6].capture_disposition == "captured"
    assert intakes[7].capture_disposition == "coverage_boundary"


def test_node_one_sided_oversize_group_is_one_content_free_control() -> None:
    intakes = _adapter().adapt_bytes(
        CROSS_LANGUAGE_OVERSIZE_BATCH.read_bytes(),
        context=CONTEXT,
    )

    assert tuple(item.kind for item in intakes) == (
        "session_started",
        "controller_failed",
        "turn_finished",
    )
    assert isinstance(intakes[1], CaptureControllerFailedIntake)
    assert intakes[1].error_code == "overflow"
    assert intakes[1].capture_disposition == "degraded"


def test_native_content_and_permitted_lineage_are_hmac_reduced_before_persistence() -> None:
    sentinels = (
        b"native-pi-session-secret",
        b"deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        b"call-secret",
        b"tool-secret",
        b"old-leaf-secret",
        b"new-leaf-secret",
        b"provider-workspace-secret",
    )
    session_id, window = (value.decode() for value in sentinels[:2])
    source = _batch(
        [
            {
                "kind": "tool_started",
                "session_id": session_id,
                "window_discriminator": window,
                "event_id": "1",
                "call_id": sentinels[2].decode(),
                "tool": sentinels[3].decode(),
                "identity_authority": "coarse",
            },
            {
                "kind": "coverage_boundary",
                "session_id": session_id,
                "window_discriminator": window,
                "event_id": "3",
                "reason": "tree",
                "old_leaf_id": sentinels[4].decode(),
                "new_leaf_id": sentinels[5].decode(),
            },
        ],
        session_id=session_id,
        window_discriminator=window,
    )
    adapter = PiCaptureAdapter(
        connection_id=CONNECTION_ID,
        bootstrap=_bootstrap(),
        project_root=Path("/") / sentinels[6].decode(),
    )

    document = json.loads(source)
    document["events"].insert(
        1,
        {
            "kind": "tool_finished",
            "session_id": session_id,
            "window_discriminator": window,
            "event_id": "2",
            "call_id": sentinels[2].decode(),
            "outcome": "succeeded",
        },
    )
    intakes = adapter.adapt_bytes(canonical_json(document), context=CONTEXT)
    persisted = b"\n".join(canonical_capture_intake(item) for item in intakes)

    assert intakes[3].kind == "turn_finished"
    assert intakes[3].capture_disposition == "coverage_boundary"
    for sentinel in sentinels:
        assert sentinel not in persisted
    assert all(sentinel.decode() not in repr(intakes) for sentinel in sentinels)


def test_same_native_session_opens_distinct_resume_and_reload_windows() -> None:
    first = _adapter().adapt_bytes(
        _batch(
            [
                {
                    **_record("session_finished", event_id="1", reason="reload"),
                    "window_discriminator": "b" * 64,
                }
            ],
            window_discriminator="b" * 64,
        ),
        context=CONTEXT,
    )
    second = _adapter().adapt_bytes(
        _batch([], window_discriminator="c" * 64),
        context=CONTEXT,
    )
    replay = _adapter().adapt_bytes(
        _batch([], window_discriminator="c" * 64, batch_id="d" * 64),
        context=CONTEXT,
    )

    assert first[0].kind == "session_started"
    assert first[-1].kind == "session_finished"
    assert first[0].session_id != second[0].session_id
    assert second[0].session_id == replay[0].session_id
    assert canonical_capture_intake(second[0]) == canonical_capture_intake(replay[0])


def test_event_ids_make_retry_replay_stable_and_conflicts_collision_ready() -> None:
    original = _adapter().adapt_bytes(
        _batch(
            [
                _record(
                    "tool_started",
                    event_id="1",
                    call_id="stable-call",
                    tool="read",
                    identity_authority="coarse",
                ),
                _record(
                    "tool_finished",
                    event_id="2",
                    call_id="stable-call",
                    outcome="succeeded",
                ),
            ],
            batch_id="1" * 64,
        ),
        context=CONTEXT,
    )[1]
    replay = _adapter().adapt_bytes(
        _batch(
            [
                _record(
                    "tool_started",
                    event_id="1",
                    call_id="stable-call",
                    tool="read",
                    identity_authority="coarse",
                ),
                _record(
                    "tool_finished",
                    event_id="2",
                    call_id="stable-call",
                    outcome="succeeded",
                ),
            ],
            batch_id="2" * 64,
        ),
        context=CONTEXT,
    )[1]
    conflict = _adapter().adapt_bytes(
        _batch(
            [
                _record(
                    "tool_started",
                    event_id="1",
                    call_id="stable-call",
                    tool="bash",
                    identity_authority="coarse",
                ),
                _record(
                    "tool_finished",
                    event_id="2",
                    call_id="stable-call",
                    outcome="succeeded",
                ),
            ],
            batch_id="3" * 64,
        ),
        context=CONTEXT,
    )[1]

    assert canonical_capture_intake(original) == canonical_capture_intake(replay)
    assert conflict.producer_event_digest == original.producer_event_digest
    assert canonical_capture_intake(conflict) != canonical_capture_intake(original)


def test_store_replays_identical_pi_evidence_and_quarantines_digest_collision(
    tmp_path: Path,
) -> None:
    original_source = _batch(
        [
            _record(
                "tool_started",
                event_id="1",
                call_id="collision-call",
                tool="read",
                identity_authority="coarse",
            ),
            _record(
                "tool_finished",
                event_id="2",
                call_id="collision-call",
                outcome="succeeded",
            ),
        ],
        batch_id="1" * 64,
    )
    conflicting_source = _batch(
        [
            _record(
                "tool_started",
                event_id="1",
                call_id="collision-call",
                tool="bash",
                identity_authority="coarse",
            ),
            _record(
                "tool_finished",
                event_id="2",
                call_id="collision-call",
                outcome="succeeded",
            ),
        ],
        batch_id="2" * 64,
    )
    original_intakes = _adapter().adapt_bytes(original_source, context=CONTEXT)
    conflicting_intakes = _adapter().adapt_bytes(conflicting_source, context=CONTEXT)
    original_chunk = _adapter().transport_chunk(original_source, context=CONTEXT)
    conflicting_chunk = _adapter().transport_chunk(conflicting_source, context=CONTEXT)
    path = tmp_path / "capture.sqlite3"
    initialize_capture_store(path)
    capability_digest = capture_capability_digest(capture_profile(PI_PROFILE))
    with CaptureStore.open(
        path,
        installation_key=KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        store.register_connection(
            connection_id=CONNECTION_ID,
            project_digest="d" * 64,
            profile_id=PI_PROFILE,
            capability_manifest_digest=capability_digest,
            host_version=PI_HOST_VERSION,
        )
        store.transition_connection(
            connection_id=CONNECTION_ID,
            expected_state=CaptureConnectionState.PENDING,
            target_state=CaptureConnectionState.ENABLED,
        )
        admitted = store.append_transport_chunk(original_chunk, original_intakes)
        replayed = store.append_transport_chunk(original_chunk, original_intakes)
        quarantined = store.append_transport_chunk(conflicting_chunk, conflicting_intakes)
        snapshot = store.snapshot_session(CONNECTION_ID, original_chunk.session_id)

    assert admitted.disposition is CaptureTransportDisposition.ADMITTED
    assert replayed.disposition is CaptureTransportDisposition.REPLAYED
    assert quarantined.disposition is CaptureTransportDisposition.QUARANTINED
    assert snapshot.state is CaptureSessionState.QUARANTINED
    assert snapshot.event_count == 3


def test_compaction_tree_gap_and_shutdown_preserve_exact_lifecycle_meanings() -> None:
    intakes = _adapt(
        [
            _record(
                "coverage_boundary",
                event_id="1",
                reason="compaction",
                compaction_reason="overflow",
                from_extension=False,
                will_retry=True,
            ),
            _record(
                "coverage_boundary",
                event_id="2",
                reason="tree",
                old_leaf_id=None,
                new_leaf_id="leaf-next",
            ),
            _record("coverage_degraded", reason="transport_gap"),
            _record("session_finished", event_id="3", reason="quit"),
        ]
    )

    assert tuple(item.kind for item in intakes) == (
        "session_started",
        "turn_finished",
        "turn_finished",
        "controller_failed",
        "session_finished",
    )
    assert intakes[1].capture_disposition == "coverage_boundary"
    assert intakes[2].capture_disposition == "coverage_boundary"
    assert intakes[3].capture_disposition == "degraded"
    gap = intakes[3]
    assert isinstance(gap, CaptureControllerFailedIntake)
    assert gap.error_code == "gap_detected"


def test_reducer_shaped_compaction_metadata_is_validated_then_only_hmac_committed() -> None:
    manual = _adapt(
        [
            _record(
                "coverage_boundary",
                event_id="1",
                reason="compaction",
                compaction_reason="manual",
                from_extension=False,
                will_retry=False,
            )
        ]
    )[1]
    overflow = _adapt(
        [
            _record(
                "coverage_boundary",
                event_id="1",
                reason="compaction",
                compaction_reason="overflow",
                from_extension=True,
                will_retry=True,
            )
        ]
    )[1]

    assert manual.kind == overflow.kind == "turn_finished"
    assert manual.capture_disposition == overflow.capture_disposition == "coverage_boundary"
    assert manual.producer_event_digest == overflow.producer_event_digest
    assert canonical_capture_intake(manual) != canonical_capture_intake(overflow)
    persisted = canonical_capture_intake(manual) + canonical_capture_intake(overflow)
    for native in (b"manual", b"overflow", b"from_extension", b"will_retry"):
        assert native not in persisted


@pytest.mark.parametrize(
    "record",
    (
        _record(
            "coverage_boundary",
            event_id="1",
            reason="compaction",
            from_extension=False,
            will_retry=False,
        ),
        _record(
            "coverage_boundary",
            event_id="1",
            reason="compaction",
            compaction_reason="future",
            from_extension=False,
            will_retry=False,
        ),
        _record(
            "coverage_boundary",
            event_id="1",
            reason="compaction",
            compaction_reason="threshold",
            from_extension=0,
            will_retry=False,
        ),
        _record(
            "coverage_boundary",
            event_id="1",
            reason="compaction",
            compaction_reason="threshold",
            from_extension=False,
            will_retry=1,
        ),
        _record(
            "coverage_boundary",
            event_id="1",
            reason="compaction",
            compaction_reason="threshold",
            from_extension=False,
            will_retry=False,
            summary="forbidden",
        ),
    ),
)
def test_reducer_shaped_compaction_rejects_missing_drifted_or_non_boolean_fields(
    record: dict[str, object],
) -> None:
    with pytest.raises(PiIntegrationError):
        _adapt([record])


def test_ambiguous_error_and_unmatched_start_degrade_without_tool_failure_claims() -> None:
    intakes = _adapt(
        [
            _record("coverage_degraded", event_id="1", reason="ambiguous_error"),
            _record("coverage_degraded", event_id="2", reason="unmatched_start"),
        ]
    )

    assert tuple(item.kind for item in intakes) == (
        "session_started",
        "controller_failed",
        "controller_failed",
    )
    for intake in intakes[1:]:
        assert isinstance(intake, CaptureControllerFailedIntake)
        assert intake.capture_disposition == "degraded"
        assert intake.error_code == "invalid_transition"
    persisted = b"".join(canonical_capture_intake(item) for item in intakes)
    assert b"tool_error" not in persisted
    assert b"ambiguous_error" not in persisted
    assert b"unmatched_start" not in persisted


def test_session_stable_gap_and_terminal_are_idempotent_across_batches() -> None:
    first = _adapter().adapt_bytes(
        _batch(
            [
                _record("coverage_degraded", reason="transport_gap"),
                _record("session_finished", event_id="1", reason="quit"),
            ],
            batch_id="1" * 64,
        ),
        context=CONTEXT,
    )
    second = _adapter().adapt_bytes(
        _batch(
            [
                _record("coverage_degraded", reason="transport_gap"),
                _record("session_finished", event_id="1", reason="quit"),
            ],
            batch_id="2" * 64,
        ),
        context=CONTEXT,
    )

    assert canonical_capture_intake(first[-2]) == canonical_capture_intake(second[-2])
    assert canonical_capture_intake(first[-1]) == canonical_capture_intake(second[-1])


def test_shutdown_is_the_unique_terminal_record_in_a_chunk() -> None:
    events = [
        _record("session_finished", event_id="1", reason="quit"),
        _record("turn_finished", event_id="2"),
    ]

    with pytest.raises(PiIntegrationError):
        _adapter().adapt_bytes(_batch(events), context=CONTEXT)


def test_shutdown_is_admitted_only_in_the_final_declared_chunk() -> None:
    source = _batch(
        [_record("session_finished", event_id="1", reason="quit")],
        chunk_index=0,
        chunk_count=2,
    )

    with pytest.raises(PiIntegrationError):
        _adapter().adapt_bytes(source, context=CONTEXT)
    with pytest.raises(PiIntegrationError):
        _adapter().transport_chunk(source, context=CONTEXT)


@pytest.mark.parametrize(
    "record",
    (
        _record(
            "tool_started",
            call_id="call",
            tool="read",
            input={"path": "forbidden"},
            identity_authority="coarse",
        ),
        _record(
            "tool_started",
            call_id="call",
            tool="read",
            identity_authority="exact",
        ),
        _record(
            "tool_started",
            call_id="",
            tool="read",
            identity_authority="coarse",
        ),
        _record("tool_finished", event_id="1", call_id="call", outcome="unknown"),
        _record("tool_finished", event_id="1", call_id="call", outcome="failed"),
        _record("coverage_boundary", event_id="1", reason="unknown"),
        _record(
            "coverage_boundary",
            event_id="1",
            reason="compaction",
            compaction_reason="manual",
            from_extension=False,
            will_retry=False,
            old_leaf_id="forbidden",
        ),
        _record("coverage_degraded", event_id="1", reason="unknown"),
        _record("controller_failed", event_id="1"),
        _record("session_finished", event_id="1", reason="startup"),
        _record("turn_finished", event_id="1", history=[]),
        _record("unknown"),
    ),
)
def test_reduced_records_are_strict_and_fail_content_free(record: dict[str, object]) -> None:
    with pytest.raises(PiIntegrationError, match="Pi capture integration is invalid") as raised:
        _adapt([record])
    assert "forbidden" not in repr(raised.value)
    assert raised.value.__cause__ is None


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value.update(schema_version="capture-batch/v2"),
        lambda value: value.update(batch_id="not-a-digest"),
        lambda value: value.update(window_discriminator="A" * 64),
        lambda value: value.update(window_discriminator="0" * 63),
        lambda value: value.update(chunk_index=True),
        lambda value: value.update(chunk_index=1),
        lambda value: value.update(extra="forbidden"),
        lambda value: value["bootstrap"].update(receipt_mac="b" * 64),
        lambda value: value["events"][0].update(session_id="other-session"),
        lambda value: value["events"][0].update(window_discriminator="c" * 64),
    ),
)
def test_batch_bootstrap_session_and_window_binding_are_closed(
    mutate: Callable[[dict[str, Any]], object],
) -> None:
    document = json.loads(_batch([_record("turn_finished", event_id="1")]))
    mutate(document)

    with pytest.raises(PiIntegrationError):
        _adapter().adapt_bytes(canonical_json(document), context=CONTEXT)


def test_duplicate_json_keys_and_unbounded_native_input_are_rejected() -> None:
    source = _batch([])
    duplicated = source[:-1] + b',"events":[]}'
    with pytest.raises(PiIntegrationError):
        _adapter().adapt_bytes(duplicated, context=CONTEXT)

    with pytest.raises(PiIntegrationError):
        _adapter().adapt_bytes(source + b" " * (2 * 1_024 * 1_024), context=CONTEXT)


@pytest.mark.parametrize(
    "events",
    (
        [
            _record("turn_finished", event_id="1"),
            _record("turn_finished", event_id="1"),
        ],
        [
            _record("turn_finished", event_id="2"),
            _record("turn_finished", event_id="1"),
        ],
        [_record("turn_finished", event_id="0")],
        [_record("turn_finished", event_id="01")],
        [_record("turn_finished", event_id="-1")],
    ),
)
def test_bridge_event_ids_are_positive_canonical_and_monotonic(
    events: list[dict[str, object]],
) -> None:
    with pytest.raises(PiIntegrationError):
        _adapter().adapt_bytes(_batch(events), context=CONTEXT)


@pytest.mark.parametrize(
    "events",
    (
        [_record("tool_finished", event_id="1", call_id="call", outcome="succeeded")],
        [
            _record(
                "tool_started",
                event_id="1",
                call_id="call",
                tool="read",
                identity_authority="coarse",
            )
        ],
        [
            _record(
                "tool_started",
                event_id="1",
                call_id="first",
                tool="read",
                identity_authority="coarse",
            ),
            _record(
                "tool_finished",
                event_id="2",
                call_id="second",
                outcome="succeeded",
            ),
        ],
        [
            _record(
                "tool_started",
                event_id="1",
                call_id="call",
                tool="read",
                identity_authority="coarse",
            ),
            _record("turn_finished", event_id="2"),
            _record(
                "tool_finished",
                event_id="3",
                call_id="call",
                outcome="succeeded",
            ),
        ],
        [
            _record(
                "tool_started",
                event_id="1",
                call_id="call",
                tool="read",
                identity_authority="coarse",
            ),
            _record(
                "tool_finished",
                event_id="3",
                call_id="call",
                outcome="succeeded",
            ),
        ],
    ),
)
def test_tool_success_records_are_atomic_adjacent_producer_pairs(
    events: list[dict[str, object]],
) -> None:
    with pytest.raises(PiIntegrationError):
        _adapter().adapt_bytes(_batch(events), context=CONTEXT)


@pytest.mark.parametrize("event_id", ("998", "9" * 16_384))
def test_event_ids_cannot_exceed_the_reducer_record_ceiling(event_id: str) -> None:
    with pytest.raises(PiIntegrationError):
        _adapt([_record("turn_finished", event_id=event_id)])


def test_event_id_at_the_reducer_record_ceiling_is_accepted() -> None:
    assert len(_adapt([_record("turn_finished", event_id="997")])) == 2


@pytest.mark.parametrize(
    ("field", "maximum", "record"),
    (
        (
            "call_id",
            16 * 1_024,
            _record(
                "tool_started",
                event_id="1",
                call_id="placeholder",
                tool="read",
                identity_authority="coarse",
            ),
        ),
        (
            "tool",
            1 * 1_024,
            _record(
                "tool_started",
                event_id="1",
                call_id="call",
                tool="placeholder",
                identity_authority="coarse",
            ),
        ),
        (
            "new_leaf_id",
            16 * 1_024,
            _record(
                "coverage_boundary",
                event_id="1",
                reason="tree",
                old_leaf_id=None,
                new_leaf_id="placeholder",
            ),
        ),
    ),
)
def test_reduced_identifier_caps_are_exact(
    field: str,
    maximum: int,
    record: dict[str, object],
) -> None:
    accepted = dict(record)
    accepted[field] = ("1" if field == "event_id" else "x") * maximum
    rejected = dict(record)
    rejected[field] = "x" * (maximum + 1)

    accepted_events = [accepted]
    rejected_events = [rejected]
    if accepted["kind"] == "tool_started":
        accepted_events.append(
            _record(
                "tool_finished",
                event_id="2",
                call_id=accepted["call_id"],
                outcome="succeeded",
            )
        )
        rejected_events.append(
            _record(
                "tool_finished",
                event_id="2",
                call_id=rejected["call_id"],
                outcome="succeeded",
            )
        )

    assert len(_adapter().adapt_bytes(_batch(accepted_events), context=CONTEXT)) >= 2
    with pytest.raises(PiIntegrationError):
        _adapter().adapt_bytes(_batch(rejected_events), context=CONTEXT)


def test_native_session_cap_is_exact_and_window_is_not_provider_identity() -> None:
    accepted = "s" * (16 * 1_024)
    accepted_event: dict[str, object] = {
        "kind": "turn_finished",
        "session_id": accepted,
        "window_discriminator": WINDOW_DISCRIMINATOR,
        "event_id": "1",
    }
    intakes = _adapter().adapt_bytes(
        _batch([accepted_event], session_id=accepted),
        context=CONTEXT,
    )
    assert tuple(item.kind for item in intakes) == ("session_started", "turn_finished")

    with pytest.raises(PiIntegrationError):
        _adapter().adapt_bytes(
            _batch([], session_id=accepted + "s"),
            context=CONTEXT,
        )


@pytest.mark.parametrize(
    "session_id",
    ("-leading", "trailing.", "contains:colon", "contains/slash", "two words", "café"),
)
def test_native_session_id_matches_the_pinned_upstream_ascii_grammar(
    session_id: str,
) -> None:
    with pytest.raises(PiIntegrationError):
        _adapter().adapt_bytes(_batch([], session_id=session_id), context=CONTEXT)


def test_transport_descriptor_commits_to_the_composite_window_and_exact_source() -> None:
    source = _batch([_record("turn_finished", event_id="1")], chunk_index=2, chunk_count=4)
    descriptor = _adapter().transport_chunk(source, context=CONTEXT)
    changed_window_source = _batch(
        [
            {
                **_record("turn_finished", event_id="1"),
                "window_discriminator": "b" * 64,
            }
        ],
        window_discriminator="b" * 64,
        chunk_index=2,
        chunk_count=4,
    )
    changed_window = _adapter().transport_chunk(changed_window_source, context=CONTEXT)

    assert descriptor.connection_id == CONNECTION_ID
    assert descriptor.batch_ref == CONTEXT.transport_batch_ref(BATCH_ID.encode())
    assert descriptor.chunk_index == 2
    assert descriptor.chunk_count == 4
    assert descriptor.chunk_digest == CONTEXT.transport_chunk_digest(source)
    assert descriptor.session_id != changed_window.session_id
    assert NATIVE_SESSION_ID not in repr(descriptor)
    assert WINDOW_DISCRIMINATOR not in repr(descriptor)


def test_oversize_record_degrades_without_truncation() -> None:
    intakes = _adapt([_record("oversize", reason="event_limit")])

    assert tuple(item.kind for item in intakes) == ("session_started", "controller_failed")
    assert intakes[1].capture_disposition == "degraded"
    overflow = intakes[1]
    assert isinstance(overflow, CaptureControllerFailedIntake)
    assert overflow.error_code == "overflow"


def test_python_and_embedded_runtime_expose_no_history_or_session_mutation_surface(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    spec = provider_installation_spec(
        project,
        environ={
            "HOME": str(tmp_path / "home"),
            "XDG_STATE_HOME": str(tmp_path / "state"),
        },
    )
    assert spec.bundle_bytes is not None
    python_source = inspect.getsource(__import__("saliencegate.integrations.pi", fromlist=["*"]))
    forbidden = (
        "sessionManager.getEntries",
        "sessionManager.getTree",
        "sessionManager.getBranch",
        "sessionManager.appendEntry",
    )
    for value in forbidden:
        assert value not in python_source
        assert value.encode() not in spec.bundle_bytes
    assert b"sourceMappingURL" not in spec.bundle_bytes
    assert os.fspath(project).encode() not in spec.bundle_bytes
    assert spec.bundle_digest == hashlib.sha256(spec.bundle_bytes).hexdigest()


@pytest.mark.skipif(os.name != "posix", reason="native Windows lifecycle is covered by R01")
def test_default_install_hook_store_status_and_disconnect_round_trip(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    pi_directory = project / ".pi"
    pi_directory.mkdir()
    foreign = pi_directory / "user-owned-trust-sentinel"
    foreign.write_bytes(b"user-owned-trust-state")
    environment = {
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "configuration"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }
    capture_executable = Path(sys.executable).resolve(strict=True)

    connected = run_connect(
        provider="pi",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    )
    spec = provider_installation_spec(project, environ=environment)
    before = run_status(
        provider="pi",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    ).providers[0]

    assert connected.capture_enabled is True
    assert connected.project_local_files == 2
    assert before.status is CaptureOperationalStatus.INSTALLED_NOT_OBSERVED
    assert before.drift == ()
    assert spec.bundle_path is not None and spec.bundle_path.read_bytes() == spec.bundle_bytes
    assert spec.bootstrap_path is not None
    installed_bootstrap = inspect_integration_bootstrap(spec.bootstrap_path)
    key = load_installation_key(environ=environment)
    identity = derive_installation_identity(spec, key)
    tampered_bootstrap = installed_bootstrap.model_dump(mode="json", warnings="error")
    tampered_bootstrap["receipt_mac"] = "f" * 64
    assert (
        run_capture_hook(
            ("--profile", PI_PROFILE.value, "--connection", identity.connection_id),
            BytesIO(_batch([], bootstrap=tampered_bootstrap)),
            environ=environment,
            capture_executable=capture_executable,
        )
        == 0
    )
    assert (
        run_capture_hook(
            ("--profile", PI_PROFILE.value, "--connection", "sg-" + "e" * 48),
            BytesIO(
                _batch(
                    [],
                    bootstrap=installed_bootstrap.model_dump(mode="json", warnings="error"),
                )
            ),
            environ=environment,
            capture_executable=capture_executable,
        )
        == 0
    )
    still_unobserved = run_status(
        provider="pi",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    ).providers[0]
    assert still_unobserved.status is CaptureOperationalStatus.INSTALLED_NOT_OBSERVED
    assert still_unobserved.session_count == 0
    source = _batch(
        [
            _record(
                "tool_started",
                event_id="1",
                call_id="runtime-call",
                tool="read",
                identity_authority="coarse",
            ),
            _record(
                "tool_finished",
                event_id="2",
                call_id="runtime-call",
                outcome="succeeded",
            ),
            _record("session_finished", event_id="3", reason="quit"),
        ],
        bootstrap=installed_bootstrap.model_dump(mode="json", warnings="error"),
    )

    assert (
        run_capture_hook(
            ("--profile", PI_PROFILE.value, "--connection", identity.connection_id),
            BytesIO(source),
            environ=environment,
            capture_executable=capture_executable,
        )
        == 0
    )

    runtime_adapter = PiCaptureAdapter(
        connection_id=identity.connection_id,
        bootstrap=installed_bootstrap,
        project_root=project,
    )
    session_id = runtime_adapter.transport_chunk(
        source,
        context=CaptureDigestContext(key),
    ).session_id
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    with CaptureStore.open(
        locations.database_path,
        installation_key=key,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        snapshot = store.snapshot_session(identity.connection_id, session_id)

    observed = run_status(
        provider="pi",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    ).providers[0]
    assert snapshot.state is CaptureSessionState.CLOSED
    assert snapshot.event_count == 4
    assert snapshot.transport_receipt_count == 1
    assert snapshot.incomplete_transport_batch_count == 0
    assert observed.status is CaptureOperationalStatus.ACTIVE_OBSERVED
    assert observed.session_count == 1
    assert foreign.read_bytes() == b"user-owned-trust-state"
    persisted = b"".join(
        candidate.read_bytes()
        for candidate in (
            locations.database_path,
            locations.database_path.with_name(locations.database_path.name + "-wal"),
            locations.database_path.with_name(locations.database_path.name + "-shm"),
        )
        if candidate.exists()
    )
    for native in (NATIVE_SESSION_ID, WINDOW_DISCRIMINATOR, "runtime-call"):
        assert native.encode() not in persisted

    disconnected = run_disconnect(
        provider="pi",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    )
    assert disconnected.disposition == "uninstalled"
    assert not spec.bundle_path.exists()
    assert not spec.bootstrap_path.exists()
    assert foreign.read_bytes() == b"user-owned-trust-state"
