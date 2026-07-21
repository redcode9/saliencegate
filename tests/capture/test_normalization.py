from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from tests.capture.store_support import (
    CONNECTION_ID,
    INSTALLATION_KEY,
    authenticated_intake,
    initialized_store,
    register_connection,
)

from saliencegate.capture.capabilities import (
    CaptureProfile,
    capture_capability_digest,
    capture_profile,
)
from saliencegate.capture.identities import CaptureDigestContext
from saliencegate.capture.normalization import (
    CaptureNormalization,
    CaptureNormalizationDiagnosticCode,
    _capture_detector_minimum,
    _structured_status_is_authorized,
    normalize_capture_session_snapshot,
    verify_capture_normalization,
)
from saliencegate.capture.publication import authenticate_capture_intake
from saliencegate.capture.schema import CaptureIntake, validate_capture_intake
from saliencegate.capture.sessions import CaptureSessionSnapshot
from saliencegate.capture.store import CaptureConnectionState
from saliencegate.capture.transport import (
    CaptureTransportChunk,
    CaptureTransportDisposition,
)
from saliencegate.domain import EventType, SignalType, canonical_json
from saliencegate.security import InstallationKey
from saliencegate.signals import DetectionStatus

_CONNECTION = "normalization-connection"
_PROJECT = "7" * 64
_ZERO_TAG = "0" * 64


def _intake(
    profile_id: CaptureProfile,
    kind: str,
    *,
    ordinal: int,
    call: str = "default",
    action: str | None = None,
    outcome_status: str | None = "failed",
    outcome_authority: str = "producer_claimed_structured",
    identity_authority: str = "exact",
    error_code: str | None = "tool_error",
    session_native: bytes = b"normalization-session",
    changes: Mapping[str, object] | None = None,
) -> CaptureIntake:
    context = CaptureDigestContext(INSTALLATION_KEY)
    manifest = capture_profile(profile_id)
    values: dict[str, object] = {
        "schema_version": "capture-intake/v1",
        "kind": kind,
        "adapter_profile": profile_id.value,
        "capability_manifest_digest": capture_capability_digest(manifest),
        "connection_id": _CONNECTION,
        "session_id": context.session_id(session_native),
        "producer_event_digest": context.producer_event(f"normalization-event-{ordinal}".encode()),
        "intake_tag": _ZERO_TAG,
        "occurred_at": None,
        "timestamp_authority": "unavailable",
        "producer_sequence": ordinal,
        "sequence_authority": "producer_exact",
        "capture_disposition": "captured",
    }
    call_ref = context.call_ref(f"normalization-call-{call}".encode())
    if kind == "action_started":
        action_name = call if action is None else action
        values.update(
            call_ref=call_ref,
            action_digest=context.action_identity(f"normalization-action-{action_name}".encode()),
            workspace_digest=context.workspace_identity(b"normalization-workspace"),
            environment_digest=context.environment_identity(b"normalization-environment"),
            tool_class="shell",
            identity_authority=identity_authority,
        )
    elif kind == "action_finished":
        if outcome_authority == "unavailable":
            outcome_status = None
            exit_status = None
            error_code = None
            failure_signature = None
        elif outcome_status == "succeeded":
            exit_status = 0
            error_code = None
            failure_signature = None
        else:
            exit_status = 1
            failure_signature = context.failure_signature(f"normalization-failure-{call}".encode())
        values.update(
            call_ref=call_ref,
            outcome_status=outcome_status,
            outcome_authority=outcome_authority,
            exit_status=exit_status,
            error_code=error_code,
            failure_signature=failure_signature,
        )
    elif kind == "permission_denied":
        values["call_ref"] = call_ref
    elif kind in {"subagent_started", "subagent_finished"}:
        values["subagent_id"] = context.subagent_id(f"normalization-subagent-{call}".encode())
    elif kind == "turn_finished":
        values["turn_id"] = context.turn_id(f"normalization-turn-{call}".encode())
    elif kind == "controller_failed":
        values.update(
            error_code="provider_callback_failed",
            failure_signature=context.failure_signature(b"normalization-controller-failure"),
        )
    elif kind not in {"session_started", "session_finished"}:
        raise AssertionError(f"unsupported test kind: {kind}")
    if changes is not None:
        values.update(changes)
    return authenticate_capture_intake(
        validate_capture_intake(values),
        context=context,
    )


def _snapshot(
    path: Path,
    profile_id: CaptureProfile,
    specs: Sequence[Mapping[str, object]],
) -> CaptureSessionSnapshot:
    manifest = capture_profile(profile_id)
    with initialized_store(path) as store:
        store.register_connection(
            connection_id=_CONNECTION,
            project_digest=_PROJECT,
            profile_id=profile_id,
            capability_manifest_digest=capture_capability_digest(manifest),
            host_version=manifest.host_version,
        )
        store.transition_connection(
            _CONNECTION,
            expected_state=CaptureConnectionState.PENDING,
            target_state=CaptureConnectionState.ENABLED,
        )
        intakes: list[CaptureIntake] = []
        for ordinal, spec in enumerate(specs, start=1):
            values = dict(spec)
            kind = values.pop("kind")
            assert type(kind) is str
            intake = _intake(profile_id, kind, ordinal=ordinal, **values)  # type: ignore[arg-type]
            intakes.append(intake)
        assert intakes
        session_id = intakes[0].session_id
        assert all(intake.session_id == session_id for intake in intakes)
        if profile_id in {
            CaptureProfile.OPENCODE_PLUGIN_V1,
            CaptureProfile.PI_EXTENSION_V1,
        }:
            context = CaptureDigestContext(INSTALLATION_KEY)
            receipt = store.append_transport_chunk(
                CaptureTransportChunk(
                    connection_id=_CONNECTION,
                    session_id=session_id,
                    batch_ref=context.transport_batch_ref(b"normalization-batch"),
                    chunk_index=0,
                    chunk_count=1,
                    chunk_digest=context.transport_chunk_digest(b"normalization-chunk"),
                ),
                tuple(intakes),
            )
            assert receipt.disposition is CaptureTransportDisposition.ADMITTED
        else:
            for intake in intakes:
                store.append(intake)
        snapshot = store.snapshot_session(_CONNECTION, session_id)
        if profile_id in {
            CaptureProfile.OPENCODE_PLUGIN_V1,
            CaptureProfile.PI_EXTENSION_V1,
        }:
            assert snapshot.transport_receipt_count == 1
        else:
            assert snapshot.transport_receipt_count == 0
        assert snapshot.incomplete_transport_batch_count == 0
        assert snapshot.coverage_degraded is False
        return snapshot


def _diagnostic_codes(
    normalization: CaptureNormalization,
) -> tuple[CaptureNormalizationDiagnosticCode, ...]:
    return tuple(item.code for item in normalization.diagnostics)


def test_maps_all_nine_capture_kinds_and_exact_tool_parents(tmp_path: Path) -> None:
    snapshot = _snapshot(
        tmp_path / "all-kinds.sqlite3",
        CaptureProfile.CLAUDE_CODE_HOOKS_V1,
        (
            {"kind": "session_started"},
            {"kind": "action_started", "call": "one", "action": "same"},
            {"kind": "action_finished", "call": "one"},
            {"kind": "action_started", "call": "two", "action": "same"},
            {"kind": "permission_denied", "call": "two"},
            {"kind": "subagent_started", "call": "child"},
            {"kind": "subagent_finished", "call": "child"},
            {"kind": "turn_finished", "call": "turn"},
            {"kind": "controller_failed"},
            {"kind": "session_finished"},
        ),
    )

    normalized = normalize_capture_session_snapshot(
        snapshot,
        installation_key=INSTALLATION_KEY,
    )

    assert normalized.semantic_coherence is True
    assert normalized.shadow_trace is not None
    assert tuple(record["kind"] for record in normalized.shadow_trace.records) == (
        "run_start",
        "action_identity",
        "tool_result",
        "action_identity",
        "tool_result",
        "controller_error",
        "run_end",
    )
    assert tuple(event.event_type for event in normalized.events) == (
        EventType.RUN_START,
        EventType.ACTION_PROPOSAL,
        EventType.TOOL_COMPLETION,
        EventType.ACTION_PROPOSAL,
        EventType.TOOL_COMPLETION,
        EventType.CONTROLLER_ERROR,
        EventType.RUN_END,
    )
    assert normalized.events[2].parent_ids == (normalized.events[1].event_id,)
    assert normalized.events[4].parent_ids == (normalized.events[3].event_id,)
    assert normalized.counts.authorized_tool_result_count == 2
    assert normalized.counts.authorized_controller_error_count == 1
    assert normalized.counts.exact_parent_classifiable_failed_result_count == 2
    assert (
        _diagnostic_codes(normalized).count(
            CaptureNormalizationDiagnosticCode.EVENT_NOT_PROJECTABLE
        )
        == 3
    )


@pytest.mark.parametrize(
    ("specs", "expected"),
    (
        (
            (
                {"kind": "session_started"},
                {"kind": "action_finished", "call": "missing"},
                {"kind": "session_finished"},
            ),
            CaptureNormalizationDiagnosticCode.MISSING_CALL_PARENT,
        ),
        (
            (
                {"kind": "session_started"},
                {"kind": "action_finished", "call": "future"},
                {"kind": "action_started", "call": "future"},
                {"kind": "session_finished"},
            ),
            CaptureNormalizationDiagnosticCode.FUTURE_CALL_PARENT,
        ),
        (
            (
                {"kind": "session_started"},
                {"kind": "action_started", "call": "duplicate", "action": "same"},
                {"kind": "action_started", "call": "duplicate", "action": "same"},
                {"kind": "action_finished", "call": "duplicate"},
                {"kind": "session_finished"},
            ),
            CaptureNormalizationDiagnosticCode.DUPLICATE_CALL_PARENT,
        ),
        (
            (
                {"kind": "session_started"},
                {"kind": "action_started", "call": "conflict", "action": "one"},
                {"kind": "action_started", "call": "conflict", "action": "two"},
                {"kind": "action_finished", "call": "conflict"},
                {"kind": "session_finished"},
            ),
            CaptureNormalizationDiagnosticCode.CONFLICTING_CALL_PARENT,
        ),
    ),
)
def test_missing_future_duplicate_and_conflicting_parents_are_ignored(
    tmp_path: Path,
    specs: Sequence[Mapping[str, object]],
    expected: CaptureNormalizationDiagnosticCode,
) -> None:
    snapshot = _snapshot(
        tmp_path / f"{expected.value}.sqlite3",
        CaptureProfile.CLAUDE_CODE_HOOKS_V1,
        specs,
    )

    normalized = normalize_capture_session_snapshot(
        snapshot,
        installation_key=INSTALLATION_KEY,
    )

    assert normalized.semantic_coherence is False
    assert normalized.counts.authorized_tool_result_count == 0
    assert expected in _diagnostic_codes(normalized)
    assert all(event.event_type is not EventType.TOOL_COMPLETION for event in normalized.events)


def test_unavailable_outcome_is_ignored_without_inventing_failure_evidence(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(
        tmp_path / "unavailable.sqlite3",
        CaptureProfile.CODEX_HOOKS_V1,
        (
            {"kind": "session_started"},
            {"kind": "action_started", "call": "one"},
            {
                "kind": "action_finished",
                "call": "one",
                "outcome_authority": "unavailable",
            },
            {"kind": "session_finished"},
        ),
    )

    normalized = normalize_capture_session_snapshot(
        snapshot,
        installation_key=INSTALLATION_KEY,
    )

    assert normalized.semantic_coherence is True
    assert normalized.counts.authorized_tool_result_count == 0
    assert CaptureNormalizationDiagnosticCode.OUTCOME_UNAVAILABLE in _diagnostic_codes(normalized)
    assert all(event.event_type is not EventType.TOOL_COMPLETION for event in normalized.events)


def test_duplicate_call_groups_and_terminal_records_are_preflighted_symmetrically(
    tmp_path: Path,
) -> None:
    duplicate_parent = _snapshot(
        tmp_path / "duplicate-no-result.sqlite3",
        CaptureProfile.CLAUDE_CODE_HOOKS_V1,
        (
            {"kind": "session_started"},
            {"kind": "action_started", "call": "duplicate", "action": "same"},
            {"kind": "action_started", "call": "duplicate", "action": "same"},
            {"kind": "session_finished"},
        ),
    )
    first_unavailable = _snapshot(
        tmp_path / "duplicate-results.sqlite3",
        CaptureProfile.CLAUDE_CODE_HOOKS_V1,
        (
            {"kind": "session_started"},
            {"kind": "action_started", "call": "one"},
            {
                "kind": "action_finished",
                "call": "one",
                "outcome_authority": "unavailable",
            },
            {"kind": "permission_denied", "call": "one"},
            {"kind": "session_finished"},
        ),
    )

    parent_normalization = normalize_capture_session_snapshot(
        duplicate_parent,
        installation_key=INSTALLATION_KEY,
    )
    result_normalization = normalize_capture_session_snapshot(
        first_unavailable,
        installation_key=INSTALLATION_KEY,
    )

    assert parent_normalization.semantic_coherence is False
    assert (
        _diagnostic_codes(parent_normalization).count(
            CaptureNormalizationDiagnosticCode.DUPLICATE_CALL_PARENT
        )
        == 2
    )
    assert result_normalization.semantic_coherence is False
    assert (
        _diagnostic_codes(result_normalization).count(
            CaptureNormalizationDiagnosticCode.DUPLICATE_CALL_RESULT
        )
        == 2
    )
    assert result_normalization.counts.authorized_tool_result_count == 0
    assert all(
        event.event_type is not EventType.TOOL_COMPLETION for event in result_normalization.events
    )


def test_ambiguous_duplicate_actions_cannot_signal_but_unrelated_actions_can(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(
        tmp_path / "duplicate-positive-isolation.sqlite3",
        CaptureProfile.CODEX_HOOKS_V1,
        (
            {"kind": "session_started"},
            {"kind": "action_started", "call": "duplicate", "action": "ambiguous"},
            {"kind": "action_started", "call": "duplicate", "action": "ambiguous"},
            {"kind": "action_started", "call": "valid-one", "action": "repeated"},
            {"kind": "action_started", "call": "valid-two", "action": "repeated"},
            {"kind": "session_finished"},
        ),
    )

    normalized = normalize_capture_session_snapshot(
        snapshot,
        installation_key=INSTALLATION_KEY,
    )

    assert normalized.semantic_coherence is False
    assert normalized.counts.action_identity_count == 2
    assert normalized.counts.exact_action_identity_count == 2
    assert (
        _diagnostic_codes(normalized).count(
            CaptureNormalizationDiagnosticCode.DUPLICATE_CALL_PARENT
        )
        == 2
    )
    assert CaptureNormalizationDiagnosticCode.SESSION_QUARANTINED not in _diagnostic_codes(
        normalized
    )
    repeated_signals = tuple(
        signal
        for report in normalized.extraction_reports
        for signal in report.signals
        if signal.signal_type is SignalType.REPEATED_ACTION
    )
    assert len(repeated_signals) == 1
    assert repeated_signals[0].evidence_event_ids == (
        normalized.events[1].event_id,
        normalized.events[2].event_id,
    )


def test_subagent_lifecycle_and_turn_identity_incoherence_is_content_free(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(
        tmp_path / "lineage.sqlite3",
        CaptureProfile.CLAUDE_CODE_HOOKS_V1,
        (
            {"kind": "session_started"},
            {"kind": "subagent_started", "call": "unclosed"},
            {"kind": "turn_finished", "call": "same"},
            {"kind": "turn_finished", "call": "same"},
            {"kind": "session_finished"},
        ),
    )

    normalized = normalize_capture_session_snapshot(
        snapshot,
        installation_key=INSTALLATION_KEY,
    )

    assert normalized.semantic_coherence is False
    assert CaptureNormalizationDiagnosticCode.INCOHERENT_SUBAGENT_LIFECYCLE in (
        _diagnostic_codes(normalized)
    )
    assert (
        _diagnostic_codes(normalized).count(CaptureNormalizationDiagnosticCode.DUPLICATE_TURN_ID)
        == 2
    )
    assert len(normalized.diagnostics) <= normalized.counts.source_event_count + 2


def test_controller_error_can_flag_but_never_meets_tool_error_absence_minimum(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(
        tmp_path / "controller-only.sqlite3",
        CaptureProfile.CLAUDE_CODE_HOOKS_V1,
        (
            {"kind": "session_started"},
            {"kind": "controller_failed"},
            {"kind": "session_finished"},
        ),
    )

    normalized = normalize_capture_session_snapshot(
        snapshot,
        installation_key=INSTALLATION_KEY,
    )
    evidence = next(
        item for item in normalized.detector_evidence if item.signal_type is SignalType.TOOL_ERROR
    )

    assert normalized.counts.authorized_controller_error_count == 1
    assert normalized.counts.authorized_tool_result_count == 0
    assert evidence.authorized_observation_count == 0
    assert evidence.minimum_observation_met is False
    assert any(
        signal.signal_type is SignalType.TOOL_ERROR
        for report in normalized.extraction_reports
        for signal in report.signals
    )


def test_mixed_exact_and_coarse_actions_do_not_authorize_an_absence_claim(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(
        tmp_path / "mixed-action-authority.sqlite3",
        CaptureProfile.CODEX_HOOKS_V1,
        (
            {"kind": "session_started"},
            {"kind": "action_started", "call": "first", "action": "first"},
            {
                "kind": "action_started",
                "call": "coarse",
                "action": "coarse",
                "identity_authority": "coarse",
            },
            {"kind": "action_started", "call": "last", "action": "last"},
            {"kind": "session_finished"},
        ),
    )

    normalized = normalize_capture_session_snapshot(
        snapshot,
        installation_key=INSTALLATION_KEY,
    )
    evidence = next(
        item
        for item in normalized.detector_evidence
        if item.signal_type is SignalType.REPEATED_ACTION
    )
    final_action = next(
        item
        for item in normalized.extraction_reports[-2].evaluations
        if item.signal_type is SignalType.REPEATED_ACTION
    )

    assert normalized.counts.exact_action_identity_count == 2
    assert final_action.outcome.status is DetectionStatus.ABSTAINED
    assert evidence.authorized_observation_count == 1
    assert evidence.unresolved_observation_count == 1
    assert evidence.minimum_observation_met is False


def test_later_unresolved_action_blocks_an_earlier_met_absence_minimum(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(
        tmp_path / "later-unresolved-action.sqlite3",
        CaptureProfile.CODEX_HOOKS_V1,
        (
            {"kind": "session_started"},
            {"kind": "action_started", "call": "first", "action": "first"},
            {"kind": "action_started", "call": "second", "action": "second"},
            {
                "kind": "action_started",
                "call": "coarse-target",
                "action": "target",
                "identity_authority": "coarse",
            },
            {"kind": "action_started", "call": "exact-target", "action": "target"},
            {"kind": "session_finished"},
        ),
    )

    normalized = normalize_capture_session_snapshot(
        snapshot,
        installation_key=INSTALLATION_KEY,
    )
    evidence = next(
        item
        for item in normalized.detector_evidence
        if item.signal_type is SignalType.REPEATED_ACTION
    )
    target_evaluation = next(
        item
        for item in normalized.extraction_reports[-2].evaluations
        if item.signal_type is SignalType.REPEATED_ACTION
    )

    assert target_evaluation.outcome.status is DetectionStatus.ABSTAINED
    assert evidence.authorized_observation_count == 2
    assert evidence.unresolved_observation_count == 1
    assert evidence.minimum_observation_met is False


def test_claude_coarse_pre_hook_identity_never_enables_repeated_failure(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(
        tmp_path / "mixed-failure-authority.sqlite3",
        CaptureProfile.CLAUDE_CODE_HOOKS_V1,
        (
            {"kind": "session_started"},
            {"kind": "action_started", "call": "first", "action": "first"},
            {"kind": "action_finished", "call": "first"},
            {
                "kind": "action_started",
                "call": "coarse",
                "action": "coarse",
                "identity_authority": "coarse",
            },
            {"kind": "action_finished", "call": "coarse"},
            {"kind": "action_started", "call": "last", "action": "last"},
            {"kind": "action_finished", "call": "last"},
            {"kind": "session_finished"},
        ),
    )

    normalized = normalize_capture_session_snapshot(
        snapshot,
        installation_key=INSTALLATION_KEY,
    )
    assert normalized.counts.exact_parent_classifiable_failed_result_count == 2
    assert all(
        item.signal_type is not SignalType.REPEATED_FAILURE for item in normalized.detector_evidence
    )
    assert all(
        item.signal_type is not SignalType.REPEATED_FAILURE
        for report in normalized.extraction_reports
        for item in report.evaluations
    )


@pytest.mark.parametrize(
    ("outcome_status", "expected_status"),
    (
        ("succeeded", DetectionStatus.NO_MATCH),
        ("failed", DetectionStatus.DETECTED),
    ),
)
def test_structured_tool_results_are_resolved_tool_error_observations(
    tmp_path: Path,
    outcome_status: str,
    expected_status: DetectionStatus,
) -> None:
    snapshot = _snapshot(
        tmp_path / f"tool-error-{outcome_status}.sqlite3",
        CaptureProfile.CLAUDE_CODE_HOOKS_V1,
        (
            {"kind": "session_started"},
            {"kind": "action_started", "call": "one"},
            {
                "kind": "action_finished",
                "call": "one",
                "outcome_status": outcome_status,
            },
            {"kind": "session_finished"},
        ),
    )

    normalized = normalize_capture_session_snapshot(
        snapshot,
        installation_key=INSTALLATION_KEY,
    )
    evidence = next(
        item for item in normalized.detector_evidence if item.signal_type is SignalType.TOOL_ERROR
    )
    result_evaluation = next(
        item
        for item in normalized.extraction_reports[-2].evaluations
        if item.signal_type is SignalType.TOOL_ERROR
    )

    assert result_evaluation.outcome.status is expected_status
    assert evidence.authorized_observation_count == 1
    assert evidence.unresolved_observation_count == 0
    assert evidence.minimum_observation_met is True


def test_structured_action_outcome_authority_is_status_specific() -> None:
    assert _structured_status_is_authorized("succeeded", frozenset({"provider_claimed_success"}))
    assert not _structured_status_is_authorized("failed", frozenset({"provider_claimed_success"}))
    assert _structured_status_is_authorized("failed", frozenset({"provider_claimed_failure"}))
    assert not _structured_status_is_authorized(
        "succeeded", frozenset({"provider_claimed_failure"})
    )
    for shared in ("provider_claimed_tool_outcome", "tool_state_discriminator"):
        assert _structured_status_is_authorized("succeeded", frozenset({shared}))
        assert _structured_status_is_authorized("failed", frozenset({shared}))
    pi_authority = frozenset({"confirmed_success_or_ambiguous_error"})
    assert _structured_status_is_authorized("succeeded", pi_authority)
    assert not _structured_status_is_authorized("failed", pi_authority)


def test_pi_success_only_authority_projects_only_confirmed_success(tmp_path: Path) -> None:
    snapshot = _snapshot(
        tmp_path / "pi-success-only.sqlite3",
        CaptureProfile.PI_EXTENSION_V1,
        (
            {"kind": "session_started"},
            {"kind": "action_started", "call": "one", "identity_authority": "coarse"},
            {
                "kind": "action_finished",
                "call": "one",
                "outcome_status": "succeeded",
            },
            {"kind": "session_finished"},
        ),
    )

    normalized = normalize_capture_session_snapshot(
        snapshot,
        installation_key=INSTALLATION_KEY,
    )
    result = next(
        event for event in normalized.events if event.event_type is EventType.TOOL_COMPLETION
    )
    outcome = result.payload["tool_outcome"]
    assert isinstance(outcome, Mapping)
    assert outcome["status"] == "succeeded"
    assert outcome["error_code"] is None
    assert outcome["exit_status"] is None
    assert outcome["failure_signature"] is None
    assert normalized.detector_evidence == ()


@pytest.mark.parametrize(
    ("profile_id", "expected_error_code"),
    (
        (CaptureProfile.CLAUDE_CODE_HOOKS_V1, "provider_error"),
        (CaptureProfile.OPENCODE_PLUGIN_V1, "tool_error"),
    ),
)
def test_structured_failures_keep_only_profile_derived_generic_details(
    tmp_path: Path,
    profile_id: CaptureProfile,
    expected_error_code: str,
) -> None:
    context = CaptureDigestContext(INSTALLATION_KEY)
    unsupported_signature = context.failure_signature(b"unsupported-result-detail")
    snapshot = _snapshot(
        tmp_path / f"generic-{profile_id.name.lower()}.sqlite3",
        profile_id,
        (
            {"kind": "session_started"},
            {"kind": "action_started", "call": "one"},
            {
                "kind": "action_finished",
                "call": "one",
                "changes": {
                    "exit_status": 23,
                    "error_code": "timeout",
                    "failure_signature": unsupported_signature,
                },
            },
            {"kind": "session_finished"},
        ),
    )

    normalized = normalize_capture_session_snapshot(
        snapshot,
        installation_key=INSTALLATION_KEY,
    )
    result = next(
        event for event in normalized.events if event.event_type is EventType.TOOL_COMPLETION
    )
    outcome = result.payload["tool_outcome"]

    assert isinstance(outcome, Mapping)
    assert outcome["status"] == "failed"
    assert outcome["exit_status"] is None
    assert outcome["error_code"] == expected_error_code
    assert outcome["failure_signature"] is None
    assert unsupported_signature not in canonical_json(result).decode("utf-8")


def test_unsupported_failure_details_cannot_manufacture_repeated_failure(
    tmp_path: Path,
) -> None:
    context = CaptureDigestContext(INSTALLATION_KEY)
    shared_signature = context.failure_signature(b"shared-unsupported-detail")
    snapshot = _snapshot(
        tmp_path / "unsupported-repeated-failure.sqlite3",
        CaptureProfile.CLAUDE_CODE_HOOKS_V1,
        (
            {"kind": "session_started"},
            {"kind": "action_started", "call": "one", "action": "one"},
            {
                "kind": "action_finished",
                "call": "one",
                "changes": {
                    "exit_status": 17,
                    "error_code": "timeout",
                    "failure_signature": shared_signature,
                },
            },
            {"kind": "action_started", "call": "two", "action": "two"},
            {
                "kind": "action_finished",
                "call": "two",
                "changes": {
                    "exit_status": 17,
                    "error_code": "timeout",
                    "failure_signature": shared_signature,
                },
            },
            {"kind": "session_finished"},
        ),
    )

    normalized = normalize_capture_session_snapshot(
        snapshot,
        installation_key=INSTALLATION_KEY,
    )

    assert all(
        signal.signal_type is not SignalType.REPEATED_FAILURE
        for report in normalized.extraction_reports
        for signal in report.signals
    )
    assert shared_signature not in canonical_json(normalized.events).decode("utf-8")


@pytest.mark.parametrize(
    ("signal_type", "minimum"),
    (
        (SignalType.REPEATED_ACTION, 2),
        (SignalType.REPEATED_FAILURE, 2),
        (SignalType.TOOL_ERROR, 1),
        (SignalType.TEST_FAILURE, 1),
    ),
)
def test_capture_detector_minimum_is_a_single_closed_contract(
    signal_type: SignalType,
    minimum: int,
) -> None:
    assert _capture_detector_minimum(signal_type) == minimum


def test_capture_detector_minimum_rejects_uninstalled_detectors() -> None:
    with pytest.raises(ValueError, match="capture normalization failed"):
        _capture_detector_minimum(SignalType.CONTEXT_SHIFT)


@pytest.mark.parametrize(
    ("profile_id", "expected"),
    (
        (CaptureProfile.CODEX_HOOKS_V1, (SignalType.REPEATED_ACTION,)),
        (
            CaptureProfile.CLAUDE_CODE_HOOKS_V1,
            (SignalType.TOOL_ERROR,),
        ),
        (
            CaptureProfile.OPENCODE_PLUGIN_V1,
            (SignalType.REPEATED_ACTION, SignalType.TOOL_ERROR),
        ),
        (
            CaptureProfile.PI_EXTENSION_V1,
            (),
        ),
    ),
)
def test_only_manifest_selected_detectors_are_evaluated(
    tmp_path: Path,
    profile_id: CaptureProfile,
    expected: tuple[SignalType, ...],
) -> None:
    snapshot = _snapshot(
        tmp_path / f"{profile_id.name.lower()}.sqlite3",
        profile_id,
        (
            {"kind": "session_started"},
            {"kind": "action_started", "call": "one", "action": "same"},
            {"kind": "action_started", "call": "two", "action": "same"},
            {"kind": "session_finished"},
        ),
    )

    normalized = normalize_capture_session_snapshot(
        snapshot,
        installation_key=INSTALLATION_KEY,
    )

    assert tuple(item.signal_type for item in normalized.detector_evidence) == expected
    assert all(
        tuple(item.signal_type for item in report.evaluations) == expected
        for report in normalized.extraction_reports
    )
    assert all(
        signal.signal_type in expected
        for report in normalized.extraction_reports
        for signal in report.signals
    )
    assert all(
        item.signal_type is not SignalType.TEST_FAILURE
        for report in normalized.extraction_reports
        for item in report.evaluations
    )
    repeated = normalized.extraction_reports[-2]
    repeated_action_detected = any(
        item.signal_type is SignalType.REPEATED_ACTION
        and item.outcome.status is DetectionStatus.DETECTED
        for item in repeated.evaluations
    )
    assert repeated_action_detected is (SignalType.REPEATED_ACTION in expected)


def test_projection_uses_fixed_logical_order_and_is_byte_stable(tmp_path: Path) -> None:
    snapshot = _snapshot(
        tmp_path / "stable.sqlite3",
        CaptureProfile.CODEX_HOOKS_V1,
        (
            {"kind": "session_started"},
            {"kind": "turn_finished", "call": "ignored"},
            {"kind": "action_started", "call": "one"},
            {"kind": "session_finished"},
        ),
    )

    first = normalize_capture_session_snapshot(
        snapshot,
        installation_key=INSTALLATION_KEY,
    )
    second = normalize_capture_session_snapshot(
        snapshot,
        installation_key=INSTALLATION_KEY,
    )
    verified = verify_capture_normalization(
        first,
        snapshot=snapshot,
        installation_key=INSTALLATION_KEY,
    )

    assert first is not second
    assert verified is not first
    assert first.run_id == second.run_id == verified.run_id
    assert first.normalization_digest == second.normalization_digest
    assert canonical_json(first.model_dump(mode="json")) == canonical_json(
        second.model_dump(mode="json")
    )
    assert tuple(event.sequence for event in first.events) == (1, 2, 3)
    assert tuple(event.timestamp.isoformat() for event in first.events) == (
        "2000-01-01T00:00:00.000001+00:00",
        "2000-01-01T00:00:00.000003+00:00",
        "2000-01-01T00:00:00.000004+00:00",
    )
    sensitive = {
        snapshot.connection_id,
        snapshot.session_id,
        snapshot.capability_manifest_digest,
        *(item.event.intake.producer_event_digest for item in snapshot.events),
    }
    assert all(
        all(value not in event.source_event_id for value in sensitive) for event in first.events
    )


def test_empty_quarantined_snapshot_normalizes_without_fabricated_trace(
    tmp_path: Path,
) -> None:
    with initialized_store(tmp_path / "quarantined.sqlite3") as store:
        register_connection(store)
        original = authenticated_intake(
            "session_started",
            session_native=b"normalization-original",
            producer_index=50,
        )
        collision = authenticated_intake(
            "session_started",
            session_native=b"normalization-collision",
            producer_index=50,
        )
        store.append(original)
        store.append(collision)
        snapshot = store.snapshot_session(CONNECTION_ID, collision.session_id)

    normalized = normalize_capture_session_snapshot(
        snapshot,
        installation_key=INSTALLATION_KEY,
    )

    assert normalized.semantic_coherence is False
    assert normalized.shadow_trace is None
    assert normalized.events == ()
    assert normalized.extraction_reports == ()
    assert normalized.counts.source_event_count == 0
    assert normalized.counts.mapped_event_count == 0
    assert _diagnostic_codes(normalized) == (
        CaptureNormalizationDiagnosticCode.SESSION_QUARANTINED,
    )


def test_normalization_verification_rejects_forged_or_wrong_key_values(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(
        tmp_path / "verify.sqlite3",
        CaptureProfile.CODEX_HOOKS_V1,
        ({"kind": "session_started"},),
    )
    normalized = normalize_capture_session_snapshot(
        snapshot,
        installation_key=INSTALLATION_KEY,
    )
    forged = normalized.model_copy(update={"semantic_coherence": False})

    with pytest.raises(ValueError, match="capture normalization failed"):
        verify_capture_normalization(
            forged,
            snapshot=snapshot,
            installation_key=INSTALLATION_KEY,
        )
    with pytest.raises(ValueError, match="capture normalization failed"):
        verify_capture_normalization(
            normalized,
            snapshot=snapshot,
            installation_key=InstallationKey(b"w" * 32),
        )
