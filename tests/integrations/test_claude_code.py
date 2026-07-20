from __future__ import annotations

import builtins
import json
import os
import signal
import socket
import subprocess
import sys
import time
import tomllib
from contextlib import suppress
from importlib import resources
from io import BytesIO
from pathlib import Path, PureWindowsPath

import pytest

import saliencegate.integrations.claude_code as claude_code_integration
from saliencegate.capture.capabilities import (
    CaptureProfile,
    CompatibilityStatus,
    capture_capability_digest,
    capture_profile,
)
from saliencegate.capture.identities import CaptureDigestContext
from saliencegate.capture.locations import resolve_capture_store_locations
from saliencegate.capture.migrations import initialize_capture_store
from saliencegate.capture.normalization import normalize_capture_session_snapshot
from saliencegate.capture.publication import verify_capture_intake_authentication
from saliencegate.capture.report import (
    CaptureReportHeadline,
    CaptureReportLimit,
    build_capture_session_report,
)
from saliencegate.capture.schema import canonical_capture_intake
from saliencegate.capture.store import (
    CaptureAppendDisposition,
    CaptureConnectionState,
    CaptureSessionState,
    CaptureStore,
    CaptureStoreMode,
)
from saliencegate.commands.capture.common import CaptureCommandConfigurationError
from saliencegate.commands.capture.connect import run_connect
from saliencegate.commands.capture.disconnect import run_disconnect
from saliencegate.commands.capture.status import (
    CaptureOperationalStatus,
    CaptureStatusDrift,
    run_status,
)
from saliencegate.domain import canonical_json
from saliencegate.integrations.claude_code import (
    CLAUDE_CODE_CONFIG_MARKER,
    CLAUDE_CODE_HOOK_EVENTS,
    CLAUDE_CODE_HOST_VERSION,
    ClaudeCodeCaptureAdapter,
    ClaudeCodeIntegrationError,
    ClaudeCodeVersionProbe,
    probe_claude_code_environment,
    probe_claude_code_version,
    provider_installation_spec,
)
from saliencegate.integrations.config_files import (
    ConfigFileError,
    ConfigSyntax,
    plan_owned_config_install,
    remove_owned_config_edit,
)
from saliencegate.integrations.hook import run_capture_hook
from saliencegate.integrations.installation import derive_installation_identity
from saliencegate.integrations.registry import ProviderInstallationKind
from saliencegate.security import InstallationKey, load_installation_key

CONNECTION_ID = "sg-" + "2" * 48
KEY = InstallationKey(b"1" * 32)
PROFILE = CaptureProfile.CLAUDE_CODE_HOOKS_V1


def _fixture_events() -> list[dict[str, object]]:
    source = (
        resources.files("saliencegate.integrations")
        .joinpath("fixtures")
        .joinpath("claude-code-hooks-v1.json")
        .read_bytes()
    )
    body = json.loads(source)
    assert type(body) is dict
    events = body["events"]
    assert type(events) is list
    return events


def _payload(event_name: str) -> dict[str, object]:
    for event in _fixture_events():
        if event["event_name"] == event_name:
            payload = event["payload"]
            assert type(payload) is dict
            return payload
    raise AssertionError(f"missing synthetic event {event_name}")


def _adapter(host_version: str = CLAUDE_CODE_HOST_VERSION) -> ClaudeCodeCaptureAdapter:
    return ClaudeCodeCaptureAdapter(
        connection_id=CONNECTION_ID,
        host_version=host_version,
    )


def _adapt(
    event_name: str,
    *,
    changes: dict[str, object] | None = None,
    adapter: ClaudeCodeCaptureAdapter | None = None,
):
    payload = _payload(event_name)
    if changes:
        payload.update(changes)
    selected = _adapter() if adapter is None else adapter
    return selected.adapt_bytes(canonical_json(payload), context=CaptureDigestContext(KEY))


def _replace_path(payload: dict[str, object], path: str, value: object) -> None:
    head, *tail = path.split(".", maxsplit=1)
    remainder = tail[0] if tail else None
    if head.endswith("[]"):
        nested = payload[head[:-2]]
        assert type(nested) is list and nested
        for item in nested:
            assert type(item) is dict
            if remainder is None:
                raise AssertionError("array member path must name a field")
            _replace_path(item, remainder, value)
        return
    if remainder is None:
        payload[head] = value
        return
    nested = payload[head]
    assert type(nested) is dict
    _replace_path(nested, remainder, value)


def _capture_executable() -> Path:
    return (Path(sys.executable).parent / "saliencegate-capture-hook").resolve(strict=True)


def _fake_claude(tmp_path: Path, version: str = CLAUDE_CODE_HOST_VERSION) -> Path:
    provider_bin = tmp_path / "provider-bin"
    provider_bin.mkdir(exist_ok=True)
    executable = provider_bin / "claude"
    executable.write_bytes(f"#!/bin/sh\nprintf '{version} (Claude Code)\\n'\n".encode())
    executable.chmod(0o700)
    return executable


def _environment(tmp_path: Path, executable: Path) -> dict[str, str]:
    return {
        "ANTHROPIC_API_KEY": "poisoned-anthropic-secret",
        "HOME": str(tmp_path / "home"),
        "PATH": str(executable.parent),
        "XDG_CONFIG_HOME": str(tmp_path / "configuration"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }


def test_capability_declaration_is_exact_and_documents_incomplete_coverage() -> None:
    declaration = _adapter().capabilities()
    manifest = capture_profile(PROFILE)

    assert declaration.profile_id is PROFILE
    assert declaration.host_version == "2.1.204"
    assert declaration.capability_digest == capture_capability_digest(manifest)
    assert manifest.complete_execution_session_coverage is False
    assert manifest.transcript_read is False
    assert manifest.raw_content_persisted is False
    assert manifest.decision_authority is False
    assert manifest.model_calls == 0
    assert manifest.coverage_exclusions == (
        "background_tasks",
        "coexisting_hooks_can_block_or_rewrite_pre_tool_use",
        "host_rejected_foreign_settings_layer",
        "permission_denials_outside_auto_mode",
        "prompt_and_output_content",
        "resumable_sessions_remain_open",
        "runtime_hook_disablement",
        "session_crons",
        "stop_and_subagent_stop_callbacks_can_continue",
        "stop_failures_before_first_prompt",
    )


def test_all_audited_events_map_with_explicit_outcome_authority() -> None:
    context = CaptureDigestContext(KEY)
    expected = {
        "SessionStart": ("session_started",),
        "PreToolUse": ("action_started",),
        "PostToolUse": ("action_finished",),
        "PostToolUseFailure": ("action_finished",),
        "PostToolBatch": (),
        "PermissionDenied": ("permission_denied",),
        "SubagentStart": ("subagent_started",),
        "SubagentStop": (),
        "Stop": (),
        "StopFailure": ("controller_failed",),
        "SessionEnd": (),
    }

    observed: set[str] = set()
    encoded: list[bytes] = []
    for event in _fixture_events():
        event_name = event["event_name"]
        payload = event["payload"]
        assert type(event_name) is str
        assert type(payload) is dict
        intakes = _adapter().adapt_bytes(canonical_json(payload), context=context)
        assert tuple(item.kind for item in intakes) == expected[event_name]
        for intake in intakes:
            assert verify_capture_intake_authentication(intake, context=context) == intake
            assert intake.capture_disposition == "captured"
            assert intake.occurred_at is None
            assert intake.timestamp_authority == "unavailable"
            assert intake.producer_sequence is None
            assert intake.sequence_authority == "unavailable"
            encoded.append(canonical_capture_intake(intake))
        observed.add(event_name)

    assert observed == set(expected)
    persisted = b"\n".join(encoded)
    assert b"synthetic-" not in persisted
    assert b"/synthetic/" not in persisted


def test_success_failure_permission_and_batch_authorities_are_closed() -> None:
    started = _adapt("PreToolUse")[0]
    succeeded = _adapt("PostToolUse")[0]
    failed = _adapt("PostToolUseFailure")[0]
    interrupted = _adapt("PostToolUseFailure", changes={"is_interrupt": True})[0]
    denied = _adapt("PermissionDenied")[0]

    assert started.call_ref == succeeded.call_ref
    assert started.tool_class == "file_read"
    assert started.identity_authority == "coarse"
    assert b"Read" not in canonical_capture_intake(started)
    assert succeeded.outcome_status == "succeeded"
    assert succeeded.outcome_authority == "producer_claimed_structured"
    assert succeeded.exit_status is None
    assert succeeded.error_code is None
    assert succeeded.failure_signature is None

    assert failed.outcome_status == "failed"
    assert failed.outcome_authority == "producer_claimed_structured"
    assert failed.error_code == "tool_error"
    assert failed.failure_signature is None
    assert interrupted.call_ref == failed.call_ref
    assert interrupted.error_code == "interrupted"

    assert denied.kind == "permission_denied"
    assert denied.call_ref != failed.call_ref
    assert _adapt("PostToolBatch") == ()
    assert _adapt("PostToolBatch") == ()
    controller = _adapt("StopFailure")[0]
    assert controller.error_code == "provider_callback_failed"
    assert controller.failure_signature is None
    with pytest.raises(ClaudeCodeIntegrationError):
        _adapt("PostToolUseFailure", changes={"is_interrupt": "true"})


def test_batch_requires_every_nested_tool_identifier_without_admitting_content() -> None:
    payload = _payload("PostToolBatch")
    calls = payload["tool_calls"]
    assert type(calls) is list and type(calls[0]) is dict
    calls[0].pop("tool_use_id")

    with pytest.raises(ClaudeCodeIntegrationError):
        _adapter().adapt_bytes(canonical_json(payload), context=CaptureDigestContext(KEY))

    assert (
        _adapt(
            "PostToolBatch",
            changes={
                "tool_calls": [
                    {
                        "tool_use_id": "synthetic-claude-call-1",
                        "tool_name": "Read",
                        "tool_input": {"prompt": "synthetic-ignored-prompt"},
                        "tool_response": "synthetic-ignored-output",
                    }
                ]
            },
        )
        == ()
    )

    forged = (
        {
            "hook_event_name": "PostToolBatch",
            "session_id": "synthetic-claude-session",
            "tool_calls[].tool_use_id": "forged",
        },
        {
            "hook_event_name": "PostToolBatch",
            "session_id": "synthetic-claude-session",
            "tool_calls": [],
            "tool_calls[].tool_use_id": "forged",
        },
        {
            "hook_event_name": "PostToolBatch",
            "session_id": "synthetic-claude-session",
            "tool_calls": [
                {"tool_use_id": "duplicate"},
                {"tool_use_id": "duplicate"},
            ],
            "tool_calls[].tool_use_id": "forged",
        },
    )
    for payload in forged:
        with pytest.raises(ClaudeCodeIntegrationError):
            _adapter().adapt_bytes(canonical_json(payload), context=CaptureDigestContext(KEY))


def test_prompt_output_response_message_reason_error_and_transcript_are_excluded() -> None:
    baselines = {
        event_name: _adapt(event_name)
        for event_name in (
            "PostToolUse",
            "PostToolUseFailure",
            "PermissionDenied",
            "SubagentStop",
            "Stop",
            "StopFailure",
            "SessionEnd",
        )
    }
    changes = {
        "PostToolUse": {
            "prompt": "changed-secret-prompt",
            "output": "changed-secret-output",
            "tool_response": {"content": "changed-secret-response"},
        },
        "PostToolUseFailure": {
            "error": "changed-secret-error",
            "duration_ms": 987654,
        },
        "PermissionDenied": {"reason": "changed-secret-reason"},
        "SubagentStop": {
            "last_assistant_message": "changed-secret-assistant-message",
            "agent_transcript_path": "/changed/secret/subagent.jsonl",
        },
        "Stop": {
            "last_assistant_message": "changed-secret-assistant-message",
            "background_tasks": ["changed-secret-task"],
            "session_crons": ["changed-secret-cron"],
        },
        "StopFailure": {
            "error": "rate_limit",
            "error_details": {"message": "changed-secret-detail"},
            "last_assistant_message": "changed-secret-assistant-message",
        },
        "SessionEnd": {
            "reason": "logout",
            "transcript_path": "/changed/secret/session.jsonl",
        },
    }

    for event_name, event_changes in changes.items():
        changed = _adapt(event_name, changes=event_changes)
        assert changed == baselines[event_name]
        assert b"changed-secret" not in b"".join(map(canonical_capture_intake, changed))


def test_every_declared_ignored_field_is_authority_free() -> None:
    for event in capture_profile(PROFILE).events:
        baseline = _adapt(event.event_name)
        for field_name in event.ignored_fields:
            payload = _payload(event.event_name)
            _replace_path(payload, field_name, f"changed-ignored-{field_name}")
            changed = _adapter().adapt_bytes(
                canonical_json(payload),
                context=CaptureDigestContext(KEY),
            )
            assert changed == baseline


def test_every_declared_optional_field_changes_admitted_evidence() -> None:
    manifest = capture_profile(PROFILE)
    assert {event.event_name: event.optional_fields for event in manifest.events} == {
        "SessionStart": (),
        "PreToolUse": ("cwd", "tool_name"),
        "PostToolUse": (),
        "PostToolUseFailure": ("is_interrupt",),
        "PostToolBatch": (),
        "PermissionDenied": (),
        "SubagentStart": (),
        "SubagentStop": (),
        "Stop": (),
        "StopFailure": (),
        "SessionEnd": (),
    }

    started = _adapt("PreToolUse")[0]
    assert _adapt("PreToolUse", changes={"cwd": "/changed/workspace"})[0] != started
    assert _adapt("PreToolUse", changes={"tool_input": {"path": "changed"}})[0] == started
    assert _adapt("PreToolUse", changes={"tool_name": "Bash"})[0] != started
    assert (
        _adapt("PostToolUseFailure", changes={"is_interrupt": True})[0]
        != _adapt("PostToolUseFailure")[0]
    )
    with pytest.raises(ClaudeCodeIntegrationError):
        _adapt("StopFailure", changes={"prompt_id": None})


def test_tool_terminal_callbacks_share_one_collision_domain_per_native_parent() -> None:
    succeeded = _adapt("PostToolUse")[0]
    failed = _adapt(
        "PostToolUseFailure",
        changes={"tool_use_id": "synthetic-claude-call-1"},
    )[0]
    denied = _adapt(
        "PermissionDenied",
        changes={"tool_use_id": "synthetic-claude-call-1"},
    )[0]
    stop_failed = _adapt(
        "StopFailure",
        changes={"prompt_id": "synthetic-claude-prompt-1"},
    )[0]

    assert succeeded.call_ref == failed.call_ref == denied.call_ref
    assert (
        succeeded.producer_event_digest
        == failed.producer_event_digest
        == denied.producer_event_digest
    )
    assert _adapt("Stop") == ()
    assert stop_failed.producer_event_digest != succeeded.producer_event_digest


def test_call_and_subagent_correlations_and_duplicates_are_session_bound() -> None:
    started = _adapt("PreToolUse")[0]
    replay = _adapt("PreToolUse")[0]
    succeeded = _adapt("PostToolUse")[0]
    conflicting = _adapt(
        "PreToolUse",
        changes={"tool_name": "Bash"},
    )[0]
    subagent_started = _adapt("SubagentStart")[0]

    assert replay == started
    assert succeeded.call_ref == started.call_ref
    assert conflicting.call_ref == started.call_ref
    assert conflicting.producer_event_digest == started.producer_event_digest
    assert conflicting.action_digest != started.action_digest
    assert _adapt("SubagentStop") == ()
    assert _adapt("Stop") == ()

    other_session = _adapt(
        "SubagentStart",
        changes={"session_id": "different-synthetic-session"},
    )[0]
    assert other_session.subagent_id != subagent_started.subagent_id


def test_resumable_session_and_attempt_callbacks_never_close_observed_window(
    tmp_path: Path,
) -> None:
    database = tmp_path / "capture.db"
    initialize_capture_store(database)
    manifest = capture_profile(PROFILE)
    started = _adapt("SessionStart")[0]
    resumed = _adapt("SessionStart", changes={"source": "resume"})[0]
    continued = _adapt(
        "PreToolUse",
        changes={
            "prompt_id": "synthetic-claude-prompt-after-resume",
            "tool_use_id": "synthetic-claude-call-after-resume",
        },
    )[0]

    assert resumed == started
    assert _adapt("SessionEnd", changes={"reason": "resume"}) == ()
    assert _adapt("Stop") == _adapt("Stop") == ()
    assert _adapt("SubagentStop") == _adapt("SubagentStop") == ()

    with CaptureStore.open(
        database,
        installation_key=KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        store.register_connection(
            connection_id=CONNECTION_ID,
            project_digest="8" * 64,
            profile_id=PROFILE,
            capability_manifest_digest=capture_capability_digest(manifest),
            host_version=CLAUDE_CODE_HOST_VERSION,
        )
        store.transition_connection(
            CONNECTION_ID,
            expected_state=CaptureConnectionState.PENDING,
            target_state=CaptureConnectionState.ENABLED,
        )
        assert store.append(started).disposition is CaptureAppendDisposition.ADMITTED
        assert store.append(resumed).disposition is CaptureAppendDisposition.REPLAYED
        assert store.append(continued).disposition is CaptureAppendDisposition.ADMITTED
        snapshot = store.snapshot_session(CONNECTION_ID, started.session_id)

    assert snapshot.state is CaptureSessionState.OPEN
    assert snapshot.event_count == 2


def test_fixture_persists_open_session_and_reports_outcome_authority(tmp_path: Path) -> None:
    database = tmp_path / "capture.db"
    initialize_capture_store(database)
    context = CaptureDigestContext(KEY)
    manifest = capture_profile(PROFILE)

    with CaptureStore.open(
        database,
        installation_key=KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        store.register_connection(
            connection_id=CONNECTION_ID,
            project_digest="8" * 64,
            profile_id=PROFILE,
            capability_manifest_digest=capture_capability_digest(manifest),
            host_version=CLAUDE_CODE_HOST_VERSION,
        )
        store.transition_connection(
            CONNECTION_ID,
            expected_state=CaptureConnectionState.PENDING,
            target_state=CaptureConnectionState.ENABLED,
        )
        admitted: list[object] = []
        for event in _fixture_events():
            event_name = event["event_name"]
            if event_name in {"PostToolUseFailure", "PermissionDenied"}:
                terminal = event["payload"]
                assert type(terminal) is dict
                parent = _payload("PreToolUse")
                parent.update(
                    tool_use_id=terminal["tool_use_id"],
                    tool_name=terminal["tool_name"],
                    tool_input=terminal["tool_input"],
                )
                admitted.extend(_adapter().adapt_bytes(canonical_json(parent), context=context))
            admitted.extend(
                _adapter().adapt_bytes(canonical_json(event["payload"]), context=context)
            )
        intakes = tuple(admitted)
        assert len(intakes) == 9
        for intake in intakes:
            store.append(intake)
        snapshot = store.snapshot_session(CONNECTION_ID, intakes[0].session_id)

    normalization = normalize_capture_session_snapshot(snapshot, installation_key=KEY)
    report = build_capture_session_report(
        snapshot,
        normalization,
        installation_key=KEY,
        spool=None,
    )
    assert snapshot.state is CaptureSessionState.OPEN
    assert report.session_state is CaptureSessionState.OPEN
    assert report.headline is CaptureReportHeadline.MEMORY_REVIEW_SUGGESTED
    assert CaptureReportLimit.SESSION_OPEN in report.coverage.limits
    assert report.counts.captured_events == 9
    assert report.counts.structured_results == 3
    assert normalization.counts.authorized_tool_result_count == 3
    assert normalization.counts.classifiable_failed_result_count == 2
    assert normalization.counts.authorized_controller_error_count == 1
    assert report.coverage.capability_exclusions == manifest.coverage_exclusions
    assert any(event.outcome_authority == "provider_claimed_success" for event in manifest.events)
    assert any(event.outcome_authority == "provider_claimed_failure" for event in manifest.events)
    assert any(event.outcome_authority == "provider_claimed_denial" for event in manifest.events)


def test_transcript_paths_are_never_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    original_open = Path.open
    original_read_bytes = Path.read_bytes

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if "synthetic/not-read" in str(path):
            raise AssertionError("excluded provider path was read")
        return original_open(path, *args, **kwargs)

    def guarded_read_bytes(path: Path) -> bytes:
        if "synthetic/not-read" in str(path):
            raise AssertionError("excluded provider path was read")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    for event in _fixture_events():
        _adapter().adapt_bytes(
            canonical_json(event["payload"]),
            context=CaptureDigestContext(KEY),
        )


@pytest.mark.parametrize(
    "payload",
    (
        {},
        {"hook_event_name": "Unknown", "session_id": "session"},
        {"hook_event_name": "SessionStart"},
        {"hook_event_name": 7, "session_id": "session"},
        {"hook_event_name": "PreToolUse", "session_id": "session"},
        {"hook_event_name": "PostToolBatch", "session_id": "session", "tool_calls": []},
    ),
)
def test_invalid_or_unknown_event_shapes_fail_closed_without_content(payload: object) -> None:
    secret = "claude-invalid-shape-secret"
    document = {**payload, "future": secret} if isinstance(payload, dict) else payload

    with pytest.raises(ClaudeCodeIntegrationError) as captured:
        _adapter().adapt_bytes(canonical_json(document), context=CaptureDigestContext(KEY))

    rendered = f"{captured.value!s}\n{captured.value!r}"
    assert secret not in rendered
    assert "session" not in rendered


def test_duplicate_keys_and_oversize_native_input_fail_content_free() -> None:
    for source in (
        b'{"hook_event_name":"SessionStart","session_id":"one","session_id":"two"}',
        b"{" + b'"future":"' + b"x" * (2 * 1024 * 1024) + b'"}',
    ):
        with pytest.raises(ClaudeCodeIntegrationError) as captured:
            _adapter().adapt_bytes(source, context=CaptureDigestContext(KEY))
        assert "one" not in str(captured.value)
        assert "two" not in repr(captured.value)


def test_version_probe_is_optional_bounded_and_exact(tmp_path: Path) -> None:
    executable = tmp_path / "claude"
    executable.write_bytes(b"synthetic")
    executable.chmod(0o700)
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def runner(command: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, b"2.1.204 (Claude Code)\n", b"")

    environment = {"PATH": str(tmp_path), "ANTHROPIC_API_KEY": "poisoned-secret"}
    probe = probe_claude_code_version(executable, runner=runner, environ=environment)

    assert probe == ClaudeCodeVersionProbe(
        host_version="2.1.204",
        compatibility=CompatibilityStatus.VERIFIED,
    )
    assert calls == [
        (
            (str(executable), "--version"),
            {
                "input": b"",
                "capture_output": True,
                "check": False,
                "timeout": 2.0,
                "env": environment,
                "cwd": str(executable.parent),
            },
        )
    ]


def test_version_probe_accepts_new_patch_and_rejects_other_or_malformed_versions(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "claude"
    executable.write_bytes(b"synthetic")
    executable.chmod(0o700)

    def completed(stdout: bytes, *, returncode: int = 0):
        def run(command: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
            return subprocess.CompletedProcess(command, returncode, stdout, b"")

        return run

    newer = probe_claude_code_version(
        executable,
        runner=completed(b"2.1.205 (Claude Code)\n"),
    )
    assert newer.compatibility is CompatibilityStatus.SCHEMA_COMPATIBLE_UNVERIFIED_VERSION

    invalid = (
        completed(b"Claude secret version\n"),
        completed(b"2.1.203 (Claude Code)\n"),
        completed(b"2.2.0 (Claude Code)\n"),
        completed(b"3.0.0 (Claude Code)\n"),
        completed(b"2.1.204 (Claude Code)\n", returncode=1),
        completed(b"x" * 4097),
    )
    for runner in invalid:
        with pytest.raises(ClaudeCodeIntegrationError) as captured:
            probe_claude_code_version(executable, runner=runner)
        assert "secret" not in str(captured.value)


@pytest.mark.skipif(os.name != "posix", reason="POSIX executable contract")
def test_environment_probe_resolves_the_normal_npm_bin_symlink(tmp_path: Path) -> None:
    package_executable = (
        tmp_path / "lib" / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
    )
    package_executable.parent.mkdir(parents=True)
    package_executable.write_bytes(b"#!/bin/sh\nprintf '2.1.204 (Claude Code)\\n'\n")
    package_executable.chmod(0o700)
    npm_bin = tmp_path / "bin"
    npm_bin.mkdir()
    npm_link = npm_bin / "claude"
    npm_link.symlink_to(os.path.relpath(package_executable, npm_bin))

    probe = probe_claude_code_environment(environ={"PATH": str(npm_bin)})

    assert probe.compatibility is CompatibilityStatus.VERIFIED


def test_windows_executable_candidates_use_only_explicit_path_and_pathext() -> None:
    current = PureWindowsPath(r"C:\untrusted-project")

    candidates = claude_code_integration._claude_executable_candidates(
        r"C:\Program Files\Claude;D:\Provider",
        windows_pathext=".CMD;.EXE",
        windows=True,
        cwd=current,
    )

    assert candidates == (
        PureWindowsPath(r"C:\Program Files\Claude\claude.CMD"),
        PureWindowsPath(r"C:\Program Files\Claude\claude.EXE"),
        PureWindowsPath(r"D:\Provider\claude.CMD"),
        PureWindowsPath(r"D:\Provider\claude.EXE"),
    )
    assert all(not candidate.is_relative_to(current) for candidate in candidates)

    explicit_current = claude_code_integration._claude_executable_candidates(
        r";C:\Trusted",
        windows_pathext=".CMD",
        windows=True,
        cwd=current,
    )
    assert explicit_current[0] == current / "claude.CMD"


@pytest.mark.skipif(os.name != "posix", reason="POSIX explicit-PATH contract")
def test_environment_probe_never_prepends_the_current_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    local_collision = project / "claude"
    local_collision.write_bytes(b"#!/bin/sh\nprintf '9.9.9 (Claude Code)\\n'\n")
    local_collision.chmod(0o700)
    trusted = _fake_claude(tmp_path)
    monkeypatch.chdir(project)

    probe = probe_claude_code_environment(environ={"PATH": str(trusted.parent)})

    assert probe.compatibility is CompatibilityStatus.VERIFIED


@pytest.mark.skipif(os.name != "posix", reason="POSIX bounded-child contract")
def test_real_version_probe_accepts_crlf_and_bounds_hangs_and_dual_streams(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "claude"
    executable.write_bytes(b"#!/bin/sh\nprintf '2.1.204 (Claude Code)\\r\\n'\n")
    executable.chmod(0o700)
    assert probe_claude_code_version(executable).compatibility is CompatibilityStatus.VERIFIED

    executable.write_bytes(b"#!/bin/sh\nsleep 30\n")
    started = time.monotonic()
    with pytest.raises(ClaudeCodeIntegrationError) as timeout_error:
        probe_claude_code_version(executable)
    assert time.monotonic() - started < 4.0
    assert str(timeout_error.value) == "Claude Code capture integration is invalid"

    executable.write_bytes(
        b"#!/bin/sh\n"
        b"i=0\n"
        b'while [ "$i" -lt 5000 ]; do\n'
        b"  printf x\n"
        b"  printf y >&2\n"
        b"  i=$((i + 1))\n"
        b"done\n"
    )
    started = time.monotonic()
    with pytest.raises(ClaudeCodeIntegrationError) as output_error:
        probe_claude_code_version(executable)
    assert time.monotonic() - started < 4.0
    assert str(output_error.value) == "Claude Code capture integration is invalid"


@pytest.mark.skipif(os.name != "posix", reason="POSIX bounded-child contract")
def test_real_version_probe_does_not_wait_on_a_detached_inherited_pipe(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "claude"
    pid_path = tmp_path / "detached.pid"
    executable.write_bytes(
        (
            f"#!{sys.executable}\n"
            "import pathlib, subprocess, sys\n"
            "child = subprocess.Popen(\n"
            "    [sys.executable, '-c', 'import time; time.sleep(30)'],\n"
            "    start_new_session=True,\n"
            ")\n"
            f"pathlib.Path({str(pid_path)!r}).write_text(str(child.pid))\n"
            "print('2.1.204 (Claude Code)')\n"
        ).encode()
    )
    executable.chmod(0o700)

    started = time.monotonic()
    try:
        with pytest.raises(ClaudeCodeIntegrationError):
            probe_claude_code_version(executable)
        assert time.monotonic() - started < 4.0
    finally:
        if pid_path.exists():
            with suppress(ProcessLookupError):
                os.kill(int(pid_path.read_text()), signal.SIGKILL)


def test_project_installation_merges_foreign_hooks_and_exactly_restores_json(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    spec = provider_installation_spec(
        project,
        environ={"HOME": str(tmp_path / "home"), "XDG_STATE_HOME": str(state)},
    )

    assert spec.installation_kind is ProviderInstallationKind.COMMAND_HOOK
    assert spec.project_local_paths == (project / ".claude" / "settings.local.json",)
    assert spec.config.syntax is ConfigSyntax.JSON_OBJECT
    assert spec.config.marker == CLAUDE_CODE_CONFIG_MARKER
    assert spec.bundle_path is None
    assert spec.bootstrap_path is None
    assert spec.receipt_path.parent == spec.launcher_path.parent
    assert spec.receipt_path.parent.is_relative_to(state / "saliencegate")
    assert not spec.receipt_path.is_relative_to(project)

    fragment = spec.config.owned_fragment
    assert fragment.count(CLAUDE_CODE_CONFIG_MARKER.encode("ascii")) == 1
    lowered = fragment.lower()
    assert b'"permissions"' not in lowered
    assert b"allow" not in lowered
    assert b"bypass" not in lowered
    assert str(spec.launcher_path).encode() in fragment

    foreign = (
        b'{\n  "model": "claude-sonnet-4-5",\n  "hooks": {'
        b'"Notification":[{"hooks":[{"type":"command","command":"foreign"}]}]},'
        b'\n  "permissions": {"allow": ["Read"]}\n}\n'
    )
    planned = plan_owned_config_install(foreign, spec.config)
    document = json.loads(planned.installed_bytes)
    assert set(document["hooks"]) == {"Notification", *CLAUDE_CODE_HOOK_EVENTS}
    assert document["hooks"]["Notification"][0]["hooks"][0]["command"] == "foreign"
    for event_name in CLAUDE_CODE_HOOK_EVENTS:
        groups = document["hooks"][event_name]
        assert type(groups) is list and len(groups) == 1
        handlers = groups[0]["hooks"]
        assert type(handlers) is list and len(handlers) == 1
        assert handlers[0]["type"] == "command"
        assert handlers[0]["timeout"] == 3
        assert type(handlers[0]["args"]) is list
    assert remove_owned_config_edit(planned.installed_bytes, planned.reverse_edit) == foreign

    for conflict in (b'{"hooks":7}', b'{"hooks":{"SessionStart":7}}'):
        with pytest.raises(ConfigFileError):
            plan_owned_config_install(conflict, spec.config)

    coexisting = {
        "hooks": {
            event_name: [{"hooks": [{"type": "command", "command": f"foreign-{event_name}"}]}]
            for event_name in CLAUDE_CODE_HOOK_EVENTS
        }
    }
    coexisting_bytes = canonical_json(coexisting)
    composed = plan_owned_config_install(coexisting_bytes, spec.config)
    composed_document = json.loads(composed.installed_bytes)
    assert len(composed.reverse_edit.additional_spans) == len(CLAUDE_CODE_HOOK_EVENTS) - 1
    assert composed.reverse_edit.json_path == ("hooks", "SessionStart")
    assert {span.json_path for span in composed.reverse_edit.additional_spans} == {
        ("hooks", event_name)
        for event_name in CLAUDE_CODE_HOOK_EVENTS
        if event_name != "SessionStart"
    }
    assert all(
        len(composed_document["hooks"][event_name]) == 2 for event_name in CLAUDE_CODE_HOOK_EVENTS
    )
    assert remove_owned_config_edit(composed.installed_bytes, composed.reverse_edit) == (
        coexisting_bytes
    )


def test_windows_exec_form_uses_absolute_powershell_not_a_cmd_command() -> None:
    launcher = PureWindowsPath(r"C:\Users\owner\State & Data\capture-hook.cmd")
    powershell = PureWindowsPath(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")
    fragment = claude_code_integration._claude_code_hook_fragment(
        launcher,
        windows_powershell=powershell,
    )
    document = json.loads(b"{" + fragment + b"}")

    assert fragment.count(CLAUDE_CODE_CONFIG_MARKER.encode("ascii")) == 1
    for event_name in CLAUDE_CODE_HOOK_EVENTS:
        handler = document["hooks"][event_name][0]["hooks"][0]
        assert handler["command"] == str(powershell)
        assert handler["command"].lower().endswith("powershell.exe")
        assert str(launcher) not in handler["args"]
        assert "-Command" in handler["args"]
        assert any(f"saliencegate-event:{event_name}" in value for value in handler["args"])
        assert handler["args"][-2] == "-Command"

    shim = PureWindowsPath(r"C:\Users\owner\npm\claude.cmd")
    version_command = claude_code_integration._windows_shim_version_command(shim, powershell)
    assert version_command[0] == str(powershell)
    assert str(shim) not in version_command
    assert version_command[-2] == "-Command"


@pytest.mark.skipif(os.name != "nt", reason="native Windows-only Claude exec smoke")
def test_native_windows_cmd_launcher_and_npm_shim_execute_through_powershell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    powershell = claude_code_integration._trusted_windows_powershell()
    launcher = tmp_path / "capture-hook.cmd"
    stdin_sentinel = tmp_path / "stdin-observed.txt"
    launcher.write_bytes(
        b"@echo off\r\n"
        b"set /p SG_LINE=\r\n"
        b'if defined SG_LINE >"%SG_SENTINEL%" echo observed\r\n'
        b"echo forbidden-output\r\n"
        b"exit /b 0\r\n"
    )
    fragment = claude_code_integration._claude_code_hook_fragment(
        launcher,
        windows_powershell=powershell,
    )
    document = json.loads(b"{" + fragment + b"}")
    handler = document["hooks"]["SessionStart"][0]["hooks"][0]
    environment = dict(os.environ)
    environment["SG_SENTINEL"] = str(stdin_sentinel)
    launched = subprocess.run(
        (handler["command"], *handler["args"]),
        input=b'{"synthetic":true}',
        capture_output=True,
        check=False,
        timeout=10,
        env=environment,
    )
    assert launched.returncode == 0
    assert launched.stdout == launched.stderr == b""
    assert stdin_sentinel.read_bytes().strip() == b"observed"

    shim = tmp_path / "claude.cmd"
    shim.write_bytes(b"@echo off\r\necho 2.1.204 ^(Claude Code^)\r\n")
    assert probe_claude_code_version(shim).compatibility is CompatibilityStatus.VERIFIED

    project_cwd = tmp_path / "project"
    provider_bin = tmp_path / "provider-bin"
    shim_bin = tmp_path / "npm-bin"
    project_cwd.mkdir()
    provider_bin.mkdir()
    shim_bin.mkdir()
    trusted_sentinel = tmp_path / "trusted-node.txt"
    malicious_sentinel = tmp_path / "malicious-node.txt"
    (provider_bin / "node.cmd").write_bytes(
        b'@echo off\r\n>"%SG_TRUSTED_NODE%" echo trusted\r\necho 2.1.204 ^(Claude Code^)\r\n'
    )
    (project_cwd / "node.cmd").write_bytes(
        b'@echo off\r\n>"%SG_MALICIOUS_NODE%" echo malicious\r\necho 9.9.9 ^(Claude Code^)\r\n'
    )
    npm_shim = shim_bin / "claude.cmd"
    npm_shim.write_bytes(b"@echo off\r\nnode %*\r\n")
    npm_environment = dict(os.environ)
    npm_environment["PATH"] = str(provider_bin) + os.pathsep + npm_environment["PATH"]
    npm_environment["PATHEXT"] = ".CMD;.EXE;.COM;.BAT"
    npm_environment["SG_TRUSTED_NODE"] = str(trusted_sentinel)
    npm_environment["SG_MALICIOUS_NODE"] = str(malicious_sentinel)
    monkeypatch.chdir(project_cwd)

    npm_probe = probe_claude_code_version(npm_shim, environ=npm_environment)

    assert npm_probe.compatibility is CompatibilityStatus.VERIFIED
    assert trusted_sentinel.exists()
    assert not malicious_sentinel.exists()

    survivor_sentinel = tmp_path / "survived.txt"
    hanging_shim = tmp_path / "claude-hanging.cmd"
    hanging_shim.write_bytes(
        (
            "@echo off\r\n"
            f'"{powershell}" -NoLogo -NoProfile -NonInteractive -Command '
            '"Start-Sleep -Seconds 5; Set-Content -LiteralPath '
            '$env:SG_SURVIVOR_SENTINEL -Value survived"\r\n'
        ).encode()
    )
    hanging_environment = dict(os.environ)
    hanging_environment["SG_SURVIVOR_SENTINEL"] = str(survivor_sentinel)
    started = time.monotonic()
    with pytest.raises(ClaudeCodeIntegrationError):
        probe_claude_code_version(hanging_shim, environ=hanging_environment)
    assert time.monotonic() - started < 4.0
    time.sleep(3.5)
    assert not survivor_sentinel.exists()


@pytest.mark.skipif(os.name != "posix", reason="native Windows lifecycle is covered by R01")
def test_dry_run_never_probes_claude_or_writes(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    executable = _fake_claude(tmp_path)
    executable.write_bytes(
        b'#!/bin/sh\n/bin/mkdir -p "$HOME"\nprintf probed > "$HOME/probed"\n'
        b"printf '2.1.204 (Claude Code)\\n'\n"
    )
    executable.chmod(0o700)
    environment = _environment(tmp_path, executable)

    report = run_connect(
        provider="claude-code",
        project=project,
        dry_run=True,
        environ=environment,
        capture_executable=_capture_executable(),
    )

    assert report.dry_run is True
    assert report.capture_enabled is False
    assert tuple(project.iterdir()) == ()
    assert not Path(environment["HOME"]).exists()
    assert not Path(environment["XDG_STATE_HOME"]).exists()


@pytest.mark.skipif(os.name != "posix", reason="native Windows lifecycle is covered by R01")
@pytest.mark.parametrize(
    "foreign",
    (
        b'{"disableAllHooks":true}\n',
        b'{"disableAllHooks":null}\n',
        b'{"hooks":null}\n',
        b'{"hooks":{"SessionStart":null}}\n',
        b'{"hooks":{"SessionStart":[42]}}\n',
        b'{"hooks":{"SessionStart":[{}]}}\n',
        b'{"hooks":{"SessionStart":[{"hooks":7}]}}\n',
        b'{"hooks":{"SessionStart":[{"hooks":[{"type":"command"}]}]}}\n',
        b'{"hooks":{"SessionStart":[{"hooks":[{"type":"unknown"}]}]}}\n',
        b'{"hooks":{"SessionStart":[{"hooks":[{"type":"http","url":"http:"}]}]}}\n',
        b'{"hooks":{"SessionStart":[{"hooks":[{"type":"command","command":"x",'
        b'"timeout":1e400}]}]}}\n',
        b'{"hooks":{"SessionStart":[{"hooks":[{"type":"command","command":"x",'
        b'"timeout":999999999999999999999999999999999999999999999999999999999999999999999'
        b"999999999999999999999999999999999999999999999999999999999999999999999999999"
        b"999999999999999999999999999999999999999999999999999999999999999999999999999"
        b"999999999999999999999999999999999999999999999999999999999999999999999999999"
        b"999999999999999999999999999999999999999999999999999999999999999999999999999"
        b"}]}]}}\n",
        b'{"hooks":{"SessionStart":[{"hooks":[{"type":"command","command":"x",'
        b'"rewakeMessage":""}]}]}}\n',
        b'{"hooks":{"SessionStart":[{"hooks":[{"type":"command","command":"x",'
        b'"timeout":null}]}]}}\n',
        b'{"hooks":{"SessionStart":[{"hooks":[{"type":"command","command":"x","args":null}]}]}}\n',
        b'{"hooks":{"SessionStart":[{"hooks":[{"type":"http","url":"https://example.com",'
        b'"headers":null}]}]}}\n',
        b'{"hooks":{"SessionStart":[{"hooks":[{"type":"http","url":"https://example.com",'
        b'"allowedEnvVars":null}]}]}}\n',
    ),
)
def test_disabled_or_malformed_hooks_fail_before_probe_without_writes(
    tmp_path: Path,
    foreign: bytes,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config = project / ".claude" / "settings.local.json"
    config.parent.mkdir()
    config.write_bytes(foreign)
    executable = _fake_claude(tmp_path)
    executable.write_bytes(
        b'#!/bin/sh\n/bin/mkdir -p "$HOME"\nprintf probed > "$HOME/probed"\n'
        b"printf '2.1.204 (Claude Code)\\n'\n"
    )
    executable.chmod(0o700)
    environment = _environment(tmp_path, executable)

    with pytest.raises(CaptureCommandConfigurationError):
        run_connect(provider="claude-code", project=project, environ=environment)

    assert config.read_bytes() == foreign
    assert not Path(environment["HOME"]).exists()
    assert not Path(environment["XDG_STATE_HOME"]).exists()


@pytest.mark.parametrize(
    "handler",
    (
        {
            "type": "command",
            "command": "",
            "args": [],
            "if": "",
            "shell": "bash",
            "timeout": 0.5,
            "statusMessage": "",
            "once": False,
            "async": True,
            "asyncRewake": False,
            "rewakeMessage": "wake",
            "rewakeSummary": "summary",
        },
        {
            "type": "prompt",
            "prompt": "",
            "if": "",
            "timeout": 1,
            "model": "",
            "continueOnBlock": False,
            "statusMessage": "",
            "once": True,
            "args": None,
        },
        {
            "type": "agent",
            "prompt": "",
            "if": "",
            "timeout": 1,
            "model": "",
            "statusMessage": "",
            "once": False,
        },
        {
            "type": "http",
            "url": "custom://",
            "if": "",
            "timeout": 1,
            "headers": {},
            "allowedEnvVars": [],
            "statusMessage": "",
            "once": False,
        },
        {
            "type": "mcp_tool",
            "server": "",
            "tool": "",
            "input": {"nested": [None, True, 1]},
            "if": "",
            "timeout": 1,
            "statusMessage": "",
            "once": False,
        },
    ),
)
def test_pinned_existing_hook_handler_shapes_are_accepted(handler: dict[str, object]) -> None:
    document = {
        "disableAllHooks": False,
        "hooks": {"SessionStart": [{"matcher": "", "hooks": [handler]}]},
    }

    assert claude_code_integration._hooks_explicitly_disabled(canonical_json(document)) is False


@pytest.mark.skipif(os.name != "posix", reason="native Windows lifecycle is covered by R01")
@pytest.mark.parametrize("scope", ("user", "project"))
def test_disabled_hooks_in_other_settings_scopes_fail_before_probe(
    tmp_path: Path,
    scope: str,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    executable = _fake_claude(tmp_path)
    executable.write_bytes(
        b'#!/bin/sh\n/bin/mkdir -p "$HOME"\nprintf probed > "$HOME/probed"\n'
        b"printf '2.1.204 (Claude Code)\\n'\n"
    )
    executable.chmod(0o700)
    environment = _environment(tmp_path, executable)
    path = (
        Path(environment["HOME"]) / ".claude" / "settings.json"
        if scope == "user"
        else project / ".claude" / "settings.json"
    )
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{"disableAllHooks":true}\n')

    with pytest.raises(CaptureCommandConfigurationError):
        run_connect(provider="claude-code", project=project, environ=environment)

    assert not (Path(environment["HOME"]) / "probed").exists()
    assert not Path(environment["XDG_STATE_HOME"]).exists()


@pytest.mark.skipif(os.name != "posix", reason="native Windows lifecycle is covered by R01")
def test_foreign_settings_schema_is_preserved_but_activation_is_not_claimed(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config = project / ".claude" / "settings.local.json"
    config.parent.mkdir()
    foreign = b'{\n  "model": 42\n}\n'
    config.write_bytes(foreign)
    executable = _fake_claude(tmp_path)
    environment = _environment(tmp_path, executable)
    capture_executable = _capture_executable()

    connected = run_connect(
        provider="claude-code",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    )
    installed = config.read_bytes()
    status = run_status(
        provider="claude-code",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    ).providers[0]

    assert connected.capture_enabled is True
    assert b'"model": 42' in installed
    assert CLAUDE_CODE_CONFIG_MARKER.encode("ascii") in installed
    assert status.status is CaptureOperationalStatus.INSTALLED_NOT_OBSERVED

    disconnected = run_disconnect(
        provider="claude-code",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    )
    assert disconnected.disposition == "uninstalled"
    assert config.read_bytes() == foreign


@pytest.mark.skipif(os.name != "posix", reason="native Windows lifecycle is covered by R01")
def test_malformed_critical_hook_degrades_the_authenticated_session(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    executable = _fake_claude(tmp_path)
    environment = _environment(tmp_path, executable)
    capture_executable = _capture_executable()
    run_connect(
        provider="claude-code",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    )
    spec = provider_installation_spec(project, environ=environment)
    key = load_installation_key(environ=environment)
    identity = derive_installation_identity(spec, key)
    arguments = ("--profile", PROFILE.value, "--connection", identity.connection_id)

    started = _payload("SessionStart")
    started["cwd"] = str(project)
    assert (
        run_capture_hook(
            arguments,
            BytesIO(canonical_json(started)),
            environ=environment,
            capture_executable=capture_executable,
        )
        == 0
    )
    malformed = _payload("PreToolUse")
    malformed["cwd"] = str(project)
    malformed.pop("tool_use_id")
    assert (
        run_capture_hook(
            arguments,
            BytesIO(canonical_json(malformed)),
            environ=environment,
            capture_executable=capture_executable,
        )
        == 0
    )

    degraded = run_status(
        provider="claude-code",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    ).providers[0]
    assert degraded.status is CaptureOperationalStatus.DEGRADED
    assert degraded.drift == (CaptureStatusDrift.SESSION_DEGRADED,)
    assert degraded.session_count == 1
    assert degraded.quarantined_sessions == 0


@pytest.mark.skipif(os.name != "posix", reason="native Windows lifecycle is covered by R01")
def test_connect_hook_status_disconnect_round_trip_is_offline_and_content_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config = project / ".claude" / "settings.local.json"
    config.parent.mkdir()
    foreign = (
        b'{\n "model":"claude-sonnet-4-5",'
        b'"hooks":{"Notification":[{"hooks":[{"type":"command","command":"foreign"}]}],'
        b'"SessionStart":[{"hooks":[{"type":"command","command":"foreign-start"}]}]},'
        b'"permissions":{"allow":["Read"]}\n}\n'
    )
    config.write_bytes(foreign)
    executable = _fake_claude(tmp_path)
    environment = _environment(tmp_path, executable)
    capture_executable = _capture_executable()

    def deny_socket(*_args: object, **_kwargs: object) -> socket.socket:
        raise AssertionError("Claude capture attempted network access")

    monkeypatch.setattr(socket, "socket", deny_socket)
    connected = run_connect(
        provider="claude-code",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    )
    assert connected.capture_enabled is True
    assert connected.project_local_files == 1

    old_spec = provider_installation_spec(project, environ=environment)
    key = load_installation_key(environ=environment)
    old_identity = derive_installation_identity(old_spec, key)
    before = run_status(
        provider="claude-code",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    ).providers[0]
    assert before.status is CaptureOperationalStatus.INSTALLED_NOT_OBSERVED

    status_probe = Path(environment["HOME"]) / "status-probe"
    executable.write_bytes(
        b'#!/bin/sh\n/bin/mkdir -p "$HOME"\nprintf probed > "$HOME/status-probe"\n'
        b"printf '2.1.205 (Claude Code)\\n'\n"
    )
    executable.chmod(0o700)
    still_read_only = run_status(
        provider="claude-code",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    ).providers[0]
    assert still_read_only.status is CaptureOperationalStatus.INSTALLED_NOT_OBSERVED
    assert not status_probe.exists()

    upgraded = run_connect(
        provider="claude-code",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    )
    assert upgraded.capture_enabled is True
    assert upgraded.disposition.value == "upgraded"
    assert status_probe.read_bytes() == b"probed"
    spec = provider_installation_spec(
        project,
        environ=environment,
        host_version="2.1.205",
    )
    identity = derive_installation_identity(spec, key)
    assert identity.connection_id != old_identity.connection_id
    assert spec.generation == old_spec.generation + 1

    nested = project / "nested"
    nested_config = nested / ".claude" / "settings.local.json"
    nested_config.parent.mkdir(parents=True)
    nested_config.write_bytes(canonical_json({"foreign": CLAUDE_CODE_CONFIG_MARKER}))
    payload = _payload("SessionStart")
    payload["cwd"] = str(nested)
    arguments = ("--profile", PROFILE.value, "--connection", identity.connection_id)
    launched = subprocess.run(
        (str(spec.launcher_path), CLAUDE_CODE_CONFIG_MARKER),
        input=canonical_json(payload),
        capture_output=True,
        check=False,
        env=environment,
        timeout=10,
    )
    assert launched.returncode == 0
    assert launched.stdout == launched.stderr == b""

    observed = run_status(
        provider="claude-code",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    ).providers[0]
    assert observed.status is CaptureOperationalStatus.ACTIVE_OBSERVED
    assert observed.session_count == 1

    assert (
        run_capture_hook(
            arguments,
            BytesIO(canonical_json(payload)),
            environ=environment,
            capture_executable=capture_executable,
        )
        == 0
    )

    disconnected = run_disconnect(
        provider="claude-code",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    )
    assert disconnected.disposition == "uninstalled"
    assert config.read_bytes() == foreign
    assert not spec.launcher_path.exists()

    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    native_session = payload["session_id"]
    assert type(native_session) is str
    with CaptureStore.open(
        locations.database_path,
        installation_key=key,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        preserved = store.snapshot_session(
            identity.connection_id,
            CaptureDigestContext(key).session_id(native_session.encode()),
        )
    assert preserved.state is CaptureSessionState.OPEN

    poison = environment["ANTHROPIC_API_KEY"].encode()
    for root in (project, Path(environment["XDG_STATE_HOME"])):
        for path in root.rglob("*"):
            if path.is_file():
                assert poison not in path.read_bytes()


def test_poisoned_anthropic_environment_socket_denial_and_import_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    original_import = builtins.__import__

    def audited_import(name: str, *args: object, **kwargs: object):
        if name == "anthropic" or name.startswith("anthropic."):
            imported.append(name)
            raise AssertionError("Anthropic package import attempted")
        return original_import(name, *args, **kwargs)

    def deny_socket(*_args: object, **_kwargs: object) -> socket.socket:
        raise AssertionError("network access attempted")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "poisoned-anthropic-secret")
    monkeypatch.setattr(builtins, "__import__", audited_import)
    monkeypatch.setattr(socket, "socket", deny_socket)
    context = CaptureDigestContext(KEY)
    encoded = b""
    for event in _fixture_events():
        intakes = _adapter().adapt_bytes(canonical_json(event["payload"]), context=context)
        encoded += b"".join(map(canonical_capture_intake, intakes))

    assert imported == []
    assert b"poisoned-anthropic-secret" not in encoded

    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    declared = canonical_json(project).lower()
    assert b"anthropic" not in declared
