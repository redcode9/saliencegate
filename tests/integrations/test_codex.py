from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from importlib import resources
from io import BytesIO
from pathlib import Path

import pytest
from scripts import smoke_capture_installed

from saliencegate.capture.capabilities import (
    CaptureProfile,
    CompatibilityStatus,
    capture_capability_digest,
    capture_profile,
)
from saliencegate.capture.health import CaptureHealthCode
from saliencegate.capture.identities import CaptureDigestContext
from saliencegate.capture.locations import resolve_capture_store_locations
from saliencegate.capture.migrations import initialize_capture_store
from saliencegate.capture.normalization import normalize_capture_session_snapshot
from saliencegate.capture.publication import verify_capture_intake_authentication
from saliencegate.capture.report import (
    CaptureReportHeadline,
    build_capture_session_report,
    encode_capture_session_report,
)
from saliencegate.capture.schema import canonical_capture_intake
from saliencegate.capture.store import (
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
from saliencegate.integrations.codex import (
    CODEX_CONFIG_MARKER,
    CODEX_HOOK_EVENTS,
    CODEX_HOST_VERSION,
    CodexCaptureAdapter,
    CodexIntegrationError,
    CodexVersionProbe,
    probe_codex_version,
    provider_installation_spec,
)
from saliencegate.integrations.config_files import (
    ConfigFileError,
    ConfigSyntax,
    plan_owned_config_install,
    remove_owned_config_edit,
)
from saliencegate.integrations.environment import environment_without_provider_credentials
from saliencegate.integrations.hook import run_capture_hook
from saliencegate.integrations.installation import derive_installation_identity
from saliencegate.integrations.registry import (
    ProviderAlias,
    ProviderInstallationKind,
    ProviderInstallationSpec,
)
from saliencegate.security import InstallationKey, load_installation_key

CONNECTION_ID = "sg-" + "1" * 48
KEY = InstallationKey(b"0" * 32)
PROFILE = CaptureProfile.CODEX_HOOKS_V1


def _fixture_events() -> list[dict[str, object]]:
    source = (
        resources.files("saliencegate.integrations")
        .joinpath("fixtures")
        .joinpath("codex-hooks-v1.json")
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


def _adapter(host_version: str = CODEX_HOST_VERSION) -> CodexCaptureAdapter:
    return CodexCaptureAdapter(
        connection_id=CONNECTION_ID,
        host_version=host_version,
    )


def _adapt(
    event_name: str,
    *,
    changes: dict[str, object] | None = None,
    adapter: CodexCaptureAdapter | None = None,
):
    payload = _payload(event_name)
    if changes:
        payload.update(changes)
    selected = _adapter() if adapter is None else adapter
    return selected.adapt_bytes(canonical_json(payload), context=CaptureDigestContext(KEY))


def test_capability_declaration_is_exact_and_documents_incomplete_coverage() -> None:
    adapter = _adapter()
    declaration = adapter.capabilities()
    manifest = capture_profile(PROFILE)

    assert declaration.profile_id is PROFILE
    assert declaration.host_version == "0.144.6"
    assert declaration.capability_digest == capture_capability_digest(manifest)
    assert manifest.complete_execution_session_coverage is False
    assert manifest.transcript_read is False
    assert manifest.decision_authority is False
    assert manifest.coverage_exclusions == (
        "hosted_tools",
        "specialized_paths_outside_local_function_tool_hooks",
        "write_stdin_is_continuation_not_a_second_action",
    )


def test_all_audited_events_map_without_inventing_boundaries_or_outcomes() -> None:
    context = CaptureDigestContext(KEY)
    expected = {
        "SessionStart": ("session_started",),
        "PreToolUse": ("action_started",),
        "PermissionRequest": (),
        "PostToolUse": ("action_finished",),
        "PreCompact": (),
        "SubagentStart": ("subagent_started",),
        "SubagentStop": ("subagent_finished",),
        "Stop": ("turn_finished",),
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


def test_every_declared_ignored_field_is_authority_free() -> None:
    manifest = capture_profile(PROFILE)
    for event in manifest.events:
        baseline = _adapt(event.event_name)
        for field_name in event.ignored_fields:
            changed = _adapt(
                event.event_name,
                changes={field_name: f"changed-ignored-{field_name}"},
            )
            assert changed == baseline


def test_every_declared_optional_field_changes_admitted_evidence() -> None:
    manifest = capture_profile(PROFILE)
    assert {event.event_name: event.optional_fields for event in manifest.events} == {
        "SessionStart": (),
        "PreToolUse": ("cwd", "tool_input", "tool_name"),
        "PermissionRequest": (),
        "PostToolUse": (),
        "PreCompact": (),
        "SubagentStart": (),
        "SubagentStop": (),
        "Stop": ("turn_id",),
    }

    pre_tool = _adapt("PreToolUse")[0]
    changed_cwd = _adapt("PreToolUse", changes={"cwd": "/changed/workspace"})[0]
    changed_input = _adapt("PreToolUse", changes={"tool_input": {"path": "changed"}})[0]
    changed_name = _adapt("PreToolUse", changes={"tool_name": "Bash"})[0]
    assert changed_cwd != pre_tool
    assert changed_input != pre_tool
    assert changed_name != pre_tool
    assert changed_cwd.workspace_digest != pre_tool.workspace_digest
    assert changed_input.action_digest != pre_tool.action_digest
    assert changed_name.action_digest != pre_tool.action_digest
    assert changed_name.tool_class == "shell"

    stop = _adapt("Stop")[0]
    changed_turn = _adapt("Stop", changes={"turn_id": "changed-turn"})[0]
    assert _adapt("Stop", changes={"turn_id": None}) == ()
    assert changed_turn != stop
    assert changed_turn.turn_id != stop.turn_id


def test_audited_fixture_persists_and_reports_deterministically(tmp_path: Path) -> None:
    database = tmp_path / "capture.db"
    initialize_capture_store(database)
    context = CaptureDigestContext(KEY)
    adapter = _adapter()
    manifest = capture_profile(PROFILE)

    with CaptureStore.open(
        database,
        installation_key=KEY,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        store.register_connection(
            connection_id=CONNECTION_ID,
            project_digest="7" * 64,
            profile_id=PROFILE,
            capability_manifest_digest=capture_capability_digest(manifest),
            host_version=CODEX_HOST_VERSION,
        )
        store.transition_connection(
            CONNECTION_ID,
            expected_state=CaptureConnectionState.PENDING,
            target_state=CaptureConnectionState.ENABLED,
        )

        intakes = tuple(
            intake
            for event in _fixture_events()
            for intake in adapter.adapt_bytes(
                canonical_json(event["payload"]),
                context=context,
            )
        )
        assert len(intakes) == 6
        for intake in intakes:
            store.append(intake)

        snapshot = store.snapshot_session(CONNECTION_ID, intakes[0].session_id)

    normalization = normalize_capture_session_snapshot(
        snapshot,
        installation_key=KEY,
    )
    first = build_capture_session_report(
        snapshot,
        normalization,
        installation_key=KEY,
        spool=None,
    )
    second = build_capture_session_report(
        snapshot,
        normalization,
        installation_key=KEY,
        spool=None,
    )

    assert encode_capture_session_report(first) == encode_capture_session_report(second)
    assert first.session_state is CaptureSessionState.OPEN
    assert first.headline is CaptureReportHeadline.INSUFFICIENT_EVIDENCE
    assert first.complete_execution_session_coverage is False
    assert first.coverage.capability_exclusions == manifest.coverage_exclusions
    assert first.counts.captured_events == 6


def test_pre_and_post_share_only_the_hmac_call_parent_and_post_has_no_outcome_claim() -> None:
    started = _adapt("PreToolUse")[0]
    finished = _adapt("PostToolUse")[0]

    assert started.kind == "action_started"
    assert finished.kind == "action_finished"
    assert started.call_ref == finished.call_ref
    assert started.producer_event_digest != finished.producer_event_digest
    assert finished.outcome_status is None
    assert finished.outcome_authority == "unavailable"
    assert finished.exit_status is None
    assert finished.error_code is None
    assert finished.failure_signature is None


def test_tool_response_and_additive_fields_never_gain_authority() -> None:
    first = _adapt("PostToolUse")[0]
    second = _adapt(
        "PostToolUse",
        changes={
            "tool_response": {
                "success": False,
                "exit_status": 97,
                "error": "provider-response-secret",
            },
            "future_additive_field": "additive-secret",
        },
    )[0]

    assert second == first
    assert b"provider-response-secret" not in canonical_capture_intake(second)
    assert b"additive-secret" not in canonical_capture_intake(second)


def test_action_identity_is_exact_coarse_or_per_call_unavailable_without_raw_input() -> None:
    exact = _adapt("PreToolUse")[0]
    coarse = _adapt("PreToolUse", changes={"tool_input": None})[0]
    missing_name_payload = _payload("PreToolUse")
    missing_name_payload.pop("tool_name")
    unavailable = _adapter().adapt_bytes(
        canonical_json(missing_name_payload),
        context=CaptureDigestContext(KEY),
    )[0]

    assert exact.identity_authority == "exact"
    assert coarse.identity_authority == "coarse"
    assert unavailable.identity_authority == "unavailable"
    assert len({exact.action_digest, coarse.action_digest, unavailable.action_digest}) == 3
    assert b"synthetic-input.txt" not in canonical_capture_intake(exact)


def test_duplicate_correlation_is_stable_but_conflicting_consumed_input_changes_intake() -> None:
    first = _adapt("PreToolUse")[0]
    replay = _adapt("PreToolUse")[0]
    conflict = _adapt(
        "PreToolUse",
        changes={"tool_input": {"path": "different-synthetic-input.txt"}},
    )[0]

    assert replay == first
    assert conflict.producer_event_digest == first.producer_event_digest
    assert conflict.call_ref == first.call_ref
    assert conflict.action_digest != first.action_digest
    assert conflict != first


def test_subagent_and_turn_correlations_are_session_bound_and_deterministic() -> None:
    started = _adapt("SubagentStart")[0]
    finished = _adapt("SubagentStop")[0]
    stopped = _adapt("Stop")[0]

    assert started.subagent_id == finished.subagent_id
    assert _adapt("Stop")[0] == stopped

    different_session = _adapt(
        "SubagentStop",
        changes={"session_id": "different-synthetic-session"},
    )[0]
    assert different_session.subagent_id != started.subagent_id


def test_transcript_and_message_paths_are_never_opened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _fixture_events()
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
    adapter = _adapter()
    context = CaptureDigestContext(KEY)
    for event in events:
        payload = event["payload"]
        assert type(payload) is dict
        adapter.adapt_bytes(canonical_json(payload), context=context)


@pytest.mark.parametrize(
    "payload",
    (
        {},
        {"hook_event_name": "Unknown", "session_id": "session"},
        {"hook_event_name": "SessionStart"},
        {"hook_event_name": 7, "session_id": "session"},
        {"hook_event_name": "PreToolUse", "session_id": "session"},
    ),
)
def test_invalid_or_unknown_event_shapes_fail_closed_without_content(payload: object) -> None:
    secret = "codex-invalid-shape-secret"
    document = {**payload, "future": secret} if isinstance(payload, dict) else payload

    with pytest.raises(CodexIntegrationError) as captured:
        _adapter().adapt_bytes(canonical_json(document), context=CaptureDigestContext(KEY))

    rendered = f"{captured.value!s}\n{captured.value!r}"
    assert secret not in rendered
    assert "session" not in rendered


def test_duplicate_keys_and_oversize_native_input_fail_content_free() -> None:
    for source in (
        b'{"hook_event_name":"SessionStart","session_id":"one","session_id":"two"}',
        b"{" + b'"future":"' + b"x" * (2 * 1024 * 1024) + b'"}',
    ):
        with pytest.raises(CodexIntegrationError) as captured:
            _adapter().adapt_bytes(source, context=CaptureDigestContext(KEY))
        assert "one" not in str(captured.value)
        assert "two" not in repr(captured.value)


def _executable(tmp_path: Path) -> Path:
    executable = tmp_path / "codex"
    executable.write_bytes(b"synthetic")
    executable.chmod(0o700)
    return executable


def _capture_executable() -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return (Path(sys.executable).parent / f"saliencegate-capture-hook{suffix}").resolve(strict=True)


def test_version_probe_is_bounded_exact_and_reports_compatibility(tmp_path: Path) -> None:
    executable = _executable(tmp_path)
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def runner(command: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, b"codex-cli 0.144.6\n", b"")

    probe_environment = {"PATH": str(tmp_path)}
    probe = probe_codex_version(
        executable,
        runner=runner,
        environ=probe_environment,
    )

    assert probe == CodexVersionProbe(
        host_version="0.144.6",
        compatibility=CompatibilityStatus.VERIFIED,
    )
    assert calls[0][0] == (str(executable), "--version")
    assert calls[0][1]["timeout"] == 2.0
    assert calls[0][1]["check"] is False
    assert calls[0][1]["capture_output"] is True
    assert calls[0][1]["input"] == b""
    assert calls[0][1]["env"] == probe_environment


def test_version_probe_classifies_new_semver_and_rejects_malformed_timeout_or_overflow(
    tmp_path: Path,
) -> None:
    executable = _executable(tmp_path)

    def completed(stdout: bytes, *, returncode: int = 0):
        def run(command: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
            return subprocess.CompletedProcess(command, returncode, stdout, b"")

        return run

    newer = probe_codex_version(executable, runner=completed(b"codex-cli 0.144.7\n"))
    assert newer.host_version == "0.144.7"
    assert newer.compatibility is CompatibilityStatus.SCHEMA_COMPATIBLE_UNVERIFIED_VERSION

    def timeout(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired((str(executable), "--version"), 2.0)

    invalid_runners = (
        completed(b"Codex version secret\n"),
        completed(b"codex-cli 0.144.6\n", returncode=1),
        completed(b"codex-cli 0.144.5\n"),
        completed(b"codex-cli 0.145.0\n"),
        completed(b"codex-cli 1.0.0\n"),
        completed(b"x" * 4097),
        timeout,
    )
    for runner in invalid_runners:
        with pytest.raises(CodexIntegrationError) as captured:
            probe_codex_version(executable, runner=runner)
        assert "secret" not in str(captured.value)


@pytest.mark.skipif(os.name != "posix", reason="native Windows probe is covered by R01")
def test_default_version_runner_retains_only_bounded_output(tmp_path: Path) -> None:
    valid = tmp_path / "codex-valid"
    valid.write_bytes(b"#!/bin/sh\nprintf 'codex-cli 0.144.6\\n'\n")
    valid.chmod(0o700)
    assert probe_codex_version(valid).compatibility is CompatibilityStatus.VERIFIED

    overflow = tmp_path / "codex-overflow"
    overflow.write_bytes(b"#!/bin/sh\nprintf '%05000d' 0\n")
    overflow.chmod(0o700)
    with pytest.raises(CodexIntegrationError):
        probe_codex_version(overflow)


def test_project_installation_spec_is_one_owned_toml_edit_without_trust_mutation(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    spec = provider_installation_spec(
        project,
        environ={
            "HOME": str(tmp_path / "home"),
            "XDG_STATE_HOME": str(state),
        },
    )

    assert spec.installation_kind is ProviderInstallationKind.COMMAND_HOOK
    assert spec.project_local_paths == (project / ".codex" / "config.toml",)
    assert spec.config.syntax is ConfigSyntax.TOML_DOCUMENT
    assert spec.config.marker == CODEX_CONFIG_MARKER
    assert spec.bundle_path is None
    assert spec.bootstrap_path is None
    assert spec.bundle_bytes is None
    assert spec.bootstrap_relative_reference is None
    assert spec.receipt_path.parent == spec.launcher_path.parent
    assert spec.receipt_path.parent.is_relative_to(state / "saliencegate")
    assert not spec.receipt_path.is_relative_to(project)

    fragment = spec.config.owned_fragment
    lowered = fragment.lower()
    assert fragment.count(CODEX_CONFIG_MARKER.encode("ascii")) == 1
    assert b"trust" not in lowered
    assert b"bypass" not in lowered
    assert b"[features]" not in lowered
    assert str(spec.launcher_path).encode() in fragment

    foreign = b'model = "gpt-5.6"\n[features]\nhooks = true\n'
    planned = plan_owned_config_install(foreign, spec.config)
    document = tomllib.loads(planned.installed_bytes.decode("utf-8"))
    assert set(document["hooks"]) == set(CODEX_HOOK_EVENTS)
    for event_name in CODEX_HOOK_EVENTS:
        groups = document["hooks"][event_name]
        assert type(groups) is list and len(groups) == 1
        handlers = groups[0]["hooks"]
        assert type(handlers) is list and len(handlers) == 1
        assert handlers[0]["type"] == "command"
        assert handlers[0]["timeout"] == 3

    assert remove_owned_config_edit(planned.installed_bytes, planned.reverse_edit) == foreign

    conflicting_hooks = (
        b'hooks = { PreToolUse = [{ matcher = "Bash", hooks = '
        b'[{ type = "command", command = "true" }] }] }\n'
    )
    with pytest.raises(ConfigFileError):
        plan_owned_config_install(conflicting_hooks, spec.config)

    for inactive_policy in (
        b"[features]\nhooks = false\n",
        b"[features]\ncodex_hooks = false\n",
        b"allow_managed_hooks_only = true\n",
    ):
        with pytest.raises(ConfigFileError):
            plan_owned_config_install(inactive_policy, spec.config)


@pytest.mark.skipif(os.name != "posix", reason="native Windows lifecycle is covered by R01")
@pytest.mark.parametrize(
    "incompatible_config",
    (
        b"[features]\nhooks = false\n",
        b"[features]\ncodex_hooks = false\n",
        b"allow_managed_hooks_only = true\n",
        (
            b'hooks = { PreToolUse = [{ matcher = "Bash", hooks = '
            b'[{ type = "command", command = "true" }] }] }\n'
        ),
    ),
)
def test_default_connect_refuses_incompatible_project_hook_config_without_writes(
    tmp_path: Path,
    incompatible_config: bytes,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config = project / ".codex" / "config.toml"
    config.parent.mkdir()
    config.write_bytes(incompatible_config)
    provider_bin = tmp_path / "provider-bin"
    provider_bin.mkdir()
    executable = provider_bin / "codex"
    executable.write_bytes(
        b'#!/bin/sh\n/bin/mkdir -p "$HOME"\nprintf probed > "$HOME/probed"\n'
        b"printf 'codex-cli 0.144.6\\n'\n"
    )
    executable.chmod(0o700)
    state = tmp_path / "state"
    environment = {
        "HOME": str(tmp_path / "home"),
        "PATH": str(provider_bin),
        "XDG_CONFIG_HOME": str(tmp_path / "configuration"),
        "XDG_STATE_HOME": str(state),
    }

    with pytest.raises(CaptureCommandConfigurationError):
        run_connect(provider="codex", project=project, environ=environment)

    assert config.read_bytes() == incompatible_config
    assert not state.exists()
    assert not Path(environment["HOME"]).exists()


@pytest.mark.skipif(os.name != "posix", reason="native Windows lifecycle is covered by R01")
def test_default_codex_dry_run_does_not_launch_the_host_or_write(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    provider_bin = tmp_path / "provider-bin"
    provider_bin.mkdir()
    executable = provider_bin / "codex"
    executable.write_bytes(
        b'#!/bin/sh\n/bin/mkdir -p "$HOME"\nprintf probed > "$HOME/probed"\n'
        b"printf 'codex-cli 0.144.6\\n'\n"
    )
    executable.chmod(0o700)
    environment = {
        "HOME": str(tmp_path / "home"),
        "PATH": str(provider_bin),
        "XDG_CONFIG_HOME": str(tmp_path / "configuration"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }

    report = run_connect(
        provider="codex",
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
@pytest.mark.parametrize("target_version", ("0.144.7", "0.144.8"))
def test_reconnect_upgrades_codex_host_generation_without_status_launching_codex(
    tmp_path: Path,
    target_version: str,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    project_config = project / ".codex" / "config.toml"
    project_config.parent.mkdir()
    project_config.write_bytes(b'model = "gpt-5.6"\n')
    provider_bin = tmp_path / "provider-bin"
    provider_bin.mkdir()
    codex_executable = provider_bin / "codex"
    codex_executable.write_bytes(b"#!/bin/sh\nprintf 'codex-cli 0.144.6\\n'\n")
    codex_executable.chmod(0o700)
    environment = {
        "HOME": str(tmp_path / "home"),
        "PATH": str(provider_bin),
        "XDG_CONFIG_HOME": str(tmp_path / "configuration"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }
    capture_executable = _capture_executable()

    first = run_connect(
        provider="codex",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    )
    assert first.capture_enabled is True
    old_spec = provider_installation_spec(
        project,
        environ=environment,
        host_version="0.144.6",
    )
    key = load_installation_key(environ=environment)
    old_identity = derive_installation_identity(old_spec, key)

    status_probe = Path(environment["HOME"]) / "status-probe"
    codex_executable.write_bytes(
        (
            '#!/bin/sh\n/bin/mkdir -p "$HOME"\nprintf probed > "$HOME/status-probe"\n'
            f"printf 'codex-cli {target_version}\\n'\n"
        ).encode()
    )
    codex_executable.chmod(0o700)
    before_reconnect = run_status(
        provider="codex",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    ).providers[0]
    assert before_reconnect.status is CaptureOperationalStatus.INSTALLED_NOT_OBSERVED
    assert before_reconnect.drift == ()
    assert not status_probe.exists()

    upgraded = run_connect(
        provider="codex",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    )
    assert upgraded.capture_enabled is True
    assert upgraded.disposition.value == "upgraded"
    assert status_probe.read_bytes() == b"probed"
    new_spec = provider_installation_spec(
        project,
        environ=environment,
        host_version=target_version,
    )
    new_identity = derive_installation_identity(new_spec, key)
    assert new_spec.generation == old_spec.generation + int(target_version.rsplit(".", 1)[1]) - 6
    assert new_identity.connection_id != old_identity.connection_id

    healthy = run_status(
        provider="codex",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    ).providers[0]
    assert healthy.status is CaptureOperationalStatus.INSTALLED_NOT_OBSERVED
    assert healthy.drift == ()
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    with CaptureStore.open(
        locations.database_path,
        installation_key=key,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        connections = store.list_connections(
            project_digest=new_identity.project_digest,
            profile_id=PROFILE,
        )
    assert {item.host_version: item.state for item in connections} == {
        "0.144.6": CaptureConnectionState.DISABLED,
        target_version: CaptureConnectionState.ENABLED,
    }

    disconnected = run_disconnect(
        provider="codex",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    )
    assert disconnected.disposition == "uninstalled"
    after_disconnect = run_status(
        provider="codex",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    ).providers[0]
    assert after_disconnect.status is CaptureOperationalStatus.NOT_INSTALLED
    assert after_disconnect.drift == ()


@pytest.mark.skipif(os.name != "posix", reason="native Windows lifecycle is covered by R01")
@pytest.mark.parametrize("start_session", (False, True))
def test_malformed_critical_hook_marks_attributable_session_coverage_degraded(
    tmp_path: Path,
    start_session: bool,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    provider_bin = tmp_path / "provider-bin"
    provider_bin.mkdir()
    codex_executable = provider_bin / "codex"
    codex_executable.write_bytes(b"#!/bin/sh\nprintf 'codex-cli 0.144.6\\n'\n")
    codex_executable.chmod(0o700)
    environment = {
        "HOME": str(tmp_path / "home"),
        "PATH": str(provider_bin),
        "XDG_CONFIG_HOME": str(tmp_path / "configuration"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }
    capture_executable = _capture_executable()
    run_connect(
        provider="codex",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    )
    key = load_installation_key(environ=environment)
    spec = provider_installation_spec(project, environ=environment)
    identity = derive_installation_identity(spec, key)
    arguments = (
        "--profile",
        PROFILE.value,
        "--connection",
        identity.connection_id,
    )
    if start_session:
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
    native_session_id = malformed["session_id"]
    assert type(native_session_id) is str
    del malformed["tool_use_id"]
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
        provider="codex",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    ).providers[0]
    assert degraded.status is CaptureOperationalStatus.DEGRADED
    assert degraded.drift == (CaptureStatusDrift.SESSION_DEGRADED,)
    assert degraded.quarantined_sessions == (0 if start_session else 1)
    locations = resolve_capture_store_locations(
        environ=environment,
        home=Path(environment["HOME"]),
    )
    session_id = CaptureDigestContext(key).session_id(native_session_id.encode("utf-8"))
    with CaptureStore.open(
        locations.database_path,
        installation_key=key,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        snapshot = store.snapshot_session(identity.connection_id, session_id)
    assert snapshot.coverage_degraded is True
    assert snapshot.state is (
        CaptureSessionState.OPEN if start_session else CaptureSessionState.QUARANTINED
    )
    assert snapshot.event_count == (1 if start_session else 0)
    assert tuple(item.code for item in snapshot.health) == (CaptureHealthCode.COVERAGE_DEGRADED,)
    assert snapshot.health[0].count == 1


@pytest.mark.skipif(
    os.name != "nt",
    reason="native installed Codex launcher execution is the remote R01 gate",
)
def test_native_windows_installed_codex_launcher_observes_one_session(
    tmp_path: Path,
) -> None:
    root = tmp_path / "launcher path & data"
    root.mkdir()
    project = root / "project"
    project.mkdir()
    home = root / "home"
    home.mkdir()
    environment = environment_without_provider_credentials(os.environ)
    environment.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "XDG_CACHE_HOME": str(home / "cache"),
            "XDG_CONFIG_HOME": str(home / "config"),
            "XDG_DATA_HOME": str(home / "data"),
            "XDG_STATE_HOME": str(home / "state"),
            "APPDATA": str(home / "appdata"),
            "LOCALAPPDATA": str(home / "localappdata"),
        }
    )
    spec = provider_installation_spec(
        project,
        environ=environment,
        host_version=CODEX_HOST_VERSION,
    )

    def resolve_spec(
        alias: ProviderAlias,
        candidate: Path,
    ) -> ProviderInstallationSpec:
        assert alias is ProviderAlias.CODEX
        assert candidate == project
        return spec

    capture_executable = _capture_executable()
    connected = run_connect(
        provider="codex",
        project=project,
        environ=environment,
        spec_resolver=resolve_spec,
        capture_executable=capture_executable,
    )
    assert connected.capture_enabled is True
    assert spec.launcher_path.is_file()

    callback = smoke_capture_installed._launcher_command(
        spec.launcher_path,
        environment=environment,
    )
    assert callback.executable is not None
    assert callback.launcher_environment is not None
    process_environment = dict(environment)
    process_environment["SALIENCEGATE_LAUNCHER"] = callback.launcher_environment
    payload = _payload("SessionStart")
    payload["cwd"] = str(project)
    completed = subprocess.run(
        callback.command,
        cwd=project,
        env=process_environment,
        executable=callback.executable,
        shell=False,
        check=False,
        capture_output=True,
        input=canonical_json(payload),
        timeout=10,
    )
    assert (completed.returncode, bool(completed.stdout), bool(completed.stderr)) == (
        0,
        False,
        False,
    )

    observed = run_status(
        provider="codex",
        project=project,
        environ=environment,
        spec_resolver=resolve_spec,
        capture_executable=capture_executable,
    ).providers[0]
    assert observed.status is CaptureOperationalStatus.ACTIVE_OBSERVED
    assert observed.session_count == 1
    assert observed.drift == ()


@pytest.mark.skipif(os.name != "posix", reason="native Windows lifecycle is covered by R01")
def test_default_codex_connect_hook_status_and_disconnect_round_trip(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    project_config = project / ".codex" / "config.toml"
    project_config.parent.mkdir()
    foreign_config = b'model = "gpt-5.6"\n'
    project_config.write_bytes(foreign_config)

    home = tmp_path / "home"
    trust_path = home / ".codex" / "hook-trust-state.json"
    trust_path.parent.mkdir(mode=0o700, parents=True)
    trust_bytes = b'{"trusted_hook_hashes":["synthetic-user-choice"]}\n'
    trust_path.write_bytes(trust_bytes)
    trust_path.chmod(0o600)
    environment = {
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(tmp_path / "configuration"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }
    provider_bin = tmp_path / "provider-bin"
    provider_bin.mkdir()
    codex_executable = provider_bin / "codex"
    codex_executable.write_bytes(b"#!/bin/sh\nprintf 'codex-cli 0.144.7\\n'\n")
    codex_executable.chmod(0o700)
    environment["PATH"] = str(provider_bin)
    capture_executable = _capture_executable()

    connected = run_connect(
        provider="codex",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    )

    assert connected.capture_enabled is True
    assert connected.project_local_files == 1
    assert trust_path.read_bytes() == trust_bytes
    spec = provider_installation_spec(
        project,
        environ=environment,
        host_version="0.144.7",
    )
    assert spec.receipt_path.is_relative_to(tmp_path / "state" / "saliencegate")
    assert spec.config_path.read_bytes().startswith(foreign_config)
    before = run_status(
        provider="codex",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    ).providers[0]
    assert before.status is CaptureOperationalStatus.INSTALLED_NOT_OBSERVED
    assert before.drift == ()

    key = load_installation_key(environ=environment)
    identity = derive_installation_identity(spec, key)
    payload = _payload("SessionStart")
    payload["cwd"] = str(project)
    arguments = (
        "--profile",
        PROFILE.value,
        "--connection",
        identity.connection_id,
    )
    launched = subprocess.run(
        (str(spec.launcher_path),),
        input=canonical_json(payload),
        capture_output=True,
        check=False,
        env=environment,
        timeout=10,
    )
    assert launched.returncode == 0
    assert launched.stdout == launched.stderr == b""

    observed = run_status(
        provider="codex",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    ).providers[0]
    assert observed.status is CaptureOperationalStatus.ACTIVE_OBSERVED
    assert observed.session_count == 1
    assert observed.drift == ()

    installed_config = project_config.read_bytes()
    disabled_config = installed_config.replace(
        foreign_config,
        foreign_config + b"[features]\nhooks = false\n",
        1,
    )
    assert disabled_config != installed_config
    project_config.write_bytes(disabled_config)
    drifted = run_status(
        provider="codex",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    ).providers[0]
    assert drifted.status is CaptureOperationalStatus.DRIFTED
    assert CaptureStatusDrift.CONFIG in drifted.drift
    project_config.write_bytes(installed_config)

    disconnected = run_disconnect(
        provider="codex",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    )
    assert disconnected.disposition == "uninstalled"
    assert project_config.read_bytes() == foreign_config
    assert trust_path.read_bytes() == trust_bytes
    assert not spec.launcher_path.exists()

    assert (
        run_capture_hook(
            arguments,
            BytesIO(canonical_json(payload)),
            environ=environment,
            capture_executable=capture_executable,
        )
        == 0
    )
    locations = resolve_capture_store_locations(
        environ=environment,
        home=home,
    )
    with CaptureStore.open(
        locations.database_path,
        installation_key=key,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        connections = store.list_connections(
            project_digest=identity.project_digest,
            profile_id=PROFILE,
        )
        sessions = store.list_sessions(
            project_digest=identity.project_digest,
            profile_id=PROFILE,
        )
    assert len(connections) == 1
    assert connections[0].host_version == "0.144.7"
    assert (
        connections[0].compatibility_status
        is CompatibilityStatus.SCHEMA_COMPATIBLE_UNVERIFIED_VERSION
    )
    assert connections[0].state is CaptureConnectionState.DISABLED
    assert len(sessions) == 1
    assert sessions[0].event_count == 1
