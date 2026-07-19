from __future__ import annotations

import hashlib
import json
from importlib import resources
from typing import Final

import pytest
from pydantic import ValidationError

from saliencegate.capture.capabilities import (
    CapabilitySupport,
    CaptureCapabilityError,
    CaptureCapabilityRegistry,
    CaptureProfile,
    CompatibilityStatus,
    capture_capability_digest,
    capture_profile,
    classify_capture_compatibility,
    load_capture_capability_registry,
    validate_capture_capability_binding,
)
from saliencegate.domain import SignalType, canonical_json

AUDIT_DATE = "2026-07-19"
PROFILES_RESOURCE = "profiles.json"

_HOSTS: Final = {
    CaptureProfile.CODEX_HOOKS_V1: (
        "Codex CLI",
        "0.144.6",
        None,
        (
            "https://learn.chatgpt.com/docs/hooks",
            "https://learn.chatgpt.com/docs/developer-commands?surface=cli#cli-codex-exec",
            "https://learn.chatgpt.com/docs/app-server",
        ),
    ),
    CaptureProfile.CLAUDE_CODE_HOOKS_V1: (
        "Claude Code",
        "2.1.204",
        None,
        (
            "https://code.claude.com/docs/en/hooks",
            "https://code.claude.com/docs/en/plugins-reference",
            "https://code.claude.com/docs/en/settings",
            "https://code.claude.com/docs/en/sessions",
        ),
    ),
    CaptureProfile.OPENCODE_PLUGIN_V1: (
        "OpenCode",
        "1.18.3",
        "127bdb30784d508cc556c71a0f32b508a3061517",
        (
            "https://opencode.ai/docs/plugins/",
            "https://opencode.ai/docs/sdk/#sessions",
            "https://opencode.ai/docs/sdk/#events",
            "https://opencode.ai/docs/config/#locations",
            "https://github.com/anomalyco/opencode/releases/tag/v1.18.3",
            "https://github.com/anomalyco/opencode/blob/127bdb30784d508cc556c71a0f32b508a3061517/packages/sdk/js/src/gen/types.gen.ts",
            "https://github.com/anomalyco/opencode/blob/127bdb30784d508cc556c71a0f32b508a3061517/packages/plugin/src/index.ts",
        ),
    ),
    CaptureProfile.PI_EXTENSION_V1: (
        "@earendil-works/pi-coding-agent",
        "0.80.10",
        "8dc78834cde4e329284cf505f9e3f99763df5529",
        (
            "https://github.com/earendil-works/pi/blob/8dc78834cde4e329284cf505f9e3f99763df5529/packages/coding-agent/docs/extensions.md",
            "https://github.com/earendil-works/pi/blob/8dc78834cde4e329284cf505f9e3f99763df5529/packages/coding-agent/docs/rpc.md",
            "https://github.com/earendil-works/pi/blob/8dc78834cde4e329284cf505f9e3f99763df5529/packages/coding-agent/docs/json.md",
            "https://github.com/earendil-works/pi/blob/8dc78834cde4e329284cf505f9e3f99763df5529/packages/coding-agent/docs/session-format.md",
            "https://github.com/earendil-works/pi/blob/8dc78834cde4e329284cf505f9e3f99763df5529/packages/coding-agent/docs/packages.md",
            "https://github.com/earendil-works/pi/blob/8dc78834cde4e329284cf505f9e3f99763df5529/packages/coding-agent/package.json",
        ),
    ),
}

_EVENTS: Final = {
    CaptureProfile.CODEX_HOOKS_V1: {
        "SessionStart": (
            ("hook_event_name", "session_id"),
            ("cwd", "source"),
            "window_open",
        ),
        "PreToolUse": (
            ("hook_event_name", "session_id", "tool_use_id"),
            ("cwd", "tool_input", "tool_name", "turn_id"),
            "action_observed",
        ),
        "PermissionRequest": (
            ("hook_event_name", "session_id"),
            ("cwd", "turn_id"),
            "coverage_only",
        ),
        "PostToolUse": (
            ("hook_event_name", "session_id", "tool_use_id"),
            ("cwd", "tool_input", "tool_name", "turn_id"),
            "action_closed_outcome_unavailable",
        ),
        "PreCompact": (
            ("hook_event_name", "session_id"),
            ("cwd", "trigger", "turn_id"),
            "coverage_boundary",
        ),
        "SubagentStart": (
            ("agent_id", "hook_event_name", "session_id"),
            ("agent_type", "cwd", "turn_id"),
            "correlation_only",
        ),
        "SubagentStop": (
            ("agent_id", "hook_event_name", "session_id"),
            ("agent_type", "cwd", "turn_id"),
            "correlation_only",
        ),
        "Stop": (
            ("hook_event_name", "session_id"),
            ("cwd", "turn_id"),
            "turn_closed",
        ),
    },
    CaptureProfile.CLAUDE_CODE_HOOKS_V1: {
        "SessionStart": (
            ("hook_event_name", "session_id"),
            ("cwd", "prompt_id", "source"),
            "window_open",
        ),
        "PreToolUse": (
            ("hook_event_name", "session_id", "tool_use_id"),
            ("cwd", "prompt_id", "tool_input", "tool_name"),
            "action_observed",
        ),
        "PostToolUse": (
            ("hook_event_name", "session_id", "tool_use_id"),
            ("cwd", "prompt_id", "tool_input", "tool_name"),
            "provider_claimed_success",
        ),
        "PostToolUseFailure": (
            ("hook_event_name", "session_id", "tool_use_id"),
            ("cwd", "is_interrupt", "prompt_id", "tool_input", "tool_name"),
            "provider_claimed_failure",
        ),
        "PostToolBatch": (
            ("hook_event_name", "session_id", "tool_calls[].tool_use_id"),
            ("cwd", "prompt_id", "tool_calls[].tool_name"),
            "reconciliation_only",
        ),
        "PermissionDenied": (
            ("hook_event_name", "session_id", "tool_use_id"),
            ("cwd", "prompt_id", "tool_input", "tool_name"),
            "provider_claimed_denial",
        ),
        "SubagentStart": (
            ("agent_id", "hook_event_name", "session_id"),
            ("agent_type", "cwd", "prompt_id"),
            "correlation_only",
        ),
        "SubagentStop": (
            ("agent_id", "hook_event_name", "session_id"),
            ("agent_type", "cwd", "prompt_id"),
            "correlation_only",
        ),
        "Stop": (
            ("hook_event_name", "session_id"),
            ("cwd", "prompt_id"),
            "turn_closed",
        ),
        "StopFailure": (
            ("hook_event_name", "session_id"),
            ("cwd", "error", "prompt_id"),
            "provider_claimed_controller_failure",
        ),
        "SessionEnd": (
            ("hook_event_name", "session_id"),
            ("cwd", "prompt_id", "reason"),
            "window_closed",
        ),
    },
    CaptureProfile.OPENCODE_PLUGIN_V1: {
        "message.part.updated": (
            (
                "id",
                "properties.part.callID",
                "properties.part.sessionID",
                "properties.part.state.status",
                "properties.part.tool",
                "properties.part.type",
                "properties.sessionID",
            ),
            ("properties.part.state.input",),
            "tool_state_discriminator",
        ),
        "session.idle": (
            ("id", "properties.sessionID"),
            (),
            "turn_closed",
        ),
        "session.error": (
            ("id",),
            ("properties.sessionID",),
            "controller_failure_when_session_correlated",
        ),
        "session.compacted": (
            ("id", "properties.sessionID"),
            (),
            "coverage_boundary",
        ),
        "session.deleted": (
            ("id", "properties.sessionID"),
            (),
            "window_closed",
        ),
        "dispose": ((), (), "flush_only"),
    },
    CaptureProfile.PI_EXTENSION_V1: {
        "session_start": (("session_id",), ("reason",), "window_open"),
        "before_agent_start": (("session_id",), (), "turn_open"),
        "tool_execution_start": (
            ("session_id", "toolCallId"),
            ("args", "toolName"),
            "action_observed",
        ),
        "tool_execution_end": (
            ("isError", "session_id", "toolCallId"),
            ("toolName",),
            "provider_claimed_tool_outcome",
        ),
        "agent_settled": (("session_id",), (), "stable_unit_closed"),
        "session_compact": (
            ("session_id",),
            ("fromExtension", "reason", "willRetry"),
            "coverage_boundary",
        ),
        "session_tree": (
            ("session_id",),
            ("fromExtension", "newLeafId", "oldLeafId"),
            "lineage_hint_only",
        ),
        "session_shutdown": (("session_id",), ("reason",), "window_closed"),
    },
}

_IGNORED_FIELDS: Final = {
    CaptureProfile.CODEX_HOOKS_V1: (
        "agent_transcript_path",
        "last_assistant_message",
        "model",
        "permission_mode",
        "stop_hook_active",
        "tool_response",
        "transcript_path",
    ),
    CaptureProfile.CLAUDE_CODE_HOOKS_V1: (
        "background_tasks",
        "effort",
        "error_details",
        "last_assistant_message",
        "permission_denied.reason_text",
        "permission_mode",
        "session_crons",
        "session_titles",
        "tool_calls[].tool_input",
        "tool_calls[].tool_response",
        "tool_failure.error_text",
        "tool_response",
        "transcript_path",
    ),
    CaptureProfile.OPENCODE_PLUGIN_V1: (
        "attachments",
        "metadata",
        "properties.part.id",
        "properties.part.messageID",
        "properties.part.state.error",
        "properties.part.state.output",
        "properties.part.state.raw",
        "properties.time",
        "session.deleted.properties.info",
        "session.error.properties.error",
        "titles",
    ),
    CaptureProfile.PI_EXTENSION_V1: (
        "compaction.entries",
        "compaction.summary",
        "images",
        "options",
        "previousSessionFile",
        "prompt",
        "result",
        "systemPrompt",
        "targetSessionFile",
    ),
}

_COVERAGE: Final = {
    CaptureProfile.CODEX_HOOKS_V1: (
        ("local_function_tool_hooks",),
        (
            "hosted_tools",
            "specialized_paths_outside_local_function_tool_hooks",
            "write_stdin_is_continuation_not_a_second_action",
        ),
    ),
    CaptureProfile.CLAUDE_CODE_HOOKS_V1: (
        ("selected_project_local_hook_events",),
        ("background_tasks", "prompt_and_output_content", "session_crons"),
    ),
    CaptureProfile.OPENCODE_PLUGIN_V1: (
        ("message.part.updated:tool",),
        ("history_materialization", "non_tool_parts", "parent_session_metadata"),
    ),
    CaptureProfile.PI_EXTENSION_V1: (
        ("observational_extension_callbacks",),
        ("blocking_or_rewriting_hooks", "history_backfill", "rpc_json_and_session_jsonl"),
    ),
}

_DETECTORS: Final = {
    CaptureProfile.CODEX_HOOKS_V1: {
        SignalType.REPEATED_ACTION: CapabilitySupport.CONDITIONAL,
        SignalType.REPEATED_FAILURE: CapabilitySupport.UNSUPPORTED,
        SignalType.TEST_FAILURE: CapabilitySupport.UNSUPPORTED,
        SignalType.TOOL_ERROR: CapabilitySupport.UNSUPPORTED,
        SignalType.CONTEXT_SHIFT: CapabilitySupport.UNSUPPORTED,
        SignalType.STALE_CONSTRAINT: CapabilitySupport.UNSUPPORTED,
        SignalType.STAGNATION: CapabilitySupport.UNSUPPORTED,
        SignalType.IRREVERSIBLE_ACTION: CapabilitySupport.UNSUPPORTED,
        SignalType.CONFLICT: CapabilitySupport.UNSUPPORTED,
    },
    CaptureProfile.CLAUDE_CODE_HOOKS_V1: {
        SignalType.REPEATED_ACTION: CapabilitySupport.CONDITIONAL,
        SignalType.REPEATED_FAILURE: CapabilitySupport.CONDITIONAL,
        SignalType.TEST_FAILURE: CapabilitySupport.UNSUPPORTED,
        SignalType.TOOL_ERROR: CapabilitySupport.SUPPORTED,
        SignalType.CONTEXT_SHIFT: CapabilitySupport.UNSUPPORTED,
        SignalType.STALE_CONSTRAINT: CapabilitySupport.UNSUPPORTED,
        SignalType.STAGNATION: CapabilitySupport.UNSUPPORTED,
        SignalType.IRREVERSIBLE_ACTION: CapabilitySupport.UNSUPPORTED,
        SignalType.CONFLICT: CapabilitySupport.UNSUPPORTED,
    },
    CaptureProfile.OPENCODE_PLUGIN_V1: {
        SignalType.REPEATED_ACTION: CapabilitySupport.CONDITIONAL,
        SignalType.REPEATED_FAILURE: CapabilitySupport.UNSUPPORTED,
        SignalType.TEST_FAILURE: CapabilitySupport.UNSUPPORTED,
        SignalType.TOOL_ERROR: CapabilitySupport.SUPPORTED,
        SignalType.CONTEXT_SHIFT: CapabilitySupport.UNSUPPORTED,
        SignalType.STALE_CONSTRAINT: CapabilitySupport.UNSUPPORTED,
        SignalType.STAGNATION: CapabilitySupport.UNSUPPORTED,
        SignalType.IRREVERSIBLE_ACTION: CapabilitySupport.UNSUPPORTED,
        SignalType.CONFLICT: CapabilitySupport.UNSUPPORTED,
    },
    CaptureProfile.PI_EXTENSION_V1: {
        SignalType.REPEATED_ACTION: CapabilitySupport.CONDITIONAL,
        SignalType.REPEATED_FAILURE: CapabilitySupport.UNSUPPORTED,
        SignalType.TEST_FAILURE: CapabilitySupport.UNSUPPORTED,
        SignalType.TOOL_ERROR: CapabilitySupport.SUPPORTED,
        SignalType.CONTEXT_SHIFT: CapabilitySupport.UNSUPPORTED,
        SignalType.STALE_CONSTRAINT: CapabilitySupport.UNSUPPORTED,
        SignalType.STAGNATION: CapabilitySupport.UNSUPPORTED,
        SignalType.IRREVERSIBLE_ACTION: CapabilitySupport.UNSUPPORTED,
        SignalType.CONFLICT: CapabilitySupport.UNSUPPORTED,
    },
}


def _installed_registry_bytes() -> bytes:
    return resources.files("saliencegate.integrations").joinpath(PROFILES_RESOURCE).read_bytes()


def _fixture_path_present(value: object, path: str) -> bool:
    head, *tail = path.split(".", maxsplit=1)
    remainder = tail[0] if tail else None
    if head.endswith("[]"):
        if type(value) is not dict:
            return False
        nested = value.get(head[:-2])
        return (
            type(nested) is list
            and bool(nested)
            and all(remainder is None or _fixture_path_present(item, remainder) for item in nested)
        )
    if type(value) is not dict or head not in value:
        return False
    return remainder is None or _fixture_path_present(value[head], remainder)


def test_registry_resource_is_canonical_strict_and_contains_only_the_four_profiles() -> None:
    source = _installed_registry_bytes()
    decoded = json.loads(source.decode("utf-8"))
    registry = load_capture_capability_registry()

    assert canonical_json(decoded) == source
    assert registry == CaptureCapabilityRegistry.model_validate_json(source)
    assert registry.schema_version == "capture-capability-registry/v1"
    assert tuple(profile.profile_id for profile in registry.profiles) == tuple(CaptureProfile)
    assert tuple(CaptureProfile) == (
        CaptureProfile.CODEX_HOOKS_V1,
        CaptureProfile.CLAUDE_CODE_HOOKS_V1,
        CaptureProfile.OPENCODE_PLUGIN_V1,
        CaptureProfile.PI_EXTENSION_V1,
    )
    assert tuple(profile.value for profile in CaptureProfile) == (
        "codex-hooks/v1",
        "claude-code-hooks/v1",
        "opencode-plugin/v1",
        "pi-extension/v1",
    )

    with pytest.raises(ValidationError):
        CaptureCapabilityRegistry.model_validate({**decoded, "unexpected": True})
    with pytest.raises(ValidationError):
        CaptureCapabilityRegistry.model_validate({**decoded, "profiles": decoded["profiles"][:-1]})
    with pytest.raises(ValidationError):
        CaptureCapabilityRegistry.model_validate(
            {**decoded, "profiles": [*decoded["profiles"], decoded["profiles"][0]]}
        )
    with pytest.raises(ValidationError):
        CaptureCapabilityRegistry.model_validate(
            {**decoded, "profiles": list(reversed(decoded["profiles"]))}
        )
    with pytest.raises(ValidationError, match="frozen"):
        registry.__setattr__("profiles", ())


@pytest.mark.parametrize("profile_id", tuple(CaptureProfile))
def test_profile_freezes_the_exact_audited_host_sources_and_security_claims(
    profile_id: CaptureProfile,
) -> None:
    profile = capture_profile(profile_id)
    host_name, host_version, upstream_revision, sources = _HOSTS[profile_id]

    assert profile.schema_version == "capture-capability-manifest/v1"
    assert profile.profile_id is profile_id
    assert profile.host_name == host_name
    assert profile.host_version == host_version
    assert profile.audit_date == AUDIT_DATE
    assert profile.upstream_revision == upstream_revision
    assert profile.official_sources == sources
    assert profile.source_authentication == "none_same_user_untrusted"
    assert profile.raw_content_persisted is False
    assert profile.transcript_read is False
    assert profile.complete_execution_session_coverage is False
    assert profile.decision_authority is False
    assert profile.model_calls == 0
    assert profile.timestamp_authority == "local_observation"
    assert profile.sequence_authority == "local_receipt_order"
    assert profile.rollback_detection == "none"
    assert profile.at_rest_integrity == "hmac_sha256_local_mutation_detection"


@pytest.mark.parametrize("profile_id", tuple(CaptureProfile))
def test_profile_event_fields_outcome_authority_omissions_and_coverage_are_exact(
    profile_id: CaptureProfile,
) -> None:
    profile = capture_profile(profile_id)
    expected = _EVENTS[profile_id]
    actual = {event.event_name: event for event in profile.events}

    assert tuple(actual) == tuple(expected)
    for event_name, (critical, optional, outcome_authority) in expected.items():
        event = actual[event_name]
        assert event.critical_fields == critical
        assert event.optional_fields == optional
        assert event.outcome_authority == outcome_authority
        assert not set(event.critical_fields) & set(event.optional_fields)
        assert not set(event.ignored_fields) & (
            set(event.critical_fields) | set(event.optional_fields)
        )

    ignored = tuple(sorted({field for event in profile.events for field in event.ignored_fields}))
    assert ignored == _IGNORED_FIELDS[profile_id]
    assert (profile.tool_coverage, profile.coverage_exclusions) == _COVERAGE[profile_id]


@pytest.mark.parametrize("profile_id", tuple(CaptureProfile))
def test_detector_matrix_is_closed_and_records_an_omission_for_every_non_supported_detector(
    profile_id: CaptureProfile,
) -> None:
    matrix = capture_profile(profile_id).detectors

    assert {item.signal_type: item.support for item in matrix} == _DETECTORS[profile_id]
    assert tuple(item.signal_type for item in matrix) == tuple(SignalType)
    assert all(
        bool(item.omissions) is (item.support is not CapabilitySupport.SUPPORTED) for item in matrix
    )


@pytest.mark.parametrize("profile_id", tuple(CaptureProfile))
def test_manifest_and_fully_synthetic_fixture_are_bound_by_canonical_sha256(
    profile_id: CaptureProfile,
) -> None:
    profile = capture_profile(profile_id)
    digest = capture_capability_digest(profile)

    assert digest == hashlib.sha256(canonical_json(profile)).hexdigest()
    assert len(profile.fixtures) == 1
    fixture = profile.fixtures[0]
    fixture_bytes = resources.files("saliencegate.integrations").joinpath(fixture.path).read_bytes()
    fixture_body = json.loads(fixture_bytes.decode("utf-8"))

    assert canonical_json(fixture_body) == fixture_bytes
    assert hashlib.sha256(fixture_bytes).hexdigest() == fixture.sha256
    assert fixture.fixture_id == f"{profile_id.value}-synthetic/v1"
    assert fixture.path.startswith("fixtures/") and fixture.path.endswith(".json")
    assert fixture.kind == "fully_synthetic_generated"
    assert fixture.transform_id == "hand_authored_from_audited_shape/v1"
    assert fixture.source_payload_retained is False
    assert fixture_body["schema_version"] == "capture-native-fixture/v1"
    assert fixture_body["profile_id"] == profile_id.value
    assert fixture_body["provenance"] == "fully_synthetic_no_provider_or_model_call"

    capabilities = {item.event_name: item for item in profile.events}
    for native_event in fixture_body["events"]:
        capability = capabilities[native_event["event_name"]]
        assert all(
            _fixture_path_present(native_event["payload"], field)
            for field in capability.critical_fields
        )


def test_binding_rejects_profile_and_manifest_digest_mismatch_without_value_leakage() -> None:
    profile = capture_profile(CaptureProfile.CODEX_HOOKS_V1)
    digest = capture_capability_digest(profile)

    assert validate_capture_capability_binding(profile.profile_id, digest) == profile

    secret = "manifest-binding-secret"
    for profile_id, declared_digest in (
        (CaptureProfile.PI_EXTENSION_V1, digest),
        (profile.profile_id, "f" * 64),
        (secret, digest),
        (profile.profile_id, secret),
    ):
        with pytest.raises(CaptureCapabilityError) as captured:
            validate_capture_capability_binding(profile_id, declared_digest)  # type: ignore[arg-type]
        assert secret not in str(captured.value)
        assert secret not in repr(captured.value)


def test_additive_host_fields_are_compatible_but_never_gain_authority() -> None:
    profile = capture_profile(CaptureProfile.CODEX_HOOKS_V1)
    event = next(item for item in profile.events if item.event_name == "PreToolUse")
    audited_fields = frozenset((*event.critical_fields, *event.optional_fields))
    additive_fields = frozenset((*audited_fields, "future_provider_field", "opaque_extension"))

    verified = classify_capture_compatibility(
        profile,
        host_version=profile.host_version,
        observed_event=event.event_name,
        observed_fields=additive_fields,
    )
    unverified = classify_capture_compatibility(
        profile,
        host_version="0.144.7",
        observed_event=event.event_name,
        observed_fields=additive_fields,
    )

    assert verified is CompatibilityStatus.VERIFIED
    assert unverified is CompatibilityStatus.SCHEMA_COMPATIBLE_UNVERIFIED_VERSION
    assert "future_provider_field" not in event.critical_fields
    assert "future_provider_field" not in event.optional_fields


@pytest.mark.parametrize("profile_id", tuple(CaptureProfile))
def test_missing_critical_field_and_unknown_event_fail_closed(profile_id: CaptureProfile) -> None:
    profile = capture_profile(profile_id)
    event = next(item for item in profile.events if item.critical_fields)
    observed = frozenset((*event.critical_fields[1:], *event.optional_fields, "additive"))

    assert (
        classify_capture_compatibility(
            profile,
            host_version=profile.host_version,
            observed_event=event.event_name,
            observed_fields=observed,
        )
        is CompatibilityStatus.INCOMPATIBLE
    )
    assert (
        classify_capture_compatibility(
            profile,
            host_version=profile.host_version,
            observed_event="unlisted-provider-event",
            observed_fields=frozenset(),
        )
        is CompatibilityStatus.INCOMPATIBLE
    )


def test_manifest_validation_errors_hide_untrusted_values() -> None:
    registry = load_capture_capability_registry()
    body = registry.model_dump(mode="json", warnings=False)
    secret = "native-profile-secret"
    body["profiles"][0]["events"][0]["unexpected"] = secret

    with pytest.raises(ValidationError) as captured:
        CaptureCapabilityRegistry.model_validate(body)

    assert secret not in str(captured.value)
    assert secret not in repr(captured.value)
