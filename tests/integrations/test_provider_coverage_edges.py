from __future__ import annotations

import importlib
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace

import pytest

import saliencegate.integrations.claude_code as claude
import saliencegate.integrations.codex as codex
import saliencegate.integrations.config_files as config_files
import saliencegate.integrations.environment as capture_environment
import saliencegate.integrations.hook as hook
import saliencegate.integrations.launcher_materialization as launcher_materialization
import saliencegate.integrations.opencode as opencode
import saliencegate.integrations.pi as pi
from saliencegate.capture.adapters import CaptureAdapterContractError
from saliencegate.capture.capabilities import (
    CaptureProfile,
    CompatibilityStatus,
    capture_capability_digest,
    capture_profile,
)
from saliencegate.capture.health import CaptureHealthCode
from saliencegate.capture.identities import CaptureDigestContext
from saliencegate.domain import canonical_json
from saliencegate.integrations.bootstrap import IntegrationBootstrap
from saliencegate.integrations.hook import CaptureHookDependencies, CaptureHookError
from saliencegate.integrations.launcher_renderer import CaptureLauncherPlatform
from saliencegate.security import InstallationKey

KEY = InstallationKey(b"e" * 32)
CONTEXT = CaptureDigestContext(KEY)
CODEX_CONNECTION = "sg-" + "c" * 48
CLAUDE_CONNECTION = "sg-" + "d" * 48
OPENCODE_CONNECTION = "sg-" + "o" * 48
PI_CONNECTION = "sg-" + "p" * 48
ZERO = "0" * 64


def _bootstrap(profile: CaptureProfile, connection_id: str, name: str) -> IntegrationBootstrap:
    return IntegrationBootstrap(
        profile=profile,
        connection_id=connection_id,
        launcher_path=Path(f"/private/tmp/saliencegate-{name}-hook"),
        capability_digest=capture_capability_digest(capture_profile(profile)),
        bundle_digest="b" * 64,
        receipt_mac="a" * 64,
    )


def _opencode_adapter() -> opencode.OpenCodeCaptureAdapter:
    return opencode.OpenCodeCaptureAdapter(
        connection_id=OPENCODE_CONNECTION,
        bootstrap=_bootstrap(CaptureProfile.OPENCODE_PLUGIN_V1, OPENCODE_CONNECTION, "opencode"),
        project_root=Path("/synthetic/opencode/project"),
    )


def _pi_adapter() -> pi.PiCaptureAdapter:
    return pi.PiCaptureAdapter(
        connection_id=PI_CONNECTION,
        bootstrap=_bootstrap(CaptureProfile.PI_EXTENSION_V1, PI_CONNECTION, "pi"),
        project_root=Path("/synthetic/pi/project"),
    )


def _opencode_batch(
    events: list[object],
    *,
    bootstrap: dict[str, object] | None = None,
    batch_id: object = "1" * 64,
    session_id: object = "opencode-session",
    chunk_index: object = 0,
    chunk_count: object = 1,
    schema_version: object = "capture-batch/v1",
    extra: Mapping[str, object] | None = None,
) -> bytes:
    sidecar = _bootstrap(
        CaptureProfile.OPENCODE_PLUGIN_V1,
        OPENCODE_CONNECTION,
        "opencode",
    ).model_dump(mode="json", warnings="error")
    document: dict[str, object] = {
        "schema_version": schema_version,
        "bootstrap": sidecar if bootstrap is None else bootstrap,
        "batch_id": batch_id,
        "session_id": session_id,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "events": events,
    }
    if extra:
        document.update(extra)
    return canonical_json(document)


def _pi_batch(
    events: list[object],
    *,
    bootstrap: dict[str, object] | None = None,
    batch_id: object = "2" * 64,
    session_id: object = "pi-session",
    window_discriminator: object = "3" * 64,
    chunk_index: object = 0,
    chunk_count: object = 1,
    schema_version: object = "capture-batch/v1",
    extra: Mapping[str, object] | None = None,
) -> bytes:
    sidecar = _bootstrap(
        CaptureProfile.PI_EXTENSION_V1,
        PI_CONNECTION,
        "pi",
    ).model_dump(mode="json", warnings="error")
    document: dict[str, object] = {
        "schema_version": schema_version,
        "bootstrap": sidecar if bootstrap is None else bootstrap,
        "batch_id": batch_id,
        "session_id": session_id,
        "window_discriminator": window_discriminator,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "events": events,
    }
    if extra:
        document.update(extra)
    return canonical_json(document)


def _opencode_event(kind: str, **values: object) -> dict[str, object]:
    return {"kind": kind, "session_id": "opencode-session", **values}


def _pi_event(kind: str, **values: object) -> dict[str, object]:
    return {
        "kind": kind,
        "session_id": "pi-session",
        "window_discriminator": "3" * 64,
        **values,
    }


@pytest.mark.parametrize(
    ("function", "value", "maximum", "expected"),
    [
        (claude._exact_text, object(), 8, None),
        (claude._exact_text, "", 8, None),
        (claude._exact_text, "abcdefghi", 8, None),
        (codex._exact_text, object(), 8, None),
        (codex._exact_text, "ok", 8, "ok"),
        (opencode._exact_text, object(), 8, None),
        (opencode._exact_text, "\ud800", 8, None),
        (opencode._exact_text, "", 8, None),
        (pi._exact_text, object(), 8, None),
        (pi._exact_text, "\ud800", 8, None),
        (pi._exact_text, "abcdefghi", 8, None),
    ],
)
def test_provider_exact_text_boundaries(function, value: object, maximum: int, expected) -> None:
    assert function(value, maximum=maximum) == expected


@pytest.mark.parametrize(
    ("function", "cases"),
    [
        (
            claude._tool_class,
            {
                "Bash": "shell",
                "Edit": "file_write",
                "Read": "file_read",
                "Glob": "search",
                "WebFetch": "network",
                "Agent": "subagent",
                "unknown": "other",
            },
        ),
        (
            codex._tool_class,
            {
                "Bash": "shell",
                "apply_patch": "file_write",
                "view_image": "file_read",
                "spawn_agent": "subagent",
                "unknown": "other",
            },
        ),
        (
            opencode._tool_class,
            {
                "Terminal": "shell",
                "MultiEdit": "file_write",
                "VIEW_IMAGE": "file_read",
                "CodeSearch": "search",
                "WebSearch": "network",
                "Task": "subagent",
                "unknown": "other",
            },
        ),
        (
            pi._tool_class,
            {
                "Terminal": "shell",
                "MultiEdit": "file_write",
                "VIEW_IMAGE": "file_read",
                "CodeSearch": "search",
                "WebSearch": "network",
                "Task": "subagent",
                "unknown": "other",
            },
        ),
    ],
)
def test_provider_tool_classification_tables(function, cases: Mapping[str, str]) -> None:
    assert {value: function(value) for value in cases} == cases


@pytest.mark.parametrize(
    ("module", "error"),
    [
        (opencode, opencode.OpenCodeIntegrationError),
        (pi, pi.PiIntegrationError),
    ],
)
def test_bridge_exact_key_boundaries(module, error: type[Exception]) -> None:
    required = frozenset({"required"})
    assert module._exact_keys({"required": 1}, required=required) == {"required": 1}
    assert module._exact_keys(
        {"required": 1, "optional": 2},
        required=required,
        optional=frozenset({"optional"}),
    ) == {"required": 1, "optional": 2}
    for value in (None, {}, {"required": 1, "extra": 2}, {1: "non-string"}):
        with pytest.raises(error):
            module._exact_keys(value, required=required)
    with pytest.raises(error):
        module._exact_keys(
            {"required": 1, 1: "non-string"},
            required=required,
            optional=frozenset({1}),  # type: ignore[arg-type]
        )


def test_environment_projection_wraps_hostile_mapping_errors() -> None:
    class HostileMapping(Mapping[str, str]):
        def __iter__(self):
            raise RuntimeError("secret-sentinel")

        def __len__(self) -> int:
            return 1

        def __getitem__(self, key: str) -> str:
            raise AssertionError(key)

    with pytest.raises(capture_environment.CaptureEnvironmentError) as raised:
        capture_environment.environment_without_provider_credentials(HostileMapping())
    assert "secret-sentinel" not in str(raised.value)

    with pytest.raises(capture_environment.CaptureEnvironmentError):
        capture_environment.environment_without_provider_credentials(object())  # type: ignore[arg-type]


def test_supported_versions_and_probe_records_reject_coercion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = (
        (claude, claude.ClaudeCodeIntegrationError, "2.1.204", "2.1.205"),
        (codex, codex.CodexIntegrationError, "0.144.6", "0.144.7"),
    )
    for module, error, audited, newer in cases:
        assert module._supported_version_parts(audited) == tuple(
            int(part) for part in audited.split(".")
        )
        for invalid in (None, True, "1.2", "01.2.3", "9.9.9"):
            with pytest.raises(error):
                module._supported_version_parts(invalid)
        probe_type = module.ClaudeCodeVersionProbe if module is claude else module.CodexVersionProbe
        with pytest.raises(error):
            probe_type(audited, CompatibilityStatus.SCHEMA_COMPATIBLE_UNVERIFIED_VERSION)
        assert (
            probe_type(
                newer,
                CompatibilityStatus.SCHEMA_COMPATIBLE_UNVERIFIED_VERSION,
            ).host_version
            == newer
        )

        class AlwaysMatch:
            @staticmethod
            def fullmatch(_value: str) -> object:
                return object()

        monkeypatch.setattr(module, "_HOST_VERSION", AlwaysMatch())
        with pytest.raises(error):
            module._supported_version_parts("not.numeric.parts")


@pytest.mark.parametrize(
    ("module", "error", "runner_name", "timeout"),
    [
        (claude, claude.ClaudeCodeIntegrationError, "_bounded_version_runner", 2.0),
        (codex, codex.CodexIntegrationError, "_bounded_version_runner", 2.0),
    ],
)
def test_version_runner_rejects_invalid_call_contracts(
    module,
    error: type[Exception],
    runner_name: str,
    timeout: float,
    tmp_path: Path,
) -> None:
    runner = getattr(module, runner_name)
    kwargs: dict[str, object] = {
        "input": b"not-empty",
        "capture_output": True,
        "check": False,
        "timeout": timeout,
        "env": {},
    }
    if module is claude:
        kwargs["cwd"] = str(tmp_path)
    with pytest.raises(error):
        runner(("provider", "--version"), **kwargs)


def test_exact_executable_and_windows_shim_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "provider"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    for module, error in (
        (claude, claude.ClaudeCodeIntegrationError),
        (codex, codex.CodexIntegrationError),
    ):
        assert module._exact_executable(executable) == executable
        for invalid in (Path("relative"), tmp_path, tmp_path / "missing"):
            with pytest.raises(error):
                module._exact_executable(invalid)
        link = tmp_path / f"{module.__name__.rsplit('.', 1)[-1]}-link"
        link.symlink_to(executable)
        with pytest.raises(error):
            module._exact_executable(link)

    shim = PureWindowsPath(r"C:\tools\claude.cmd")
    shell = PureWindowsPath(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")
    command = claude._windows_shim_version_command(shim, shell)
    assert command[0] == str(shell)
    assert command[-2] == "-Command"
    for bad_shim, bad_shell in (
        (PureWindowsPath(r"relative\claude.cmd"), shell),
        (PureWindowsPath(r"C:\tools\claude.exe"), shell),
        (shim, PureWindowsPath(r"C:\tools\cmd.exe")),
    ):
        with pytest.raises(claude.ClaudeCodeIntegrationError):
            claude._windows_shim_version_command(bad_shim, bad_shell)

    monkeypatch.setattr(
        claude.base64, "b64encode", lambda _value: (_ for _ in ()).throw(TypeError())
    )
    with pytest.raises(claude.ClaudeCodeIntegrationError):
        claude._windows_shim_version_command(shim, shell)


def test_claude_path_candidate_validation_is_platform_explicit() -> None:
    cwd = PurePosixPath("/workspace")
    assert claude._claude_executable_candidates(
        '"/one":relative:',
        windows_pathext=None,
        windows=False,
        cwd=cwd,
    ) == (
        Path("/one/claude"),
        Path("/workspace/relative/claude"),
        Path("/workspace/claude"),
    )
    windows = claude._claude_executable_candidates(
        r'"C:\one";bin',
        windows_pathext=".EXE;.CMD;.EXE",
        windows=True,
        cwd=PureWindowsPath(r"C:\workspace"),
    )
    assert windows == (
        PureWindowsPath(r"C:\one\claude.EXE"),
        PureWindowsPath(r"C:\one\claude.CMD"),
        PureWindowsPath(r"C:\workspace\bin\claude.EXE"),
        PureWindowsPath(r"C:\workspace\bin\claude.CMD"),
    )
    invalid_calls = (
        (object(), None, False, cwd),
        ("x", None, False, PurePosixPath("relative")),
        ("x", "." + "X" * 4_096, True, PureWindowsPath(r"C:\workspace")),
        ("x", "BAD", True, PureWindowsPath(r"C:\workspace")),
        ("bad\x00path", None, False, cwd),
        (":".join("x" for _ in range(2_049)), None, False, cwd),
    )
    for configured_path, pathext, windows_flag, selected_cwd in invalid_calls:
        with pytest.raises(claude.ClaudeCodeIntegrationError):
            claude._claude_executable_candidates(
                configured_path,  # type: ignore[arg-type]
                windows_pathext=pathext,
                windows=windows_flag,
                cwd=selected_cwd,
            )


def test_toml_and_hook_fragments_reject_invalid_render_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert codex._toml_string("quoted\nvalue") == '"quoted\\nvalue"'
    for value in ("", "bad\x00value"):
        with pytest.raises(codex.CodexIntegrationError):
            codex._toml_string(value)

    launcher = Path("/absolute/capture-hook")
    fragment = codex._codex_hook_fragment(launcher)
    assert fragment.count(b"[[hooks.") == len(codex.CODEX_HOOK_EVENTS) * 2
    assert b'matcher = "startup|resume|clear|compact"' in fragment

    claude_fragment = claude._claude_code_hook_fragment(PurePosixPath("/absolute/capture-hook"))
    assert claude_fragment.count(b'"type":"command"') == len(claude.CLAUDE_CODE_HOOK_EVENTS)
    for invalid in (PurePosixPath("relative"), PurePosixPath("/"), PurePosixPath("/bad\x00name")):
        with pytest.raises(claude.ClaudeCodeIntegrationError):
            claude._claude_code_hook_fragment(invalid)

    windows_fragment = claude._claude_code_hook_fragment(
        PureWindowsPath(r"C:\state\capture-hook.cmd"),
        windows_powershell=PureWindowsPath(
            r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
        ),
    )
    assert b'"-NonInteractive"' in windows_fragment
    with pytest.raises(claude.ClaudeCodeIntegrationError):
        claude._claude_code_hook_fragment(
            PureWindowsPath(r"C:\state\capture-hook.exe"),
            windows_powershell=PureWindowsPath(r"C:\Windows\System32\cmd.exe"),
        )

    monkeypatch.setattr(claude, "canonical_json", lambda _value: b"[]")
    with pytest.raises(claude.ClaudeCodeIntegrationError):
        claude._claude_code_hook_fragment(PurePosixPath("/absolute/capture-hook"))


def _provider_fixture(provider: str, event_name: str) -> dict[str, object]:
    fixture = Path(__file__).parents[2] / "src" / "saliencegate" / "integrations" / "fixtures"
    document = json.loads((fixture / f"{provider}-hooks-v1.json").read_bytes())
    for entry in document["events"]:
        if entry["event_name"] == event_name:
            return dict(entry["payload"])
    raise AssertionError(event_name)


def test_claude_hook_policy_schema_accepts_each_pinned_handler_type() -> None:
    handlers = (
        {"type": "command", "command": "/bin/true", "args": ["safe"], "shell": "bash"},
        {"type": "prompt", "prompt": "safe", "continueOnBlock": False},
        {"type": "agent", "prompt": "safe", "once": True},
        {
            "type": "http",
            "url": "https://example.invalid/hook",
            "headers": {"x-safe": "value"},
            "allowedEnvVars": ["SAFE"],
        },
        {"type": "mcp_tool", "server": "safe", "tool": "safe", "input": {}},
    )
    for handler in handlers:
        claude._validate_existing_hook_groups(
            {"hooks": {"PreToolUse": [{"matcher": "*", "hooks": [handler]}]}}
        )
    claude._validate_existing_hook_groups({})


@pytest.mark.parametrize(
    "document",
    [
        {"hooks": []},
        {"hooks": {"PreToolUse": {}}},
        {"hooks": {"PreToolUse": [None]}},
        {"hooks": {"PreToolUse": [{"matcher": 1, "hooks": []}]}},
        {"hooks": {"PreToolUse": [{}]}},
        {"hooks": {"PreToolUse": [{"hooks": [None]}]}},
        {"hooks": {"PreToolUse": [{"hooks": [{"type": 1}]}]}},
        {"hooks": {"PreToolUse": [{"hooks": [{"type": "unknown"}]}]}},
        {"hooks": {"PreToolUse": [{"hooks": [{"type": "command"}]}]}},
        {
            "hooks": {
                "PreToolUse": [{"hooks": [{"type": "command", "command": "/bin/true", "if": 1}]}]
            }
        },
        {
            "hooks": {
                "PreToolUse": [{"hooks": [{"type": "command", "command": "/bin/true", "once": 1}]}]
            }
        },
        {
            "hooks": {
                "PreToolUse": [
                    {"hooks": [{"type": "command", "command": "/bin/true", "rewakeMessage": ""}]}
                ]
            }
        },
        {
            "hooks": {
                "PreToolUse": [
                    {"hooks": [{"type": "command", "command": "/bin/true", "timeout": object()}]}
                ]
            }
        },
        {
            "hooks": {
                "PreToolUse": [
                    {"hooks": [{"type": "command", "command": "/bin/true", "timeout": 0}]}
                ]
            }
        },
        {
            "hooks": {
                "PreToolUse": [
                    {"hooks": [{"type": "command", "command": "/bin/true", "args": "bad"}]}
                ]
            }
        },
        {
            "hooks": {
                "PreToolUse": [
                    {"hooks": [{"type": "command", "command": "/bin/true", "args": [1]}]}
                ]
            }
        },
        {
            "hooks": {
                "PreToolUse": [
                    {"hooks": [{"type": "command", "command": "/bin/true", "shell": "zsh"}]}
                ]
            }
        },
        {
            "hooks": {
                "PreToolUse": [
                    {"hooks": [{"type": "http", "url": "https://example.invalid", "headers": []}]}
                ]
            }
        },
        {
            "hooks": {
                "PreToolUse": [
                    {
                        "hooks": [
                            {
                                "type": "http",
                                "url": "https://example.invalid",
                                "headers": {"x": 1},
                            }
                        ]
                    }
                ]
            }
        },
        {
            "hooks": {
                "PreToolUse": [
                    {
                        "hooks": [
                            {
                                "type": "http",
                                "url": "https://example.invalid",
                                "allowedEnvVars": "bad",
                            }
                        ]
                    }
                ]
            }
        },
        {
            "hooks": {
                "PreToolUse": [
                    {
                        "hooks": [
                            {
                                "type": "http",
                                "url": "https://example.invalid",
                                "allowedEnvVars": [1],
                            }
                        ]
                    }
                ]
            }
        },
        {"hooks": {"PreToolUse": [{"hooks": [{"type": "http", "url": "not a URL"}]}]}},
        {
            "hooks": {
                "PreToolUse": [
                    {"hooks": [{"type": "mcp_tool", "server": "s", "tool": "t", "input": []}]}
                ]
            }
        },
    ],
)
def test_claude_hook_policy_schema_rejects_ambiguous_handlers(document: object) -> None:
    with pytest.raises(claude.ClaudeCodeIntegrationError):
        claude._validate_existing_hook_groups(document)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (b"{}", False),
        (b'{"disableAllHooks":false}', False),
        (b'{"disableAllHooks":true}', True),
    ],
)
def test_claude_hook_disablement_is_strict(source: bytes, expected: bool) -> None:
    assert claude._hooks_explicitly_disabled(source) is expected


@pytest.mark.parametrize(
    "source",
    [
        b"[]",
        b'{"disableAllHooks":1}',
        b'{"duplicate":1,"duplicate":2}',
        b'{"value":NaN}',
        b"not-json",
    ],
)
def test_claude_hook_disablement_rejects_noncanonical_policy(source: bytes) -> None:
    with pytest.raises(claude.ClaudeCodeIntegrationError):
        claude._hooks_explicitly_disabled(source)


def test_claude_json_callbacks_reject_duplicate_and_constants() -> None:
    assert claude._strict_json_object([("a", 1)]) == {"a": 1}
    with pytest.raises(claude.ClaudeCodeIntegrationError):
        claude._strict_json_object([("a", 1), ("a", 2)])
    with pytest.raises(claude.ClaudeCodeIntegrationError):
        claude._reject_json_constant("NaN")


def test_provider_action_identity_fallbacks_are_pseudonymous() -> None:
    call_material = b"call-material"
    unavailable, name, authority = claude._action_identity(
        CONTEXT,
        document={},
        call_material=call_material,
    )
    assert name is None and authority == "unavailable"
    coarse, name, authority = claude._action_identity(
        CONTEXT,
        document={"tool_name": "Read"},
        call_material=call_material,
    )
    assert name == "Read" and authority == "coarse" and coarse != unavailable

    unavailable, name, authority = codex._action_identity(
        CONTEXT,
        document={},
        call_material=call_material,
    )
    assert name is None and authority == "unavailable"
    coarse, _, authority = codex._action_identity(
        CONTEXT,
        document={"tool_name": "Read"},
        call_material=call_material,
    )
    exact, _, exact_authority = codex._action_identity(
        CONTEXT,
        document={"tool_name": "Read", "tool_input": {"path": "synthetic"}},
        call_material=call_material,
    )
    fallback, _, fallback_authority = codex._action_identity(
        CONTEXT,
        document={"tool_name": "Read", "tool_input": object()},
        call_material=call_material,
    )
    assert authority == "coarse"
    assert exact_authority == "exact"
    assert fallback_authority == "unavailable"
    assert fallback == unavailable
    assert len({unavailable, coarse, exact}) == 3


def test_claude_batch_field_validation_is_conservative() -> None:
    assert claude._validated_batch_fields({"tool_calls[].tool_use_id": True}) == frozenset()
    assert claude._validated_batch_fields({"other": 1}) == {"other"}
    assert claude._validated_batch_fields({"tool_calls": ()}) == {"tool_calls"}
    assert claude._validated_batch_fields({"tool_calls": (None,)}) == {"tool_calls"}
    assert claude._validated_batch_fields({"tool_calls": ({},)}) == {"tool_calls"}
    assert claude._validated_batch_fields(
        {"tool_calls": ({"tool_use_id": "same"}, {"tool_use_id": "same"})}
    ) == {"tool_calls"}
    assert claude._validated_batch_fields(
        {"tool_calls": ({"tool_use_id": "one"}, {"tool_use_id": "two"})}
    ) == {"tool_calls", "tool_calls[].tool_use_id"}


def test_codex_and_claude_adapter_error_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    codex_adapter = codex.CodexCaptureAdapter(connection_id=CODEX_CONNECTION)
    claude_adapter = claude.ClaudeCodeCaptureAdapter(connection_id=CLAUDE_CONNECTION)
    assert repr(codex_adapter) == str(codex_adapter) == "CodexCaptureAdapter(<redacted>)"
    assert repr(claude_adapter) == str(claude_adapter) == "ClaudeCodeCaptureAdapter(<redacted>)"
    assert codex_adapter.capabilities().profile_id is CaptureProfile.CODEX_HOOKS_V1
    assert claude_adapter.capabilities().profile_id is CaptureProfile.CLAUDE_CODE_HOOKS_V1

    for constructor, error in (
        (codex.CodexCaptureAdapter, codex.CodexIntegrationError),
        (claude.ClaudeCodeCaptureAdapter, claude.ClaudeCodeIntegrationError),
    ):
        with pytest.raises(error):
            constructor(connection_id="short")

    with pytest.raises(codex.CodexIntegrationError):
        codex_adapter.adapt_bytes(b"{}", context=object())  # type: ignore[arg-type]
    with pytest.raises(claude.ClaudeCodeIntegrationError):
        claude_adapter.adapt_bytes(b"{}", context=object())  # type: ignore[arg-type]

    codex_bad = (
        ("PreToolUse", {"tool_use_id": ""}),
        ("SubagentStart", {"agent_id": ""}),
    )
    for event_name, mutation in codex_bad:
        document = _provider_fixture("codex", event_name)
        document.update(mutation)
        with pytest.raises(codex.CodexIntegrationError):
            codex_adapter.adapt_bytes(canonical_json(document), context=CONTEXT)

    claude_bad = (
        ("PostToolBatch", {"tool_calls": []}),
        ("PreToolUse", {"tool_use_id": ""}),
        ("PostToolUseFailure", {"is_interrupt": "yes"}),
        ("SubagentStart", {"agent_id": ""}),
        ("StopFailure", {"prompt_id": ""}),
    )
    for event_name, mutation in claude_bad:
        document = _provider_fixture("claude-code", event_name)
        document.update(mutation)
        with pytest.raises(claude.ClaudeCodeIntegrationError):
            claude_adapter.adapt_bytes(canonical_json(document), context=CONTEXT)

    monkeypatch.setattr(
        codex, "read_bounded_json", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError())
    )
    with pytest.raises(codex.CodexIntegrationError):
        codex_adapter.adapt_bytes(b"{}", context=CONTEXT)
    monkeypatch.setattr(
        claude,
        "read_bounded_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError()),
    )
    with pytest.raises(claude.ClaudeCodeIntegrationError):
        claude_adapter.adapt_bytes(b"{}", context=CONTEXT)


def test_adapter_exception_wrappers_and_terminal_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    codex_adapter = codex.CodexCaptureAdapter(connection_id=CODEX_CONNECTION)
    claude_adapter = claude.ClaudeCodeCaptureAdapter(connection_id=CLAUDE_CONNECTION)
    with monkeypatch.context() as scoped:
        scoped.setattr(
            codex,
            "CaptureAdapterCapabilities",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError()),
        )
        with pytest.raises(codex.CodexIntegrationError):
            codex_adapter.capabilities()
    with monkeypatch.context() as scoped:
        scoped.setattr(
            claude,
            "CaptureAdapterCapabilities",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError()),
        )
        with pytest.raises(claude.ClaudeCodeIntegrationError):
            claude_adapter.capabilities()

    with monkeypatch.context() as scoped:
        scoped.setattr(
            codex, "capture_profile", lambda _profile: (_ for _ in ()).throw(RuntimeError())
        )
        with pytest.raises(codex.CodexIntegrationError):
            codex.CodexCaptureAdapter(connection_id=CODEX_CONNECTION)
    with monkeypatch.context() as scoped:
        scoped.setattr(
            claude, "capture_profile", lambda _profile: (_ for _ in ()).throw(RuntimeError())
        )
        with pytest.raises(claude.ClaudeCodeIntegrationError):
            claude.ClaudeCodeCaptureAdapter(connection_id=CLAUDE_CONNECTION)

    batch = _provider_fixture("claude-code", "PostToolBatch")
    batch["tool_calls"] = []
    with monkeypatch.context() as scoped:
        scoped.setattr(
            claude,
            "classify_capture_compatibility",
            lambda *_args, **_kwargs: CompatibilityStatus.VERIFIED,
        )
        with pytest.raises(claude.ClaudeCodeIntegrationError):
            claude_adapter.adapt_bytes(canonical_json(batch), context=CONTEXT)

    impossible = canonical_json(
        {"hook_event_name": "Impossible", "session_id": "session", "cwd": "/workspace"}
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(claude, "_event_capability", lambda _event: object())
        scoped.setattr(
            claude,
            "classify_capture_compatibility",
            lambda *_args, **_kwargs: CompatibilityStatus.VERIFIED,
        )
        with pytest.raises(claude.ClaudeCodeIntegrationError):
            claude_adapter.adapt_bytes(impossible, context=CONTEXT)
    with monkeypatch.context() as scoped:
        scoped.setattr(codex, "_event_capability", lambda _event: object())
        scoped.setattr(
            codex,
            "classify_capture_compatibility",
            lambda *_args, **_kwargs: CompatibilityStatus.VERIFIED,
        )
        with pytest.raises(codex.CodexIntegrationError):
            codex_adapter.adapt_bytes(impossible, context=CONTEXT)


def test_probe_wrappers_reject_runner_shape_without_leaking_output(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "provider"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    cases = (
        (
            claude.probe_claude_code_version,
            claude.ClaudeCodeIntegrationError,
            b"2.1.204 (Claude Code)\n",
        ),
        (codex.probe_codex_version, codex.CodexIntegrationError, b"codex-cli 0.144.6\n"),
    )
    for probe, error, valid_stdout in cases:
        with pytest.raises(error):
            probe(executable, runner=None)  # type: ignore[arg-type]
        for completed in (
            subprocess.CompletedProcess((str(executable),), 1, valid_stdout, b""),
            subprocess.CompletedProcess((str(executable),), 0, "not-bytes", b""),
            subprocess.CompletedProcess((str(executable),), 0, b"malformed", b""),
        ):
            with pytest.raises(error):
                probe(executable, runner=lambda *_args, result=completed, **_kwargs: result)


def test_real_version_runners_cover_bounded_process_failures(tmp_path: Path) -> None:
    claude_valid = tmp_path / "claude-valid"
    claude_valid.write_bytes(b"#!/bin/sh\nprintf '2.1.204 (Claude Code)\\n'\n")
    claude_valid.chmod(0o700)
    completed = claude._bounded_version_runner(
        (str(claude_valid),),
        input=b"",
        capture_output=True,
        check=False,
        timeout=claude.CLAUDE_CODE_VERSION_TIMEOUT_SECONDS,
        env={},
        cwd=str(tmp_path),
    )
    assert completed.stdout == b"2.1.204 (Claude Code)\n"

    codex_timeout = tmp_path / "codex-timeout"
    codex_timeout.write_bytes(b"#!/bin/sh\nsleep 3\n")
    codex_timeout.chmod(0o700)
    with pytest.raises(subprocess.TimeoutExpired):
        codex._bounded_version_runner(
            (str(codex_timeout),),
            input=b"",
            capture_output=True,
            check=False,
            timeout=codex.CODEX_VERSION_TIMEOUT_SECONDS,
            env={},
        )


def test_claude_runner_missing_pipe_and_termination_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    terminated: list[object] = []

    class MissingPipeProcess:
        stdout = None
        stderr = None
        pid = 42

        @staticmethod
        def poll() -> None:
            return None

    process = MissingPipeProcess()
    monkeypatch.setattr(claude.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(claude, "_terminate_version_process", terminated.append)
    with pytest.raises(claude.ClaudeCodeIntegrationError):
        claude._bounded_version_runner(
            ("provider",),
            input=b"",
            capture_output=True,
            check=False,
            timeout=claude.CLAUDE_CODE_VERSION_TIMEOUT_SECONDS,
            env={},
            cwd=str(tmp_path),
        )
    assert terminated == [process]

    class WaitFails:
        pid = 43

        @staticmethod
        def poll() -> None:
            return None

        @staticmethod
        def wait(*, timeout: float) -> None:
            del timeout
            raise OSError

    monkeypatch.setattr(claude.os, "killpg", lambda *_args: None)
    claude._terminate_version_process(WaitFails())  # type: ignore[arg-type]


def test_claude_resolver_and_environment_failure_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    provider_bin = tmp_path / "bin"
    provider_bin.mkdir()
    non_executable = provider_bin / "claude"
    non_executable.write_bytes(b"not executable")
    non_executable.chmod(0o600)
    with pytest.raises(claude.ClaudeCodeIntegrationError):
        claude._resolve_claude_executable(str(provider_bin), environment={})
    with pytest.raises(claude.ClaudeCodeIntegrationError):
        claude.probe_claude_code_environment(environ={})

    with monkeypatch.context() as scoped:
        scoped.setattr(
            claude,
            "environment_without_provider_credentials",
            lambda _environ: (_ for _ in ()).throw(RuntimeError()),
        )
        with pytest.raises(claude.ClaudeCodeIntegrationError):
            claude.probe_claude_code_environment(environ={})

    extensions = ";".join(f".X{index}" for index in range(64))
    configured_path = ";".join(f"C:\\p{index}" for index in range(129))
    with pytest.raises(claude.ClaudeCodeIntegrationError):
        claude._claude_executable_candidates(
            configured_path,
            windows_pathext=extensions,
            windows=True,
            cwd=PureWindowsPath(r"C:\workspace"),
        )


def test_provider_spec_exception_wrappers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    environment = {"HOME": str(tmp_path / "home"), "XDG_STATE_HOME": str(tmp_path / "state")}
    assert codex.provider_installation_spec(project, environ=environment).provider_id == "codex"
    assert (
        claude.provider_installation_spec(project, environ=environment).provider_id == "claude-code"
    )
    for provider, error in (
        (codex, codex.CodexIntegrationError),
        (claude, claude.ClaudeCodeIntegrationError),
    ):
        with pytest.raises(error):
            provider.provider_installation_spec(Path("relative"))
        with monkeypatch.context() as scoped:
            scoped.setattr(
                provider,
                "environment_without_provider_credentials",
                lambda _environ: (_ for _ in ()).throw(RuntimeError()),
            )
            with pytest.raises(error):
                provider.provider_installation_spec(project, environ=environment)


def test_codex_project_policy_and_discovery_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    environment = {"HOME": str(tmp_path / "home"), "XDG_STATE_HOME": str(tmp_path / "state")}
    spec = codex.provider_installation_spec(project, environ=environment)
    with pytest.raises(codex.CodexIntegrationError):
        codex._validate_project_hook_policy(
            SimpleNamespace(config_path=None, config=None)  # type: ignore[arg-type]
        )

    assert spec.config_path is not None
    spec.config_path.parent.mkdir()
    codex._validate_project_hook_policy(spec)
    spec.config_path.write_text("[features]\nhooks = true\n")
    codex._validate_project_hook_policy(spec)
    spec.config_path.write_text("features = true\n")
    with pytest.raises(codex.CodexIntegrationError):
        codex._validate_project_hook_policy(spec)

    assert spec.config is not None
    marker = spec.config.marker.encode("ascii")
    spec.config_path.write_bytes(b"# " + marker + b"\n# " + marker + b"\n")
    with pytest.raises(codex.CodexIntegrationError):
        codex._validate_project_hook_policy(spec)
    spec.config_path.write_bytes(b"# " + marker + b"\n")
    with pytest.raises(codex.CodexIntegrationError):
        codex._validate_project_hook_policy(spec)
    spec.receipt_path.parent.mkdir(parents=True)
    spec.receipt_path.mkdir()
    with pytest.raises(codex.CodexIntegrationError):
        codex._validate_project_hook_policy(spec)

    nested = project / "src"
    nested.mkdir()
    assert codex._discover_codex_project({"cwd": str(nested)}) == project
    for document in ({}, {"cwd": "relative"}, {"cwd": str(tmp_path / "missing")}):
        with pytest.raises(codex.CodexIntegrationError):
            codex._discover_codex_project(document)
    bad_project = tmp_path / "bad-project"
    bad_project.mkdir()
    (bad_project / ".codex").write_text("not a directory")
    with pytest.raises(codex.CodexIntegrationError):
        codex._discover_codex_project({"cwd": str(bad_project)})

    with monkeypatch.context() as scoped:
        scoped.setattr(
            config_files,
            "read_config_bytes",
            lambda _path: (_ for _ in ()).throw(RuntimeError()),
        )
        with pytest.raises(codex.CodexIntegrationError):
            codex._discover_codex_project({"cwd": str(project)})


def test_claude_project_policy_and_candidate_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    nested = project / "src"
    nested.mkdir(parents=True)
    environment = {"HOME": str(tmp_path / "home"), "XDG_STATE_HOME": str(tmp_path / "state")}
    spec = claude.provider_installation_spec(project, environ=environment)
    with pytest.raises(claude.ClaudeCodeIntegrationError):
        claude._validate_project_hook_policy(
            SimpleNamespace(config_path=None, config=None),  # type: ignore[arg-type]
            environment,
        )

    assert spec.config_path is not None and spec.config is not None
    spec.config_path.parent.mkdir()
    claude._validate_project_hook_policy(spec, environment)
    marker = spec.config.marker.encode("ascii")
    spec.config_path.write_bytes(b'{"x":"' + marker + b'","y":"' + marker + b'"}')
    with pytest.raises(claude.ClaudeCodeIntegrationError):
        claude._validate_project_hook_policy(spec, environment)
    spec.config_path.write_bytes(b'{"x":"' + marker + b'"}')
    with pytest.raises(claude.ClaudeCodeIntegrationError):
        claude._validate_project_hook_policy(spec, environment)
    spec.receipt_path.parent.mkdir(parents=True)
    spec.receipt_path.mkdir()
    with pytest.raises(claude.ClaudeCodeIntegrationError):
        claude._validate_project_hook_policy(spec, environment)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            claude,
            "read_config_bytes",
            lambda path: (
                claude.CLAUDE_CODE_CONFIG_MARKER.encode("ascii")
                if path.parent == project / ".claude"
                else None
            ),
        )
        assert claude._claude_code_project_candidates({"cwd": str(nested)}) == (project,)

    for document in ({}, {"cwd": "relative"}, {"cwd": str(tmp_path / "missing")}):
        with pytest.raises(claude.ClaudeCodeIntegrationError):
            claude._claude_code_project_candidates(document)
    bad_project = tmp_path / "bad-project"
    bad_project.mkdir()
    (bad_project / ".claude").write_text("not a directory")
    with pytest.raises(claude.ClaudeCodeIntegrationError):
        claude._claude_code_project_candidates({"cwd": str(bad_project)})

    with monkeypatch.context() as scoped:
        scoped.setattr(
            claude,
            "read_config_bytes",
            lambda _path: (_ for _ in ()).throw(claude.ConfigFileError()),
        )
        with pytest.raises(claude.ClaudeCodeIntegrationError):
            claude._claude_code_project_candidates({"cwd": str(project)})


def test_provider_dependency_builders_reject_invalid_boundaries() -> None:
    cases = (
        (codex.build_capture_hook_dependencies, codex.CodexIntegrationError, CODEX_CONNECTION),
        (
            claude.build_capture_hook_dependencies,
            claude.ClaudeCodeIntegrationError,
            CLAUDE_CONNECTION,
        ),
        (
            opencode.build_capture_hook_dependencies,
            opencode.OpenCodeIntegrationError,
            OPENCODE_CONNECTION,
        ),
        (pi.build_capture_hook_dependencies, pi.PiIntegrationError, PI_CONNECTION),
    )
    for builder, error, connection_id in cases:
        with pytest.raises(error):
            builder("not-bytes", connection_id=connection_id)  # type: ignore[arg-type]


def test_opencode_reduced_event_matrix_covers_every_declared_disposition() -> None:
    adapter = _opencode_adapter()
    events = [
        _opencode_event(
            "tool_started",
            event_id="start-exact",
            call_id="call-exact",
            tool="bash",
            identity_authority="exact",
            input={"command": "synthetic"},
        ),
        _opencode_event(
            "tool_started",
            call_id="call-unavailable",
            tool="read",
            identity_authority="unavailable",
        ),
        _opencode_event(
            "tool_finished",
            event_id="finish-success",
            call_id="call-exact",
            outcome="succeeded",
        ),
        _opencode_event("tool_finished", call_id="call-unavailable", outcome="failed"),
        *(
            _opencode_event("coverage_degraded", reason=reason)
            for reason in ("invalid_transition", "missing_field", "overflow", "transport_gap")
        ),
        _opencode_event("turn_finished", event_id="turn"),
        _opencode_event("controller_failed"),
        _opencode_event("coverage_boundary", event_id="boundary"),
        _opencode_event("session_finished"),
        _opencode_event("oversize", reason="event_limit"),
    ]
    intakes = adapter.adapt_bytes(_opencode_batch(events), context=CONTEXT)
    assert {intake.capture_disposition for intake in intakes} == {
        "captured",
        "degraded",
        "coverage_boundary",
    }
    assert {intake.kind for intake in intakes} == {
        "session_started",
        "action_started",
        "action_finished",
        "controller_failed",
        "turn_finished",
        "session_finished",
    }
    assert adapter.transport_chunk(_opencode_batch([]), context=CONTEXT).connection_id == (
        OPENCODE_CONNECTION
    )


def test_opencode_event_validation_edges() -> None:
    adapter = _opencode_adapter()
    batch = opencode._parse_batch(_opencode_batch([]))
    invalid_events: tuple[object, ...] = (
        None,
        {1: "non-string-key"},
        _opencode_event(""),
        _opencode_event("oversize", reason="wrong"),
        _opencode_event(
            "tool_started",
            call_id="",
            tool="read",
            identity_authority="unavailable",
        ),
        _opencode_event(
            "tool_started",
            call_id="call",
            tool="",
            identity_authority="unavailable",
        ),
        _opencode_event(
            "tool_started",
            call_id="call",
            tool="read",
            identity_authority="coarse",
        ),
        _opencode_event(
            "tool_started",
            call_id="call",
            tool="read",
            identity_authority="exact",
        ),
        _opencode_event(
            "tool_started",
            event_id="",
            call_id="call",
            tool="read",
            identity_authority="unavailable",
        ),
        _opencode_event("tool_finished", call_id="", outcome="succeeded"),
        _opencode_event("tool_finished", call_id="call", outcome="unknown"),
        _opencode_event("tool_finished", event_id="", call_id="call", outcome="succeeded"),
        _opencode_event("coverage_degraded", reason="unknown"),
        _opencode_event("unknown"),
        _opencode_event("turn_finished", event_id=""),
    )
    for event in invalid_events:
        with pytest.raises(opencode.OpenCodeIntegrationError):
            adapter._event_intake(event, batch=batch, event_index=0, context=CONTEXT)

    for invalid in (None, {1: "non-string-key"}, _opencode_event("turn_finished", session_id="x")):
        with pytest.raises(opencode.OpenCodeIntegrationError):
            adapter._validate_event_session(invalid, "opencode-session")


def test_opencode_batch_constructor_transport_and_wrapper_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_batches = (
        _opencode_batch([], schema_version="wrong"),
        _opencode_batch([], batch_id="bad"),
        _opencode_batch([], session_id=""),
        _opencode_batch([], chunk_index=True),
        _opencode_batch([], chunk_index=-1),
        _opencode_batch([], chunk_index=1, chunk_count=1),
        _opencode_batch([], extra={"extra": True}),
    )
    for source in (*invalid_batches, b"not-json"):
        with pytest.raises(opencode.OpenCodeIntegrationError):
            opencode._parse_batch(source)

    adapter = _opencode_adapter()
    wrong_bootstrap = _bootstrap(
        CaptureProfile.PI_EXTENSION_V1,
        PI_CONNECTION,
        "pi",
    ).model_dump(mode="json", warnings="error")
    with pytest.raises(opencode.OpenCodeIntegrationError):
        adapter.adapt_bytes(_opencode_batch([], bootstrap=wrong_bootstrap), context=CONTEXT)
    with pytest.raises(opencode.OpenCodeIntegrationError):
        adapter.transport_chunk(_opencode_batch([]), context=object())  # type: ignore[arg-type]
    with pytest.raises(opencode.OpenCodeIntegrationError):
        adapter.adapt_bytes(_opencode_batch([]), context=object())  # type: ignore[arg-type]

    monkeypatch.setattr(
        adapter.__class__, "_validated_batch", lambda *_args: (_ for _ in ()).throw(RuntimeError())
    )
    with pytest.raises(opencode.OpenCodeIntegrationError):
        adapter.transport_chunk(b"{}", context=CONTEXT)
    with pytest.raises(opencode.OpenCodeIntegrationError):
        adapter.adapt_bytes(b"{}", context=CONTEXT)


def test_opencode_constructor_capabilities_bundle_and_spec_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid_bootstrap = _bootstrap(
        CaptureProfile.OPENCODE_PLUGIN_V1,
        OPENCODE_CONNECTION,
        "opencode",
    )
    invalid_arguments = (
        {"connection_id": "short", "bootstrap": valid_bootstrap, "project_root": tmp_path},
        {
            "connection_id": OPENCODE_CONNECTION,
            "bootstrap": _bootstrap(CaptureProfile.PI_EXTENSION_V1, PI_CONNECTION, "pi"),
            "project_root": tmp_path,
        },
        {
            "connection_id": OPENCODE_CONNECTION,
            "bootstrap": valid_bootstrap,
            "project_root": Path("relative"),
        },
        {
            "connection_id": OPENCODE_CONNECTION,
            "bootstrap": valid_bootstrap,
            "project_root": tmp_path,
            "host_version": "wrong",
        },
        {
            "connection_id": OPENCODE_CONNECTION,
            "bootstrap": valid_bootstrap.model_copy(update={"capability_digest": ZERO}),
            "project_root": tmp_path,
        },
    )
    for arguments in invalid_arguments:
        with pytest.raises(opencode.OpenCodeIntegrationError):
            opencode.OpenCodeCaptureAdapter(**arguments)

    adapter = _opencode_adapter()
    monkeypatch.setattr(
        opencode,
        "CaptureAdapterCapabilities",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError()),
    )
    with pytest.raises(opencode.OpenCodeIntegrationError):
        adapter.capabilities()

    project = tmp_path / "project"
    project.mkdir()
    environment = {"HOME": str(tmp_path / "home"), "XDG_STATE_HOME": str(tmp_path / "state")}
    assert opencode.provider_installation_spec(project, environ=environment).bundle_bytes
    for invalid_project, host_version in (
        (Path("relative"), opencode.OPENCODE_HOST_VERSION),
        (tmp_path / "missing", opencode.OPENCODE_HOST_VERSION),
        (project, "wrong"),
    ):
        with pytest.raises(opencode.OpenCodeIntegrationError):
            opencode.provider_installation_spec(invalid_project, host_version=host_version)

    class EmptyAsset:
        def joinpath(self, _name: str) -> EmptyAsset:
            return self

        @staticmethod
        def read_bytes() -> bytes:
            return b""

    monkeypatch.setattr(opencode.resources, "files", lambda _package: EmptyAsset())
    with pytest.raises(opencode.OpenCodeIntegrationError):
        opencode._bundle_bytes()


@pytest.mark.parametrize(
    ("module", "error", "batch_factory", "adapter_factory"),
    [
        (opencode, opencode.OpenCodeIntegrationError, _opencode_batch, _opencode_adapter),
        (pi, pi.PiIntegrationError, _pi_batch, _pi_adapter),
    ],
)
def test_bridge_exception_wrappers_and_intake_ceiling(
    module,
    error: type[Exception],
    batch_factory,
    adapter_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as scoped:
        scoped.setattr(
            module,
            "read_bounded_json",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(error()),
        )
        with pytest.raises(error):
            module._canonical_batch(b"{}")
    with monkeypatch.context() as scoped:
        scoped.setattr(
            module,
            "_exact_keys",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(error()),
        )
        with pytest.raises(error):
            module._bootstrap_from_document({})
    with monkeypatch.context() as scoped:
        scoped.setattr(
            module,
            "canonical_json",
            lambda _value: (_ for _ in ()).throw(RuntimeError()),
        )
        with pytest.raises(error):
            module._bootstrap_from_document({})

    adapter = adapter_factory()
    assert "<redacted>" in repr(adapter)
    batch = module._parse_batch(batch_factory([]))
    with monkeypatch.context() as scoped:
        scoped.setattr(module, "MAX_CAPTURE_TRANSPORT_CHUNKS_PER_SESSION", 0)
        scoped.setattr(adapter.__class__, "_validated_batch", lambda *_args: batch)
        with pytest.raises(error):
            adapter.adapt_bytes(b"{}", context=CONTEXT)


def test_bridge_bundle_spec_and_constructor_generic_wrappers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    cases = (
        (
            opencode,
            opencode.OpenCodeIntegrationError,
            opencode.provider_installation_spec,
            opencode.OpenCodeCaptureAdapter,
            _bootstrap(CaptureProfile.OPENCODE_PLUGIN_V1, OPENCODE_CONNECTION, "opencode"),
            OPENCODE_CONNECTION,
        ),
        (
            pi,
            pi.PiIntegrationError,
            pi.provider_installation_spec,
            pi.PiCaptureAdapter,
            _bootstrap(CaptureProfile.PI_EXTENSION_V1, PI_CONNECTION, "pi"),
            PI_CONNECTION,
        ),
    )
    for module, error, spec_builder, constructor, bootstrap, connection_id in cases:
        with monkeypatch.context() as scoped:
            scoped.setattr(
                module.resources,
                "files",
                lambda _package: (_ for _ in ()).throw(RuntimeError()),
            )
            with pytest.raises(error):
                module._bundle_bytes()
        with monkeypatch.context() as scoped:
            scoped.setattr(
                module,
                "environment_without_provider_credentials",
                lambda _environ: (_ for _ in ()).throw(RuntimeError()),
            )
            with pytest.raises(error):
                spec_builder(project, environ={})
        with monkeypatch.context() as scoped:
            scoped.setattr(
                module,
                "capture_capability_digest",
                lambda _profile: (_ for _ in ()).throw(RuntimeError()),
            )
            with pytest.raises(error):
                constructor(
                    connection_id=connection_id,
                    bootstrap=bootstrap,
                    project_root=project,
                )


def test_pi_reduced_event_matrix_covers_every_declared_disposition() -> None:
    adapter = _pi_adapter()
    paired = [
        _pi_event(
            "tool_started",
            event_id="1",
            call_id="call",
            tool="bash",
            identity_authority="coarse",
        ),
        _pi_event("tool_finished", event_id="2", call_id="call", outcome="succeeded"),
        _pi_event("coverage_degraded", event_id="3", reason="invalid_transition"),
        _pi_event("coverage_degraded", event_id="4", reason="missing_field"),
        _pi_event("coverage_degraded", event_id="5", reason="overflow"),
        _pi_event("coverage_degraded", event_id="6", reason="ambiguous_error"),
        _pi_event("coverage_degraded", event_id="7", reason="unmatched_start"),
        _pi_event(
            "coverage_boundary",
            event_id="8",
            reason="compaction",
            compaction_reason="manual",
            from_extension=False,
            will_retry=True,
        ),
        _pi_event(
            "coverage_boundary",
            event_id="9",
            reason="tree",
            old_leaf_id=None,
            new_leaf_id="leaf",
        ),
        _pi_event("turn_finished", event_id="10"),
        _pi_event("session_finished", event_id="11", reason="quit"),
    ]
    intakes = adapter.adapt_bytes(_pi_batch(paired), context=CONTEXT)
    assert {intake.capture_disposition for intake in intakes} == {
        "captured",
        "degraded",
        "coverage_boundary",
    }
    transport_gap = adapter.adapt_bytes(
        _pi_batch([_pi_event("coverage_degraded", reason="transport_gap")]),
        context=CONTEXT,
    )
    oversize = adapter.adapt_bytes(
        _pi_batch([_pi_event("oversize", reason="event_limit")]),
        context=CONTEXT,
    )
    assert transport_gap[-1].error_code == "gap_detected"
    assert oversize[-1].error_code == "overflow"
    assert adapter.transport_chunk(_pi_batch([]), context=CONTEXT).connection_id == PI_CONNECTION


def test_pi_event_validation_edges() -> None:
    adapter = _pi_adapter()
    batch = pi._parse_batch(_pi_batch([]))
    invalid_events: tuple[object, ...] = (
        None,
        {1: "non-string-key"},
        _pi_event(""),
        _pi_event("oversize", reason="wrong"),
        _pi_event(
            "tool_started", event_id="", call_id="call", tool="read", identity_authority="coarse"
        ),
        _pi_event(
            "tool_started", event_id="1", call_id="", tool="read", identity_authority="coarse"
        ),
        _pi_event(
            "tool_started", event_id="1", call_id="call", tool="", identity_authority="coarse"
        ),
        _pi_event(
            "tool_started", event_id="1", call_id="call", tool="read", identity_authority="exact"
        ),
        _pi_event("tool_finished", event_id="1", call_id="", outcome="succeeded"),
        _pi_event("tool_finished", event_id="1", call_id="call", outcome="failed"),
        _pi_event("coverage_degraded", event_id="1", reason="unknown"),
        _pi_event(
            "coverage_boundary",
            event_id="1",
            reason="compaction",
            compaction_reason="unknown",
            from_extension=False,
            will_retry=False,
        ),
        _pi_event(
            "coverage_boundary",
            event_id="1",
            reason="compaction",
            compaction_reason="manual",
            from_extension=0,
            will_retry=False,
        ),
        _pi_event(
            "coverage_boundary",
            event_id="1",
            reason="tree",
            old_leaf_id=object(),
            new_leaf_id=None,
        ),
        _pi_event("coverage_boundary", event_id="1", reason="unknown"),
        _pi_event("unknown", event_id="1"),
        _pi_event("session_finished", event_id="1", reason="unknown"),
    )
    for event in invalid_events:
        with pytest.raises(pi.PiIntegrationError):
            adapter._event_intake(event, batch=batch, event_index=0, context=CONTEXT)

    for invalid in (
        None,
        {1: "non-string-key"},
        _pi_event("turn_finished", session_id="x"),
        _pi_event(""),
    ):
        with pytest.raises(pi.PiIntegrationError):
            adapter._validate_event_coordinates(invalid, batch)


def test_pi_batch_pair_order_terminal_and_wrapper_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_batches = (
        _pi_batch([], schema_version="wrong"),
        _pi_batch([], batch_id="bad"),
        _pi_batch([], session_id="bad session"),
        _pi_batch([], window_discriminator="bad"),
        _pi_batch([], chunk_index=True),
        _pi_batch([], chunk_index=-1),
        _pi_batch([], chunk_index=1, chunk_count=1),
        _pi_batch([], extra={"extra": True}),
    )
    for source in (*invalid_batches, b"not-json"):
        with pytest.raises(pi.PiIntegrationError):
            pi._parse_batch(source)

    adapter = _pi_adapter()
    wrong_bootstrap = _bootstrap(
        CaptureProfile.OPENCODE_PLUGIN_V1,
        OPENCODE_CONNECTION,
        "opencode",
    ).model_dump(mode="json", warnings="error")
    invalid_sequences = (
        _pi_batch([], bootstrap=wrong_bootstrap),
        _pi_batch(
            [_pi_event("turn_finished", event_id="2"), _pi_event("turn_finished", event_id="1")]
        ),
        _pi_batch(
            [
                _pi_event(
                    "tool_started",
                    event_id="1",
                    call_id="c",
                    tool="read",
                    identity_authority="coarse",
                )
            ]
        ),
        _pi_batch(
            [
                _pi_event(
                    "tool_started",
                    event_id="1",
                    call_id="c",
                    tool="read",
                    identity_authority="coarse",
                ),
                _pi_event("turn_finished", event_id="2"),
            ]
        ),
        _pi_batch([_pi_event("tool_finished", event_id="1", call_id="c", outcome="succeeded")]),
        _pi_batch(
            [
                _pi_event("turn_finished", event_id="1"),
                _pi_event("tool_finished", event_id="2", call_id="c", outcome="succeeded"),
            ]
        ),
        _pi_batch(
            [_pi_event("session_finished", event_id="1", reason="quit")],
            chunk_index=0,
            chunk_count=2,
        ),
    )
    for source in invalid_sequences:
        with pytest.raises(pi.PiIntegrationError):
            adapter.adapt_bytes(source, context=CONTEXT)

    with pytest.raises(pi.PiIntegrationError):
        adapter.transport_chunk(_pi_batch([]), context=object())  # type: ignore[arg-type]
    with pytest.raises(pi.PiIntegrationError):
        adapter.adapt_bytes(_pi_batch([]), context=object())  # type: ignore[arg-type]
    monkeypatch.setattr(
        adapter.__class__, "_validated_batch", lambda *_args: (_ for _ in ()).throw(RuntimeError())
    )
    with pytest.raises(pi.PiIntegrationError):
        adapter.transport_chunk(b"{}", context=CONTEXT)
    with pytest.raises(pi.PiIntegrationError):
        adapter.adapt_bytes(b"{}", context=CONTEXT)


def test_pi_constructor_capabilities_bundle_and_spec_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid_bootstrap = _bootstrap(CaptureProfile.PI_EXTENSION_V1, PI_CONNECTION, "pi")
    invalid_arguments = (
        {"connection_id": "short", "bootstrap": valid_bootstrap, "project_root": tmp_path},
        {
            "connection_id": PI_CONNECTION,
            "bootstrap": _bootstrap(
                CaptureProfile.OPENCODE_PLUGIN_V1,
                OPENCODE_CONNECTION,
                "opencode",
            ),
            "project_root": tmp_path,
        },
        {
            "connection_id": PI_CONNECTION,
            "bootstrap": valid_bootstrap,
            "project_root": Path("relative"),
        },
        {
            "connection_id": PI_CONNECTION,
            "bootstrap": valid_bootstrap,
            "project_root": tmp_path,
            "host_version": "wrong",
        },
        {
            "connection_id": PI_CONNECTION,
            "bootstrap": valid_bootstrap.model_copy(update={"capability_digest": ZERO}),
            "project_root": tmp_path,
        },
    )
    for arguments in invalid_arguments:
        with pytest.raises(pi.PiIntegrationError):
            pi.PiCaptureAdapter(**arguments)

    adapter = _pi_adapter()
    monkeypatch.setattr(
        pi,
        "CaptureAdapterCapabilities",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError()),
    )
    with pytest.raises(pi.PiIntegrationError):
        adapter.capabilities()

    project = tmp_path / "project"
    project.mkdir()
    environment = {"HOME": str(tmp_path / "home"), "XDG_STATE_HOME": str(tmp_path / "state")}
    assert pi.provider_installation_spec(project, environ=environment).bundle_bytes
    for invalid_project, host_version in (
        (Path("relative"), pi.PI_HOST_VERSION),
        (tmp_path / "missing", pi.PI_HOST_VERSION),
        (project, "wrong"),
    ):
        with pytest.raises(pi.PiIntegrationError):
            pi.provider_installation_spec(invalid_project, host_version=host_version)

    class EmptyAsset:
        def joinpath(self, _name: str) -> EmptyAsset:
            return self

        @staticmethod
        def read_bytes() -> bytes:
            return b""

    monkeypatch.setattr(pi.resources, "files", lambda _package: EmptyAsset())
    with pytest.raises(pi.PiIntegrationError):
        pi._bundle_bytes()


def _codex_source(event_name: str = "SessionStart") -> bytes:
    return canonical_json(_provider_fixture("codex", event_name))


def _hook_dependencies(
    adapter: object,
    store: object,
    spool: object,
    *,
    context: object = CONTEXT,
    registry: object | None = None,
    receipt: object | None = None,
    connection: object | None = None,
    health: list[CaptureHealthCode] | None = None,
) -> CaptureHookDependencies:
    registry_evidence = object() if registry is None else registry
    receipt_evidence = object() if receipt is None else receipt
    connection_evidence = object() if connection is None else connection

    return CaptureHookDependencies(
        validate_registry=lambda _profile: registry_evidence,
        validate_receipt=lambda _profile, _connection, _registry: receipt_evidence,
        validate_connection=lambda _profile, _connection, _registry, _receipt: connection_evidence,
        load_context=lambda _connection: context,  # type: ignore[return-value]
        resolve_adapter=lambda _connection: adapter,
        open_store=lambda _connection: store,  # type: ignore[return-value]
        open_spool=lambda _connection: spool,  # type: ignore[return-value]
        mark_health=lambda _connection, code: None if health is None else health.append(code),
    )


class _HookAdapter:
    def __init__(self, delegate: object, *, result: object | None = None) -> None:
        self.delegate = delegate
        self.result = result

    def capabilities(self):
        return self.delegate.capabilities()

    def adapt_bytes(self, source: bytes, *, context: CaptureDigestContext):
        if self.result is not None:
            return self.result
        return self.delegate.adapt_bytes(source, context=context)

    def transport_chunk(self, source: bytes, *, context: CaptureDigestContext):
        return self.delegate.transport_chunk(source, context=context)


class _HookStore:
    def __init__(self, *, close_error: bool = False) -> None:
        self.appended: list[object] = []
        self.close_error = close_error

    def append(self, intake: object) -> object:
        self.appended.append(intake)
        return SimpleNamespace(disposition="admitted")

    def append_transport_chunk(self, descriptor: object, intakes: tuple[object, ...]) -> object:
        self.appended.extend((descriptor, *intakes))
        return SimpleNamespace(disposition="admitted")

    def close(self) -> None:
        if self.close_error:
            raise RuntimeError("close failed")


class _HookSpool:
    def __init__(
        self,
        *,
        disposition: object = "admitted",
        failure: Exception | None = None,
        close_error: bool = False,
    ) -> None:
        self.disposition = disposition
        self.failure = failure
        self.close_error = close_error

    def admit(self, store: _HookStore, intake: object) -> object:
        if self.failure is not None:
            raise self.failure
        store.append(intake)
        return SimpleNamespace(disposition=self.disposition)

    def admit_transport(
        self,
        store: _HookStore,
        descriptor: object,
        intakes: tuple[object, ...],
        _fallback: tuple[object, ...],
    ) -> tuple[object, ...]:
        if self.failure is not None:
            raise self.failure
        store.append_transport_chunk(descriptor, intakes)
        return (SimpleNamespace(disposition=self.disposition),)

    def close(self) -> None:
        if self.close_error:
            raise RuntimeError("close failed")


def test_hook_dependency_sentinels_and_argument_parser_edges() -> None:
    with pytest.raises(CaptureHookError):
        CaptureHookDependencies(
            validate_registry=None,  # type: ignore[arg-type]
            validate_receipt=lambda *_args: object(),
            validate_connection=lambda *_args: object(),
            load_context=lambda *_args: CONTEXT,
            resolve_adapter=lambda *_args: object(),
            open_store=lambda *_args: object(),  # type: ignore[arg-type]
            open_spool=lambda *_args: object(),  # type: ignore[arg-type]
            mark_health=lambda *_args: None,
        )

    unavailable_calls = (
        lambda: hook._unavailable_registry(CaptureProfile.CODEX_HOOKS_V1),
        lambda: hook._unavailable_receipt(
            CaptureProfile.CODEX_HOOKS_V1, CODEX_CONNECTION, object()
        ),
        lambda: hook._unavailable_connection(
            CaptureProfile.CODEX_HOOKS_V1,
            CODEX_CONNECTION,
            object(),
            object(),
        ),
        lambda: hook._unavailable_context(object()),
        lambda: hook._unavailable_adapter(object()),
        lambda: hook._unavailable_store(object()),
        lambda: hook._unavailable_spool(object()),
    )
    for call in unavailable_calls:
        with pytest.raises(CaptureHookError):
            call()
    assert hook._unavailable_health(object(), CaptureHealthCode.COVERAGE_DEGRADED) is None

    valid_orders = (
        ("--profile", "codex-hooks/v1", "--connection", CODEX_CONNECTION),
        ("--connection", CODEX_CONNECTION, "--profile", "codex-hooks/v1"),
    )
    for arguments in valid_orders:
        parsed = hook.parse_capture_hook_arguments(arguments)
        assert parsed.profile is CaptureProfile.CODEX_HOOKS_V1
        assert parsed.connection_id == CODEX_CONNECTION

    invalid_arguments: tuple[object, ...] = (
        "not-a-sequence",
        (),
        ("--profile", "codex-hooks/v1", "--profile", "codex-hooks/v1"),
        ("--unknown", "codex-hooks/v1", "--connection", CODEX_CONNECTION),
        ("--profile", "unknown", "--connection", CODEX_CONNECTION),
        ("--profile", "codex-hooks/v1", "--connection", "short"),
        ("--profile", 1, "--connection", CODEX_CONNECTION),
    )
    for arguments in invalid_arguments:
        with pytest.raises(CaptureHookError):
            hook.parse_capture_hook_arguments(arguments)  # type: ignore[arg-type]

    class HostileSequence:
        def __len__(self) -> int:
            raise RuntimeError("provider content")

    with pytest.raises(CaptureHookError):
        hook.parse_capture_hook_arguments(HostileSequence())  # type: ignore[arg-type]


def test_hook_bounded_reader_and_json_bounds_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    assert hook._bounded_read(BytesIO(b"{}")) == b"{}"
    for stream in (
        object(),
        SimpleNamespace(read=lambda _remaining: "not-bytes"),
        SimpleNamespace(read=lambda remaining: b"x" * (remaining + 1)),
        BytesIO(b""),
        BytesIO(b"x" * (hook._MAX_CAPTURE_NATIVE_BYTES + 1)),
    ):
        with pytest.raises(CaptureHookError):
            hook._bounded_read(stream)  # type: ignore[arg-type]

    assert hook._unique_json_object([("a", 1)]) == {"a": 1}
    with pytest.raises(ValueError):
        hook._unique_json_object([("a", 1), ("a", 2)])
    with pytest.raises(ValueError):
        hook._reject_json_constant("NaN")

    assert hook._json_within_capture_bounds({"string": "value", "list": [None, True, 1, 1.5]})
    assert not hook._json_within_capture_bounds({1: "non-string-key"})
    assert not hook._json_within_capture_bounds({"value": float("inf")})
    assert not hook._json_within_capture_bounds({"value": object()})
    assert not hook._json_within_capture_bounds({"value": "\ud800"})
    monkeypatch.setattr(hook, "_MAX_CAPTURE_JSON_ITEMS", 2)
    assert not hook._json_within_capture_bounds([1, 2])
    monkeypatch.setattr(hook, "_MAX_CAPTURE_JSON_ITEMS", 10_000)
    monkeypatch.setattr(hook, "_MAX_CAPTURE_JSON_DEPTH", 1)
    assert not hook._json_within_capture_bounds([[None]])
    monkeypatch.setattr(hook, "_MAX_CAPTURE_JSON_DEPTH", 32)
    monkeypatch.setattr(hook, "_MAX_CAPTURE_JSON_STRING_BYTES", 2)
    assert not hook._json_within_capture_bounds({"long": "value"})

    for source in (b"[]", b'{"a":1,"a":2}', b'{"a":NaN}', b"not-json"):
        with pytest.raises(CaptureHookError):
            hook.read_capture_hook_document(BytesIO(source))


def test_default_dependency_resolution_is_shape_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    dependencies = _hook_dependencies(
        codex.CodexCaptureAdapter(connection_id=CODEX_CONNECTION),
        _HookStore(),
        _HookSpool(),
    )
    imported: list[str] = []

    def fake_import(name: str) -> object:
        imported.append(name)
        return SimpleNamespace(
            build_capture_hook_dependencies=lambda *_args, **_kwargs: dependencies
        )

    monkeypatch.setattr(importlib, "import_module", fake_import)
    codex_source = canonical_json(
        {"hook_event_name": "SessionStart", "session_id": "session", "cwd": "/workspace"}
    )
    assert (
        hook._default_dependencies(
            profile="codex-hooks/v1",
            connection_id=CODEX_CONNECTION,
            source=codex_source,
            environ={},
            capture_executable=None,
        )
        is dependencies
    )
    assert imported == ["saliencegate.integrations.codex"]

    bridge_source = _opencode_batch([])
    assert (
        hook._default_dependencies(
            profile="opencode-plugin/v1",
            connection_id=OPENCODE_CONNECTION,
            source=bridge_source,
            environ={},
            capture_executable=None,
        )
        is dependencies
    )

    invalid = (
        ("unknown", codex_source),
        ("codex-hooks/v1", b"[]"),
        ("codex-hooks/v1", canonical_json({"hook_event_name": "Unknown"})),
        (
            "codex-hooks/v1",
            canonical_json(
                {"hook_event_name": "SessionStart", "session_id": "s", "cwd": "relative"}
            ),
        ),
        ("opencode-plugin/v1", canonical_json({"schema_version": "capture-batch/v1"})),
        ("opencode-plugin/v1", b"not-json"),
    )
    for profile, source in invalid:
        assert (
            hook._default_dependencies(
                profile=profile,
                connection_id=CODEX_CONNECTION,
                source=source,
                environ={},
                capture_executable=None,
            )
            is None
        )

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: SimpleNamespace(build_capture_hook_dependencies=None),
    )
    assert (
        hook._default_dependencies(
            profile="codex-hooks/v1",
            connection_id=CODEX_CONNECTION,
            source=codex_source,
            environ={},
            capture_executable=None,
        )
        is None
    )


def test_hook_adapter_intake_and_resource_helpers_cover_fail_closed_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegate = codex.CodexCaptureAdapter(connection_id=CODEX_CONNECTION)
    intakes = delegate.adapt_bytes(_codex_source(), context=CONTEXT)
    adapter = _HookAdapter(delegate)
    assert (
        hook._adapter_intakes(
            adapter,
            _codex_source(),
            profile=CaptureProfile.CODEX_HOOKS_V1,
            connection_id=CODEX_CONNECTION,
            context=CONTEXT,
        )
        == intakes
    )
    with pytest.raises(CaptureHookError):
        hook._adapter_intakes(
            adapter,
            _codex_source(),
            profile=CaptureProfile.CLAUDE_CODE_HOOKS_V1,
            connection_id=CODEX_CONNECTION,
            context=CONTEXT,
        )
    with pytest.raises(CaptureAdapterContractError):
        hook._adapter_intakes(
            SimpleNamespace(capabilities=delegate.capabilities),
            _codex_source(),
            profile=CaptureProfile.CODEX_HOOKS_V1,
            connection_id=CODEX_CONNECTION,
            context=CONTEXT,
        )
    import saliencegate.capture.adapters as adapter_contract

    monkeypatch.setattr(
        adapter_contract, "validated_capture_adapter", lambda _adapter: delegate.capabilities()
    )
    with pytest.raises(CaptureHookError):
        hook._adapter_intakes(
            SimpleNamespace(adapt_bytes=None),
            _codex_source(),
            profile=CaptureProfile.CODEX_HOOKS_V1,
            connection_id=CODEX_CONNECTION,
            context=CONTEXT,
        )
    with pytest.raises(CaptureHookError):
        hook._adapter_intakes(
            _HookAdapter(delegate, result=[]),
            _codex_source(),
            profile=CaptureProfile.CODEX_HOOKS_V1,
            connection_id=CODEX_CONNECTION,
            context=CONTEXT,
        )
    with pytest.raises(CaptureHookError):
        hook._adapter_intakes(
            _HookAdapter(delegate, result=intakes),
            _codex_source(),
            profile=CaptureProfile.CODEX_HOOKS_V1,
            connection_id=CLAUDE_CONNECTION,
            context=CONTEXT,
        )

    with pytest.raises(CaptureHookError):
        hook._require_evidence(None)
    evidence = object()
    assert hook._require_evidence(evidence) is evidence

    assert hook._close_resource(None)
    assert hook._close_resource(object())
    assert not hook._close_resource(
        SimpleNamespace(close=lambda: (_ for _ in ()).throw(RuntimeError()))
    )
    assert hook._spool_disposition(SimpleNamespace(disposition="queued")) == "queued"
    assert hook._spool_disposition(SimpleNamespace(disposition=1)) is None

    class HostileReceipt:
        @property
        def disposition(self) -> object:
            raise RuntimeError

    assert hook._spool_disposition(HostileReceipt()) is None


def test_hook_transport_gap_and_fallback_are_bounded() -> None:
    adapter = _opencode_adapter()
    source = _opencode_batch([_opencode_event("session_finished")])
    intakes = adapter.adapt_bytes(source, context=CONTEXT)
    descriptor = adapter.transport_chunk(source, context=CONTEXT)
    gap = hook._transport_gap_intake(intakes, descriptor, context=CONTEXT)
    assert gap.kind == "controller_failed"
    assert gap.error_code == "gap_detected"
    assert hook._bounded_transport_fallback(intakes, gap) == (intakes[0], gap, intakes[-1])

    without_terminal = adapter.adapt_bytes(_opencode_batch([]), context=CONTEXT)
    assert hook._bounded_transport_fallback(without_terminal, gap) == (without_terminal[0], gap)
    for invalid_intakes, invalid_descriptor, invalid_context in (
        ((), descriptor, CONTEXT),
        (intakes, object(), CONTEXT),
        (intakes, descriptor, object()),
        (intakes[1:], descriptor, CONTEXT),
    ):
        with pytest.raises(CaptureHookError):
            hook._transport_gap_intake(
                invalid_intakes,
                invalid_descriptor,
                context=invalid_context,  # type: ignore[arg-type]
            )


def test_run_capture_hook_nonbridge_and_bridge_failure_matrix() -> None:
    from saliencegate.capture.spool import CaptureSpoolError

    arguments = ("--profile", "codex-hooks/v1", "--connection", CODEX_CONNECTION)
    adapter = _HookAdapter(codex.CodexCaptureAdapter(connection_id=CODEX_CONNECTION))
    health: list[CaptureHealthCode] = []
    store = _HookStore()
    spool = _HookSpool(disposition="dropped_quota")
    dependencies = _hook_dependencies(adapter, store, spool, health=health)
    assert (
        hook.run_capture_hook(arguments, BytesIO(_codex_source()), dependencies=dependencies) == 0
    )
    assert health == [CaptureHealthCode.SPOOL_QUOTA]
    assert len(store.appended) == 1

    failure_cases = (
        _hook_dependencies(adapter, object(), spool, health=health),
        _hook_dependencies(adapter, store, object(), health=health),
        _hook_dependencies(adapter, store, _HookSpool(failure=CaptureSpoolError()), health=health),
        _hook_dependencies(adapter, store, spool, context=object(), health=health),
    )
    for selected in failure_cases:
        assert (
            hook.run_capture_hook(arguments, BytesIO(_codex_source()), dependencies=selected) == 0
        )

    close_health: list[CaptureHealthCode] = []
    assert (
        hook.run_capture_hook(
            arguments,
            BytesIO(_codex_source()),
            dependencies=_hook_dependencies(
                adapter,
                _HookStore(close_error=True),
                _HookSpool(close_error=True),
                health=close_health,
            ),
        )
        == 0
    )
    assert CaptureHealthCode.COVERAGE_DEGRADED in close_health

    bridge_source = _opencode_batch([_opencode_event("session_finished")])
    bridge_adapter = _HookAdapter(_opencode_adapter())
    bridge_arguments = (
        "--profile",
        "opencode-plugin/v1",
        "--connection",
        OPENCODE_CONNECTION,
    )
    bridge_health: list[CaptureHealthCode] = []
    assert (
        hook.run_capture_hook(
            bridge_arguments,
            BytesIO(bridge_source),
            dependencies=_hook_dependencies(
                bridge_adapter,
                _HookStore(),
                _HookSpool(disposition="dropped_quota"),
                health=bridge_health,
            ),
        )
        == 0
    )
    assert bridge_health == [CaptureHealthCode.SPOOL_QUOTA]
    assert hook.run_capture_hook(arguments, BytesIO(_codex_source()), dependencies=object()) == 0  # type: ignore[arg-type]


def test_health_entrypoint_and_standard_stream_failure_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[CaptureHealthCode] = []
    dependencies = _hook_dependencies(object(), object(), object(), health=observed)
    hook._mark_health(dependencies, None, CaptureHealthCode.COVERAGE_DEGRADED)
    hook._mark_coverage_degraded(dependencies, None)
    assert observed == []

    failing = _hook_dependencies(object(), object(), object())
    failing = CaptureHookDependencies(
        **{
            **{
                name: getattr(failing, name)
                for name in (
                    "validate_registry",
                    "validate_receipt",
                    "validate_connection",
                    "load_context",
                    "resolve_adapter",
                    "open_store",
                    "open_spool",
                )
            },
            "mark_health": lambda *_args: (_ for _ in ()).throw(RuntimeError()),
        }
    )
    hook._mark_health(failing, object(), CaptureHealthCode.COVERAGE_DEGRADED)

    closed: list[int] = []
    fake_os = SimpleNamespace(
        devnull="/dev/null",
        O_WRONLY=1,
        open=lambda *_args: 99,
        dup2=lambda *_args: (_ for _ in ()).throw(OSError()),
        close=closed.append,
    )
    monkeypatch.setattr(hook, "os", fake_os)
    assert not hook._silence_standard_streams()
    assert closed == [99]

    monkeypatch.setattr(hook, "_silence_standard_streams", lambda: False)
    assert hook.entrypoint(()) == 0
    monkeypatch.setattr(hook, "_silence_standard_streams", lambda: True)
    monkeypatch.setattr(
        hook, "run_capture_hook", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError())
    )
    assert hook.entrypoint(()) == 0


def test_launcher_materialization_success_and_failure_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    spec = opencode.provider_installation_spec(
        project,
        environ={"HOME": str(tmp_path / "home"), "XDG_STATE_HOME": str(tmp_path / "state")},
    )
    executable = tmp_path / "capture-hook"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    materialized = launcher_materialization.materialize_provider_launcher(
        spec,
        KEY,
        capture_executable=executable,
    )
    assert materialized.launcher_bytes != spec.launcher_bytes
    assert str(executable) in materialized.launcher_bytes.decode("utf-8")
    assert launcher_materialization._trusted_launcher_watchdog(
        CaptureLauncherPlatform.POSIX
    ).is_absolute()

    monkeypatch.setattr(
        launcher_materialization.sysconfig,
        "get_path",
        lambda _name: str(tmp_path / "missing-scripts"),
    )
    for invalid in (None, object(), tmp_path / "missing", project, Path("relative-hook")):
        with pytest.raises(Exception) as raised:
            launcher_materialization.materialize_provider_launcher(
                spec,
                KEY,
                capture_executable=invalid,
            )
        assert raised.type.__name__ == "CaptureCommandUnavailableError"

    monkeypatch.setattr(launcher_materialization.os, "access", lambda *_args: False)
    with pytest.raises(Exception) as raised:
        launcher_materialization._trusted_launcher_watchdog(CaptureLauncherPlatform.POSIX)
    assert raised.type.__name__ == "CaptureCommandUnavailableError"


def test_launcher_materialization_ignores_hostile_cwd_and_path_for_default_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    spec = opencode.provider_installation_spec(
        project,
        environ={"HOME": str(tmp_path / "home"), "XDG_STATE_HOME": str(tmp_path / "state")},
    )
    trusted_scripts = tmp_path / "trusted scripts"
    trusted_scripts.mkdir()
    executable_name = (
        "saliencegate-capture-hook.exe"
        if launcher_materialization.os.name == "nt"
        else "saliencegate-capture-hook"
    )
    trusted = trusted_scripts / executable_name
    trusted.write_bytes(b"#!/bin/sh\nexit 0\n")
    trusted.chmod(0o700)
    hostile = project / executable_name
    hostile.write_bytes(b"#!/bin/sh\nexit 99\n")
    hostile.chmod(0o700)
    monkeypatch.chdir(project)
    monkeypatch.setenv("PATH", os.fspath(project))
    monkeypatch.setattr(
        launcher_materialization.sysconfig,
        "get_path",
        lambda name: str(trusted_scripts) if name == "scripts" else None,
    )
    monkeypatch.setattr(
        launcher_materialization,
        "_posix_executable_boundary_is_trusted",
        lambda candidate: candidate == trusted.resolve(),
    )
    monkeypatch.setattr(
        launcher_materialization,
        "_windows_executable_boundary_is_trusted",
        lambda candidate: candidate == trusted.resolve(),
    )

    materialized = launcher_materialization.materialize_provider_launcher(spec, KEY)
    rendered = materialized.launcher_bytes.decode("utf-8")

    assert str(trusted.resolve()) in rendered
    assert str(hostile.resolve()) not in rendered


def test_interpreter_hook_boundary_uses_exact_windows_console_script_name(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    scripts = tmp_path / "Scripts"
    project.mkdir()
    scripts.mkdir()
    command_wrapper = scripts / "saliencegate-capture-hook.cmd"
    command_wrapper.write_bytes(b"synthetic wrapper")

    with pytest.raises(Exception) as raised:
        launcher_materialization._interpreter_capture_executable(
            project_root=project,
            scripts_directory=scripts,
            native_windows=True,
            candidate_is_trusted=lambda _candidate: True,
        )
    assert raised.type.__name__ == "CaptureCommandUnavailableError"

    executable = scripts / "saliencegate-capture-hook.exe"
    executable.write_bytes(b"synthetic executable")

    assert (
        launcher_materialization._interpreter_capture_executable(
            project_root=project,
            scripts_directory=scripts,
            native_windows=True,
            candidate_is_trusted=lambda _candidate: True,
        )
        == executable.resolve()
    )


def test_interpreter_hook_boundary_rejects_project_local_virtual_environment(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    scripts = project / ".venv" / "bin"
    scripts.mkdir(parents=True)
    executable = scripts / "saliencegate-capture-hook"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    observed: list[Path] = []

    with pytest.raises(Exception) as raised:
        launcher_materialization._interpreter_capture_executable(
            project_root=project,
            scripts_directory=scripts,
            native_windows=False,
            candidate_is_trusted=lambda candidate: observed.append(candidate) is None,
        )

    assert raised.type.__name__ == "CaptureCommandUnavailableError"
    assert observed == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership and mode boundary")
def test_interpreter_hook_boundary_rejects_world_writable_ancestry(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    with tempfile.TemporaryDirectory(prefix="saliencegate-hostile-hook-", dir="/tmp") as raw:
        scripts = Path(raw)
        executable = scripts / "saliencegate-capture-hook"
        executable.write_bytes(b"#!/bin/sh\nexit 0\n")
        executable.chmod(0o700)

        with pytest.raises(Exception) as raised:
            launcher_materialization._interpreter_capture_executable(
                project_root=project,
                scripts_directory=scripts,
                native_windows=False,
            )

    assert raised.type.__name__ == "CaptureCommandUnavailableError"


def test_explicit_windows_capture_executable_preserves_com_support(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "capture-hook.com"
    executable.write_bytes(b"synthetic executable")

    assert (
        launcher_materialization._explicit_capture_executable(
            executable,
            native_windows=True,
        )
        == executable.resolve()
    )
