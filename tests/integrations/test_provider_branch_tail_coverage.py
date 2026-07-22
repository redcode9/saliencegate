"""Coverage-oriented regressions for otherwise rare contract edges."""

from __future__ import annotations

import subprocess as subprocess_module
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from tests.integrations.test_claude_code import (
    _capture_executable as claude_capture_executable,
)
from tests.integrations.test_claude_code import _environment as claude_environment
from tests.integrations.test_claude_code import _fake_claude
from tests.integrations.test_claude_code import _payload as claude_payload
from tests.integrations.test_codex import _capture_executable as codex_capture_executable
from tests.integrations.test_codex import _payload as codex_payload

import saliencegate.integrations.claude_code as claude
import saliencegate.integrations.codex as codex
import saliencegate.integrations.installation as installation_module
import saliencegate.integrations.registry as registry_module
from saliencegate.capture.capabilities import CaptureProfile
from saliencegate.capture.store import CaptureConnectionState, CaptureStore
from saliencegate.commands.capture.connect import run_connect
from saliencegate.domain import canonical_json
from saliencegate.integrations.installation import derive_installation_identity
from saliencegate.security import load_installation_key


@dataclass(frozen=True)
class _ProviderCase:
    module: ModuleType
    error: type[Exception]
    provider: str
    profile: CaptureProfile
    host_version: str
    project: Path
    environment: dict[str, str]
    capture_executable: Path
    source: bytes
    connection_id: str


def _claude_case(tmp_path: Path) -> _ProviderCase:
    project = tmp_path / "claude-project"
    project.mkdir()
    executable = _fake_claude(tmp_path)
    environment = claude_environment(tmp_path, executable)
    capture_executable = claude_capture_executable()
    run_connect(
        provider="claude-code",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    )
    spec = claude.provider_installation_spec(project, environ=environment)
    key = load_installation_key(environ=environment)
    identity = derive_installation_identity(spec, key)
    payload = claude_payload("SessionStart")
    payload["cwd"] = str(project)
    return _ProviderCase(
        module=claude,
        error=claude.ClaudeCodeIntegrationError,
        provider="claude-code",
        profile=CaptureProfile.CLAUDE_CODE_HOOKS_V1,
        host_version=claude.CLAUDE_CODE_HOST_VERSION,
        project=project,
        environment=environment,
        capture_executable=capture_executable,
        source=canonical_json(payload),
        connection_id=identity.connection_id,
    )


def _codex_case(tmp_path: Path) -> _ProviderCase:
    project = tmp_path / "codex-project"
    project.mkdir()
    provider_bin = tmp_path / "codex-bin"
    provider_bin.mkdir()
    executable = provider_bin / "codex"
    executable.write_bytes(b"#!/bin/sh\nprintf 'codex-cli 0.144.6\\n'\n")
    executable.chmod(0o700)
    environment = {
        "HOME": str(tmp_path / "codex-home"),
        "PATH": str(provider_bin),
        "XDG_CONFIG_HOME": str(tmp_path / "codex-configuration"),
        "XDG_STATE_HOME": str(tmp_path / "codex-state"),
    }
    capture_executable = codex_capture_executable()
    run_connect(
        provider="codex",
        project=project,
        environ=environment,
        capture_executable=capture_executable,
    )
    spec = codex.provider_installation_spec(project, environ=environment)
    key = load_installation_key(environ=environment)
    identity = derive_installation_identity(spec, key)
    payload = codex_payload("SessionStart")
    payload["cwd"] = str(project)
    return _ProviderCase(
        module=codex,
        error=codex.CodexIntegrationError,
        provider="codex",
        profile=CaptureProfile.CODEX_HOOKS_V1,
        host_version=codex.CODEX_HOST_VERSION,
        project=project,
        environment=environment,
        capture_executable=capture_executable,
        source=canonical_json(payload),
        connection_id=identity.connection_id,
    )


def _build_dependencies(case: _ProviderCase):
    return case.module.build_capture_hook_dependencies(
        case.source,
        connection_id=case.connection_id,
        environ=case.environment,
        capture_executable=case.capture_executable,
    )


def _exercise_dependency_guards(case: _ProviderCase) -> None:
    dependencies = _build_dependencies(case)
    registry = dependencies.validate_registry(case.profile)
    receipt = dependencies.validate_receipt(case.profile, case.connection_id, registry)
    runtime = dependencies.validate_connection(
        case.profile,
        case.connection_id,
        registry,
        receipt,
    )

    invalid_calls = (
        lambda: dependencies.validate_registry(CaptureProfile.PI_EXTENSION_V1),
        lambda: dependencies.validate_receipt(case.profile, "wrong", registry),
        lambda: dependencies.validate_connection(
            case.profile,
            case.connection_id,
            registry,
            object(),
        ),
        lambda: dependencies.load_context(object()),
    )
    for call in invalid_calls:
        with pytest.raises(case.error):
            call()

    original_key = runtime.key
    object.__setattr__(runtime, "key", object())
    try:
        with pytest.raises(case.error):
            dependencies.load_context(runtime)
    finally:
        object.__setattr__(runtime, "key", original_key)

    original_connection = runtime.connection
    object.__setattr__(runtime, "connection", object())
    try:
        with pytest.raises(case.error):
            dependencies.resolve_adapter(runtime)
    finally:
        object.__setattr__(runtime, "connection", original_connection)

    original_locations = runtime.locations
    object.__setattr__(runtime, "locations", object())
    try:
        with pytest.raises(case.error):
            dependencies.open_store(runtime)
        with pytest.raises(case.error):
            dependencies.open_spool(runtime)
    finally:
        object.__setattr__(runtime, "locations", original_locations)

    with pytest.raises(case.error):
        dependencies.mark_health(runtime, object())


def _copy_connection(connection: object, **updates: object) -> object:
    model_copy = getattr(connection, "model_copy", None)
    if callable(model_copy):
        return model_copy(update=updates)
    return replace(connection, **updates)


def _patch_connection(
    scoped: pytest.MonkeyPatch,
    case: _ProviderCase,
    **updates: object,
) -> None:
    method_name = "get_connection" if case.provider == "claude-code" else "_get_hook_connection"
    original = getattr(CaptureStore, method_name)

    def changed(store: CaptureStore, connection_id: str) -> object:
        return _copy_connection(original(store, connection_id), **updates)

    scoped.setattr(CaptureStore, method_name, changed)


def _assert_builder_failure(case: _ProviderCase) -> None:
    with pytest.raises(case.error):
        _build_dependencies(case)


def _exercise_builder_guards(case: _ProviderCase, monkeypatch: pytest.MonkeyPatch) -> None:
    registry_type = type(registry_module.BUILTIN_PROVIDER_REGISTRY)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            registry_type,
            "resolve",
            lambda *_args, **_kwargs: SimpleNamespace(
                profile=CaptureProfile.PI_EXTENSION_V1,
                host_version=case.host_version,
            ),
        )
        _assert_builder_failure(case)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            registry_type,
            "resolve",
            lambda *_args, **_kwargs: SimpleNamespace(
                profile=case.profile,
                host_version=case.host_version,
            ),
        )
        _assert_builder_failure(case)

    if case.provider == "claude-code":
        with monkeypatch.context() as scoped:
            _patch_connection(scoped, case, profile_id=CaptureProfile.PI_EXTENSION_V1)
            _assert_builder_failure(case)
        with monkeypatch.context() as scoped:
            scoped.setattr(claude, "_claude_code_project_candidates", lambda _document: ())
            _assert_builder_failure(case)

    with monkeypatch.context() as scoped:
        _patch_connection(scoped, case, project_digest="f" * 64)
        _assert_builder_failure(case)

    original_derive = installation_module.derive_installation_identity
    with monkeypatch.context() as scoped:
        calls = 0

        def drifted_identity(spec: object, key: object):
            nonlocal calls
            calls += 1
            identity = original_derive(spec, key)
            if calls == 2:
                return identity.model_copy(update={"connection_id": "sg-" + "f" * 48})
            return identity

        scoped.setattr(installation_module, "derive_installation_identity", drifted_identity)
        _assert_builder_failure(case)

    original_inspect = installation_module.inspect_provider_installation
    with monkeypatch.context() as scoped:

        def invalid_installation(spec: object, key: object):
            status = original_inspect(spec, key)
            return status.model_copy(update={"installed": False})

        scoped.setattr(installation_module, "inspect_provider_installation", invalid_installation)
        _assert_builder_failure(case)

    with monkeypatch.context() as scoped:
        _patch_connection(scoped, case, state=CaptureConnectionState.DISABLED)
        _assert_builder_failure(case)


def test_claude_dependency_builder_closes_every_runtime_evidence_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _claude_case(tmp_path)
    _exercise_dependency_guards(case)
    _exercise_builder_guards(case, monkeypatch)


def test_codex_dependency_builder_closes_every_runtime_evidence_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _codex_case(tmp_path)
    _exercise_dependency_guards(case)
    _exercise_builder_guards(case, monkeypatch)


class _VersionStream:
    def __init__(self, value: object, *, callable_read: bool = True) -> None:
        self._value = value
        if not callable_read:
            self.read = None  # type: ignore[assignment]

    def read(self, _size: int) -> object:
        return self._value

    def close(self) -> None:
        pass


class _VersionProcess:
    def __init__(self, stdout: object, stderr: object) -> None:
        self.stdout = stdout
        self.stderr = stderr

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0

    def poll(self) -> int:
        return 0

    def kill(self) -> None:
        pass


def _run_codex_version_runner() -> None:
    codex._bounded_version_runner(
        ("codex", "--version"),
        input=b"",
        capture_output=True,
        check=False,
        timeout=codex.CODEX_VERSION_TIMEOUT_SECONDS,
        env={},
    )


def test_codex_version_runner_rejects_missing_noncallable_and_nonbyte_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = (
        _VersionProcess(None, None),
        _VersionProcess(_VersionStream(b"", callable_read=False), _VersionStream(b"")),
        _VersionProcess(_VersionStream("not-bytes"), _VersionStream(b"")),
    )
    for process in cases:
        with monkeypatch.context() as scoped:
            scoped.setattr(
                subprocess_module,
                "Popen",
                lambda *_args, selected=process, **_kwargs: selected,
            )
            with pytest.raises(codex.CodexIntegrationError):
                _run_codex_version_runner()


def test_provider_environment_policy_and_project_tail_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claude_project = tmp_path / "claude-policy"
    claude_project.mkdir()
    claude_environment_value = {
        "HOME": str(tmp_path / "home"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }
    claude_spec = claude.provider_installation_spec(
        claude_project,
        environ=claude_environment_value,
    )
    with pytest.raises(claude.ClaudeCodeIntegrationError):
        claude._validate_project_hook_policy(claude_spec, {"HOME": 1})  # type: ignore[dict-item]
    (claude_project / ".claude").write_bytes(b"not-a-directory")
    with pytest.raises(claude.ClaudeCodeIntegrationError):
        claude._validate_project_hook_policy(claude_spec, claude_environment_value)

    codex_project = tmp_path / "codex-policy"
    codex_project.mkdir()
    codex_spec = codex.provider_installation_spec(
        codex_project,
        environ={"HOME": str(tmp_path / "codex-home")},
    )
    (codex_project / ".codex").write_bytes(b"not-a-directory")
    with pytest.raises(codex.CodexIntegrationError):
        codex._validate_project_hook_policy(codex_spec)

    for module, project, error in (
        (claude, tmp_path / "claude-home", claude.ClaudeCodeIntegrationError),
        (codex, tmp_path / "codex-home-project", codex.CodexIntegrationError),
    ):
        project.mkdir(exist_ok=True)
        with monkeypatch.context() as scoped:
            scoped.setattr(
                module,
                "environment_without_provider_credentials",
                lambda _environment: {"HOME": 1},
            )
            with pytest.raises(error):
                module.provider_installation_spec(project, environ={})

    with monkeypatch.context() as scoped:
        scoped.setattr(
            codex,
            "environment_without_provider_credentials",
            lambda _environment: {"PATH": 1},
        )
        with pytest.raises(codex.CodexIntegrationError):
            codex.probe_codex_environment(environ={})


def test_provider_project_discovery_rejects_files_and_bounds_parent_walks(
    tmp_path: Path,
) -> None:
    regular_file = tmp_path / "regular-file"
    regular_file.write_bytes(b"content")
    with pytest.raises(claude.ClaudeCodeIntegrationError):
        claude._claude_code_project_candidates({"cwd": str(regular_file)})
    with pytest.raises(codex.CodexIntegrationError):
        codex._discover_codex_project({"cwd": str(regular_file)})

    claude_deep = tmp_path / "claude-deep"
    claude_deep = claude_deep.joinpath(*(f"d{index}" for index in range(130)))
    claude_deep.mkdir(parents=True)
    with pytest.raises(claude.ClaudeCodeIntegrationError):
        claude._claude_code_project_candidates({"cwd": str(claude_deep)})

    codex_deep = tmp_path / "codex-deep"
    codex_deep = codex_deep.joinpath(*(f"d{index}" for index in range(130)))
    codex_deep.mkdir(parents=True)
    with pytest.raises(codex.CodexIntegrationError):
        codex._discover_codex_project({"cwd": str(codex_deep)})


def test_claude_project_candidate_walk_skips_unmarked_inner_configuration(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    nested = project / "nested"
    nested.mkdir(parents=True)
    for directory in (project / ".claude", nested / ".claude"):
        directory.mkdir()
    (nested / ".claude" / "settings.local.json").write_bytes(b"{}")
    (project / ".claude" / "settings.local.json").write_bytes(
        canonical_json({"marker": claude.CLAUDE_CODE_CONFIG_MARKER})
    )

    assert claude._claude_code_project_candidates({"cwd": str(nested)}) == (project,)
