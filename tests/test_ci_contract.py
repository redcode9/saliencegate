from __future__ import annotations

import inspect
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import textwrap
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
import scripts.smoke_capture_installed as capture_smoke
import scripts.smoke_shadow_installed as shadow_smoke
import scripts.verify_built_artifacts as artifact_verifier
import scripts.verify_connector_artifacts as connector_artifact_verifier
from scripts.verify_built_artifacts import (
    DOCUMENTED_COMMAND_CASES,
    EXECUTED_COMMAND_CASES,
    SUPPLEMENTAL_ARTIFACT_PROOFS,
    _run,
    artifact_compatible_commands,
)

from saliencegate.security import inspect_private_file_location

ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
MAKEFILE = ROOT / "Makefile"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"
README = ROOT / "README.md"
SOCKET_GUARD = ROOT / "scripts" / "run_without_sockets.py"
CORE_IMPORT_SMOKE = ROOT / "scripts" / "smoke_core_imports.py"
MODEL_RUNTIME_SMOKE = ROOT / "scripts" / "smoke_model_runtime.py"
LAUNCH_CONTRACT_SMOKE = ROOT / "scripts" / "smoke_launch_contracts.py"
PACKAGE_IMPORT_SMOKE = ROOT / "scripts" / "smoke_package_imports.py"
SHADOW_INSTALLED_SMOKE = ROOT / "scripts" / "smoke_shadow_installed.py"
CAPTURE_INSTALLED_SMOKE = ROOT / "scripts" / "smoke_capture_installed.py"
ARTIFACT_SOCKET_GUARD = ROOT / "scripts" / "artifact_socket_guard.py"
SHADOW_TRACE_BENCHMARK = ROOT / "scripts" / "benchmark_shadow_trace.py"
ARTIFACT_VERIFIER = ROOT / "scripts" / "verify_built_artifacts.py"
CONNECTOR_ARTIFACT_VERIFIER = ROOT / "scripts" / "verify_connector_artifacts.py"
CONNECTOR_NETWORK_GUARD = ROOT / "connectors" / "scripts" / "deny-network.mjs"
FORBIDDEN_CORE_MODULES = ("anthropic", "harbor", "httpx", "openai", "openai_harmony")
PROVIDER_CREDENTIAL_KEYS = (
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_ORGANIZATION",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT",
    "OPENAI_PROJECT_ID",
)
NATIVE_RUNNER_MATRIX = (
    ("ubuntu-24.04", "ubuntu-24.04"),
    ("macos-15", "macos-15"),
    ("windows-2025", "windows-2025"),
)

CORE_GATE_TARGETS = (
    "format",
    "lint",
    "typecheck",
    "test",
    "coverage",
    "docs-check",
    "build",
    "audit",
)
CAPTURE_GATE_TARGETS = (
    "capture-check",
    "connector-node-preflight",
    "connector-install",
    "connector-source-check",
    "connectors-check",
    "connector-artifact-smoke",
)
REQUIRED_TARGETS = (
    *CORE_GATE_TARGETS,
    *CAPTURE_GATE_TARGETS,
    "check",
)
CHECK_DEPENDENCIES = (
    "format",
    "lint",
    "typecheck",
    "capture-check",
    "test",
    "coverage",
    "docs-check",
    "connectors-check",
    "build",
    "artifact-smoke",
    "connector-artifact-smoke",
    "audit",
)


def _read(relative: str) -> str:
    path = ROOT / relative
    assert path.is_file(), f"required repository file is missing: {relative}"
    return path.read_text(encoding="utf-8")


class _UnreadableProviderEnvironment(Mapping[str, str]):
    def __init__(self) -> None:
        self.read_keys: list[str] = []
        self._keys = ("PATH", *(key.swapcase() for key in PROVIDER_CREDENTIAL_KEYS))

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, key: str) -> str:
        self.read_keys.append(key)
        if key.upper() in PROVIDER_CREDENTIAL_KEYS:
            raise AssertionError("provider credential value was read")
        if key == "PATH":
            return "/synthetic/bin"
        raise KeyError(key)


def test_artifact_environment_projection_never_reads_provider_credentials() -> None:
    core_environment = _UnreadableProviderEnvironment()
    assert artifact_verifier._environment_without_provider_credentials(core_environment) == {
        "PATH": "/synthetic/bin"
    }
    assert core_environment.read_keys == ["PATH"]

    connector_environment = _UnreadableProviderEnvironment()
    assert connector_artifact_verifier._project_environment_without_provider_values(
        connector_environment
    ) == {"PATH": "/synthetic/bin"}
    assert connector_environment.read_keys == ["PATH"]

    capture_environment = _UnreadableProviderEnvironment()
    projected = capture_smoke._subprocess_environment(capture_environment)
    assert projected["PATH"] == "/synthetic/bin"
    assert projected["SALIENCEGATE_ARTIFACT_SOCKET_DENIAL"] == "1"
    assert all(
        projected[key] == capture_smoke._POISONED_CREDENTIAL for key in PROVIDER_CREDENTIAL_KEYS
    )
    assert capture_environment.read_keys == ["PATH"]


def test_artifact_startup_guard_rejects_provider_credential_keys(tmp_path: Path) -> None:
    startup_log = tmp_path / "artifact-startups.log"
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    (site_packages / "saliencegate_artifact_socket_guard.py").write_bytes(
        ARTIFACT_SOCKET_GUARD.read_bytes()
    )
    (site_packages / "saliencegate_artifact_socket_guard.pth").write_bytes(
        b"import saliencegate_artifact_socket_guard\n"
    )
    environment = artifact_verifier._environment_without_provider_credentials(os.environ)
    environment.update(
        {
            "SALIENCEGATE_ARTIFACT_SOCKET_DENIAL": "1",
            "SALIENCEGATE_ARTIFACT_SOCKET_STARTUP_LOG": str(startup_log),
        }
    )
    command = (
        sys.executable,
        "-I",
        "-S",
        "-c",
        "import site,sys; site.addsitedir(sys.argv[1]); print('artifact-pth-continued')",
        str(site_packages),
    )

    for credential_key in PROVIDER_CREDENTIAL_KEYS:
        sentinel = f"credential-value-sentinel-{credential_key}".encode()
        poisoned_environment = {**environment, credential_key.swapcase(): sentinel.decode()}
        rejected = subprocess.run(
            command,
            env=poisoned_environment,
            check=False,
            capture_output=True,
            timeout=10,
        )
        assert rejected.returncode != 0
        assert b"artifact-pth-continued" not in rejected.stdout + rejected.stderr
        assert b"retained provider credentials" in rejected.stderr
        assert sentinel not in rejected.stdout + rejected.stderr
        assert not startup_log.exists()

    accepted = subprocess.run(
        command,
        env=environment,
        check=False,
        capture_output=True,
        timeout=10,
    )
    assert accepted.returncode == 0
    assert accepted.stdout == b"artifact-pth-continued\n"
    assert accepted.stderr == b""
    assert startup_log.read_bytes() == b"installed-artifact-socket-denial-active\n"


def test_artifact_guard_canary_receives_a_scrubbed_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[dict[str, str]] = []

    def fake_run(command: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        environment = kwargs.get("env")
        assert isinstance(environment, dict)
        observed.append(environment)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=b"installed-artifact-child-network-and-credential-denial-ok\n",
            stderr=b"",
        )

    monkeypatch.setattr(artifact_verifier, "_run", fake_run)
    source_environment = {
        "PATH": "/synthetic/bin",
        "SALIENCEGATE_ARTIFACT_SOCKET_DENIAL": "1",
        "SALIENCEGATE_ARTIFACT_SOCKET_STARTUP_LOG": str(tmp_path / "startups.log"),
        **{key.swapcase(): "credential-value-must-not-be-read" for key in PROVIDER_CREDENTIAL_KEYS},
    }

    artifact_verifier._prove_artifact_socket_guard(
        python=tmp_path / "python",
        cwd=tmp_path,
        environment=source_environment,
    )

    assert len(observed) == 1
    assert observed[0]["PATH"] == "/synthetic/bin"
    assert observed[0]["SALIENCEGATE_ARTIFACT_SOCKET_DENIAL"] == "1"
    assert all(key.upper() not in PROVIDER_CREDENTIAL_KEYS for key in observed[0])


def test_artifact_failure_diagnostics_redact_all_controlled_capture_sentinels() -> None:
    sentinels = (
        artifact_verifier._POISONED_PROVIDER_CREDENTIAL,
        *artifact_verifier._CONTROLLED_CAPTURE_SENTINELS,
    )
    diagnostic = artifact_verifier._diagnostic_excerpt(("visible " + " ".join(sentinels)).encode())

    assert diagnostic.startswith("visible ")
    assert diagnostic.count("<redacted>") == len(sentinels)
    assert all(sentinel not in diagnostic for sentinel in sentinels)


@pytest.mark.parametrize(
    "sentinel",
    (
        artifact_verifier._POISONED_PROVIDER_CREDENTIAL,
        artifact_verifier._CONTROLLED_CAPTURE_SENTINELS[0],
    ),
)
def test_capture_quickstart_rejects_sensitive_data_in_successful_stdout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sentinel: str,
) -> None:
    monkeypatch.setattr(
        artifact_verifier,
        "_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            (),
            0,
            stdout=f"status {sentinel}\n".encode(),
            stderr=b"",
        ),
    )

    with pytest.raises(RuntimeError, match="exposed sensitive native data") as raised:
        artifact_verifier._run_capture_cli(
            label="wheel",
            python=tmp_path / "python",
            guard=tmp_path / "guard.py",
            saliencegate=tmp_path / "saliencegate",
            arguments=("status",),
            cwd=tmp_path,
            environment={},
        )

    assert sentinel not in str(raised.value)


def test_installed_capture_artifact_proof_has_the_closed_provider_matrix() -> None:
    source = CAPTURE_INSTALLED_SMOKE.read_text(encoding="utf-8")
    assert "sys.flags.isolated != 1" in source
    assert 'os.environ.get("PYTHONPATH", "")' in source
    assert "socket.AF_INET" in source
    assert "did not import the installed package" in source
    assert "runtime unexpectedly exposes Node or npm" in source
    assert "probe_host=True" in inspect.getsource(capture_smoke._exercise_provider)
    assert "probe_host=True" in inspect.getsource(capture_smoke._emit_codex_fixture)
    assert "_UnreadableCredentialEnvironment" in source
    assert "capture-session-report/v1" in source
    assert "capture artifact report evidence is invalid" in source
    assert "hmac_sha256_local_mutation_detection" in source
    assert "report-after-disconnect.json" in source
    assert "persisted a raw sentinel" in source
    assert '"installed-e2e"' in source
    assert "capture-installed-connectors-e2e-ok" in source
    assert "capture-installed-bridge-callbacks-ok" in source
    assert 'import { fileURLToPath } from "node:url";' in source
    assert "fileURLToPath(imported.saliencegateBootstrap)" in source
    assert "fileURLToPath(bootstrapHref)" in source
    assert "saliencegateBootstrap.href !== bootstrapHref" not in source
    assert "SALIENCEGATE_ARTIFACT_SOCKET_DENIAL" in source
    socket_guard = ARTIFACT_SOCKET_GUARD.read_text(encoding="utf-8")
    assert "installed artifact child retained provider credentials" in socket_guard
    assert "PROVIDER_CREDENTIAL_DENIAL_ACTIVE" in socket_guard
    assert "for await (const chunk of process.stdin)" in source
    assert "Windows launcher metacharacter proof is incomplete" in source
    assert 'status.providers[0].status.value != "active_observed"' in source
    assert "status.providers[0].drift" in source
    for callback in (
        'invoke("session_start"',
        'invoke("before_agent_start"',
        'invoke("tool_execution_start"',
        'invoke("tool_execution_end"',
        'invoke("agent_settled"',
        'invoke("session_shutdown"',
    ):
        assert callback in source
    for provider in ("codex", "claude-code", "opencode", "pi"):
        assert f'"{provider}"' in source

    verifier = ARTIFACT_VERIFIER.read_text(encoding="utf-8")
    assert '"scripts/smoke_capture_installed.py"' in verifier
    assert 'b"capture-installed-artifact-ok\\n"' in verifier
    assert '"capture-lifecycle"' in verifier
    assert '"capture-quickstart"' in verifier
    assert '"emit-codex-fixture"' in verifier
    assert '"validate-codex-report"' in verifier
    for command_arguments in (
        '("connect", "codex", "--project", project, "--dry-run")',
        '("connect", "codex", "--project", project)',
        '("doctor", "--capture")',
        '("status", "codex", "--project", project)',
        '("sessions", "--limit", "20")',
        '("report", "--latest", "--output", report_path)',
        '("disconnect", "codex", "--project", project)',
    ):
        assert command_arguments in verifier
    assert '"codex.cmd"' in verifier
    assert '"claude.cmd"' in verifier
    assert "os.pathsep.join((os.fspath(fake_hosts), os.fspath(python.parent)))" in verifier
    assert "_environment_without_provider_credentials(environment)" in inspect.getsource(
        artifact_verifier._prove_artifact_socket_guard
    )
    connector_e2e = inspect.getsource(artifact_verifier._prove_installed_connector_e2e)
    assert 'del lifecycle_environment["SALIENCEGATE_ARTIFACT_SOCKET_DENIAL"]' in connector_e2e
    assert "env=lifecycle_environment" in connector_e2e

    connector_verifier = CONNECTOR_ARTIFACT_VERIFIER.read_text(encoding="utf-8")
    assert 'NATIVE_SECRET_SENTINEL = "connector-native-secret-sentinel"' in connector_verifier
    assert "persisted sensitive native data" in connector_verifier


def test_installed_callback_command_is_platform_exact(tmp_path: Path) -> None:
    launcher = tmp_path / "capture hook"
    launcher.write_bytes(b"synthetic launcher")
    launcher.chmod(0o700)

    posix = capture_smoke._launcher_command(
        launcher,
        environment={},
        platform="posix",
    )
    assert posix == capture_smoke._InstalledCommandCallback(
        command=(str(launcher.resolve(strict=True)),)
    )

    system_root = tmp_path / "Windows Root & Data"
    command = system_root / "System32" / "cmd.exe"
    command.parent.mkdir(parents=True)
    command.write_bytes(b"synthetic command interpreter")
    windows = capture_smoke._launcher_command(
        launcher,
        environment={"SystemRoot": str(system_root)},
        platform="nt",
    )
    resolved_command = str(command.resolve(strict=True))
    resolved_launcher = str(launcher.resolve(strict=True))
    assert windows == capture_smoke._InstalledCommandCallback(
        command=(f'"{resolved_command}" /d /v:off /s /c ""%SALIENCEGATE_LAUNCHER%""'),
        executable=resolved_command,
        launcher_environment=resolved_launcher,
    )
    naive_sequence = (
        resolved_command,
        "/d",
        "/v:off",
        "/s",
        "/c",
        '""%SALIENCEGATE_LAUNCHER%""',
    )
    assert subprocess.list2cmdline(naive_sequence) == (
        f'"{resolved_command}" /d /v:off /s /c '
        r"\"\"%SALIENCEGATE_LAUNCHER%\"\""
    )
    assert subprocess.list2cmdline(naive_sequence) != windows.command

    with pytest.raises(RuntimeError, match="platform is invalid"):
        capture_smoke._launcher_command(launcher, environment={}, platform="vms")


def test_installed_windows_callback_passes_raw_command_line_and_executable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launcher = tmp_path / "launcher path & data" / "capture-hook.cmd"
    launcher.parent.mkdir()
    launcher.write_bytes(b"@exit /b 0\r\n")
    launcher.chmod(0o700)
    system_root = tmp_path / "Windows Root & Data"
    command = system_root / "System32" / "cmd.exe"
    command.parent.mkdir(parents=True)
    command.write_bytes(b"synthetic command interpreter")
    callback = capture_smoke._launcher_command(
        launcher,
        environment={"SystemRoot": str(system_root)},
        platform="nt",
    )
    observed: list[tuple[object, dict[str, object]]] = []

    def fake_run(
        invoked: object,
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        observed.append((invoked, kwargs))
        return subprocess.CompletedProcess(invoked, 0, stdout=b"", stderr=b"")

    starts = iter((7, 8))
    monkeypatch.setattr(capture_smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(
        capture_smoke,
        "_socket_guard_start_count",
        lambda _environment: next(starts),
    )

    capture_smoke._invoke_installed_command_callback(
        callback=callback,
        payload=b'{"hook_event_name":"SessionStart"}\n',
        project=tmp_path,
        environment={"SystemRoot": str(system_root)},
    )

    assert len(observed) == 1
    invoked, kwargs = observed[0]
    assert invoked == callback.command
    assert isinstance(invoked, str)
    assert kwargs["executable"] == str(command.resolve(strict=True))
    assert kwargs["shell"] is False
    process_environment = kwargs["env"]
    assert isinstance(process_environment, dict)
    assert process_environment["SALIENCEGATE_LAUNCHER"] == str(launcher.resolve(strict=True))


def test_installed_codex_callback_is_derived_from_the_managed_config(tmp_path: Path) -> None:
    launcher = tmp_path / "capture-hook"
    launcher.write_bytes(b"synthetic launcher")
    launcher.chmod(0o700)
    config_path = tmp_path / "config.toml"
    marker = "saliencegate-managed-codex-hooks-v1"

    def config(command: str) -> bytes:
        groups = []
        for event_name in ("SessionStart", "PreToolUse", "PostToolUse"):
            groups.extend(
                (
                    f"[[hooks.{event_name}]]",
                    f"[[hooks.{event_name}.hooks]]",
                    'type = "command"',
                    f"command = {json.dumps(command)}",
                    "timeout = 3",
                )
            )
        return (f"# {marker}\n" + "\n".join(groups) + "\n").encode()

    spec = SimpleNamespace(
        bootstrap_path=None,
        bundle_path=None,
        config=SimpleNamespace(marker=marker),
        config_path=config_path,
        launcher_path=launcher,
    )
    config_path.write_bytes(config(str(launcher.resolve(strict=True))))
    callbacks = capture_smoke._installed_command_callbacks(
        capture_smoke._CASES[0],
        spec,
        environment={},
    )
    assert set(callbacks) == {"SessionStart", "PreToolUse", "PostToolUse"}
    assert all(
        callback.command == (str(launcher.resolve(strict=True)),) for callback in callbacks.values()
    )

    config_path.write_bytes(config(str(tmp_path / "wrong-launcher")))
    with pytest.raises(RuntimeError, match="config binding is invalid"):
        capture_smoke._installed_command_callbacks(
            capture_smoke._CASES[0],
            spec,
            environment={},
        )


def test_connector_artifact_verifier_materializes_native_launchers(tmp_path: Path) -> None:
    posix_workspace = tmp_path / "posix"
    posix_workspace.mkdir()
    posix_launcher = connector_artifact_verifier._materialize_launcher(
        posix_workspace,
        connector="opencode",
        platform="posix",
    )
    assert posix_launcher.name == "capture-launcher-opencode"
    assert posix_launcher.read_bytes() == connector_artifact_verifier.POSIX_LAUNCHER_SOURCE.replace(
        b"{provider}",
        b"opencode",
    )
    assert posix_launcher.read_bytes().endswith(b'"opencode"\n')
    assert posix_launcher.stat().st_mode & 0o100
    assert (posix_launcher.parent / "capture-launcher.mjs").is_file()

    windows_workspace = tmp_path / "windows"
    windows_workspace.mkdir()
    windows_launcher = connector_artifact_verifier._materialize_launcher(
        windows_workspace,
        connector="pi",
        platform="nt",
    )
    assert windows_launcher.name == "capture-launcher-pi.cmd"
    assert windows_launcher.read_bytes() == (
        connector_artifact_verifier.WINDOWS_LAUNCHER_SOURCE.replace(b"{provider}", b"pi")
    )
    assert windows_launcher.read_bytes().startswith(b"@echo off\r\n")
    assert windows_launcher.read_bytes().endswith(b'"pi"\r\n')
    assert b"%~n0" not in windows_launcher.read_bytes()
    assert windows_launcher.parent.name == "launcher path & data"
    assert " " in str(windows_launcher)
    assert "&" in str(windows_launcher)
    assert (windows_workspace / "proof" / "capture-launcher.mjs").is_file()

    with pytest.raises(
        connector_artifact_verifier.VerificationError,
        match="platform is unsupported",
    ):
        connector_artifact_verifier._materialize_launcher(
            tmp_path / "unsupported",
            connector="opencode",
            platform="vms",
        )


def test_connector_artifact_launch_log_requires_both_exact_providers(tmp_path: Path) -> None:
    launch_log = tmp_path / "launches.ndjson"

    def record(provider: str) -> str:
        return json.dumps(
            {
                "network_denial": True,
                "provider": provider,
                "provider_credential_keys_present": [],
                "schema_version": "connector-artifact-launch/v1",
                "stdin_bytes": 1,
            }
        )

    launch_log.write_text(
        "\n".join(record("opencode") for _ in range(3)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        connector_artifact_verifier.VerificationError,
        match="did not exercise both launch paths",
    ):
        connector_artifact_verifier._validate_launch_log(launch_log)

    launch_log.write_text(
        "\n".join((record("opencode"), record("opencode"), record("pi"))) + "\n",
        encoding="utf-8",
    )
    connector_artifact_verifier._validate_launch_log(launch_log)


def test_connector_artifact_verifier_resolves_npm_cli_without_executing_cmd(
    tmp_path: Path,
) -> None:
    tool_root = tmp_path / "toolchain"
    node = tool_root / "node.exe"
    npm = tool_root / "npm.cmd"
    npm_cli = tool_root / "node_modules" / "npm" / "bin" / "npm-cli.js"
    package = npm_cli.parent.parent / "package.json"
    npm_cli.parent.mkdir(parents=True)
    node.write_bytes(b"synthetic-node")
    npm.write_bytes(b"@echo off\r\n")
    npm_cli.write_bytes(b"synthetic-npm-cli")
    package.write_text('{"name":"npm","version":"10.9.3"}', encoding="utf-8")
    node.chmod(0o700)
    npm.chmod(0o700)

    assert connector_artifact_verifier._resolve_regular_command(
        str(node),
        label="Node.js",
    ) == node.resolve(strict=True)
    assert connector_artifact_verifier._validated_npm_cli(str(npm)) == npm_cli.resolve(strict=True)

    package.write_text('{"name":"npm","version":"10.9.2"}', encoding="utf-8")
    with pytest.raises(
        connector_artifact_verifier.VerificationError,
        match=r"could not validate npm-cli\.js",
    ):
        connector_artifact_verifier._validated_npm_cli(str(npm))


def test_connector_artifact_timeout_and_redaction_are_content_free(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    node = tmp_path / "node"
    npm_cli = tmp_path / "npm-cli.js"
    monkeypatch.setattr(
        connector_artifact_verifier,
        "_resolve_regular_command",
        lambda *_args, **_kwargs: node,
    )
    monkeypatch.setattr(
        connector_artifact_verifier,
        "_validated_npm_cli",
        lambda *_args, **_kwargs: npm_cli,
    )
    calls = 0

    def fake_run(*args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(args[0], 0, "v22.19.0\n", "")
        if calls == 2:
            return subprocess.CompletedProcess(args[0], 0, "10.9.3\n", "")
        raise subprocess.TimeoutExpired(
            args[0],
            30,
            output=connector_artifact_verifier.NATIVE_SECRET_SENTINEL,
            stderr=connector_artifact_verifier.POISONED_CREDENTIAL,
        )

    monkeypatch.setattr(connector_artifact_verifier.subprocess, "run", fake_run)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    guard = tmp_path / "deny-network.mjs"

    with pytest.raises(
        connector_artifact_verifier.VerificationError,
        match="connector artifact smoke timed out",
    ) as raised:
        connector_artifact_verifier._run_node_smoke(
            node="node",
            npm="npm",
            guard=guard,
            assets={"opencode": tmp_path / "opencode.js", "pi": tmp_path / "pi.js"},
            launch_log=tmp_path / "launches.ndjson",
            workspace=workspace,
        )

    rendered = str(raised.value)
    assert connector_artifact_verifier.NATIVE_SECRET_SENTINEL not in rendered
    assert connector_artifact_verifier.POISONED_CREDENTIAL not in rendered
    assert calls == 3
    assert (
        connector_artifact_verifier._redacted(
            " ".join(
                (
                    connector_artifact_verifier.NATIVE_SECRET_SENTINEL,
                    connector_artifact_verifier.POISONED_CREDENTIAL,
                )
            )
        )
        == "<redacted> <redacted>"
    )


@pytest.mark.parametrize(
    ("payload", "platform", "expected"),
    (
        (b'{"ok":true}\n', "posix", b'{"ok":true}\n'),
        (b'{"ok":true}\n', "nt", b'{"ok":true}\n'),
        (b'{"ok":true}\r\n', "nt", b'{"ok":true}\n'),
    ),
)
def test_core_artifact_verifier_normalizes_only_one_windows_transport_ending(
    payload: bytes,
    platform: str,
    expected: bytes,
) -> None:
    assert artifact_verifier._normalize_transport_stdout(payload, platform=platform) == expected


@pytest.mark.parametrize(
    "payload",
    (
        b'{"ok":true}\r\n\r\n',
        b'{"ok":true}\r\nextra',
        b'{"ok":true}\r\r\n',
    ),
)
def test_core_artifact_verifier_rejects_noncanonical_windows_transport(
    payload: bytes,
) -> None:
    with pytest.raises(RuntimeError, match="non-canonical Windows transport ending"):
        artifact_verifier._normalize_transport_stdout(payload, platform="nt")


@pytest.mark.parametrize(
    ("payload", "platform", "expected"),
    (
        (b"first\nsecond\n", "posix", b"first\nsecond\n"),
        (b"first\nsecond\n", "nt", b"first\nsecond\n"),
        (b"first\r\nsecond\r\n", "nt", b"first\nsecond\n"),
    ),
)
def test_core_artifact_verifier_normalizes_multiline_terminal_output(
    payload: bytes,
    platform: str,
    expected: bytes,
) -> None:
    assert artifact_verifier._normalize_terminal_stdout(payload, platform=platform) == expected


@pytest.mark.parametrize("payload", (b"first\rsecond\n", b"first\r\r\n"))
def test_core_artifact_verifier_rejects_noncanonical_windows_terminal_output(
    payload: bytes,
) -> None:
    with pytest.raises(RuntimeError, match="non-canonical Windows ending"):
        artifact_verifier._normalize_terminal_stdout(payload, platform="nt")


def test_installed_shadow_smoke_routes_windows_private_io_through_security_helpers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, Path, object]] = []
    path = tmp_path / "private" / "payload.json"

    def forbidden_posix_primitive(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("POSIX ownership primitive reached Windows branch")

    monkeypatch.setattr(shadow_smoke.os, "getuid", forbidden_posix_primitive, raising=False)
    monkeypatch.setattr(shadow_smoke.os, "fchmod", forbidden_posix_primitive, raising=False)
    monkeypatch.setattr(
        shadow_smoke,
        "_authorize_windows_private_parent",
        lambda parent: calls.append(("authorize", parent, None)),
    )
    monkeypatch.setattr(
        shadow_smoke,
        "_publish_windows_private_file",
        lambda target, data: calls.append(("publish", target, data)),
    )
    monkeypatch.setattr(
        shadow_smoke,
        "_read_windows_private_file",
        lambda target, *, maximum_bytes: (
            calls.append(("read", target, maximum_bytes)),
            b'{"ok":true}\n',
        )[1],
    )
    monkeypatch.setattr(
        shadow_smoke,
        "_ensure_windows_smoke_directory",
        lambda target: calls.append(("ensure", target, None)),
    )

    shadow_smoke._write_private(path, b'{"ok":true}\n', platform="nt")
    assert shadow_smoke._read_private(path, maximum_bytes=64, platform="nt") == b'{"ok":true}\n'
    shadow_smoke._prepare_private_directory(path.parent, platform="nt")

    assert calls == [
        ("authorize", path.parent, None),
        ("publish", path, b'{"ok":true}\n'),
        ("authorize", path.parent, None),
        ("read", path, 64),
        ("ensure", path.parent, None),
    ]


def test_core_artifact_verifier_routes_windows_private_publication_through_installed_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
    payload = b'{"private":true}\n'
    environment = {"PATH": "/synthetic/bin"}

    def forbidden_outer_private_write(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("outer POSIX private writer reached Windows branch")

    def fake_run(
        command: object,
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        normalized = tuple(str(item) for item in command)  # type: ignore[union-attr]
        calls.append((normalized, kwargs))
        return subprocess.CompletedProcess(
            normalized,
            0,
            stdout=b"shadow-private-file-ok\r\n",
            stderr=b"",
        )

    monkeypatch.setattr(artifact_verifier, "_write_private", forbidden_outer_private_write)
    monkeypatch.setattr(artifact_verifier, "_run", fake_run)
    artifact_verifier._publish_installed_private_file(
        tmp_path / "shadow" / "report.json",
        payload,
        python=tmp_path / "venv" / "Scripts" / "python.exe",
        guard=tmp_path / "package" / "scripts" / "run_without_sockets.py",
        validator=tmp_path / "package" / "scripts" / "smoke_shadow_installed.py",
        cwd=tmp_path / "case",
        environment=environment,
        platform="nt",
    )

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[-2:] == (
        "publish-private-file",
        str(tmp_path / "shadow" / "report.json"),
    )
    assert kwargs == {
        "cwd": tmp_path / "case",
        "env": environment,
        "capture_output": True,
        "input_bytes": payload,
    }
    proof_body = inspect.getsource(artifact_verifier._prove_documented_cases)
    assert "_write_private(" not in proof_body
    assert proof_body.count("_publish_installed_private_file(") == 3


def test_installed_shadow_smoke_private_stdin_uses_platform_private_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = b'{"private":true}\n'
    target = tmp_path / "private" / "payload.json"
    writes: list[tuple[Path, bytes]] = []

    class BinaryInput:
        buffer = io.BytesIO(payload)

    monkeypatch.setattr(shadow_smoke.sys, "stdin", BinaryInput())
    monkeypatch.setattr(
        shadow_smoke,
        "_write_private",
        lambda path, data: writes.append((path, data)),
    )

    shadow_smoke._publish_private_stdin(target)

    assert writes == [(target, payload)]


def test_installed_shadow_smoke_validates_windows_key_through_private_reader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import saliencegate.security as security

    key_path = tmp_path / "config" / "saliencegate" / "installation.key"
    guard = shadow_smoke._EnvironmentReadGuard({"HOME": str(tmp_path)})
    assert guard["HOME"] == str(tmp_path)
    reads: list[tuple[Path, int, str | None]] = []
    monkeypatch.setattr(security, "default_installation_key_path", lambda: key_path)
    monkeypatch.setattr(
        shadow_smoke,
        "_read_private",
        lambda path, *, maximum_bytes, platform=None: (
            reads.append((path, maximum_bytes, platform)),
            b"k" * 32,
        )[1],
    )

    shadow_smoke._validate_local_key(guard, platform="nt")
    assert reads == [(key_path, 32, "nt")]


def test_core_artifact_verifier_resolves_native_venv_and_cli_paths(tmp_path: Path) -> None:
    environment = tmp_path / "venv"
    assert artifact_verifier._venv_python(environment, platform="posix") == (
        environment / "bin" / "python"
    )
    assert artifact_verifier._venv_python(environment, platform="nt") == (
        environment / "Scripts" / "python.exe"
    )

    posix_case = tmp_path / "posix-case"
    posix_case.mkdir()
    posix_python = environment / "bin" / "python"
    assert (
        artifact_verifier._installed_cli_script(
            python=posix_python,
            case_root=posix_case,
            platform="posix",
        )
        == environment / "bin" / "saliencegate"
    )

    windows_case = tmp_path / "windows-case"
    windows_case.mkdir()
    windows_cli = artifact_verifier._installed_cli_script(
        python=environment / "Scripts" / "python.exe",
        case_root=windows_case,
        platform="nt",
    )
    assert windows_cli == windows_case / "saliencegate-command.py"
    assert windows_cli.read_bytes() == artifact_verifier._WINDOWS_CLI_SHIM
    assert b"raise SystemExit(entrypoint())" in windows_cli.read_bytes()

    with pytest.raises(RuntimeError, match="platform is unsupported"):
        artifact_verifier._venv_python(environment, platform="vms")


@pytest.mark.parametrize(
    "member_name",
    (
        "saliencegate-0.1.0/C:/evil",
        "saliencegate-0.1.0/name:stream",
        "saliencegate-0.1.0/CON",
        "saliencegate-0.1.0/aux/file.txt",
        "saliencegate-0.1.0/trailing.",
    ),
)
def test_core_artifact_extraction_rejects_windows_unsafe_members_on_every_platform(
    tmp_path: Path,
    member_name: str,
) -> None:
    sdist = tmp_path / "crafted.tar.gz"
    payload = b"synthetic"
    with tarfile.open(sdist, mode="w:gz") as archive:
        member = tarfile.TarInfo(member_name)
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    destination = tmp_path / "extracted"
    with pytest.raises(RuntimeError, match="Windows-unsafe path"):
        artifact_verifier._extract_sdist(sdist, destination)
    assert destination.is_dir()
    assert tuple(destination.iterdir()) == ()


def _issue_field(text: str, field_id: str) -> str:
    match = re.search(
        rf"(?ms)^  - type: [^\n]+\n    id: {re.escape(field_id)}\n"
        r"(?P<body>.*?)(?=^  - type:|\Z)",
        text,
    )
    assert match is not None, f"required issue field is missing: {field_id}"
    return match.group("body")


def _job_block(text: str, job_id: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(job_id)}:\n.*?(?=^  [a-z0-9][a-z0-9-]*:\n|\Z)",
        text,
    )
    assert match is not None, f"required CI job is missing: {job_id}"
    return match.group(0)


def _step_containing(job: str, needle: str) -> str:
    blocks = re.findall(
        r"(?ms)^      - name: .*?(?=^      - name:|\Z)",
        job,
    )
    matches = [block for block in blocks if needle in block]
    assert len(matches) == 1, f"expected one CI step containing {needle!r}"
    return matches[0]


def _assert_native_runner_matrix(job: str) -> None:
    assert "runs-on: ${{ matrix.runner }}" in job
    assert "fail-fast: false" in job
    assert (
        tuple(
            zip(
                re.findall(r"(?m)^          - platform: ([^\n]+)$", job),
                re.findall(r"(?m)^            runner: ([^\n]+)$", job),
                strict=True,
            )
        )
        == NATIVE_RUNNER_MATRIX
    )


def _assert_inline_python_step(job: str, needle: str) -> None:
    step = _step_containing(job, needle)
    assert "shell: python" in step
    marker = "        run: |\n"
    assert marker in step
    source = textwrap.dedent(step.split(marker, maxsplit=1)[1])
    compile(source, f"<{needle}>", "exec")


def _assert_installed_shadow_smoke(job: str, *, prefix: str) -> None:
    extraction = _step_containing(job, "tar -xzf dist/*.tar.gz")
    shadow = _step_containing(job, '" shadow analyze "')
    packaged_root = f'"$RUNNER_TEMP/{prefix}-package"'
    shadow_root = f'"$RUNNER_TEMP/{prefix}-shadow"'

    assert "--strip-components=1" in extraction
    assert packaged_root in extraction
    assert f'"$RUNNER_TEMP/{prefix}-package/examples/shadow_asyncio.py"' in shadow
    assert re.search(rf"(?m)^\s+mkdir -p [^\n]*{re.escape(shadow_root)}", shadow)
    assert re.search(rf"(?m)^\s+chmod 700 [^\n]*{re.escape(shadow_root)}", shadow)
    assert "umask 077" in shadow
    assert "scripts/smoke_shadow_installed.py write-trace" in shadow
    assert f'"$RUNNER_TEMP/{prefix}-shadow/events.ndjson"' in shadow
    assert f'"$RUNNER_TEMP/{prefix}-shadow/shadow-report.json"' in shadow
    assert f'"$RUNNER_TEMP/{prefix}-shadow/shadow-command.json"' in shadow
    assert "--capture-scope complete_run_declared" in shadow
    assert "scripts/smoke_shadow_installed.py validate-report" in shadow
    assert f'"$RUNNER_TEMP/{prefix}-package/examples/atif-shadow/one_call.py"' in shadow
    assert (
        f'"$RUNNER_TEMP/{prefix}-package/examples/atif-shadow/codex-minimal.trajectory.json"'
    ) in shadow
    assert (
        f'"$RUNNER_TEMP/{prefix}-package/examples/atif-shadow/terminus-minimal.trajectory.json"'
    ) in shadow
    assert shadow.count(" shadow analyze-atif ") == 2
    assert shadow.count(" validate-public-atif ") == 2
    assert "--profile harbor-codex-v1" in shadow
    assert "--profile harbor-terminus-2-v1" in shadow
    assert "harbor-codex/v1" in shadow
    assert "harbor-terminus-2/v1" in shadow
    assert "scripts/smoke_shadow_installed.py exercise-atif" in shadow
    assert 'OPENAI_API_KEY="provider-credential-read-must-fail"' in shadow
    assert 'ANTHROPIC_API_KEY="provider-credential-read-must-fail"' in shadow
    assert "tests/fixtures" not in shadow

    guarded_lines = [line for line in shadow.splitlines() if 'bin/python"' in line]
    assert len(guarded_lines) == 10
    assert all("-I scripts/run_without_sockets.py" in line for line in guarded_lines)


def test_artifact_verifier_reports_captured_failures_without_credentials() -> None:
    credential = "provider-credential-read-must-fail"
    source = (
        "import sys; "
        f"print({credential!r}); "
        "print('actionable stderr', file=sys.stderr); "
        "raise SystemExit(7)"
    )

    try:
        _run((sys.executable, "-I", "-c", source), capture_output=True)
    except RuntimeError as error:
        message = str(error)
    else:
        raise AssertionError("the failing proof command unexpectedly succeeded")

    assert "exited with 7" in message
    assert "actionable stderr" in message
    assert credential not in message
    assert "<redacted>" in message


def test_artifact_failure_diagnostics_never_read_ambient_provider_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile_environment = _UnreadableProviderEnvironment()

    class HostileEnvironmentOS:
        environ = hostile_environment

        @staticmethod
        def fspath(value: str | Path) -> str:
            return str(value)

    monkeypatch.setattr(artifact_verifier, "os", HostileEnvironmentOS)
    source = "import sys; print('provider-credential-read-must-fail'); raise SystemExit(9)"

    with pytest.raises(RuntimeError, match="exited with 9") as failure:
        _run(
            (sys.executable, "-I", "-c", source),
            capture_output=True,
        )

    assert "provider-credential-read-must-fail" not in str(failure.value)
    assert hostile_environment.read_keys == ["PATH"]


def test_artifact_workspace_canonicalizes_a_symlinked_temp_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    actual_parent = tmp_path / "actual"
    actual_parent.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual_parent, target_is_directory=True)
    created = actual_parent / "proof"
    created.mkdir(mode=0o700)
    returned = alias / "proof"
    monkeypatch.setattr(
        artifact_verifier.tempfile,
        "mkdtemp",
        lambda *, prefix: str(returned),
    )

    with artifact_verifier._workspace(None) as workspace:
        assert workspace == created.resolve(strict=True)
        assert workspace.is_dir()
        inspect_private_file_location(workspace / "absent")
        assert not (workspace / "absent").exists()

    assert not created.exists()
    assert actual_parent.is_dir()
    assert alias.is_symlink()


def test_artifact_main_places_default_workspace_beside_distribution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dist_dir = tmp_path / "checkout" / "dist"
    dist_dir.mkdir(parents=True)
    created = dist_dir.parent / "synthetic-smoke"
    observed: dict[str, object] = {}

    def make_temp_directory(*, prefix: str, **options: object) -> str:
        observed["prefix"] = prefix
        observed["parent"] = options.get("dir")
        created.mkdir(mode=0o700)
        return str(created)

    def verify(**options: object) -> None:
        observed["work_root"] = options["work_root"]
        observed["dist_dir"] = options["dist_dir"]

    monkeypatch.setattr(artifact_verifier.tempfile, "mkdtemp", make_temp_directory)
    monkeypatch.setattr(artifact_verifier, "verify_built_artifacts", verify)

    assert artifact_verifier.main(("--dist-dir", str(dist_dir))) == 0
    assert observed == {
        "prefix": "saliencegate-artifact-smoke-",
        "parent": dist_dir.parent.resolve(strict=True),
        "work_root": created.resolve(),
        "dist_dir": dist_dir.resolve(strict=True),
    }
    assert not created.exists()


def test_makefile_exposes_noninteractive_gates_in_the_authoritative_order() -> None:
    text = _read("Makefile")

    for target in REQUIRED_TARGETS:
        assert re.search(rf"(?m)^{re.escape(target)}\s*:", text)
    assert ".PHONY:" in text
    phony = next(line for line in text.splitlines() if line.startswith(".PHONY:"))
    assert all(target in phony.split() for target in REQUIRED_TARGETS)
    assert re.search(r"(?m)^\.NOTPARALLEL\s*:\s*$", text)
    assert re.search(r"(?m)^artifact-smoke\s*:\s*$", text)
    assert "artifact-smoke" in phony.split()

    docs_check = re.search(r"(?ms)^docs-check:\n(?P<body>.*?)(?=^[a-z-]+:)", text)
    assert docs_check is not None
    assert re.findall(r"(?m)^\s+(.+)$", docs_check.group("body")) == [
        "uv run --locked python scripts/check_public_tree.py",
        "uv run --locked python scripts/check_public_docs.py",
        "uv run --locked python scripts/check_readme_visuals.py",
    ]

    check = re.search(r"(?m)^check\s*:\s*(?P<dependencies>[^\n]+)$", text)
    assert check is not None
    assert tuple(check.group("dependencies").split()) == CHECK_DEPENDENCIES

    preflight = re.search(r"(?ms)^connector-node-preflight:\n(?P<body>.*?)(?=^[a-z-]+:)", text)
    assert preflight is not None
    assert "Node.js %s" in preflight.group("body")
    assert "npm %s" in preflight.group("body")
    assert 'test "$$node_version" = "v22.19.0"' in preflight.group("body")
    assert 'test "$$npm_version" = "10.9.3"' in preflight.group("body")
    assert re.search(r"(?m)^connector-install: connector-node-preflight$", text)
    assert re.search(
        r"(?m)^connector-source-check: "
        r"connector-test connector-benchmark connector-build connector-audit$",
        text,
    )
    assert "npx --yes --package=node@22.19.0 --package=npm@10.9.3 -c" in text
    assert "$(CONNECTOR_TOOLCHAIN) 'make connector-source-check'" in text
    assert "npm ci --no-audit --no-fund" in text
    assert "npm run connector:test" in text
    assert "npm run connector:benchmark -- --assert-budgets" in text
    assert "npm run connector:build:check" in text
    assert "npm run connector:audit" in text
    for credential_key in PROVIDER_CREDENTIAL_KEYS:
        assert (
            f"$(CONNECTOR_GATE_TARGETS): export {credential_key} := "
            "provider-credential-read-must-fail"
        ) in text
    assert "$(CONNECTOR_GATE_TARGETS): export" in text
    assert "${" not in "\n".join(
        line for line in text.splitlines() if "provider-credential-read-must-fail" in line
    )
    assert re.search(r"(?m)^artifact-smoke:\s*$", text)
    assert re.search(r"(?m)^connector-artifact-smoke:\s*$", text)

    required_commands = (
        "ruff format --check .",
        "ruff check .",
        "mypy src/saliencegate",
        "pytest --cov=saliencegate --cov-branch",
        "python scripts/check_public_tree.py",
        "python scripts/check_public_docs.py",
        "python scripts/check_readme_visuals.py",
        "uv lock --check",
        "uv build --no-build-isolation --clear --no-create-gitignore",
        "uv export --locked --all-extras --all-groups --no-emit-project",
        "SALIENCEGATE_REQUIRE_DISTRIBUTIONS=1",
        "pytest -q tests/test_package.py",
        "python scripts/verify_built_artifacts.py --dist-dir dist",
        "python scripts/verify_connector_artifacts.py --dist-dir dist --node node --npm npm",
        (
            "python scripts/verify_built_artifacts.py --dist-dir dist --node node "
            "--capture-connectors-only"
        ),
        "python scripts/run_capture_hook_benchmark.py --assert-budgets",
        "python scripts/benchmark_capture_report.py --assert-budgets",
        "tests/capture/test_store_concurrency.py",
        "tests/capture/test_properties.py",
        "tests/test_capture_*benchmark.py",
        "set -eu",
        "pip-audit --strict",
        "--progress-spinner off",
        "--disable-pip",
    )
    assert all(command in text for command in required_commands)


def test_ci_is_least_privilege_pinned_and_covers_supported_python() -> None:
    text = _read(".github/workflows/ci.yml")
    test_job = _job_block(text, "test")

    assert re.search(r"(?m)^permissions:\s*\n\s+contents:\s*read\s*$", text)
    assert len(re.findall(r"(?m)^permissions:", text)) == 1
    assert not re.search(r"(?m)^[ \t]+permissions:", text)
    assert not re.search(r"(?mi)^permissions:\s*(?:write-all|\{[^\n]*write)", text)
    assert not re.search(r"(?m)^\s+[a-z-]+:\s*write\s*$", text)
    assert "pull_request:" in text
    assert "push:" in text
    assert "workflow_dispatch:" in text
    assert "fail-fast: false" in text
    assert "timeout-minutes:" in text
    assert re.search(r"(?m)^    timeout-minutes: 90$", test_job)

    strategy = re.search(
        r"(?ms)^    strategy:\n(?P<body>.*?)(?=^    steps:)",
        text,
    )
    assert strategy is not None
    matrix_versions = re.findall(
        r'^          - "([0-9]+\.[0-9]+)"$',
        strategy.group("body"),
        flags=re.MULTILINE,
    )
    assert matrix_versions == ["3.11", "3.12", "3.13"]
    assert "UV_PYTHON: ${{ matrix.python-version }}" in test_job
    assert "sys.version_info[:2]" in test_job

    uses_lines = re.findall(r"(?m)^\s*-?\s*uses:\s*([^\s#]+)(?:\s+#\s*(\S.*))?$", text)
    assert uses_lines
    for reference, version_comment in uses_lines:
        assert re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", reference)
        assert re.search(r"\bv\d", version_comment)

    assert "astral-sh/setup-uv@" in text
    assert "enable-cache: true" in text
    assert "cache-dependency-glob: uv.lock" in text
    assert 'UV_LOCKED: "1"' in text
    assert "UV_FROZEN" not in text
    first_sync = "uv sync --locked --all-extras --dev --no-install-project"
    second_sync = "uv sync --locked --all-extras --dev --no-build-isolation"
    for job_id in ("test", "quality", "coverage", "build"):
        job = _job_block(text, job_id)
        assert job.count(first_sync) == 1
        assert job.count(second_sync) == 1
        assert job.index(first_sync) < job.index(second_sync)
    assert text.count(first_sync) == 4
    assert text.count(second_sync) == 4
    benchmark_first_sync = "uv sync --locked --dev --no-install-project"
    benchmark_second_sync = "uv sync --locked --dev --no-build-isolation"
    for job_id in (
        "shadow-benchmark-evidence",
        "capture-benchmark-contracts",
        "capture-platform-contract",
    ):
        performance = _job_block(text, job_id)
        assert performance.count(benchmark_first_sync) == 1
        assert performance.count(benchmark_second_sync) == 1
        assert performance.index(benchmark_first_sync) < performance.index(benchmark_second_sync)
    assert text.count(benchmark_first_sync) == 3
    assert text.count(benchmark_second_sync) == 3
    build_command = "uv build --no-build-isolation --clear --no-create-gitignore"
    build = _job_block(text, "build")
    assert build.count(build_command) == 1
    assert text.count(build_command) == 1
    for alternative in ("hatch build", "pip wheel", "python -m build", "python -m hatchling"):
        assert alternative not in text


def test_ci_separates_static_quality_from_authoritative_coverage() -> None:
    text = _read(".github/workflows/ci.yml")
    test_job = _job_block(text, "test")
    quality = _job_block(text, "quality")
    coverage = _job_block(text, "coverage")
    build = _job_block(text, "build")

    assert re.search(r"(?m)^    timeout-minutes: 30$", quality)
    assert re.findall(r"(?m)^        run: (make [^\n]+)$", quality) == [
        "make format lint typecheck docs-check audit"
    ]
    assert 'python-version: "3.12"' in coverage
    assert re.search(r"(?m)^    timeout-minutes: 30$", coverage)
    assert "needs:\n      - test" in coverage
    assert "coverage combine coverage-data" in coverage
    assert "coverage report --show-missing --fail-under=0" in coverage
    assert "coverage json --fail-under=0 -o .coverage.json" in coverage
    assert "scripts/check_coverage_thresholds.py .coverage.json --minimum 95" in coverage
    assert "actions/download-artifact@" in coverage
    assert "make coverage" not in coverage
    assert "coverage" not in re.findall(r"(?m)^        run: (make [^\n]+)$", quality)[0].split()

    assert test_job.count("- name: core") == 1
    assert test_job.count("- name: state-decay-v2") == 1
    assert test_job.count("- name: review-io") == 1
    assert test_job.count("- name: review-cli-pack") == 1
    assert re.search(
        r"(?m)^\s+tests$\n^\s+--ignore=tests/benchmarks/state_decay_v2$",
        test_job,
    )
    for review_file in (
        "test_public_review_io.py",
        "test_public_review.py",
        "test_public_review_cli.py",
        "test_public_review_pack.py",
    ):
        assert test_job.count(review_file) == 2
    assert "COVERAGE_FILE: .coverage.${{ matrix.shard.name }}" in test_job
    assert "include-hidden-files: true" in test_job

    needs = re.search(r"(?ms)^    needs:\n(?P<body>.*?)(?=^    [a-z-]+:)", build)
    assert needs is not None
    assert re.findall(r"(?m)^      - ([a-z][a-z-]+)$", needs.group("body")) == [
        "test",
        "quality",
        "coverage",
        "shadow-benchmark-evidence",
        "capture-benchmark-contracts",
        "connectors",
        "capture-platform-contract",
    ]

    makefile = _read("Makefile")
    check = re.search(r"(?m)^check\s*:\s*(?P<dependencies>[^\n]+)$", makefile)
    assert check is not None
    assert tuple(check.group("dependencies").split()) == CHECK_DEPENDENCIES
    assert "pytest --cov=saliencegate --cov-branch" in makefile
    assert "--cov-report=json:.coverage.json --cov-fail-under=0" in makefile
    assert "python scripts/check_coverage_thresholds.py .coverage.json --minimum 95" in makefile
    assert "fail_under = 95" in _read("pyproject.toml")


def test_ci_verifies_the_shadow_benchmark_contract_and_sealed_evidence() -> None:
    text = _read(".github/workflows/ci.yml")
    evidence = _job_block(text, "shadow-benchmark-evidence")
    build = _job_block(text, "build")

    assert "name: Shadow benchmark contract and evidence" in evidence
    assert "runs-on: ubuntu-24.04" in evidence
    assert 'python-version: "3.12"' in evidence
    assert re.search(r"(?m)^    timeout-minutes: 30$", evidence)
    assert "uv sync --locked --dev --no-install-project" in evidence
    assert "uv sync --locked --dev --no-build-isolation" in evidence
    assert "tests/test_shadow_trace_benchmark.py" in evidence
    assert "tests/test_readme_visuals.py" in evidence
    assert "uv run --locked python scripts/check_readme_visuals.py" in evidence
    assert "benchmark_shadow_trace.py --assert-budgets" not in evidence
    assert "SALIENCEGATE_BENCHMARK_RUNNER_IMAGE" not in evidence
    assert "- shadow-benchmark-evidence" in build

    benchmark = SHADOW_TRACE_BENCHMARK.read_text(encoding="utf-8")
    for required in (
        "1_000",
        "5.0",
        "15.0",
        "512",
        "median",
        "peak_rss",
        "cpu",
        "runner_image",
        "--assert-budgets",
    ):
        assert required in benchmark


def test_ci_verifies_capture_benchmark_contracts_and_connector_performance() -> None:
    text = _read(".github/workflows/ci.yml")
    capture = _job_block(text, "capture-benchmark-contracts")
    connectors = _job_block(text, "connectors")
    build = _job_block(text, "build")

    assert "name: Capture benchmark contracts" in capture
    assert "runs-on: ubuntu-24.04" in capture
    assert 'python-version: "3.12"' in capture
    assert re.search(r"(?m)^    timeout-minutes: 30$", capture)
    assert "tests/test_capture_hook_benchmark.py" in capture
    assert "tests/test_capture_report_benchmark.py" in capture
    assert "run_capture_hook_benchmark.py --assert-budgets" not in capture
    assert "benchmark_capture_report.py --assert-budgets" not in capture
    assert "SALIENCEGATE_CAPTURE_BENCHMARK_RUNNER_IMAGE" not in capture
    assert "provider-credential-read-must-fail" in capture

    setup_node = "actions/setup-node@48b55a011bda9f5d6aeb4c2d9c7362e8dae4041e"
    assert setup_node in connectors
    assert "# v6.4.0" in connectors
    assert "node-version: 22.19.0" in connectors
    assert "check-latest: false" in connectors
    assert "package-manager-cache: false" in connectors
    assert "run: make connector-source-check" in connectors
    assert "provider-credential-read-must-fail" in connectors
    assert "- capture-benchmark-contracts" in build
    assert "- connectors" in build

    package = _read("package.json")
    assert "connector:network-denial:selftest" in package
    assert "connector:benchmark" in package
    assert "connector:audit" in package
    assert "--import ./connectors/scripts/deny-network.mjs" in package


def test_ci_declares_targeted_native_capture_contracts() -> None:
    text = _read(".github/workflows/ci.yml")
    platform = _job_block(text, "capture-platform-contract")
    build = _job_block(text, "build")

    _assert_native_runner_matrix(platform)
    assert re.search(r"(?m)^    timeout-minutes: 60$", platform)
    for test_path in (
        "tests/security/test_windows.py",
        "tests/capture/test_store_security.py",
        "tests/capture/test_spool.py",
        "tests/integrations/test_hook.py",
        "tests/integrations/test_installation.py",
        "tests/integrations/test_launcher_renderer.py",
    ):
        assert test_path in platform
    assert "HOME: ${{ runner.temp }}/capture-platform-home" in platform
    assert "USERPROFILE: ${{ runner.temp }}/capture-platform-home" in platform
    assert "SALIENCEGATE_CI_ROOT=$root" in platform
    assert "/inheritance:r /grant:r" in platform
    assert '"*${ownerSid}:(OI)(CI)F"' in platform
    assert "test_native_windows_operations_authorize_a_real_private_directory" in platform
    assert "test_read_only_audit_authenticates_an_empty_spool_without_creating_a_lock" in platform
    assert "test_windows_launcher_preserves_stdin_argv_silence_and_timeout" in platform
    assert platform.count("test_native_windows_") == 5
    assert "$env:HOME = $captureHome" in platform
    assert '$env:TEMP = Join-Path $env:SALIENCEGATE_CI_ROOT "temp"' in platform
    assert '--basetemp (Join-Path $env:SALIENCEGATE_CI_ROOT "pytest")' in platform
    assert "provider-credential-read-must-fail" in platform
    assert "- capture-platform-contract" in build
    assert "--disable-socket" in _read("pyproject.toml")


def test_ci_builds_once_and_gates_distribution_membership_before_upload() -> None:
    text = _read(".github/workflows/ci.yml")
    test_job = _job_block(text, "test")
    build = _job_block(text, "build")

    assert text.count("actions/upload-artifact@") == 2
    assert test_job.count("actions/upload-artifact@") == 1
    assert build.count("actions/upload-artifact@") == 1
    assert "SALIENCEGATE_REQUIRE_DISTRIBUTIONS=1" in build
    assert "uv run --locked pytest -q tests/test_package.py" in build
    assert build.index("SALIENCEGATE_REQUIRE_DISTRIBUTIONS=1") < build.index(
        "actions/upload-artifact@"
    )
    assert "*.tar.gz" in text

    step_blocks = re.findall(
        r"(?ms)^      - name: .*?(?=^      - name:|^  [a-z][a-z-]+:|\Z)",
        text,
    )
    upload_steps = [block for block in step_blocks if "actions/upload-artifact@" in block]
    assert len(upload_steps) == 2
    build_upload = next(block for block in upload_steps if "name: python-distributions" in block)
    coverage_upload = next(block for block in upload_steps if "coverage-${{" in block)
    uploaded_paths = re.findall(r"(?m)^\s+path:\s*([^\n]+)$", build_upload)
    assert uploaded_paths == ["dist/"]
    assert "path: .coverage.${{ matrix.shard.name }}" in coverage_upload
    assert "include-hidden-files: true" in coverage_upload
    assert all(
        forbidden not in "\n".join(uploaded_paths)
        for forbidden in ("trace", "runs", ".artifacts", "reports")
    )

    assert "Require exactly one wheel and one source distribution" in text
    assert "find dist -mindepth 1 -maxdepth 1" in text
    assert "-name '*.whl'" in text
    assert "-name '*.tar.gz'" in text


def test_ci_proves_public_atif_semantics_from_artifacts_without_checkout() -> None:
    text = _read(".github/workflows/ci.yml")
    artifact = _job_block(text, "artifact-only-atif")

    assert "needs:\n      - build" in artifact
    _assert_native_runner_matrix(artifact)
    assert "Prove ATIF from artifacts only (${{ matrix.platform }})" in artifact
    assert 'PYTHONPATH: ""' in artifact
    assert "actions/download-artifact@" in artifact
    assert "actions/checkout@" not in artifact
    assert "GITHUB_WORKSPACE" not in artifact
    assert "tests/fixtures" not in artifact
    assert "${{ runner.temp }}/saliencegate-artifact-only" in artifact
    assert artifact.count("shell: python") == 2
    _assert_inline_python_step(artifact, "Recover the shared artifact verifier")
    _assert_inline_python_step(artifact, "Prove documented commands")
    assert 'tarfile.open(sdists[0], mode="r:gz")' in artifact
    assert '("scripts", "verify_built_artifacts.py")' in artifact
    assert 'os.environ["ARTIFACT_ROOT"]' in artifact
    assert "sys.executable" in artifact
    assert '"--dist-dir",' in artifact
    assert '"--work-dir",' in artifact
    assert '"--python",' in artifact
    assert '"3.12",' in artifact
    assert "check=True" in artifact
    for unix_only in ("tar -xzf", "find ", "mkdir -p", "chmod ", "umask "):
        assert unix_only not in artifact
    assert " shadow analyze-atif " not in artifact
    assert " validate-public-atif " not in artifact

    assert ARTIFACT_VERIFIER.is_file()
    assert set(DOCUMENTED_COMMAND_CASES) | SUPPLEMENTAL_ARTIFACT_PROOFS == EXECUTED_COMMAND_CASES
    documented: list[str] = []
    for path in (README, ROOT / "examples" / "atif-shadow" / "README.md"):
        documented.extend(artifact_compatible_commands(path.read_text(encoding="utf-8")))
    assert set(documented) == set(DOCUMENTED_COMMAND_CASES.values())

    verifier = ARTIFACT_VERIFIER.read_text(encoding="utf-8")
    installed_smoke = SHADOW_INSTALLED_SMOKE.read_text(encoding="utf-8")
    assert '"prepare-private-directory"' in verifier
    assert "_normalize_transport_stdout(native.stdout)" in verifier
    assert "_normalize_transport_stdout(completed.stdout)" in verifier
    assert 'sys.argv[1] == "prepare-private-directory"' in installed_smoke
    for windows_security_boundary in (
        "NativeWindowsSecurityOperations",
        "authorize_windows_private_path",
        "ensure_windows_private_directory",
        "publish_private_file",
        "read_private_file",
    ):
        assert windows_security_boundary in installed_smoke


def test_ci_proves_connector_bundles_from_artifacts_without_checkout() -> None:
    text = _read(".github/workflows/ci.yml")
    artifact = _job_block(text, "artifact-only-connectors")

    assert "needs:\n      - build" in artifact
    _assert_native_runner_matrix(artifact)
    assert "Prove connectors from artifacts only (${{ matrix.platform }})" in artifact
    assert 'PYTHONPATH: ""' in artifact
    assert "actions/download-artifact@" in artifact
    assert "actions/checkout@" not in artifact
    assert "GITHUB_WORKSPACE" not in artifact
    assert "tests/fixtures" not in artifact
    assert "${{ runner.temp }}/saliencegate-connector-artifact-only" in artifact
    assert artifact.count("shell: python") == 2
    _assert_inline_python_step(artifact, "Recover the connector artifact verifiers")
    _assert_inline_python_step(artifact, "Prove installed callbacks and bundles")
    assert 'tarfile.open(sdists[0], mode="r:gz")' in artifact
    assert '("scripts", "verify_connector_artifacts.py")' in artifact
    assert '("scripts", "verify_built_artifacts.py")' in artifact
    assert 'os.environ["ARTIFACT_ROOT"]' in artifact
    assert "sys.executable" in artifact
    assert '"--dist-dir",' in artifact
    assert '"--work-dir",' in artifact
    assert '"--node",' in artifact
    assert '"node",' in artifact
    assert '"--npm",' in artifact
    assert '"npm",' in artifact
    assert '"--capture-connectors-only",' in artifact
    assert '"3.12",' in artifact
    assert "check=True" in artifact
    for unix_only in ("tar -xzf", "find ", "mkdir -p", "chmod ", "umask "):
        assert unix_only not in artifact
    assert "node-version: 22.19.0" in artifact
    assert "package-manager-cache: false" in artifact
    assert "astral-sh/setup-uv@" in artifact
    assert 'version: "0.11.28"' in artifact
    assert "provider-credential-read-must-fail" in artifact
    assert "USERPROFILE:" in artifact
    assert "APPDATA:" in artifact
    assert "LOCALAPPDATA:" in artifact
    installed_verifier = ARTIFACT_VERIFIER.read_text(encoding="utf-8")
    assert 'case_root = work_root / label / "launcher path & data"' in installed_verifier

    assert CONNECTOR_ARTIFACT_VERIFIER.is_file()
    assert CONNECTOR_NETWORK_GUARD.is_file()
    assert connector_artifact_verifier.EXPECTED_NODE_VERSION == "v22.19.0"
    assert connector_artifact_verifier.EXPECTED_NPM_VERSION == "10.9.3"
    assert set(connector_artifact_verifier.BUNDLES) == {"opencode", "pi"}
    assert connector_artifact_verifier.PROVIDER_CREDENTIAL_KEYS == PROVIDER_CREDENTIAL_KEYS
    verifier = CONNECTOR_ARTIFACT_VERIFIER.read_text(encoding="utf-8")
    assert "ERR_SALIENCEGATE_NETWORK_DISABLED" in verifier
    assert "process.getBuiltinModule" in verifier
    assert "wheel and sdist connector bundles are not byte-identical" in verifier
    assert "dist must contain exactly one wheel and one sdist" in verifier
    assert "persisted sensitive native data" in verifier
    assert connector_artifact_verifier.NATIVE_SECRET_SENTINEL in verifier
    assert "os.environ.copy()" not in verifier
    assert "if key.upper() in PROVIDER_CREDENTIAL_KEY_SET" in verifier
    assert "provider credential reached connector launcher" in verifier
    assert "network denial was not inherited by connector launcher" in verifier
    assert "openCodeHooks.event" in verifier
    assert 'piHandlers.get("session_start")' in verifier
    assert "built connector runtime did not exercise both launch paths" in verifier
    assert "if observed_providers != set(RUNTIME_PROFILES)" in verifier
    assert "launcher = _materialize_launcher(workspace, connector=connector)" in verifier
    assert 'f"capture-launcher-{connector}.cmd"' in verifier
    assert '"launcher path & data"' in verifier
    assert "WINDOWS_LAUNCHER_SOURCE" in verifier
    assert "_validated_npm_cli" in verifier
    assert '"npm-cli.js"' in verifier
    assert '(resolved_node, npm_cli, "--version")' in verifier
    assert '(npm, "--version")' not in verifier
    assert "SALIENCEGATE_ARTIFACT_NODE" in verifier


def test_artifact_sentinel_failures_are_content_free(tmp_path: Path) -> None:
    capture_sentinel = b"artifact-codex-raw-content-sentinel"
    capture_path = tmp_path / "capture" / "state.sqlite3"
    capture_path.parent.mkdir()
    capture_path.write_bytes(capture_sentinel)
    with pytest.raises(RuntimeError, match="persisted a raw sentinel") as capture_failure:
        capture_smoke._scan_for_sentinels(tmp_path / "capture", (capture_sentinel,))
    assert capture_sentinel.decode() not in str(capture_failure.value)
    capture_path.unlink()

    sibling_state = tmp_path / "state" / "saliencegate" / "capture.sqlite3"
    sibling_state.parent.mkdir(parents=True)
    sibling_state.write_bytes(capture_sentinel)
    with pytest.raises(RuntimeError, match="persisted a raw sentinel"):
        capture_smoke._scan_for_sentinels(tmp_path, (capture_sentinel,))

    connector_path = tmp_path / "connector" / "launches.ndjson"
    connector_path.parent.mkdir()
    connector_path.write_text(
        connector_artifact_verifier.NATIVE_SECRET_SENTINEL,
        encoding="utf-8",
    )
    with pytest.raises(
        connector_artifact_verifier.VerificationError,
        match="persisted sensitive native data",
    ) as connector_failure:
        connector_artifact_verifier._assert_poison_absent(tmp_path / "connector")
    assert connector_artifact_verifier.NATIVE_SECRET_SENTINEL not in str(connector_failure.value)


def test_ci_exercises_the_installed_core_wheel_without_optional_runtime() -> None:
    text = _read(".github/workflows/ci.yml")
    core = _job_block(text, "installed-core-wheel")

    assert "needs:\n      - build" in core
    assert 'PYTHONPATH: ""' in core
    assert "actions/download-artifact@" in core
    assert (
        "uv export --locked --no-dev --no-emit-project "
        '--output-file "$RUNNER_TEMP/core-runtime-requirements.txt" --quiet'
    ) in core
    assert '--require-hashes -r "$RUNNER_TEMP/core-runtime-requirements.txt"' in core
    assert "--no-deps dist/*.whl" in core
    assert '"$RUNNER_TEMP/core-wheel-venv/bin/python" -I scripts/run_without_sockets.py' in core
    assert "scripts/smoke_core_imports.py" in core
    assert core.count("scripts/run_without_sockets.py") >= 19
    import_smoke = CORE_IMPORT_SMOKE.read_text(encoding="utf-8")
    assert "import saliencegate.cli" in import_smoke
    assert "import saliencegate.shadow" in import_smoke
    assert "find_spec(optional_module)" in import_smoke
    assert "optional_module in sys.modules" in import_smoke
    for optional_module in FORBIDDEN_CORE_MODULES:
        assert f'"{optional_module}"' in import_smoke
    for command in (
        '"$RUNNER_TEMP/core-wheel-venv/bin/saliencegate" doctor',
        '"$RUNNER_TEMP/core-wheel-venv/bin/saliencegate" replay',
        '"$RUNNER_TEMP/core-wheel-venv/bin/saliencegate" validate',
        '"$RUNNER_TEMP/core-wheel-venv/bin/saliencegate" benchmark state-decay-smoke',
        '"$RUNNER_TEMP/core-wheel-venv/bin/saliencegate" algorithm replay',
    ):
        assert command in core
    assert "tests/fixtures/runs/basic.jsonl" in core
    assert "tests/fixtures/runs/paper_two_phase_basic.jsonl" in core
    assert "tests/fixtures/models/paper_two_phase_fixed_step_responses.jsonl" in core
    assert "--condition fixed_step" in core
    assert '"$RUNNER_TEMP/core-wheel-output/algorithm/manifest.json"' in core
    assert (
        '"$RUNNER_TEMP/core-wheel-venv/bin/saliencegate" demo --json '
        '> "$RUNNER_TEMP/core-wheel-output/launch-demo.json"'
    ) in core
    assert (
        '"$RUNNER_TEMP/core-wheel-venv/bin/saliencegate-review" --help '
        '> "$RUNNER_TEMP/core-wheel-output/review-help.txt"'
    ) in core
    for command in ("build-pack", "review", "status", "build-envelope"):
        assert (
            f'"$RUNNER_TEMP/core-wheel-venv/bin/saliencegate-review" {command} --help '
            '>> "$RUNNER_TEMP/core-wheel-output/review-help.txt"'
        ) in core
    assert (
        "scripts/smoke_launch_contracts.py "
        '"$RUNNER_TEMP/core-wheel-output/launch-demo.json" '
        '"$RUNNER_TEMP/core-wheel-output/review-help.txt"'
    ) in core
    launch_smoke = LAUNCH_CONTRACT_SMOKE.read_text(encoding="utf-8")
    assert "launch-contracts-ok" in launch_smoke
    assert "finalize" in launch_smoke
    assert "build-envelope" in launch_smoke
    _assert_installed_shadow_smoke(core, prefix="core-wheel")


def test_ci_exercises_the_installed_sdist_from_locked_build_dependencies() -> None:
    text = _read(".github/workflows/ci.yml")
    sdist = _job_block(text, "installed-sdist")

    assert "needs:\n      - build" in sdist
    assert 'PYTHONPATH: ""' in sdist
    assert "actions/download-artifact@" in sdist
    assert "--all-extras" not in sdist
    assert "--all-groups" not in sdist
    assert (
        "uv export --locked --no-dev --no-emit-project "
        '--output-file "$RUNNER_TEMP/sdist-core-runtime-requirements.txt" --quiet'
    ) in sdist
    assert (
        "uv export --locked --only-dev --no-emit-project "
        '--output-file "$RUNNER_TEMP/sdist-build-constraints.txt" --quiet'
    ) in sdist
    assert '--require-hashes -r "$RUNNER_TEMP/sdist-core-runtime-requirements.txt"' in sdist
    assert '--require-hashes --constraints "$RUNNER_TEMP/sdist-build-constraints.txt"' in sdist
    lock = _read("uv.lock")
    hatchling = re.search(r'(?ms)^\[\[package\]\]\nname = "hatchling"\nversion = "([^"]+)"', lock)
    assert hatchling is not None
    assert f"hatchling=={hatchling.group(1)}" in sdist
    assert "--no-deps --no-build-isolation dist/*.tar.gz" in sdist
    assert '"$RUNNER_TEMP/sdist-venv/bin/python" -I scripts/run_without_sockets.py' in sdist
    assert "scripts/smoke_package_imports.py" in sdist
    assert sdist.count("scripts/run_without_sockets.py") >= 19
    import_smoke = PACKAGE_IMPORT_SMOKE.read_text(encoding="utf-8")
    assert "import saliencegate.capture" in import_smoke
    assert "import saliencegate.shadow" in import_smoke
    assert "discover_capture_migrations" in import_smoke
    assert "capture migration resources are incomplete" in import_smoke
    assert "load_capture_capability_registry" in import_smoke
    assert "find_spec(optional_module)" in import_smoke
    assert "optional_module in sys.modules" in import_smoke
    for optional_module in FORBIDDEN_CORE_MODULES:
        assert f'"{optional_module}"' in import_smoke
    for command in (
        '"$RUNNER_TEMP/sdist-venv/bin/saliencegate" doctor',
        '"$RUNNER_TEMP/sdist-venv/bin/saliencegate" replay',
        '"$RUNNER_TEMP/sdist-venv/bin/saliencegate" validate',
        '"$RUNNER_TEMP/sdist-venv/bin/saliencegate" benchmark state-decay-smoke',
        '"$RUNNER_TEMP/sdist-venv/bin/saliencegate" algorithm replay',
    ):
        assert command in sdist
    assert "tests/fixtures/runs/basic.jsonl" in sdist
    assert "tests/fixtures/runs/paper_two_phase_basic.jsonl" in sdist
    assert "tests/fixtures/models/paper_two_phase_fixed_step_responses.jsonl" in sdist
    assert "--condition fixed_step" in sdist
    assert '"$RUNNER_TEMP/sdist-output/algorithm/manifest.json"' in sdist
    assert 'chmod 700 "$RUNNER_TEMP/sdist-home" "$RUNNER_TEMP/sdist-output"' in sdist
    assert (
        '"$RUNNER_TEMP/sdist-venv/bin/saliencegate" demo --json '
        '> "$RUNNER_TEMP/sdist-output/launch-demo.json"'
    ) in sdist
    assert (
        '"$RUNNER_TEMP/sdist-venv/bin/saliencegate-review" --help '
        '> "$RUNNER_TEMP/sdist-output/review-help.txt"'
    ) in sdist
    for command in ("build-pack", "review", "status", "build-envelope"):
        assert (
            f'"$RUNNER_TEMP/sdist-venv/bin/saliencegate-review" {command} --help '
            '>> "$RUNNER_TEMP/sdist-output/review-help.txt"'
        ) in sdist
    assert (
        "scripts/smoke_launch_contracts.py "
        '"$RUNNER_TEMP/sdist-output/launch-demo.json" '
        '"$RUNNER_TEMP/sdist-output/review-help.txt"'
    ) in sdist
    _assert_installed_shadow_smoke(sdist, prefix="sdist")


def test_installed_shadow_smoke_is_private_core_only_and_self_validating() -> None:
    source = SHADOW_INSTALLED_SMOKE.read_text(encoding="utf-8")

    for optional_module in FORBIDDEN_CORE_MODULES:
        assert f'"{optional_module}"' in source
    assert "find_spec(optional_module)" in source
    assert "optional_module in sys.modules" in source
    assert "os.O_EXCL" in source
    assert "0o600" in source
    assert "os.fsync" in source
    assert "StableReadPolicy.PRIVATE_OWNER" in source
    assert "decode_shadow_run_report" in source
    assert "decode_shadow_trace_report" in source
    assert "canonical_json(report" in source
    assert "canonical_json(command_report)" in source
    assert "_EnvironmentReadGuard" in source
    assert "_PROVIDER_CREDENTIAL_KEYS" in source
    assert "_assert_network_is_denied" in source
    assert '"exercise-atif"' in source
    assert '"harbor-codex-v1"' in source
    assert '"harbor-terminus-2-v1"' in source
    for zero_field in (
        "model_calls",
        "budget_reservations",
        "cycles_created",
        "memory_revisions",
        "interventions",
        "delivery_authorizations",
        "deliveries",
        "intervention_outcomes",
    ):
        assert f'"{zero_field}"' in source


def test_ci_keeps_the_installed_model_runtime_smoke_config_only() -> None:
    text = _read(".github/workflows/ci.yml")
    runtime = _job_block(text, "installed-model-runtime")

    assert "needs:\n      - build" in runtime
    assert 'PYTHONPATH: ""' in runtime
    assert "actions/download-artifact@" in runtime
    assert (
        "uv export --locked --all-extras --no-dev --no-emit-project "
        '--output-file "$RUNNER_TEMP/model-runtime-requirements.txt" --quiet'
    ) in runtime
    assert '--require-hashes -r "$RUNNER_TEMP/model-runtime-requirements.txt"' in runtime
    assert "--no-deps dist/*.whl" in runtime
    assert (
        '"$RUNNER_TEMP/model-runtime-venv/bin/python" -I scripts/run_without_sockets.py' in runtime
    )
    assert runtime.count("scripts/run_without_sockets.py") == 1
    assert "scripts/smoke_model_runtime.py" in runtime
    smoke = MODEL_RUNTIME_SMOKE.read_text(encoding="utf-8")
    assert "from saliencegate.models.openai_compatible import OpenAICompatibleConfig" in smoke
    assert 'base_url="http://127.0.0.1:11434/v1"' in smoke
    assert 'model="gpt-oss:20b"' in smoke
    assert "import saliencegate.commands.pilot" in smoke
    assert "OpenAICompatibleClient" not in smoke
    assert "saliencegate pilot" not in runtime
    assert "/chat/completions" not in runtime


def test_ci_installed_jobs_share_only_the_built_distributions() -> None:
    text = _read(".github/workflows/ci.yml")
    installed = tuple(
        _job_block(text, job_id)
        for job_id in ("installed-core-wheel", "installed-sdist", "installed-model-runtime")
    )

    assert text.count("actions/download-artifact@") == 6
    for job in installed:
        assert "name: python-distributions" in job
        assert "path: dist/" in job
        assert "persist-credentials: false" in job
        for line in job.splitlines():
            if '/bin/saliencegate"' in line or '/bin/saliencegate-review"' in line:
                assert "scripts/run_without_sockets.py" in line


def test_ci_fetches_complete_head_history_for_the_public_tree_guard() -> None:
    text = _read(".github/workflows/ci.yml")
    checkout_count = text.count("uses: actions/checkout@")

    assert checkout_count > 0
    assert text.count("persist-credentials: false") == checkout_count
    assert text.count("fetch-depth: 0") == checkout_count


def test_socket_guard_allows_only_local_socket_pairs(tmp_path: Path) -> None:
    harmless = tmp_path / "harmless.py"
    harmless.write_text(
        "import _socket\nimport asyncio\nimport socket\n"
        "left, right = socket.socketpair()\nleft.close()\nright.close()\n"
        "left, right = _socket.socketpair()\nleft.close()\nright.close()\n"
        "asyncio.run(asyncio.sleep(0))\n"
        "print('offline-ok')\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        (sys.executable, "-I", str(SOCKET_GUARD), str(harmless)),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0
    assert completed.stdout == "offline-ok\n"
    assert completed.stderr == ""

    blocked_sources = {
        "public-ipv4": "import socket\nsocket.socket(socket.AF_INET)\n",
        "public-ipv6": "import socket\nsocket.socket(socket.AF_INET6)\n",
        "private-ipv4": "import _socket\n_socket.socket(_socket.AF_INET)\n",
        "private-ipv6": "import _socket\n_socket.socket(_socket.AF_INET6)\n",
        "public-resolver": "import socket\nsocket.getaddrinfo('localhost', 80)\n",
        "private-resolver": "import _socket\n_socket.getaddrinfo('localhost', 80)\n",
        "public-unix": "import socket\nsocket.socket(socket.AF_UNIX)\n",
        "private-unix": "import _socket\n_socket.socket(_socket.AF_UNIX)\n",
        "public-inherited-ipv4": (
            "import os\nimport socket\nread, write = os.pipe()\n"
            "socket.socket(socket.AF_INET, fileno=read)\n"
        ),
        "private-inherited-ipv4": (
            "import _socket\nimport os\nread, write = os.pipe()\n"
            "_socket.socket(_socket.AF_INET, fileno=read)\n"
        ),
    }
    for name, source in blocked_sources.items():
        blocked = tmp_path / f"blocked-{name}.py"
        blocked.write_text(source, encoding="utf-8")
        refused = subprocess.run(
            (sys.executable, "-I", str(SOCKET_GUARD), str(blocked)),
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert refused.returncode != 0, name
        assert "SocketAccessError" in refused.stderr, name
        assert "socket or resolver access is disabled for this smoke" in refused.stderr, name


def test_dependabot_and_collaboration_templates_require_reproducible_evidence() -> None:
    dependabot = _read(".github/dependabot.yml")
    assert 'package-ecosystem: "uv"' in dependabot
    assert 'package-ecosystem: "github-actions"' in dependabot
    assert dependabot.count('interval: "weekly"') == 2
    assert dependabot.count('directory: "/"') == 2

    bug = _read(".github/ISSUE_TEMPLATE/bug.yml")
    for field in ("reproduction", "expected", "actual", "environment"):
        assert field in bug.lower()
    assert "secret" in bug.lower()

    benchmark = _read(".github/ISSUE_TEMPLATE/benchmark.yml")
    required_benchmark_fields = (
        "evidence-kind",
        "model-runtime",
        "hardware",
        "configuration-digest",
        "seed",
        "artifact-digest",
        "reproduction-command",
        "results",
    )
    for field_id in required_benchmark_fields:
        field = _issue_field(benchmark, field_id)
        assert re.search(
            r"(?m)^    validations:\n      required: true$",
            field,
        )
    assert "external" in benchmark.lower() and "synthetic" in benchmark.lower()

    for field_id in (
        "reproduction",
        "expected",
        "actual",
        "environment",
    ):
        field = _issue_field(bug, field_id)
        assert re.search(r"(?m)^    validations:\n      required: true$", field)

    pull_request = _read(".github/pull_request_template.md").lower()
    assert "make check" in pull_request
    assert "secret" in pull_request
    assert "claim" in pull_request


def test_contributing_documents_the_same_exact_gate_order() -> None:
    text = CONTRIBUTING.read_text(encoding="utf-8")
    heading = "## Required gate order"
    assert heading in text
    section = text.split(heading, maxsplit=1)[1]
    positions = [section.index(f"`make {target}`") for target in CORE_GATE_TARGETS]
    assert positions == sorted(positions)
    assert "`make check` runs this sequence" in section
    first_sync = "uv sync --locked --all-extras --dev --no-install-project"
    second_sync = "uv sync --locked --all-extras --dev --no-build-isolation"
    assert first_sync in text
    assert second_sync in text
    assert text.index(first_sync) < text.index(second_sync)

    readme = README.read_text(encoding="utf-8")
    assert first_sync in readme
    assert second_sync in readme
