from __future__ import annotations

import ast
import os
import shlex
import subprocess
import sys
import time
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest
from tests.capture.store_support import (
    CONNECTION_ID,
    authenticated_intake,
    capture_context,
)

from saliencegate.capture.adapters import (
    CAPTURE_ADAPTER_PROTOCOL_VERSION,
    CaptureAdapterCapabilities,
)
from saliencegate.capture.capabilities import (
    CaptureProfile,
    capture_capability_digest,
    capture_profile,
)
from saliencegate.capture.health import CaptureHealthCode
from saliencegate.capture.identities import CaptureDigestContext
from saliencegate.capture.schema import MAX_CAPTURE_NATIVE_BYTES, CaptureIntake
from saliencegate.capture.spool import CaptureSpoolUnavailableError
from saliencegate.capture.store import CaptureStoreBusyError, CaptureStoreStateError
from saliencegate.integrations.hook import (
    CaptureHookArguments,
    CaptureHookDependencies,
    CaptureHookError,
    parse_capture_hook_arguments,
    read_capture_hook_document,
    run_capture_hook,
)
from saliencegate.integrations.launcher_renderer import (
    CaptureLauncherPlatform,
    render_capture_launcher,
)

HOOK_SOURCE = Path("src/saliencegate/integrations/hook.py")
POSIX_LAUNCHER = Path("src/saliencegate/integrations/launchers/posix.sh")
WINDOWS_LAUNCHER = Path("src/saliencegate/integrations/launchers/windows.cmd")
PROFILE = CaptureProfile.CODEX_HOOKS_V1
PROFILE_VALUE = PROFILE.value


def _exact_limit_document() -> bytes:
    # Consume the complete JSON item budget while keeping aggregate JSON strings
    # at their independent 1 MiB limit. The remaining bytes are canonical integers.
    integer_count = 9_997
    string_value = b"x" * (1_024 * 1_024 - 2)
    prefix = b'{"n":['
    suffix = b'],"s":"' + string_value + b'"}'
    digit_bytes = MAX_CAPTURE_NATIVE_BYTES - len(prefix) - len(suffix) - integer_count + 1
    width, wider_count = divmod(digit_bytes, integer_count)
    integers = [b"1" * (width + (index < wider_count)) for index in range(integer_count)]
    document = prefix + b",".join(integers) + suffix
    assert len(document) == MAX_CAPTURE_NATIVE_BYTES
    return document


def _capabilities() -> CaptureAdapterCapabilities:
    manifest = capture_profile(PROFILE)
    return CaptureAdapterCapabilities(
        protocol_version=CAPTURE_ADAPTER_PROTOCOL_VERSION,
        profile_id=PROFILE,
        capability_digest=capture_capability_digest(manifest),
        host_version=manifest.host_version,
    )


class _Adapter:
    def __init__(
        self,
        calls: list[str],
        intakes: tuple[CaptureIntake, ...] = (),
        *,
        failure: Exception | None = None,
    ) -> None:
        self.calls = calls
        self.intakes = intakes
        self.failure = failure

    def capabilities(self) -> CaptureAdapterCapabilities:
        self.calls.append("capabilities")
        return _capabilities()

    def adapt_bytes(self, source: bytes, *, context: object) -> tuple[CaptureIntake, ...]:
        assert source == b"{}"
        assert context is capture_context_object
        self.calls.append("adapt")
        if self.failure is not None:
            raise self.failure
        return self.intakes


class _Store:
    def __init__(self, calls: list[str], failure: Exception | None = None) -> None:
        self.calls = calls
        self.failure = failure

    def append(self, intake: CaptureIntake) -> object:
        del intake
        self.calls.append("append")
        if self.failure is not None:
            raise self.failure
        return object()

    def close(self) -> None:
        self.calls.append("close_store")


class _Spool:
    def __init__(self, calls: list[str], failure: Exception | None = None) -> None:
        self.calls = calls
        self.failure = failure

    def enqueue(self, intake: CaptureIntake) -> object:
        del intake
        self.calls.append("enqueue")
        if self.failure is not None:
            raise self.failure
        return object()

    def admit(self, store: object, intake: CaptureIntake) -> object:
        self.calls.append("admit")
        if self.failure is not None:
            raise self.failure
        append = getattr(store, "append", None)
        assert callable(append)
        try:
            return append(intake)
        except CaptureStoreBusyError:
            return self.enqueue(intake)


registry_evidence = object()
receipt_evidence = object()
connection_evidence = object()
capture_context_object = capture_context()


def _dependencies(
    calls: list[str],
    *,
    adapter: _Adapter,
    store: _Store,
    spool: _Spool | None = None,
) -> CaptureHookDependencies:
    selected_spool = _Spool(calls) if spool is None else spool

    def validate_registry(profile: CaptureProfile) -> object:
        assert profile is PROFILE
        calls.append("registry")
        return registry_evidence

    def validate_receipt(
        profile: CaptureProfile,
        connection_id: str,
        registry: object,
    ) -> object:
        assert (profile, connection_id, registry) == (
            PROFILE,
            CONNECTION_ID,
            registry_evidence,
        )
        calls.append("receipt")
        return receipt_evidence

    def validate_connection(
        profile: CaptureProfile,
        connection_id: str,
        registry: object,
        receipt: object,
    ) -> object:
        assert (profile, connection_id, registry, receipt) == (
            PROFILE,
            CONNECTION_ID,
            registry_evidence,
            receipt_evidence,
        )
        calls.append("connection")
        return connection_evidence

    def load_context(connection: object) -> CaptureDigestContext:
        assert connection is connection_evidence
        calls.append("context")
        return capture_context_object

    def resolve_adapter(connection: object) -> object:
        assert connection is connection_evidence
        calls.append("adapter")
        return adapter

    def open_store(connection: object) -> _Store:
        assert connection is connection_evidence
        calls.append("open_store")
        return store

    def open_spool(connection: object) -> _Spool:
        assert connection is connection_evidence
        calls.append("open_spool")
        return selected_spool

    def mark_health(connection: object, code: CaptureHealthCode) -> None:
        assert connection is connection_evidence
        calls.append(f"health:{code.value}")

    return CaptureHookDependencies(
        validate_registry=validate_registry,
        validate_receipt=validate_receipt,
        validate_connection=validate_connection,
        load_context=load_context,
        resolve_adapter=resolve_adapter,
        open_store=open_store,
        open_spool=open_spool,
        mark_health=mark_health,
    )


def test_transport_parser_accepts_only_the_two_fixed_non_sensitive_arguments() -> None:
    assert parse_capture_hook_arguments(
        ("--profile", PROFILE_VALUE, "--connection", CONNECTION_ID)
    ) == CaptureHookArguments(profile=PROFILE, connection_id=CONNECTION_ID)
    assert parse_capture_hook_arguments(
        ("--connection", CONNECTION_ID, "--profile", PROFILE_VALUE)
    ) == CaptureHookArguments(profile=PROFILE, connection_id=CONNECTION_ID)

    invalid = (
        (),
        ("--profile", PROFILE_VALUE),
        ("--profile", PROFILE_VALUE, "--connection", CONNECTION_ID, "raw-secret"),
        ("--profile", PROFILE_VALUE, "--profile", PROFILE_VALUE),
        ("--profile=" + PROFILE_VALUE, "--connection", CONNECTION_ID),
        ("--profile", "codex-hooks/v2", "--connection", CONNECTION_ID),
        ("--profile", PROFILE_VALUE, "--connection", "$(provider-secret)"),
    )
    for arguments in invalid:
        with pytest.raises(CaptureHookError):
            parse_capture_hook_arguments(arguments)


def test_transport_reader_accepts_the_exact_two_mib_canonical_boundary() -> None:
    document = _exact_limit_document()

    assert read_capture_hook_document(BytesIO(document)) == document


@pytest.mark.parametrize("document", (b"{}\n", b'{ "z": 0, "a": 1 }'))
def test_transport_reader_preserves_valid_noncanonical_provider_json(document: bytes) -> None:
    assert read_capture_hook_document(BytesIO(document)) == document


@pytest.mark.parametrize(
    "document",
    (
        b'{"s":"' + (b"x" * (1_024 * 1_024)) + b'"}',
        b'{"n":[' + (b"0," * 9_998) + b"0]}",
        (b'{"nested":' * 33) + b"0" + (b"}" * 33),
        b'{"n":NaN}',
        b'{"n":Infinity}',
    ),
)
def test_transport_reader_rejects_string_item_depth_and_number_limit_overflow(
    document: bytes,
) -> None:
    with pytest.raises(CaptureHookError, match="capture hook failed"):
        read_capture_hook_document(BytesIO(document))


@pytest.mark.parametrize(
    "document",
    (
        b"",
        b"[]",
        b'{"duplicate":1,"duplicate":2}',
        b"{}{}",
        b"\xff",
        b" " * (MAX_CAPTURE_NATIVE_BYTES + 1),
    ),
)
def test_transport_reader_rejects_nonobject_duplicate_trailing_and_oversized_input(
    document: bytes,
) -> None:
    with pytest.raises(CaptureHookError, match="capture hook failed"):
        read_capture_hook_document(BytesIO(document))


def test_registry_receipt_and_connection_are_validated_before_native_adapter_dispatch() -> None:
    calls: list[str] = []
    adapter = _Adapter(calls)
    dependencies = _dependencies(calls, adapter=adapter, store=_Store(calls))

    assert (
        run_capture_hook(
            ("--profile", PROFILE_VALUE, "--connection", CONNECTION_ID),
            BytesIO(b"{}"),
            dependencies=dependencies,
        )
        == 0
    )
    assert calls == [
        "registry",
        "receipt",
        "connection",
        "context",
        "adapter",
        "capabilities",
        "adapt",
        "open_store",
        "close_store",
    ]


def test_receipt_failure_is_silent_and_never_resolves_or_dispatches_the_adapter() -> None:
    calls: list[str] = []
    dependencies = _dependencies(calls, adapter=_Adapter(calls), store=_Store(calls))

    def reject_receipt(
        profile: CaptureProfile,
        connection_id: str,
        registry: object,
    ) -> object:
        del profile, connection_id, registry
        calls.append("receipt_rejected")
        raise RuntimeError("raw-provider-receipt-secret")

    dependencies = replace(dependencies, validate_receipt=reject_receipt)

    assert (
        run_capture_hook(
            ("--profile", PROFILE_VALUE, "--connection", CONNECTION_ID),
            BytesIO(b"{}"),
            dependencies=dependencies,
        )
        == 0
    )
    assert calls == ["registry", "receipt_rejected"]


def test_configured_admission_uses_the_spool_ordering_fence_and_falls_back_on_busy() -> None:
    calls: list[str] = []
    intake = authenticated_intake("session_started")
    dependencies = _dependencies(
        calls,
        adapter=_Adapter(calls, (intake,)),
        store=_Store(calls, CaptureStoreBusyError()),
    )

    assert (
        run_capture_hook(
            ("--profile", PROFILE_VALUE, "--connection", CONNECTION_ID),
            BytesIO(b"{}"),
            dependencies=dependencies,
        )
        == 0
    )
    assert calls[-6:] == [
        "open_store",
        "open_spool",
        "admit",
        "append",
        "enqueue",
        "close_store",
    ]


def test_spool_open_failure_marks_unavailable_without_direct_store_append() -> None:
    calls: list[str] = []
    intake = authenticated_intake("session_started")
    dependencies = _dependencies(
        calls,
        adapter=_Adapter(calls, (intake,)),
        store=_Store(calls),
    )

    def unavailable_spool(_connection: object) -> _Spool:
        calls.append("open_spool")
        raise CaptureSpoolUnavailableError()

    dependencies = replace(dependencies, open_spool=unavailable_spool)

    assert (
        run_capture_hook(
            ("--profile", PROFILE_VALUE, "--connection", CONNECTION_ID),
            BytesIO(b"{}"),
            dependencies=dependencies,
        )
        == 0
    )
    assert calls[-4:] == [
        "open_store",
        "open_spool",
        "health:spool_unavailable",
        "close_store",
    ]
    assert "append" not in calls


def test_spool_lock_failure_marks_unavailable_without_direct_store_append() -> None:
    calls: list[str] = []
    intake = authenticated_intake("session_started")
    dependencies = _dependencies(
        calls,
        adapter=_Adapter(calls, (intake,)),
        store=_Store(calls),
        spool=_Spool(calls, CaptureSpoolUnavailableError()),
    )

    assert (
        run_capture_hook(
            ("--profile", PROFILE_VALUE, "--connection", CONNECTION_ID),
            BytesIO(b"{}"),
            dependencies=dependencies,
        )
        == 0
    )
    assert calls[-5:] == [
        "open_store",
        "open_spool",
        "admit",
        "health:spool_unavailable",
        "close_store",
    ]
    assert "append" not in calls


def test_nonbusy_store_failure_marks_content_free_health_without_enqueueing() -> None:
    calls: list[str] = []
    intake = authenticated_intake("session_started")
    dependencies = _dependencies(
        calls,
        adapter=_Adapter(calls, (intake,)),
        store=_Store(calls, CaptureStoreStateError()),
    )

    assert (
        run_capture_hook(
            ("--profile", PROFILE_VALUE, "--connection", CONNECTION_ID),
            BytesIO(b"{}"),
            dependencies=dependencies,
        )
        == 0
    )
    assert "open_spool" in calls
    assert "admit" in calls
    assert "enqueue" not in calls
    assert "health:coverage_degraded" in calls


def test_adapter_and_spool_failures_are_absorbed_without_rendering_provider_content() -> None:
    adapter_calls: list[str] = []
    adapter_dependencies = _dependencies(
        adapter_calls,
        adapter=_Adapter(
            adapter_calls,
            failure=RuntimeError("raw-provider-adapter-secret"),
        ),
        store=_Store(adapter_calls),
    )
    assert (
        run_capture_hook(
            ("--profile", PROFILE_VALUE, "--connection", CONNECTION_ID),
            BytesIO(b"{}"),
            dependencies=adapter_dependencies,
        )
        == 0
    )
    assert adapter_calls[-1] == "health:coverage_degraded"

    spool_calls: list[str] = []
    intake = authenticated_intake("session_started")
    spool_dependencies = _dependencies(
        spool_calls,
        adapter=_Adapter(spool_calls, (intake,)),
        store=_Store(spool_calls, CaptureStoreBusyError()),
        spool=_Spool(spool_calls, CaptureSpoolUnavailableError()),
    )
    assert (
        run_capture_hook(
            ("--profile", PROFILE_VALUE, "--connection", CONNECTION_ID),
            BytesIO(b"{}"),
            dependencies=spool_dependencies,
        )
        == 0
    )
    assert spool_calls[-4:] == [
        "open_spool",
        "admit",
        "health:spool_unavailable",
        "close_store",
    ]
    assert "append" not in spool_calls


def test_provider_facing_entrypoint_is_always_zero_with_empty_standard_streams() -> None:
    command = (
        sys.executable,
        "-c",
        "from saliencegate.integrations.hook import entrypoint; "
        "raise SystemExit(entrypoint(['--profile','codex-hooks/v1',"
        "'--connection','connection-one']))",
    )
    completed = subprocess.run(
        command,
        input=b'{"secret":"provider-owned"}\n',
        capture_output=True,
        check=False,
        timeout=5,
    )

    assert completed.returncode == 0
    assert completed.stdout == b""
    assert completed.stderr == b""


def test_default_transport_absorbs_malformed_input_without_loading_runtime() -> None:
    assert (
        run_capture_hook(
            ("--profile", PROFILE_VALUE, "--connection", CONNECTION_ID),
            BytesIO(b"not-json"),
        )
        == 0
    )


def _render_posix_launcher(
    destination: Path,
    *,
    executable: Path,
    profile: str = PROFILE_VALUE,
    connection_id: str = CONNECTION_ID,
) -> Path:
    source = POSIX_LAUNCHER.read_text(encoding="utf-8")
    rendered = (
        source.replace("__SALIENCEGATE_EXECUTABLE_SHELL__", shlex.quote(str(executable)))
        .replace("__SALIENCEGATE_WATCHDOG_SHELL__", shlex.quote("/bin/sleep"))
        .replace("__SALIENCEGATE_PROFILE_SHELL__", shlex.quote(profile))
        .replace("__SALIENCEGATE_CONNECTION_SHELL__", shlex.quote(connection_id))
    )
    destination.write_text(rendered, encoding="utf-8")
    destination.chmod(0o700)
    return destination


def _python_target(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{body}\n", encoding="utf-8")
    path.chmod(0o700)
    return path


@pytest.mark.skipif(os.name == "nt", reason="POSIX launcher contract")
def test_posix_launcher_preserves_stdin_fixed_argv_and_absorbs_child_output_and_exit(
    tmp_path: Path,
) -> None:
    captured_stdin = tmp_path / "captured stdin"
    captured_argv = tmp_path / "captured argv"
    executable = _python_target(
        tmp_path / "capture $() ; target",
        "import pathlib, sys\n"
        f"pathlib.Path({str(captured_stdin)!r}).write_bytes(sys.stdin.buffer.read())\n"
        f"pathlib.Path({str(captured_argv)!r}).write_text(repr(sys.argv[1:]), encoding='utf-8')\n"
        "print('provider-stdout-secret')\n"
        "print('provider-stderr-secret', file=sys.stderr)\n"
        "raise SystemExit(73)",
    )
    launcher = _render_posix_launcher(tmp_path / "launcher", executable=executable)
    payload = b'{"canonical":"payload"}'

    completed = subprocess.run(
        (str(launcher), "ignored-provider-argv"),
        input=payload,
        capture_output=True,
        check=False,
        timeout=5,
    )

    assert completed.returncode == 0
    assert completed.stdout == b""
    assert completed.stderr == b""
    assert captured_stdin.read_bytes() == payload
    assert captured_argv.read_text(encoding="utf-8") == repr(
        ["--profile", PROFILE_VALUE, "--connection", CONNECTION_ID]
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX launcher contract")
@pytest.mark.parametrize("executable_mode", (None, 0o600))
def test_posix_launcher_is_zero_and_silent_for_missing_or_nonexecutable_target(
    tmp_path: Path,
    executable_mode: int | None,
) -> None:
    executable = tmp_path / "missing capture hook"
    if executable_mode is not None:
        executable.write_text("#!/bin/sh\nexit 91\n", encoding="utf-8")
        executable.chmod(executable_mode)
    launcher = _render_posix_launcher(tmp_path / "launcher", executable=executable)

    completed = subprocess.run(
        (str(launcher),),
        input=b"{}",
        capture_output=True,
        check=False,
        timeout=5,
    )

    assert (completed.returncode, completed.stdout, completed.stderr) == (0, b"", b"")


@pytest.mark.skipif(os.name == "nt", reason="POSIX launcher contract")
def test_posix_launcher_enforces_the_two_second_fail_open_timeout(tmp_path: Path) -> None:
    executable = _python_target(
        tmp_path / "slow-capture-hook",
        "import time\ntime.sleep(30)",
    )
    launcher = _render_posix_launcher(tmp_path / "launcher", executable=executable)
    hostile_path = tmp_path / "hostile-path"
    hostile_path.mkdir()
    fake_sleep = _python_target(
        hostile_path / "sleep",
        "import time\ntime.sleep(30)",
    )
    assert fake_sleep.exists()
    environment = os.environ.copy()
    environment["PATH"] = str(hostile_path)

    started = time.monotonic()
    completed = subprocess.run(
        (str(launcher),),
        input=b"{}",
        capture_output=True,
        check=False,
        env=environment,
        timeout=5,
    )
    elapsed = time.monotonic() - started

    assert (completed.returncode, completed.stdout, completed.stderr) == (0, b"", b"")
    assert 1.5 <= elapsed < 4.0


def test_windows_launcher_encodes_the_native_contract() -> None:
    source = WINDOWS_LAUNCHER.read_text(encoding="utf-8")

    assert "setlocal DisableDelayedExpansion" in source
    assert 'set "capture_executable=__SALIENCEGATE_EXECUTABLE_BATCH__"' in source
    assert 'set "capture_powershell=__SALIENCEGATE_WATCHDOG_BATCH__"' in source
    assert 'set "capture_profile=__SALIENCEGATE_PROFILE_BATCH__"' in source
    assert 'set "capture_connection=__SALIENCEGATE_CONNECTION_BATCH__"' in source
    assert "2000-$clock.ElapsedMilliseconds" in source
    assert "WaitForExit([int]$remaining)" in source
    assert ".Kill()" in source
    assert "call :capture_main >nul 2>nul" in source
    assert '"%capture_powershell%" -NoLogo' in source
    assert "\npowershell.exe " not in source
    assert source.count("exit /b 0") >= 2
    assert "%*" not in source


@pytest.mark.skipif(
    os.name != "nt",
    reason="native batch execution is the remote R01 gate",
)
def test_windows_launcher_preserves_stdin_argv_silence_and_timeout_with_pinned_watchdog(
    tmp_path: Path,
) -> None:
    system_root = Path(os.environ["SYSTEMROOT"])
    powershell = (
        system_root / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    ).resolve(strict=True)
    command_processor = (system_root / "System32" / "cmd.exe").resolve(strict=True)
    source = tmp_path / "capture target.cs"
    executable = tmp_path / "capture target.exe"
    source.write_text(
        """
using System;
using System.IO;
using System.Threading;

public static class CaptureTarget {
    public static int Main(string[] args) {
        string argvPath = Environment.GetEnvironmentVariable("SG_CAPTURE_TEST_ARGV");
        if (Environment.GetEnvironmentVariable("SG_CAPTURE_TEST_MODE") == "sleep") {
            File.WriteAllText(argvPath, "started");
            Thread.Sleep(30000);
            return 0;
        }
        string stdinPath = Environment.GetEnvironmentVariable("SG_CAPTURE_TEST_STDIN");
        using (Stream input = Console.OpenStandardInput())
        using (FileStream output = new FileStream(stdinPath, FileMode.Create, FileAccess.Write)) {
            input.CopyTo(output);
        }
        File.WriteAllLines(argvPath, args);
        Console.Out.Write("provider-stdout-secret");
        Console.Error.Write("provider-stderr-secret");
        return 73;
    }
}
""".strip(),
        encoding="utf-8",
    )
    compile_script = tmp_path / "compile target.ps1"
    compile_script.write_text(
        "Add-Type -TypeDefinition "
        "(Get-Content -Raw -LiteralPath $env:SG_CAPTURE_TEST_SOURCE) "
        "-Language CSharp -OutputAssembly $env:SG_CAPTURE_TEST_EXECUTABLE "
        "-OutputType ConsoleApplication\n",
        encoding="utf-8",
    )
    compile_environment = os.environ.copy()
    compile_environment["SG_CAPTURE_TEST_SOURCE"] = str(source)
    compile_environment["SG_CAPTURE_TEST_EXECUTABLE"] = str(executable)
    compiled = subprocess.run(
        (
            str(powershell),
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(compile_script),
        ),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        env=compile_environment,
        timeout=30,
    )
    assert compiled.returncode == 0, compiled.stderr.decode(errors="replace")
    assert executable.is_file()

    launcher = tmp_path / "capture launcher.cmd"
    launcher.write_bytes(
        render_capture_launcher(
            executable=executable,
            profile=PROFILE,
            connection_id=CONNECTION_ID,
            platform=CaptureLauncherPlatform.WINDOWS,
            watchdog_executable=powershell,
        )
    )
    hostile = tmp_path / "hostile cwd"
    hostile.mkdir()
    (hostile / "powershell.exe").write_bytes(b"hostile-path-shadow")
    captured_stdin = tmp_path / "captured stdin.bin"
    captured_argv = tmp_path / "captured argv.txt"
    environment = os.environ.copy()
    environment["PATH"] = str(hostile)
    environment["SG_CAPTURE_TEST_STDIN"] = str(captured_stdin)
    environment["SG_CAPTURE_TEST_ARGV"] = str(captured_argv)
    environment["SG_CAPTURE_TEST_MODE"] = "record"
    payload = b'{"canonical":"payload"}'

    completed = subprocess.run(
        (
            str(command_processor),
            "/d",
            "/s",
            "/c",
            f'""{launcher}" ignored-provider-argv"',
        ),
        input=payload,
        capture_output=True,
        check=False,
        cwd=hostile,
        env=environment,
        timeout=8,
    )

    assert (completed.returncode, completed.stdout, completed.stderr) == (0, b"", b"")
    assert captured_stdin.read_bytes() == payload
    assert captured_argv.read_text(encoding="utf-8").splitlines() == [
        "--profile",
        PROFILE_VALUE,
        "--connection",
        CONNECTION_ID,
    ]

    captured_argv.unlink()
    environment["SG_CAPTURE_TEST_MODE"] = "sleep"
    started = time.monotonic()
    timed = subprocess.run(
        (str(command_processor), "/d", "/s", "/c", f'""{launcher}""'),
        input=b"{}",
        capture_output=True,
        check=False,
        cwd=hostile,
        env=environment,
        timeout=8,
    )
    elapsed = time.monotonic() - started

    assert (timed.returncode, timed.stdout, timed.stderr) == (0, b"", b"")
    assert captured_argv.read_text(encoding="utf-8") == "started"
    assert 1.5 <= elapsed < 6.0


def test_hook_source_has_no_network_model_runtime_or_environment_access() -> None:
    source = HOOK_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.partition(".")[0])

    assert imported_roots.isdisjoint(
        {"anthropic", "httpx", "openai", "openai_harmony", "requests", "socket", "urllib"}
    )
    assert "os.environ" not in source
    assert "os.getenv" not in source


def test_hook_source_import_does_not_load_capture_or_pydantic_runtime() -> None:
    source = HOOK_SOURCE.resolve()
    command = (
        sys.executable,
        "-I",
        "-c",
        "import importlib.util,sys;"
        f"spec=importlib.util.spec_from_file_location('_isolated_capture_hook',{str(source)!r});"
        "module=importlib.util.module_from_spec(spec);"
        "sys.modules[spec.name]=module;"
        "spec.loader.exec_module(module);"
        "heavy=[name for name in sys.modules "
        "if name=='pydantic' or name.startswith('saliencegate.capture')];"
        "raise SystemExit(bool(heavy))",
    )

    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
        timeout=5,
    )

    assert completed.returncode == 0
    assert completed.stdout == b""
    assert completed.stderr == b""


def test_packaged_hook_import_does_not_load_network_or_model_runtime() -> None:
    command = (
        sys.executable,
        "-I",
        "-c",
        "import sys;"
        "import saliencegate.integrations.hook;"
        "forbidden=('anthropic','httpx','openai','openai_harmony','pydantic',"
        "'requests','socket','urllib');"
        "loaded={name.partition('.')[0] for name in sys.modules};"
        "raise SystemExit(bool(loaded.intersection(forbidden)))",
    )

    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
        timeout=5,
    )

    assert completed.returncode == 0
    assert completed.stdout == b""
    assert completed.stderr == b""


def test_default_fail_open_path_does_not_load_capture_or_pydantic_runtime() -> None:
    command = (
        sys.executable,
        "-I",
        "-c",
        "import sys;from io import BytesIO;"
        "from saliencegate.integrations.hook import run_capture_hook;"
        "result=run_capture_hook("
        "('--profile','codex-hooks/v1','--connection','connection-one'),"
        "BytesIO(b'{}'));"
        "heavy=[name for name in sys.modules "
        "if name=='pydantic' or name.startswith('saliencegate.capture')];"
        "raise SystemExit(result != 0 or bool(heavy))",
    )

    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
        timeout=5,
    )

    assert completed.returncode == 0
    assert completed.stdout == b""
    assert completed.stderr == b""


def test_unknown_default_event_does_not_load_provider_or_capture_runtime() -> None:
    command = (
        sys.executable,
        "-I",
        "-c",
        "import sys;from io import BytesIO;"
        "from saliencegate.integrations.hook import run_capture_hook;"
        "result=run_capture_hook("
        "('--profile','codex-hooks/v1','--connection','connection-one'),"
        'BytesIO(b\'{"hook_event_name":"Unknown","session_id":"one",\''
        'b\'"cwd":"/provider-controlled"}\'));'
        "heavy=[name for name in sys.modules if name=='pydantic' "
        "or name.startswith('saliencegate.capture') "
        "or name=='saliencegate.integrations.codex'];"
        "raise SystemExit(result != 0 or bool(heavy))",
    )

    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
        timeout=5,
    )

    assert completed.returncode == 0
    assert completed.stdout == b""
    assert completed.stderr == b""
