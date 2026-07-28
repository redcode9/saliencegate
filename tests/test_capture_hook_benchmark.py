from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import pytest
from scripts import benchmark_capture_hook as benchmark
from scripts import run_capture_hook_benchmark as registered

from saliencegate.domain import canonical_json

SCRIPT = Path("scripts/benchmark_capture_hook.py")
REGISTERED_SCRIPT = Path("scripts/run_capture_hook_benchmark.py")


class _PressureSpool:
    def __init__(self) -> None:
        self._counter_lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def _admit(self) -> str:
        with self._counter_lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.002)
        with self._counter_lock:
            self.active -= 1
        return "admitted"

    def admit(self, _store: object, _intake: object) -> str:
        return self._admit()

    def enqueue(self, _intake: object) -> str:
        return self._admit()

    def admit_transport(
        self,
        _store: object,
        _chunk: object,
        _intakes: tuple[object, ...],
        _fallback: tuple[object, ...],
    ) -> tuple[str, ...]:
        return (self._admit(),)


def test_shared_spool_proxy_serializes_every_admission_boundary() -> None:
    spool = _PressureSpool()
    proxy = registered._NonClosingSpoolProxy(spool)  # type: ignore[arg-type]
    barrier = threading.Barrier(24)

    def invoke(ordinal: int) -> object:
        barrier.wait(timeout=10.0)
        method = ordinal % 3
        if method == 0:
            return proxy.admit(object(), object())
        if method == 1:
            return proxy.enqueue(object())
        return proxy.admit_transport(object(), object(), (), ())

    with ThreadPoolExecutor(max_workers=24) as executor:
        results = tuple(executor.map(invoke, range(24)))

    assert results.count("admitted") == 16
    assert results.count(("admitted",)) == 8
    assert spool.active == 0
    assert spool.peak == 1


def _sample(
    duration_ms: float,
    *,
    returncode: int | None = 0,
    stdout_empty: bool = True,
    stderr_empty: bool = True,
    timed_out: bool = False,
) -> benchmark.CaptureHookBenchmarkSample:
    return benchmark.CaptureHookBenchmarkSample(
        duration_ms=duration_ms,
        returncode=returncode,
        stdout_empty=stdout_empty,
        stderr_empty=stderr_empty,
        timed_out=timed_out,
    )


def test_protocol_runs_exactly_30_cold_then_200_warm_fresh_process_samples(
    tmp_path: Path,
) -> None:
    calls: list[tuple[benchmark.BenchmarkPhase, int]] = []
    preparations: list[int] = []

    def sample_runner(
        phase: benchmark.BenchmarkPhase,
        ordinal: int,
    ) -> benchmark.CaptureHookBenchmarkSample:
        calls.append((phase, ordinal))
        return _sample(100.0 if phase == "cold" else 20.0)

    report = benchmark.run_benchmark(
        sample_runner,
        metadata_path=tmp_path,
        cold_preparer=preparations.append,
    )

    assert calls == [
        *(("cold", ordinal) for ordinal in range(1, 31)),
        *(("warm", ordinal) for ordinal in range(1, 201)),
    ]
    assert preparations == list(range(1, 31))
    assert report["schema_version"] == "capture-hook-benchmark/v1"
    assert report["protocol"] == {
        "cold_measurements": 30,
        "warm_measurements": 200,
        "phase_order": ["cold", "warm"],
        "cold_preparer_invoked_before_each_sample": True,
    }
    assert report["budgets_ms"] == {
        "cold_p95": 750.0,
        "warm_p95": 250.0,
        "warm_p99": 500.0,
        "overall_max": 2_000.0,
    }
    assert report["cold"]["count"] == 30
    assert report["warm"]["count"] == 200
    assert report["cold"]["p95_ms"] == 100.0
    assert report["warm"]["p99_ms"] == 20.0
    assert report["passed"] is True


def test_nearest_rank_percentiles_and_functional_failures_are_gating(tmp_path: Path) -> None:
    warm_durations = [float(value) for value in range(1, 201)]

    def sample_runner(
        phase: benchmark.BenchmarkPhase,
        ordinal: int,
    ) -> benchmark.CaptureHookBenchmarkSample:
        if phase == "cold":
            return _sample(float(ordinal))
        if ordinal == 200:
            return _sample(
                warm_durations[ordinal - 1],
                returncode=None,
                stdout_empty=False,
                timed_out=True,
            )
        return _sample(warm_durations[ordinal - 1])

    report = benchmark.run_benchmark(
        sample_runner,
        metadata_path=tmp_path,
        cold_preparer=lambda _ordinal: None,
    )

    assert report["cold"]["p95_ms"] == 29.0
    assert report["cold"]["p99_ms"] == 30.0
    assert report["warm"]["p95_ms"] == 190.0
    assert report["warm"]["p99_ms"] == 198.0
    assert report["warm"]["max_ms"] == 200.0
    assert report["warm"]["failed_invocations"] == 1
    assert report["warm"]["timeouts"] == 1
    assert report["passed"] is False


def test_metadata_records_runner_os_and_filesystem_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SALIENCEGATE_CAPTURE_BENCHMARK_RUNNER_IMAGE", "registered-runner")
    metadata = benchmark.runner_metadata(tmp_path)

    assert metadata["system"]
    assert metadata["platform"]
    assert metadata["machine"]
    assert metadata["python_version"]
    assert metadata["runner_image"] == "registered-runner"
    filesystem = metadata["filesystem"]
    assert isinstance(filesystem, dict)
    assert filesystem["device"] is not None
    assert filesystem["block_size"] is not None
    assert filesystem["type"]


def test_darwin_filesystem_type_uses_the_longest_matching_mount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Completed:
        returncode = 0
        stdout = "\n".join(
            (
                "/dev/disk1 on / (apfs, sealed, local)",
                f"/dev/disk2 on {tmp_path} (hfs, local)",
            )
        )

    monkeypatch.setattr(benchmark.subprocess, "run", lambda *_args, **_kwargs: _Completed())

    assert benchmark._darwin_filesystem_type(tmp_path) == "hfs"


def test_metadata_uses_native_windows_filesystem_statistics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        benchmark,
        "_windows_filesystem_statistics",
        lambda _path: {
            "type": "NTFS",
            "filesystem_id": 1234,
            "block_size": 4096,
        },
    )

    metadata = benchmark.runner_metadata(tmp_path)

    assert metadata["filesystem"] == {
        "device": tmp_path.stat().st_dev,
        "type": "NTFS",
        "filesystem_id": 1234,
        "block_size": 4096,
    }


def test_launcher_invocation_passes_only_fixed_launcher_path_and_redacts_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = tmp_path / "provider launcher"
    launcher.write_bytes(b"synthetic")
    launcher.chmod(0o700)
    calls: list[dict[str, object]] = []

    def fake_run(
        command: tuple[str, ...],
        **kwargs: object,
    ) -> benchmark._SilentProcessResult:
        calls.append({"command": command, **kwargs})
        return benchmark._SilentProcessResult(
            returncode=0,
            stdout_empty=True,
            stderr_empty=True,
            timed_out=False,
        )

    monkeypatch.setattr(benchmark, "_run_silent_process", fake_run)

    sample = benchmark.invoke_launcher(launcher, b"{}")

    assert sample.passed is True
    assert calls == [
        {
            "command": (str(launcher),),
            "input_data": b"{}",
            "timeout_seconds": benchmark.SUBPROCESS_TIMEOUT_SECONDS,
        }
    ]


def test_launcher_invocation_passes_an_exact_isolated_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = tmp_path / "provider-launcher"
    launcher.write_bytes(b"synthetic")
    launcher.chmod(0o700)
    calls: list[dict[str, object]] = []

    def fake_run(command: tuple[str, ...], **kwargs: object) -> benchmark._SilentProcessResult:
        calls.append({"command": command, **kwargs})
        return benchmark._SilentProcessResult(0, True, True, False)

    monkeypatch.setattr(benchmark, "_run_silent_process", fake_run)
    environment = {"HOME": str(tmp_path / "home"), "OPENAI_API_KEY": "poisoned"}

    sample = benchmark.invoke_launcher(launcher, b"{}", environment=environment)

    assert sample.passed is True
    assert calls == [
        {
            "command": (str(launcher),),
            "input_data": b"{}",
            "timeout_seconds": benchmark.SUBPROCESS_TIMEOUT_SECONDS,
            "environment": environment,
        }
    ]


def test_silent_process_discards_large_child_output_without_pipe_deadlock() -> None:
    result = benchmark._run_silent_process(
        (
            sys.executable,
            "-c",
            (
                "import sys;"
                "sys.stdout.buffer.write(b'x' * (4 * 1024 * 1024));"
                "sys.stderr.buffer.write(b'y' * (4 * 1024 * 1024));"
                "sys.stdin.buffer.read()"
            ),
        ),
        input_data=b"{}",
        timeout_seconds=5.0,
    )

    assert result == benchmark._SilentProcessResult(
        returncode=0,
        stdout_empty=False,
        stderr_empty=False,
        timed_out=False,
    )


def test_silent_process_waits_once_in_a_blocking_waiter_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_popen = subprocess.Popen
    waits: list[tuple[str, float | None]] = []

    class ObservedProcess:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.delegate = real_popen(*args, **kwargs)  # type: ignore[arg-type]
            self.stdin = self.delegate.stdin
            self.stdout = self.delegate.stdout
            self.stderr = self.delegate.stderr

        @property
        def returncode(self) -> int | None:
            return self.delegate.returncode

        def wait(self, timeout: float | None = None) -> int:
            waits.append((threading.current_thread().name, timeout))
            return self.delegate.wait(timeout=timeout)

        def kill(self) -> None:
            self.delegate.kill()

    monkeypatch.setattr(benchmark.subprocess, "Popen", ObservedProcess)

    result = benchmark._run_silent_process(
        (sys.executable, "-c", "pass"),
        input_data=None,
        timeout_seconds=5.0,
    )

    assert result == benchmark._SilentProcessResult(0, True, True, False)
    assert waits == [("capture-hook-process-waiter", None)]


def test_silent_process_timeout_kills_and_reaps_the_blocking_waiter() -> None:
    started = time.monotonic()
    result = benchmark._run_silent_process(
        (sys.executable, "-c", "import time; time.sleep(30)"),
        input_data=None,
        timeout_seconds=0.05,
    )

    assert time.monotonic() - started < 5.0
    assert result.timed_out is True
    assert result.returncode is not None
    assert result.stdout_empty is True
    assert result.stderr_empty is True
    assert not any(thread.name == "capture-hook-process-waiter" for thread in threading.enumerate())


def test_silent_process_wait_failure_kills_before_raising_content_free_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedWaitProcess:
        stdin = None
        stdout = BytesIO()
        stderr = BytesIO()
        returncode = None

        def __init__(self) -> None:
            self.killed = False

        def wait(self) -> int:
            raise OSError

        def kill(self) -> None:
            self.killed = True

    process = FailedWaitProcess()
    monkeypatch.setattr(benchmark.subprocess, "Popen", lambda *_args, **_kwargs: process)

    with pytest.raises(benchmark.CaptureHookBenchmarkError) as captured:
        benchmark._run_silent_process(
            ("synthetic-capture-hook",),
            input_data=None,
            timeout_seconds=1.0,
        )

    assert str(captured.value) == "capture hook benchmark failed"
    assert process.killed is True


def test_external_cold_reinitializer_is_silent_bounded_and_runs_before_every_cold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "cold-reinitializer"
    executable.write_bytes(b"executable")
    executable.chmod(0o700)
    calls: list[dict[str, object]] = []

    def fake_run(
        command: tuple[str, ...],
        **kwargs: object,
    ) -> benchmark._SilentProcessResult:
        calls.append({"command": command, **kwargs})
        return benchmark._SilentProcessResult(
            returncode=0,
            stdout_empty=True,
            stderr_empty=True,
            timed_out=False,
        )

    monkeypatch.setattr(benchmark, "_run_silent_process", fake_run)
    prepare = benchmark._external_cold_preparer(executable)

    prepare(1)

    assert calls == [
        {
            "command": (str(executable),),
            "input_data": None,
            "timeout_seconds": benchmark.COLD_REINITIALIZER_TIMEOUT_SECONDS,
        }
    ]


def test_launcher_report_states_the_measured_scope_and_artifact_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = tmp_path / "launcher"
    cold_reinitializer = tmp_path / "cold-reinitializer"
    launcher.write_bytes(b"launcher-v1")
    launcher.chmod(0o700)
    cold_reinitializer.write_bytes(b"reset-v1")
    cold_reinitializer.chmod(0o700)

    def fake_benchmark(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {"protocol": {}, "passed": True}

    monkeypatch.setattr(benchmark, "run_benchmark", fake_benchmark)
    report = benchmark.run_launcher_benchmark(
        launcher,
        b"{}",
        cold_reinitializer=cold_reinitializer,
    )

    assert report["protocol"] == {
        "fresh_launcher_process_per_sample": True,
        "external_reinitializer_invoked_before_each_cold_sample": True,
        "cold_state_reinitialization_verified": False,
        "workload": "rendered_launcher_stdin_contract",
        "capture_admission_verified": False,
        "pass_scope": "launcher_process_contract_only",
    }
    artifacts = report["artifacts"]
    assert isinstance(artifacts, dict)
    assert artifacts["payload"] == {
        "sha256": hashlib.sha256(b"{}").hexdigest(),
        "size_bytes": 2,
    }
    for name, data in (("launcher", b"launcher-v1"), ("cold_reinitializer", b"reset-v1")):
        evidence = artifacts[name]
        assert isinstance(evidence, dict)
        assert evidence["sha256"] == hashlib.sha256(data).hexdigest()
        assert evidence["size_bytes"] == len(data)
        assert evidence["device"] is not None
        assert evidence["file_id"] is not None


def test_cli_emits_one_canonical_report_and_assert_flag_controls_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    launcher = tmp_path / "launcher"
    payload = tmp_path / "payload.json"
    cold_reinitializer = tmp_path / "cold-reinitializer"
    launcher.write_bytes(b"launcher")
    launcher.chmod(0o700)
    payload.write_bytes(b"{}")
    cold_reinitializer.write_bytes(b"cold")
    cold_reinitializer.chmod(0o700)
    monkeypatch.setattr(
        benchmark,
        "run_launcher_benchmark",
        lambda *_args, **_kwargs: {"passed": False},
    )

    arguments = [
        "--launcher",
        str(launcher),
        "--payload-file",
        str(payload),
        "--cold-reinitializer",
        str(cold_reinitializer),
    ]
    assert benchmark.main(arguments) == 0
    assert benchmark.main([*arguments, "--assert-budgets"]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert [json.loads(line) for line in lines] == [{"passed": False}, {"passed": False}]
    assert lines == ['{"passed":false}', '{"passed":false}']


def test_cli_rejects_an_invalid_payload_without_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    launcher = tmp_path / "launcher"
    payload = tmp_path / "payload.json"
    cold_reinitializer = tmp_path / "cold-reinitializer"
    launcher.write_bytes(b"launcher")
    launcher.chmod(0o700)
    payload.write_bytes(b"{")
    cold_reinitializer.write_bytes(b"cold")
    cold_reinitializer.chmod(0o700)

    assert (
        benchmark.main(
            [
                "--launcher",
                str(launcher),
                "--payload-file",
                str(payload),
                "--cold-reinitializer",
                str(cold_reinitializer),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_benchmark_has_no_network_or_model_runtime_imports() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.partition(".")[0])

    assert imported_roots.isdisjoint(
        {"anthropic", "httpx", "openai", "openai_harmony", "requests", "socket", "urllib"}
    )


def test_registered_gate_locks_exact_workloads_and_provider_timeout() -> None:
    assert benchmark.COLD_MEASUREMENTS == 30
    assert benchmark.WARM_MEASUREMENTS == 200
    assert registered.CONCURRENT_HOOK_INVOCATIONS == 64
    assert registered.PROVIDER_TIMEOUT_BUDGET_MS == 2_000.0


def test_registered_concurrency_provider_timeout_is_gating() -> None:
    samples = tuple(
        (0, 2_000.001 if ordinal == 17 else 10.0)
        for ordinal in range(registered.CONCURRENT_HOOK_INVOCATIONS)
    )

    report = registered._concurrency_measurements(
        seed_returncode=0,
        seed_duration_ms=10.0,
        samples=samples,
    )

    assert report["max_ms"] == 2_000.001
    assert report["provider_timeout_budget_ms"] == 2_000.0
    assert report["provider_timeout_passed"] is False
    assert report["failed_invocations"] == 0
    assert report["passed"] is False


def test_registered_concurrency_seed_timeout_is_gating() -> None:
    report = registered._concurrency_measurements(
        seed_returncode=0,
        seed_duration_ms=2_000.001,
        samples=tuple((0, 10.0) for _ordinal in range(registered.CONCURRENT_HOOK_INVOCATIONS)),
    )

    assert report["max_ms"] == 10.0
    assert report["provider_timeout_budget_ms"] == 2_000.0
    assert report["provider_timeout_passed"] is False
    assert report["passed"] is False


def test_registered_gate_never_reads_ambient_provider_credentials(tmp_path: Path) -> None:
    provider_keys = tuple(registered._PROVIDER_CREDENTIAL_NAMES)
    mixed_case_keys = tuple(key.lower() for key in provider_keys)
    observed: list[str] = []

    class HostileEnvironment(Mapping[str, str]):
        def __iter__(self) -> Iterator[str]:
            return iter(("PATH", *mixed_case_keys))

        def __len__(self) -> int:
            return len(mixed_case_keys) + 1

        def __getitem__(self, key: str) -> str:
            observed.append(key)
            if key.upper() in provider_keys:
                raise AssertionError("provider credential value was read")
            if key == "PATH":
                return "/synthetic/bin"
            raise KeyError(key)

    copied = registered._environment_without_provider_credentials(HostileEnvironment())

    assert copied == {"PATH": "/synthetic/bin"}
    assert observed == ["PATH"]

    launcher = tmp_path / "launcher"
    launcher.write_bytes(b"synthetic")
    launcher.chmod(0o700)
    observed.clear()
    with pytest.raises(ValueError):
        benchmark.invoke_launcher(  # type: ignore[arg-type]
            launcher,
            b"{}",
            environment=HostileEnvironment(),
        )
    assert observed == []


def test_registered_gate_main_emits_one_canonical_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = {
        "schema_version": "registered-capture-hook-benchmark/v1",
        "passed": True,
    }
    monkeypatch.setattr(
        registered,
        "run_registered_capture_hook_benchmark",
        lambda _root: expected,
    )

    assert registered.main(["--assert-budgets"]) == 0

    output = capsys.readouterr()
    assert output.err == ""
    assert output.out.encode() == canonical_json(expected) + b"\n"


def test_registered_gate_admits_all_64_contention_events_in_a_fresh_process() -> None:
    program = """
import tempfile
from pathlib import Path

from scripts import benchmark_capture_hook as engine
from scripts import run_capture_hook_benchmark as gate
from saliencegate.domain import canonical_json

engine.COLD_MEASUREMENTS = 2
engine.WARM_MEASUREMENTS = 3
with tempfile.TemporaryDirectory(prefix="capture-hook-regression-") as raw:
    report = gate.run_registered_capture_hook_benchmark(Path(raw).resolve(strict=True))
summary = {
    "capture_admission_verified": report["protocol"]["capture_admission_verified"],
    "concurrency": report["concurrency"],
}
print(canonical_json(summary).decode("utf-8"))
"""

    completed = subprocess.run(
        (sys.executable, "-c", program),
        capture_output=True,
        check=False,
        env={
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("COV_CORE_", "COVERAGE_PROCESS_"))
        },
        timeout=60.0,
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    assert completed.stderr == b""
    summary = json.loads(completed.stdout)
    assert summary["capture_admission_verified"] is True
    assert summary["concurrency"]["invocations"] == 64
    assert summary["concurrency"]["concurrent_workers"] == 64
    assert summary["concurrency"]["distinct_payloads"] == 64
    assert summary["concurrency"]["mode"] == "in_process_hook_invocations"
    assert summary["concurrency"]["launcher_processes_started"] == 0
    assert summary["concurrency"]["simultaneous_barrier_release"] is True
    assert summary["concurrency"]["spool_process_local_fence"] is True
    assert summary["concurrency"]["authenticated_event_count_verified"] == 65
    assert summary["concurrency"]["capture_admission_verified"] is True
    assert summary["concurrency"]["failed_invocations"] == 0
    assert summary["concurrency"]["provider_timeout_budget_ms"] == 2_000.0
    assert summary["concurrency"]["provider_timeout_passed"] is True
    assert summary["concurrency"]["passed"] is True


def test_registered_gate_declares_real_admission_and_has_no_network_or_model_imports() -> None:
    source = REGISTERED_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.partition(".")[0])

    assert "run_connect(" in source
    assert '"capture_admission_verified": True' in source
    assert "verify_capture_session_snapshot(" in source
    assert "spool.drain(store)" in source
    assert imported_roots.isdisjoint(
        {"anthropic", "httpx", "openai", "openai_harmony", "requests", "socket", "urllib"}
    )
