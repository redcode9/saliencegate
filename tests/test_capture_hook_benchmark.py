from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import pytest
from scripts import benchmark_capture_hook as benchmark

SCRIPT = Path("scripts/benchmark_capture_hook.py")


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


def test_metadata_records_runner_os_and_filesystem_identity(tmp_path: Path) -> None:
    metadata = benchmark.runner_metadata(tmp_path)

    assert metadata["system"]
    assert metadata["platform"]
    assert metadata["machine"]
    assert metadata["python_version"]
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
