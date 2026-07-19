from __future__ import annotations

import ast
import hashlib
import json
import platform
import sys
from pathlib import Path

import pytest
from scripts import benchmark_shadow_trace as benchmark

from saliencegate.shadow import (
    ATIFProfile,
    ATIFShadowAdapter,
    ShadowEnvironmentBinding,
)
from saliencegate.shadow.trace import ATIFShadowDiagnostics

SCRIPT = Path("scripts/benchmark_shadow_trace.py")
FORBIDDEN_IMPORTS = frozenset(
    {
        "anthropic",
        "harbor",
        "httpx",
        "openai",
        "openai_harmony",
        "requests",
        "socket",
        "urllib",
    }
)


def _sample(
    backend: benchmark.BenchmarkBackend,
    *,
    duration: float,
    peak_rss: float,
    source_digest: str,
    mapped_record_count: int = benchmark.MAPPED_RECORD_COUNT,
) -> benchmark.BenchmarkSample:
    return benchmark.BenchmarkSample(
        backend=backend,
        duration_seconds=duration,
        peak_rss_mib=peak_rss,
        mapped_record_count=mapped_record_count,
        appended_event_count=mapped_record_count,
        batch_call_count=1,
        batch_mutation_count=1,
        source_sha256=source_digest,
        report_digest="a" * 64,
    )


def test_synthetic_atif_workload_deterministically_maps_exactly_1000_records() -> None:
    source = benchmark.build_synthetic_atif_source()

    assert source == benchmark.build_synthetic_atif_source()
    payload = json.loads(source)
    agent_step = payload["steps"][1]
    assert len(agent_step["tool_calls"]) == 499
    assert len(agent_step["observation"]["results"]) == 499

    adapter = ATIFShadowAdapter(
        profile=ATIFProfile.HARBOR_CODEX_V1,
        environment=ShadowEnvironmentBinding(
            default_working_directory="/synthetic/saliencegate-benchmark",
            environment_digest="e" * 64,
        ),
    )
    trace = adapter.adapt_bytes(
        source,
        run_id=benchmark._RUN_ID,
        task_scope_digest="1" * 64,
        lineage_scope_digest="2" * 64,
        capture_manifest_digest="3" * 64,
    )

    assert type(trace.diagnostics) is ATIFShadowDiagnostics
    assert trace.diagnostics.mapped_shadow_record_count == 1_000
    assert dict(trace.diagnostics.tool_call_disposition_counts)["mapped_action"] == 499
    assert dict(trace.diagnostics.result_disposition_counts)["mapped_structured_outcome"] == 499


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
def test_small_worker_path_performs_one_fresh_batch_mutation(
    backend: benchmark.BenchmarkBackend,
) -> None:
    sample = benchmark._measure_backend(backend, mapped_record_count=4)

    assert sample.backend == backend
    assert sample.mapped_record_count == 4
    assert sample.appended_event_count == 4
    assert sample.batch_call_count == 1
    assert sample.batch_mutation_count == 1
    assert sample.duration_seconds > 0
    assert sample.peak_rss_mib > 0


def test_protocol_uses_warmup_plus_five_isolated_measurements_and_fixed_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_digests = {
        record_count: hashlib.sha256(
            benchmark.build_synthetic_atif_source(record_count)
        ).hexdigest()
        for record_count in (
            benchmark.SCALING_BASELINE_RECORD_COUNT,
            benchmark.MAPPED_RECORD_COUNT,
        )
    }
    calls: list[tuple[benchmark.BenchmarkBackend, int]] = []

    def fake_isolated_sample(
        backend: benchmark.BenchmarkBackend,
        *,
        mapped_record_count: int,
    ) -> benchmark.BenchmarkSample:
        assert mapped_record_count in source_digests
        call = (backend, mapped_record_count)
        calls.append(call)
        ordinal = calls.count(call)
        if mapped_record_count == benchmark.SCALING_BASELINE_RECORD_COUNT:
            duration = 30.0 if ordinal == 1 else 20.0
            peak_rss = 900.0
        else:
            duration = (
                30.0 if ordinal == 1 else float(ordinal - 1 if backend == "memory" else ordinal + 8)
            )
            peak_rss = 500.0 if ordinal == 1 else 200.0
        return _sample(
            backend,
            duration=duration,
            peak_rss=peak_rss,
            source_digest=source_digests[mapped_record_count],
            mapped_record_count=mapped_record_count,
        )

    monkeypatch.setattr(benchmark, "_assert_provider_free", lambda: None)
    monkeypatch.setattr(benchmark, "_run_isolated_sample", fake_isolated_sample)
    monkeypatch.setenv("SALIENCEGATE_BENCHMARK_RUNNER_IMAGE", "ubuntu-24.04")

    report = benchmark.run_benchmark()

    assert calls == (
        [("memory", 250)] * 6
        + [("memory", 1_000)] * 6
        + [("sqlite", 250)] * 6
        + [("sqlite", 1_000)] * 6
    )
    assert report["schema_version"] == "shadow-trace-benchmark/v1"
    assert report["passed"] is True
    metadata = report["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["runner_image"] == "ubuntu-24.04"
    backends = report["backends"]
    assert isinstance(backends, dict)
    memory = backends["memory"]
    sqlite = backends["sqlite"]
    assert isinstance(memory, dict)
    assert isinstance(sqlite, dict)
    assert len(memory["measurements"]) == 5
    assert len(sqlite["measurements"]) == 5
    assert memory["median_seconds"] == 3.0
    assert sqlite["median_seconds"] == 12.0
    assert memory["median_budget_seconds"] == 5.0
    assert sqlite["median_budget_seconds"] == 15.0
    assert memory["maximum_peak_rss_mib"] == 500.0
    assert sqlite["maximum_peak_rss_mib"] == 500.0
    assert memory["scaling_ratio_250_to_1000"] == 3.0 / 20.0
    assert sqlite["scaling_ratio_250_to_1000"] == 12.0 / 20.0
    assert memory["scaling_ratio_gating"] is False
    assert sqlite["scaling_ratio_gating"] is False
    assert memory["scaling_baseline"]["median_seconds"] == 20.0
    assert sqlite["scaling_baseline"]["median_seconds"] == 20.0
    assert memory["scaling_baseline"]["maximum_peak_rss_mib"] == 900.0
    assert sqlite["scaling_baseline"]["maximum_peak_rss_mib"] == 900.0
    assert memory["scaling_baseline"]["gating"] is False
    assert sqlite["scaling_baseline"]["gating"] is False
    assert report["protocol"]["scaling_ratio_gating"] is False
    assert report["workload"]["scaling_baseline_mapped_record_count"] == 250


def test_workers_are_isolated_under_socket_guard_and_script_has_no_provider_imports() -> None:
    command = benchmark._worker_command("memory", mapped_record_count=1_000)

    assert command[:2] == (sys.executable, "-I")
    assert command[2].endswith("scripts/run_without_sockets.py")
    assert command[3].endswith("scripts/benchmark_shadow_trace.py")
    assert command[4:] == (
        "--worker",
        "memory",
        "--mapped-record-count",
        "1000",
    )

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.partition(".")[0])
    assert imported_roots.isdisjoint(FORBIDDEN_IMPORTS)


def test_linux_metadata_prefers_proc_cpu_model_and_assert_flag_controls_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor : 0\nmodel name : Contract CPU 9000\n", encoding="utf-8")
    monkeypatch.setattr(benchmark, "_LINUX_CPUINFO", cpuinfo)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    assert benchmark._cpu_model() == "Contract CPU 9000"

    monkeypatch.setattr(benchmark, "run_benchmark", lambda: {"passed": False})
    assert benchmark.main([]) == 0
    assert benchmark.main(["--assert-budgets"]) == 1
    output_lines = capsys.readouterr().out.splitlines()
    assert output_lines == ['{"passed":false}', '{"passed":false}']


def test_runner_metadata_prefers_concrete_hosted_identity_and_declares_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ImageOS", "ubuntu24")
    monkeypatch.setenv("ImageVersion", "20260717.1")
    monkeypatch.setenv("SALIENCEGATE_BENCHMARK_RUNNER_IMAGE", "ubuntu-24.04")

    hosted = benchmark._metadata()

    assert hosted["runner_image"] == "ubuntu24@20260717.1"
    assert hosted["runner_image_identity_source"] == "github_hosted_environment"
    assert hosted["runner_image_os"] == "ubuntu24"
    assert hosted["runner_image_version"] == "20260717.1"
    assert hosted["runner_image_declared_fallback"] == "ubuntu-24.04"

    monkeypatch.delenv("ImageOS")
    monkeypatch.delenv("ImageVersion")
    fallback = benchmark._metadata()

    assert fallback["runner_image"] == "ubuntu-24.04"
    assert fallback["runner_image_identity_source"] == "declared_fallback"
    assert fallback["runner_image_os"] is None
    assert fallback["runner_image_version"] is None
    assert fallback["runner_image_declared_fallback"] == "ubuntu-24.04"
