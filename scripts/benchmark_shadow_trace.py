from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal
from unittest.mock import patch
from uuid import UUID

from saliencegate.domain import canonical_json
from saliencegate.ports.repository import (
    ConditionalAppendOperation,
    ConditionalBatchReceipt,
    LedgerHead,
)
from saliencegate.security import InstallationKey
from saliencegate.shadow import (
    ATIFProfile,
    ATIFShadowAdapter,
    ShadowAnalyzer,
    ShadowEnvironmentBinding,
    ShadowSession,
    ShadowTraceReport,
)
from saliencegate.shadow.trace import ATIFShadowDiagnostics

BenchmarkBackend = Literal["memory", "sqlite"]

BENCHMARK_SCHEMA_VERSION: Final = "shadow-trace-benchmark/v1"
SAMPLE_SCHEMA_VERSION: Final = "shadow-trace-benchmark-sample/v1"
MAPPED_RECORD_COUNT: Final = 1_000
SCALING_BASELINE_RECORD_COUNT: Final = 250
MEASUREMENT_COUNT: Final = 5
MEMORY_MEDIAN_BUDGET_SECONDS: Final = 5.0
SQLITE_MEDIAN_BUDGET_SECONDS: Final = 15.0
PEAK_RSS_BUDGET_MIB: Final = 512.0

_RUNNER_IMAGE_ENVIRONMENT_KEY: Final = "SALIENCEGATE_BENCHMARK_RUNNER_IMAGE"
_GITHUB_IMAGE_OS_ENVIRONMENT_KEY: Final = "ImageOS"
_GITHUB_IMAGE_VERSION_ENVIRONMENT_KEY: Final = "ImageVersion"
_RUN_ID: Final = UUID("8a7e465f-186d-4c62-9d12-8d9afb474b6b")
_INSTALLATION_KEY_BYTES: Final = bytes.fromhex("42" * 32)
_WORKING_DIRECTORY: Final = "/synthetic/saliencegate-benchmark"
_ENVIRONMENT_DIGEST: Final = "e" * 64
_TASK_SCOPE_DIGEST: Final = "1" * 64
_LINEAGE_SCOPE_DIGEST: Final = "2" * 64
_CAPTURE_MANIFEST_DIGEST: Final = "3" * 64
_FORBIDDEN_PROVIDER_MODULES: Final = (
    "anthropic",
    "harbor",
    "httpx",
    "openai",
    "openai_harmony",
)
_BACKENDS: Final[tuple[BenchmarkBackend, ...]] = ("memory", "sqlite")
_MEDIAN_BUDGETS: Final[dict[BenchmarkBackend, float]] = {
    "memory": MEMORY_MEDIAN_BUDGET_SECONDS,
    "sqlite": SQLITE_MEDIAN_BUDGET_SECONDS,
}
_ROOT: Final = Path(__file__).resolve().parents[1]
_SOCKET_GUARD: Final = Path(__file__).resolve().with_name("run_without_sockets.py")
_LINUX_CPUINFO: Final = Path("/proc/cpuinfo")


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    """One isolated, fresh-process measurement."""

    backend: BenchmarkBackend
    duration_seconds: float
    peak_rss_mib: float
    mapped_record_count: int
    appended_event_count: int
    batch_call_count: int
    batch_mutation_count: int
    source_sha256: str
    report_digest: str

    def as_json(self) -> dict[str, object]:
        return {
            "schema_version": SAMPLE_SCHEMA_VERSION,
            "backend": self.backend,
            "duration_seconds": self.duration_seconds,
            "peak_rss_mib": self.peak_rss_mib,
            "mapped_record_count": self.mapped_record_count,
            "appended_event_count": self.appended_event_count,
            "batch_call_count": self.batch_call_count,
            "batch_mutation_count": self.batch_mutation_count,
            "source_sha256": self.source_sha256,
            "report_digest": self.report_digest,
        }


@dataclass(slots=True)
class _BatchProbe:
    call_count: int = 0
    mutation_count: int = 0


def _action_count(mapped_record_count: int) -> int:
    if (
        type(mapped_record_count) is not int
        or mapped_record_count < 4
        or mapped_record_count > MAPPED_RECORD_COUNT
        or mapped_record_count % 2 != 0
    ):
        raise ValueError("mapped record count must be an even integer from 4 through 1000")
    return (mapped_record_count - 2) // 2


def build_synthetic_atif_source(
    mapped_record_count: int = MAPPED_RECORD_COUNT,
) -> bytes:
    """Build the deterministic Codex ATIF workload without executing its commands."""

    action_count = _action_count(mapped_record_count)
    tool_calls: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    tool_call_details: dict[str, object] = {}
    for ordinal in range(1, action_count + 1):
        call_id = f"benchmark-call-{ordinal:04d}"
        tool_calls.append(
            {
                "arguments": {
                    "cmd": f"printf benchmark-record-{ordinal:04d}",
                    "login": False,
                    "sandbox_permissions": "workspace-write",
                    "shell": "/bin/sh",
                    "tty": False,
                    "workdir": _WORKING_DIRECTORY,
                    "yield_time_ms": 1_000,
                },
                "function_name": "exec_command",
                "tool_call_id": call_id,
            }
        )
        results.append(
            {
                "content": f"synthetic-output-{ordinal:04d}",
                "source_call_id": call_id,
            }
        )
        tool_call_details[call_id] = {
            "metadata": {"exit_code": 0},
            "status": "completed",
        }

    return canonical_json(
        {
            "agent": {
                "model_name": "provider-free-synthetic-model",
                "name": "codex",
                "version": "benchmark-v1",
            },
            "schema_version": "ATIF-v1.7",
            "session_id": "saliencegate-shadow-benchmark-v1",
            "steps": [
                {
                    "message": "Deterministic provider-free benchmark workload",
                    "source": "user",
                    "step_id": 1,
                },
                {
                    "extra": {"tool_call_details": tool_call_details},
                    "message": "Map and analyze every synthetic command",
                    "observation": {"results": results},
                    "source": "agent",
                    "step_id": 2,
                    "tool_calls": tool_calls,
                },
            ],
        }
    )


def _loaded_provider_modules() -> tuple[str, ...]:
    return tuple(
        provider
        for provider in _FORBIDDEN_PROVIDER_MODULES
        if any(name == provider or name.startswith(f"{provider}.") for name in sys.modules)
    )


def _assert_provider_free() -> None:
    loaded = _loaded_provider_modules()
    if loaded:
        raise RuntimeError(f"benchmark imported forbidden provider modules: {', '.join(loaded)}")


def _environment() -> ShadowEnvironmentBinding:
    return ShadowEnvironmentBinding(
        default_working_directory=_WORKING_DIRECTORY,
        environment_digest=_ENVIRONMENT_DIGEST,
    )


async def _analyze_fresh_trace(
    backend: BenchmarkBackend,
    source: bytes,
    *,
    sqlite_path: Path | None,
) -> tuple[ShadowTraceReport, _BatchProbe]:
    adapter = ATIFShadowAdapter(
        profile=ATIFProfile.HARBOR_CODEX_V1,
        environment=_environment(),
    )
    trace = adapter.adapt_bytes(
        source,
        run_id=_RUN_ID,
        task_scope_digest=_TASK_SCOPE_DIGEST,
        lineage_scope_digest=_LINEAGE_SCOPE_DIGEST,
        capture_manifest_digest=_CAPTURE_MANIFEST_DIGEST,
    )
    if type(trace.diagnostics) is not ATIFShadowDiagnostics:
        raise RuntimeError("benchmark adapter did not produce ATIF diagnostics")

    key = InstallationKey(_INSTALLATION_KEY_BYTES)
    if backend == "memory":
        session = ShadowSession.in_memory_for_trace(
            run_id=_RUN_ID,
            trace_binding=trace.binding,
            installation_key=key,
        )
    elif backend == "sqlite" and sqlite_path is not None:
        session = ShadowSession.sqlite_for_trace(
            sqlite_path,
            run_id=_RUN_ID,
            trace_binding=trace.binding,
            installation_key=key,
        )
    else:
        raise ValueError("benchmark backend configuration is invalid")

    original_append = ShadowSession._append_trace_batch_locked
    probe = _BatchProbe()

    async def counted_append(
        bound_session: ShadowSession,
        operations: tuple[ConditionalAppendOperation, ...],
        *,
        expected_head: LedgerHead | None,
    ) -> ConditionalBatchReceipt:
        probe.call_count += 1
        receipt = await original_append(
            bound_session,
            operations,
            expected_head=expected_head,
        )
        initial_count = 0 if receipt.initial_head is None else receipt.initial_head.entry_count
        if receipt.final_head.entry_count > initial_count:
            probe.mutation_count += 1
        return receipt

    with patch.object(ShadowSession, "_append_trace_batch_locked", new=counted_append):
        async with session:
            report = await ShadowAnalyzer(session).analyze(trace)
    return report, probe


def _peak_rss_mib() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return peak / divisor


def _measure_backend(
    backend: BenchmarkBackend,
    *,
    mapped_record_count: int,
) -> BenchmarkSample:
    source = build_synthetic_atif_source(mapped_record_count)
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    sqlite_path: Path | None = None
    if backend == "sqlite":
        temporary_directory = tempfile.TemporaryDirectory(prefix="saliencegate-shadow-benchmark-")
        private_parent = Path(temporary_directory.name).resolve(strict=True)
        sqlite_path = private_parent / "shadow.sqlite3"

    try:
        started_ns = time.perf_counter_ns()
        report, probe = asyncio.run(
            _analyze_fresh_trace(
                backend,
                source,
                sqlite_path=sqlite_path,
            )
        )
        elapsed_seconds = (time.perf_counter_ns() - started_ns) / 1_000_000_000.0
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()

    diagnostics = report.diagnostics
    if type(diagnostics) is not ATIFShadowDiagnostics:
        raise RuntimeError("benchmark report did not preserve ATIF diagnostics")
    shadow_report = report.shadow_report
    if (
        diagnostics.mapped_shadow_record_count != mapped_record_count
        or shadow_report.input_row_count != mapped_record_count
        or shadow_report.unique_input_event_count != mapped_record_count
        or shadow_report.appended_event_count != mapped_record_count
        or shadow_report.preexisting_event_count != 0
        or shadow_report.initial_ledger_entry_count != 0
        or any(row.persistence_disposition != "appended" for row in shadow_report.rows)
        or probe.call_count != 1
        or probe.mutation_count != 1
    ):
        raise RuntimeError("benchmark did not perform exactly one fresh batch mutation")

    return BenchmarkSample(
        backend=backend,
        duration_seconds=elapsed_seconds,
        peak_rss_mib=_peak_rss_mib(),
        mapped_record_count=diagnostics.mapped_shadow_record_count,
        appended_event_count=shadow_report.appended_event_count,
        batch_call_count=probe.call_count,
        batch_mutation_count=probe.mutation_count,
        source_sha256=hashlib.sha256(source).hexdigest(),
        report_digest=report.report_digest,
    )


def _parse_number(value: object, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("benchmark sample number is invalid")
    result = float(value)
    if not math.isfinite(result) or (result <= 0.0 if positive else result < 0.0):
        raise ValueError("benchmark sample number is invalid")
    return result


def _parse_digest(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("benchmark sample digest is invalid")
    return value


def _parse_sample(payload: object, *, expected_backend: BenchmarkBackend) -> BenchmarkSample:
    expected_keys = frozenset(
        BenchmarkSample(
            expected_backend,
            1.0,
            1.0,
            4,
            4,
            1,
            1,
            "0" * 64,
            "0" * 64,
        ).as_json()
    )
    if type(payload) is not dict or set(payload) != expected_keys:
        raise ValueError("benchmark sample shape is invalid")
    if payload["schema_version"] != SAMPLE_SCHEMA_VERSION or payload["backend"] != expected_backend:
        raise ValueError("benchmark sample identity is invalid")
    integer_fields = (
        "mapped_record_count",
        "appended_event_count",
        "batch_call_count",
        "batch_mutation_count",
    )
    if any(type(payload[field]) is not int for field in integer_fields):
        raise ValueError("benchmark sample count is invalid")
    return BenchmarkSample(
        backend=expected_backend,
        duration_seconds=_parse_number(payload["duration_seconds"], positive=True),
        peak_rss_mib=_parse_number(payload["peak_rss_mib"], positive=True),
        mapped_record_count=payload["mapped_record_count"],
        appended_event_count=payload["appended_event_count"],
        batch_call_count=payload["batch_call_count"],
        batch_mutation_count=payload["batch_mutation_count"],
        source_sha256=_parse_digest(payload["source_sha256"]),
        report_digest=_parse_digest(payload["report_digest"]),
    )


def _worker_command(backend: BenchmarkBackend, *, mapped_record_count: int) -> tuple[str, ...]:
    return (
        sys.executable,
        "-I",
        str(_SOCKET_GUARD),
        str(Path(__file__).resolve()),
        "--worker",
        backend,
        "--mapped-record-count",
        str(mapped_record_count),
    )


def _run_isolated_sample(
    backend: BenchmarkBackend,
    *,
    mapped_record_count: int,
) -> BenchmarkSample:
    completed = subprocess.run(
        _worker_command(backend, mapped_record_count=mapped_record_count),
        cwd=_ROOT,
        check=False,
        capture_output=True,
        timeout=120,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")[-4_000:]
        raise RuntimeError(f"isolated {backend} benchmark failed: {stderr}")
    try:
        payload = json.loads(completed.stdout)
        return _parse_sample(payload, expected_backend=backend)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError, TypeError):
        raise RuntimeError(f"isolated {backend} benchmark returned an invalid report") from None


def _physical_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    if type(page_size) is not int or type(page_count) is not int or min(page_size, page_count) < 1:
        return None
    return page_size * page_count


def _cpu_model() -> str:
    if platform.system() == "Linux":
        try:
            cpuinfo = _LINUX_CPUINFO.read_text(encoding="utf-8", errors="replace")
        except OSError:
            cpuinfo = ""
        fields: dict[str, str] = {}
        for line in cpuinfo.splitlines():
            key, separator, value = line.partition(":")
            if separator and value.strip():
                fields.setdefault(key.strip().lower(), value.strip())
        for field_name in ("model name", "hardware", "processor"):
            if model := fields.get(field_name):
                return model
    return platform.processor() or platform.machine() or "unknown"


def _metadata() -> dict[str, object]:
    logical_cores = os.cpu_count()
    memory_bytes = _physical_memory_bytes()
    image_os = os.environ.get(_GITHUB_IMAGE_OS_ENVIRONMENT_KEY)
    image_version = os.environ.get(_GITHUB_IMAGE_VERSION_ENVIRONMENT_KEY)
    declared_fallback = os.environ.get(_RUNNER_IMAGE_ENVIRONMENT_KEY, "unspecified")
    if image_os and image_version:
        runner_image = f"{image_os}@{image_version}"
        runner_image_identity_source = "github_hosted_environment"
    else:
        runner_image = declared_fallback
        runner_image_identity_source = (
            "declared_fallback" if declared_fallback != "unspecified" else "unavailable"
        )
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "cpu_model": _cpu_model(),
        "logical_core_count": logical_cores if logical_cores is not None else 0,
        "memory_capacity_bytes": memory_bytes,
        "memory_capacity_mib": (None if memory_bytes is None else memory_bytes / (1024.0 * 1024.0)),
        "platform": platform.platform(),
        "system": platform.system() or "unknown",
        "machine": platform.machine() or "unknown",
        "runner_image": runner_image,
        "runner_image_identity_source": runner_image_identity_source,
        "runner_image_os": image_os,
        "runner_image_version": image_version,
        "runner_image_declared_fallback": declared_fallback,
    }


def _summarize_backend(
    backend: BenchmarkBackend,
    warmup: BenchmarkSample,
    measurements: tuple[BenchmarkSample, ...],
    *,
    mapped_record_count: int,
) -> dict[str, object]:
    if len(measurements) != MEASUREMENT_COUNT:
        raise ValueError("benchmark measurement count is invalid")
    samples = (warmup, *measurements)
    if any(
        sample.backend != backend
        or sample.mapped_record_count != mapped_record_count
        or sample.appended_event_count != mapped_record_count
        or sample.batch_call_count != 1
        or sample.batch_mutation_count != 1
        for sample in samples
    ):
        raise RuntimeError("benchmark sample violated the workload contract")
    median_seconds = statistics.median(sample.duration_seconds for sample in measurements)
    maximum_peak_rss_mib = max(sample.peak_rss_mib for sample in samples)
    duration_passed = median_seconds <= _MEDIAN_BUDGETS[backend]
    rss_passed = maximum_peak_rss_mib <= PEAK_RSS_BUDGET_MIB
    return {
        "warmup": warmup.as_json(),
        "measurements": [sample.as_json() for sample in measurements],
        "median_seconds": median_seconds,
        "median_budget_seconds": _MEDIAN_BUDGETS[backend],
        "maximum_peak_rss_mib": maximum_peak_rss_mib,
        "peak_rss_budget_mib": PEAK_RSS_BUDGET_MIB,
        "duration_passed": duration_passed,
        "rss_passed": rss_passed,
        "passed": duration_passed and rss_passed,
    }


def _summarize_scaling_baseline(
    backend: BenchmarkBackend,
    warmup: BenchmarkSample,
    measurements: tuple[BenchmarkSample, ...],
) -> dict[str, object]:
    if len(measurements) != MEASUREMENT_COUNT:
        raise ValueError("benchmark measurement count is invalid")
    samples = (warmup, *measurements)
    if any(
        sample.backend != backend
        or sample.mapped_record_count != SCALING_BASELINE_RECORD_COUNT
        or sample.appended_event_count != SCALING_BASELINE_RECORD_COUNT
        or sample.batch_call_count != 1
        or sample.batch_mutation_count != 1
        for sample in samples
    ):
        raise RuntimeError("benchmark scaling sample violated the workload contract")
    return {
        "warmup": warmup.as_json(),
        "measurements": [sample.as_json() for sample in measurements],
        "median_seconds": statistics.median(sample.duration_seconds for sample in measurements),
        "maximum_peak_rss_mib": max(sample.peak_rss_mib for sample in samples),
        "gating": False,
    }


def run_benchmark() -> dict[str, object]:
    """Measure 250-to-1000 scaling and gate only maximum-size absolute budgets."""

    _assert_provider_free()
    expected_source_digests = {
        record_count: hashlib.sha256(build_synthetic_atif_source(record_count)).hexdigest()
        for record_count in (SCALING_BASELINE_RECORD_COUNT, MAPPED_RECORD_COUNT)
    }
    summaries: dict[str, object] = {}
    all_samples: list[BenchmarkSample] = []
    for backend in _BACKENDS:
        scaling_warmup = _run_isolated_sample(
            backend,
            mapped_record_count=SCALING_BASELINE_RECORD_COUNT,
        )
        scaling_measurements = tuple(
            _run_isolated_sample(
                backend,
                mapped_record_count=SCALING_BASELINE_RECORD_COUNT,
            )
            for _ in range(MEASUREMENT_COUNT)
        )
        maximum_warmup = _run_isolated_sample(backend, mapped_record_count=MAPPED_RECORD_COUNT)
        maximum_measurements = tuple(
            _run_isolated_sample(backend, mapped_record_count=MAPPED_RECORD_COUNT)
            for _ in range(MEASUREMENT_COUNT)
        )
        samples = (
            scaling_warmup,
            *scaling_measurements,
            maximum_warmup,
            *maximum_measurements,
        )
        all_samples.extend(samples)
        scaling = _summarize_scaling_baseline(
            backend,
            scaling_warmup,
            scaling_measurements,
        )
        maximum = _summarize_backend(
            backend,
            maximum_warmup,
            maximum_measurements,
            mapped_record_count=MAPPED_RECORD_COUNT,
        )
        scaling_median = scaling["median_seconds"]
        maximum_median = maximum["median_seconds"]
        if not isinstance(scaling_median, int | float) or not isinstance(
            maximum_median, int | float
        ):
            raise RuntimeError("benchmark scaling summary is invalid")
        summaries[backend] = {
            **maximum,
            "scaling_baseline": scaling,
            "scaling_ratio_250_to_1000": maximum_median / scaling_median,
            "scaling_ratio_gating": False,
        }

    if any(
        sample.source_sha256 != expected_source_digests.get(sample.mapped_record_count)
        for sample in all_samples
    ) or any(
        len(
            {
                sample.report_digest
                for sample in all_samples
                if sample.mapped_record_count == record_count
            }
        )
        != 1
        for record_count in expected_source_digests
    ):
        raise RuntimeError("isolated benchmark workload was not deterministic")
    _assert_provider_free()
    passed = all(bool(summaries[backend]["passed"]) for backend in _BACKENDS)  # type: ignore[index]
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "metadata": _metadata(),
        "protocol": {
            "warmup_processes_per_backend": 1,
            "measurement_processes_per_backend": MEASUREMENT_COUNT,
            "isolated_process_per_sample": True,
            "network_access": "socket_and_resolver_denied",
            "provider_modules_imported": [],
            "scaling_ratio_gating": False,
        },
        "workload": {
            "profile": ATIFProfile.HARBOR_CODEX_V1.value,
            "atif_step_count": 2,
            "mapped_record_count": MAPPED_RECORD_COUNT,
            "mapped_action_count": _action_count(MAPPED_RECORD_COUNT),
            "mapped_outcome_count": _action_count(MAPPED_RECORD_COUNT),
            "source_sha256": expected_source_digests[MAPPED_RECORD_COUNT],
            "scaling_baseline_mapped_record_count": SCALING_BASELINE_RECORD_COUNT,
            "scaling_baseline_source_sha256": expected_source_digests[
                SCALING_BASELINE_RECORD_COUNT
            ],
            "fresh_batch_mutation_count_per_sample": 1,
        },
        "budgets": {
            "memory_median_seconds": MEMORY_MEDIAN_BUDGET_SECONDS,
            "sqlite_median_seconds": SQLITE_MEDIAN_BUDGET_SECONDS,
            "peak_rss_mib": PEAK_RSS_BUDGET_MIB,
        },
        "backends": summaries,
        "passed": passed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark a deterministic 1000-record provider-free ATIF Shadow trace."
    )
    parser.add_argument(
        "--assert-budgets",
        action="store_true",
        help="exit non-zero when a median runtime or peak RSS budget is exceeded",
    )
    parser.add_argument("--worker", choices=_BACKENDS, help=argparse.SUPPRESS)
    parser.add_argument("--mapped-record-count", type=int, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    worker = arguments.worker
    mapped_record_count = arguments.mapped_record_count
    if worker is not None:
        if arguments.assert_budgets or mapped_record_count is None:
            raise SystemExit(2)
        _assert_provider_free()
        sample = _measure_backend(worker, mapped_record_count=mapped_record_count)
        _assert_provider_free()
        sys.stdout.buffer.write(canonical_json(sample.as_json()) + b"\n")
        return 0
    if mapped_record_count is not None:
        raise SystemExit(2)

    report = run_benchmark()
    sys.stdout.buffer.write(canonical_json(report) + b"\n")
    return 0 if not arguments.assert_budgets or report["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
