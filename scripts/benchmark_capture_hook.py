from __future__ import annotations

import argparse
import hashlib
import math
import os
import platform
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Event, Thread
from typing import BinaryIO, Final, Literal

from saliencegate.domain import canonical_json
from saliencegate.integrations.hook import read_capture_hook_document

BenchmarkPhase = Literal["cold", "warm"]
SampleRunner = Callable[[BenchmarkPhase, int], "CaptureHookBenchmarkSample"]
ColdPreparer = Callable[[int], None]

BENCHMARK_SCHEMA_VERSION: Final = "capture-hook-benchmark/v1"
COLD_MEASUREMENTS: Final = 30
WARM_MEASUREMENTS: Final = 200
COLD_P95_BUDGET_MS: Final = 750.0
WARM_P95_BUDGET_MS: Final = 250.0
WARM_P99_BUDGET_MS: Final = 500.0
OVERALL_MAX_BUDGET_MS: Final = 2_000.0
SUBPROCESS_TIMEOUT_SECONDS: Final = 3.0
COLD_REINITIALIZER_TIMEOUT_SECONDS: Final = 30.0


class CaptureHookBenchmarkError(RuntimeError):
    """A content-free benchmark protocol failure."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("capture hook benchmark failed")


@dataclass(frozen=True, slots=True)
class CaptureHookBenchmarkSample:
    """Content-free outcome of one provider-local launcher process."""

    duration_ms: float
    returncode: int | None
    stdout_empty: bool
    stderr_empty: bool
    timed_out: bool

    def __post_init__(self) -> None:
        if (
            type(self.duration_ms) is not float
            or not math.isfinite(self.duration_ms)
            or self.duration_ms <= 0.0
            or (self.returncode is not None and type(self.returncode) is not int)
            or type(self.stdout_empty) is not bool
            or type(self.stderr_empty) is not bool
            or type(self.timed_out) is not bool
        ):
            raise ValueError("capture hook benchmark sample is invalid")

    @property
    def passed(self) -> bool:
        return (
            not self.timed_out and self.returncode == 0 and self.stdout_empty and self.stderr_empty
        )

    def as_json(self) -> dict[str, object]:
        return {
            "duration_ms": self.duration_ms,
            "returncode": self.returncode,
            "stdout_empty": self.stdout_empty,
            "stderr_empty": self.stderr_empty,
            "timed_out": self.timed_out,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class _SilentProcessResult:
    returncode: int | None
    stdout_empty: bool
    stderr_empty: bool
    timed_out: bool


def _drain_process_output(stream: BinaryIO, observed: Event) -> None:
    try:
        while stream.read(64 * 1_024):
            observed.set()
    except (OSError, ValueError):
        return


def _write_process_input(stream: BinaryIO, payload: bytes) -> None:
    try:
        stream.write(payload)
        stream.flush()
    except (BrokenPipeError, OSError, ValueError):
        pass
    finally:
        with suppress(OSError, ValueError):
            stream.close()


def _run_silent_process(
    command: tuple[str, ...],
    *,
    input_data: bytes | None,
    timeout_seconds: float,
) -> _SilentProcessResult:
    """Run a child while draining and discarding output in fixed-size chunks."""

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE if input_data is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    if process.stdout is None or process.stderr is None:
        with suppress(OSError):
            process.kill()
        raise CaptureHookBenchmarkError()
    stdout_observed = Event()
    stderr_observed = Event()
    readers = (
        Thread(
            target=_drain_process_output,
            args=(process.stdout, stdout_observed),
            daemon=True,
        ),
        Thread(
            target=_drain_process_output,
            args=(process.stderr, stderr_observed),
            daemon=True,
        ),
    )
    for reader in readers:
        reader.start()
    writer: Thread | None = None
    if input_data is not None:
        if process.stdin is None:
            with suppress(OSError):
                process.kill()
            raise CaptureHookBenchmarkError()
        writer = Thread(
            target=_write_process_input,
            args=(process.stdin, input_data),
            daemon=True,
        )
        writer.start()

    timed_out = False
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        with suppress(OSError):
            process.kill()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=1.0)
    finally:
        if process.stdin is not None:
            with suppress(OSError, ValueError):
                process.stdin.close()
        if writer is not None:
            writer.join(timeout=1.0)
        for reader in readers:
            reader.join(timeout=1.0)
        for stream in (process.stdout, process.stderr):
            with suppress(OSError, ValueError):
                stream.close()
        for reader in readers:
            reader.join(timeout=1.0)
    if (writer is not None and writer.is_alive()) or any(reader.is_alive() for reader in readers):
        raise CaptureHookBenchmarkError()
    return _SilentProcessResult(
        returncode=process.returncode,
        stdout_empty=not stdout_observed.is_set(),
        stderr_empty=not stderr_observed.is_set(),
        timed_out=timed_out,
    )


def _linux_filesystem_type(path: Path) -> str | None:
    try:
        lines = (
            Path("/proc/self/mountinfo")
            .read_text(
                encoding="utf-8",
                errors="strict",
            )
            .splitlines()
        )
        resolved = str(path.resolve(strict=True))
        candidates: list[tuple[int, str]] = []
        for line in lines:
            left, separator, right = line.partition(" - ")
            left_fields = left.split()
            right_fields = right.split()
            if not separator or len(left_fields) < 5 or not right_fields:
                continue
            mount_point = (
                left_fields[4]
                .replace(r"\040", " ")
                .replace(r"\011", "\t")
                .replace(r"\012", "\n")
                .replace(r"\134", "\\")
            )
            if resolved == mount_point or resolved.startswith(mount_point.rstrip("/") + "/"):
                candidates.append((len(mount_point), right_fields[0]))
        return max(candidates)[1] if candidates else None
    except (OSError, ValueError):
        return None


def _darwin_filesystem_type(path: Path) -> str | None:
    try:
        completed = subprocess.run(
            ("/sbin/mount",),
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
        if completed.returncode != 0 or len(completed.stdout) > 1 * 1_024 * 1_024:
            return None
        resolved = str(path.resolve(strict=True))
        candidates: list[tuple[int, str]] = []
        for line in completed.stdout.splitlines():
            _source, separator, mounted = line.partition(" on ")
            mount_point, options_separator, options = mounted.rpartition(" (")
            if not separator or not options_separator or not options.endswith(")"):
                continue
            mount_point = mount_point.replace(r"\040", " ").replace(r"\134", "\\")
            filesystem_type = options[:-1].partition(",")[0].strip()
            if not filesystem_type:
                continue
            if resolved == mount_point or resolved.startswith(mount_point.rstrip("/") + "/"):
                candidates.append((len(mount_point), filesystem_type))
        return max(candidates)[1] if candidates else None
    except (OSError, subprocess.SubprocessError):
        return None


def _filesystem_type(path: Path) -> str:
    system = platform.system()
    if system == "Linux":
        return _linux_filesystem_type(path) or "unknown"
    if system == "Darwin":
        return _darwin_filesystem_type(path) or "unknown"
    return "unknown"


def _windows_filesystem_statistics(path: Path) -> dict[str, object]:
    """Read native volume identity and allocation size without POSIX APIs."""

    try:  # pragma: no cover - exercised by native Windows CI
        import ctypes

        resolved = path.resolve(strict=True)
        root = resolved.anchor
        if not root:
            raise OSError
        sectors_per_cluster = ctypes.c_ulong()
        bytes_per_sector = ctypes.c_ulong()
        free_clusters = ctypes.c_ulong()
        total_clusters = ctypes.c_ulong()
        if not ctypes.windll.kernel32.GetDiskFreeSpaceW(  # type: ignore[attr-defined]
            root,
            ctypes.byref(sectors_per_cluster),
            ctypes.byref(bytes_per_sector),
            ctypes.byref(free_clusters),
            ctypes.byref(total_clusters),
        ):
            raise OSError
        volume_serial = ctypes.c_ulong()
        filesystem_name = ctypes.create_unicode_buffer(256)
        if not ctypes.windll.kernel32.GetVolumeInformationW(  # type: ignore[attr-defined]
            root,
            None,
            0,
            ctypes.byref(volume_serial),
            None,
            None,
            filesystem_name,
            len(filesystem_name),
        ):
            raise OSError
        block_size = sectors_per_cluster.value * bytes_per_sector.value
        if block_size <= 0:
            raise OSError
        return {
            "type": filesystem_name.value or "unknown",
            "filesystem_id": int(volume_serial.value),
            "block_size": int(block_size),
        }
    except (AttributeError, OSError, TypeError, ValueError):
        raise CaptureHookBenchmarkError() from None


def runner_metadata(filesystem_path: Path) -> dict[str, object]:
    """Record reproducible runner and measured-filesystem identity."""

    if not isinstance(filesystem_path, Path) or not filesystem_path.is_absolute():
        raise ValueError("benchmark filesystem path must be absolute")
    file_stat = filesystem_path.stat()
    if platform.system() == "Windows":
        filesystem = _windows_filesystem_statistics(filesystem_path)
    else:
        filesystem_stat = os.statvfs(filesystem_path)
        filesystem = {
            "type": _filesystem_type(filesystem_path),
            "filesystem_id": getattr(filesystem_stat, "f_fsid", None),
            "block_size": filesystem_stat.f_bsize,
        }
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "system": platform.system() or "unknown",
        "release": platform.release() or "unknown",
        "platform": platform.platform() or "unknown",
        "machine": platform.machine() or "unknown",
        "runner_image": os.environ.get("SALIENCEGATE_BENCHMARK_RUNNER_IMAGE", "unspecified"),
        "filesystem": {
            "device": file_stat.st_dev,
            **filesystem,
        },
    }


def invoke_launcher(launcher: Path, payload: bytes) -> CaptureHookBenchmarkSample:
    """Measure one launcher without retaining or rendering provider-owned output."""

    if (
        not isinstance(launcher, Path)
        or not launcher.is_absolute()
        or launcher.is_symlink()
        or not launcher.is_file()
        or (os.name != "nt" and not os.access(launcher, os.X_OK))
        or type(payload) is not bytes
        or not payload
    ):
        raise ValueError("capture hook benchmark input is invalid")
    started = time.perf_counter_ns()
    try:
        completed = _run_silent_process(
            (str(launcher),),
            input_data=payload,
            timeout_seconds=SUBPROCESS_TIMEOUT_SECONDS,
        )
        duration_ms = max((time.perf_counter_ns() - started) / 1_000_000.0, 0.001)
        return CaptureHookBenchmarkSample(
            duration_ms=duration_ms,
            returncode=completed.returncode,
            stdout_empty=completed.stdout_empty,
            stderr_empty=completed.stderr_empty,
            timed_out=completed.timed_out,
        )
    except OSError:
        duration_ms = max((time.perf_counter_ns() - started) / 1_000_000.0, 0.001)
        return CaptureHookBenchmarkSample(
            duration_ms=duration_ms,
            returncode=None,
            stdout_empty=True,
            stderr_empty=True,
            timed_out=False,
        )


def _nearest_rank(values: tuple[float, ...], percentile: float) -> float:
    if (
        not values
        or type(percentile) is not float
        or not 0.0 < percentile <= 1.0
        or any(type(value) is not float or not math.isfinite(value) for value in values)
    ):
        raise ValueError("capture hook percentile input is invalid")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _phase_summary(
    phase: BenchmarkPhase,
    samples: tuple[CaptureHookBenchmarkSample, ...],
) -> dict[str, object]:
    expected = COLD_MEASUREMENTS if phase == "cold" else WARM_MEASUREMENTS
    if len(samples) != expected or any(
        type(sample) is not CaptureHookBenchmarkSample for sample in samples
    ):
        raise ValueError("capture hook benchmark phase is invalid")
    durations = tuple(sample.duration_ms for sample in samples)
    return {
        "count": len(samples),
        "measurements": [sample.as_json() for sample in samples],
        "p50_ms": _nearest_rank(durations, 0.50),
        "p95_ms": _nearest_rank(durations, 0.95),
        "p99_ms": _nearest_rank(durations, 0.99),
        "max_ms": max(durations),
        "failed_invocations": sum(not sample.passed for sample in samples),
        "timeouts": sum(sample.timed_out for sample in samples),
    }


def run_benchmark(
    sample_runner: SampleRunner,
    *,
    metadata_path: Path,
    cold_preparer: ColdPreparer,
) -> dict[str, object]:
    """Run the locked 30-cold/200-warm protocol and evaluate all budgets."""

    if not callable(sample_runner) or not callable(cold_preparer):
        raise ValueError("capture hook benchmark runner is invalid")
    cold_results: list[CaptureHookBenchmarkSample] = []
    for ordinal in range(1, COLD_MEASUREMENTS + 1):
        cold_preparer(ordinal)
        cold_results.append(sample_runner("cold", ordinal))
    cold_samples = tuple(cold_results)
    warm_samples = tuple(
        sample_runner("warm", ordinal) for ordinal in range(1, WARM_MEASUREMENTS + 1)
    )
    cold = _phase_summary("cold", cold_samples)
    warm = _phase_summary("warm", warm_samples)
    cold_p95_ms = cold["p95_ms"]
    warm_p95_ms = warm["p95_ms"]
    warm_p99_ms = warm["p99_ms"]
    cold_max_ms = cold["max_ms"]
    warm_max_ms = warm["max_ms"]
    cold_failures = cold["failed_invocations"]
    warm_failures = warm["failed_invocations"]
    if (
        type(cold_p95_ms) is not float
        or type(warm_p95_ms) is not float
        or type(warm_p99_ms) is not float
        or type(cold_max_ms) is not float
        or type(warm_max_ms) is not float
        or type(cold_failures) is not int
        or type(warm_failures) is not int
    ):
        raise RuntimeError("capture hook benchmark summary is invalid")
    cold_p95_passed = cold_p95_ms <= COLD_P95_BUDGET_MS
    warm_p95_passed = warm_p95_ms <= WARM_P95_BUDGET_MS
    warm_p99_passed = warm_p99_ms <= WARM_P99_BUDGET_MS
    overall_max_ms = max(cold_max_ms, warm_max_ms)
    overall_max_passed = overall_max_ms <= OVERALL_MAX_BUDGET_MS
    functional_passed = cold_failures == warm_failures == 0
    passed = all(
        (
            cold_p95_passed,
            warm_p95_passed,
            warm_p99_passed,
            overall_max_passed,
            functional_passed,
        )
    )
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "metadata": runner_metadata(metadata_path),
        "protocol": {
            "cold_measurements": COLD_MEASUREMENTS,
            "warm_measurements": WARM_MEASUREMENTS,
            "phase_order": ["cold", "warm"],
            "cold_preparer_invoked_before_each_sample": True,
        },
        "budgets_ms": {
            "cold_p95": COLD_P95_BUDGET_MS,
            "warm_p95": WARM_P95_BUDGET_MS,
            "warm_p99": WARM_P99_BUDGET_MS,
            "overall_max": OVERALL_MAX_BUDGET_MS,
        },
        "cold": cold,
        "warm": warm,
        "overall_max_ms": overall_max_ms,
        "budget_results": {
            "cold_p95_passed": cold_p95_passed,
            "warm_p95_passed": warm_p95_passed,
            "warm_p99_passed": warm_p99_passed,
            "overall_max_passed": overall_max_passed,
            "functional_contract_passed": functional_passed,
        },
        "passed": passed,
    }


def _artifact_evidence(path: Path, *, maximum_bytes: int) -> dict[str, object]:
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or path.is_symlink()
            or before.st_nlink != 1
            or not 1 <= before.st_size <= maximum_bytes
        ):
            raise CaptureHookBenchmarkError()
        with path.open("rb") as stream:
            data = stream.read(maximum_bytes + 1)
        after = path.lstat()
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_mode,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_mode,
        )
        if len(data) != before.st_size or identity_after != identity_before:
            raise CaptureHookBenchmarkError()
        return {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "device": before.st_dev,
            "file_id": before.st_ino,
        }
    except CaptureHookBenchmarkError:
        raise
    except (OSError, TypeError, ValueError):
        raise CaptureHookBenchmarkError() from None


def _external_cold_preparer(executable: Path) -> ColdPreparer:
    if (
        not isinstance(executable, Path)
        or not executable.is_absolute()
        or executable.is_symlink()
        or not executable.is_file()
        or (os.name != "nt" and not os.access(executable, os.X_OK))
    ):
        raise CaptureHookBenchmarkError()

    def prepare(_ordinal: int) -> None:
        try:
            completed = _run_silent_process(
                (str(executable),),
                input_data=None,
                timeout_seconds=COLD_REINITIALIZER_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError, CaptureHookBenchmarkError):
            raise CaptureHookBenchmarkError() from None
        if (
            completed.timed_out
            or completed.returncode != 0
            or not completed.stdout_empty
            or not completed.stderr_empty
        ):
            raise CaptureHookBenchmarkError()

    return prepare


def run_launcher_benchmark(
    launcher: Path,
    payload: bytes,
    *,
    cold_reinitializer: Path,
    payload_evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Run the protocol against one fully rendered provider-local launcher."""

    try:
        canonical_payload = read_capture_hook_document(BytesIO(payload))
    except Exception:
        raise CaptureHookBenchmarkError() from None
    canonical_payload_evidence: dict[str, object] = {
        "sha256": hashlib.sha256(canonical_payload).hexdigest(),
        "size_bytes": len(canonical_payload),
    }
    if payload_evidence is not None:
        if (
            not isinstance(payload_evidence, Mapping)
            or payload_evidence.get("sha256") != canonical_payload_evidence["sha256"]
            or payload_evidence.get("size_bytes") != canonical_payload_evidence["size_bytes"]
            or type(payload_evidence.get("device")) is not int
            or type(payload_evidence.get("file_id")) is not int
        ):
            raise CaptureHookBenchmarkError()
        canonical_payload_evidence.update(
            device=payload_evidence["device"],
            file_id=payload_evidence["file_id"],
        )
    launcher_before = _artifact_evidence(launcher, maximum_bytes=1 * 1_024 * 1_024)
    reinitializer_before = _artifact_evidence(
        cold_reinitializer,
        maximum_bytes=1 * 1_024 * 1_024,
    )
    report = run_benchmark(
        lambda _phase, _ordinal: invoke_launcher(launcher, canonical_payload),
        metadata_path=launcher.parent,
        cold_preparer=_external_cold_preparer(cold_reinitializer),
    )
    if launcher_before != _artifact_evidence(
        launcher,
        maximum_bytes=1 * 1_024 * 1_024,
    ) or reinitializer_before != _artifact_evidence(
        cold_reinitializer,
        maximum_bytes=1 * 1_024 * 1_024,
    ):
        raise CaptureHookBenchmarkError()
    protocol = report["protocol"]
    if type(protocol) is not dict:
        raise CaptureHookBenchmarkError()
    protocol.update(
        {
            "fresh_launcher_process_per_sample": True,
            "external_reinitializer_invoked_before_each_cold_sample": True,
            "cold_state_reinitialization_verified": False,
            "workload": "rendered_launcher_stdin_contract",
            "capture_admission_verified": False,
            "pass_scope": "launcher_process_contract_only",
        }
    )
    report["artifacts"] = {
        "launcher": launcher_before,
        "payload": canonical_payload_evidence,
        "cold_reinitializer": reinitializer_before,
    }
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark a rendered passive capture-hook launcher.",
        allow_abbrev=False,
    )
    parser.add_argument("--launcher", required=True, type=Path)
    parser.add_argument("--payload-file", required=True, type=Path)
    parser.add_argument("--cold-reinitializer", required=True, type=Path)
    parser.add_argument("--assert-budgets", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    launcher = parsed.launcher.absolute()
    payload_path = parsed.payload_file.absolute()
    cold_reinitializer = parsed.cold_reinitializer.absolute()
    try:
        payload_evidence = _artifact_evidence(
            payload_path,
            maximum_bytes=2 * 1_024 * 1_024,
        )
        with payload_path.open("rb") as stream:
            payload = stream.read(2 * 1_024 * 1_024 + 1)
        if (
            payload_evidence != _artifact_evidence(payload_path, maximum_bytes=2 * 1_024 * 1_024)
            or hashlib.sha256(payload).hexdigest() != payload_evidence["sha256"]
            or len(payload) != payload_evidence["size_bytes"]
        ):
            raise CaptureHookBenchmarkError()
        report = run_launcher_benchmark(
            launcher,
            payload,
            cold_reinitializer=cold_reinitializer,
            payload_evidence=payload_evidence,
        )
    except (CaptureHookBenchmarkError, OSError, TypeError, ValueError):
        return 2
    sys.stdout.write(canonical_json(report).decode("utf-8") + "\n")
    return 0 if not parsed.assert_budgets or report["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
