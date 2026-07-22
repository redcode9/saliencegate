from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from saliencegate.capture.capabilities import (
    CaptureProfile,
    capture_capability_digest,
    capture_profile,
)
from saliencegate.capture.identities import CaptureDigestContext
from saliencegate.capture.migrations import initialize_capture_store
from saliencegate.capture.normalization import normalize_capture_session_snapshot
from saliencegate.capture.publication import authenticate_capture_intake
from saliencegate.capture.report import (
    build_capture_session_report,
    decode_capture_session_report,
    encode_capture_session_report,
)
from saliencegate.capture.schema import CaptureIntake, validate_capture_intake
from saliencegate.capture.sessions import verify_capture_session_snapshot
from saliencegate.capture.store import (
    MAX_CAPTURE_EVENTS_PER_SESSION,
    CaptureConnectionState,
    CaptureStore,
    CaptureStoreMode,
)
from saliencegate.domain import canonical_json
from saliencegate.security import InstallationKey

CAPTURE_REPORT_BENCHMARK_SCHEMA_VERSION: Final = "capture-report-benchmark/v1"
CAPTURE_REPORT_EVENT_COUNT: Final = 1_000
CAPTURE_REPORT_DURATION_BUDGET_MS: Final = 2_000.0
CAPTURE_REPORT_PEAK_RSS_BUDGET_BYTES: Final = 128 * 1_024 * 1_024
CAPTURE_REPORT_WORKER_TIMEOUT_SECONDS: Final = 60.0
_CONNECTION_ID: Final = "capture-report-benchmark"
_PROJECT_DIGEST: Final = "b" * 64
_SESSION_NATIVE: Final = b"capture-report-benchmark-session"
_KEY_MATERIAL: Final = hashlib.sha256(b"saliencegate:capture-report-benchmark:v1").digest()
_POISON_VALUE: Final = "provider-credential-read-must-fail"
_PROVIDER_CREDENTIAL_NAMES: Final = (
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_ORGANIZATION",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT",
    "OPENAI_PROJECT_ID",
)
_FORBIDDEN_RUNTIME_MODULE_ROOTS: Final = frozenset(
    {"anthropic", "httpx", "openai", "openai_harmony", "requests", "urllib3"}
)


class CaptureReportBenchmarkError(RuntimeError):
    """A content-free failure of the isolated report benchmark."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("capture report benchmark failed")


def _installation_key() -> InstallationKey:
    return InstallationKey(_KEY_MATERIAL)


def _benchmark_intake(
    kind: str,
    *,
    ordinal: int,
    context: CaptureDigestContext,
) -> CaptureIntake:
    profile = capture_profile(CaptureProfile.CODEX_HOOKS_V1)
    values: dict[str, object] = {
        "schema_version": "capture-intake/v1",
        "kind": kind,
        "adapter_profile": profile.profile_id.value,
        "capability_manifest_digest": capture_capability_digest(profile),
        "connection_id": _CONNECTION_ID,
        "session_id": context.session_id(_SESSION_NATIVE),
        "producer_event_digest": context.producer_event(f"report-event-{ordinal}".encode()),
        "intake_tag": "0" * 64,
        "occurred_at": None,
        "timestamp_authority": "unavailable",
        "producer_sequence": ordinal,
        "sequence_authority": "producer_exact",
        "capture_disposition": "captured",
    }
    if kind == "turn_finished":
        values["turn_id"] = context.turn_id(f"report-turn-{ordinal}".encode())
    elif kind not in {"session_started", "session_finished"}:
        raise CaptureReportBenchmarkError()
    return authenticate_capture_intake(validate_capture_intake(values), context=context)


def prepare_capture_report_fixture(database: Path, *, event_count: int) -> str:
    """Create a real authenticated store outside the measured worker phase."""

    if (
        not isinstance(database, Path)
        or not database.is_absolute()
        or type(event_count) is not int
        or not 3 <= event_count <= MAX_CAPTURE_EVENTS_PER_SESSION
    ):
        raise CaptureReportBenchmarkError()
    key = _installation_key()
    context = CaptureDigestContext(key)
    manifest = capture_profile(CaptureProfile.CODEX_HOOKS_V1)
    initialize_capture_store(database)
    with CaptureStore.open(
        database,
        installation_key=key,
        busy_timeout_ms=60_000,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        store.register_connection(
            connection_id=_CONNECTION_ID,
            project_digest=_PROJECT_DIGEST,
            profile_id=manifest.profile_id,
            capability_manifest_digest=capture_capability_digest(manifest),
            host_version=manifest.host_version,
        )
        store.transition_connection(
            _CONNECTION_ID,
            expected_state=CaptureConnectionState.PENDING,
            target_state=CaptureConnectionState.ENABLED,
        )
        for ordinal in range(1, event_count + 1):
            kind = (
                "session_started"
                if ordinal == 1
                else "session_finished"
                if ordinal == event_count
                else "turn_finished"
            )
            store.append(_benchmark_intake(kind, ordinal=ordinal, context=context))
    return context.session_id(_SESSION_NATIVE)


def _peak_rss_bytes() -> int:
    if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = (
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            )

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
        if not ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
            process,
            ctypes.byref(counters),
            counters.cb,
        ):
            raise CaptureReportBenchmarkError()
        return int(counters.PeakWorkingSetSize)

    import resource

    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if type(maximum) not in (int, float) or not math.isfinite(maximum) or maximum <= 0:
        raise CaptureReportBenchmarkError()
    multiplier = 1 if platform.system() == "Darwin" else 1_024
    return int(maximum * multiplier)


def _network_denial_is_active() -> bool:
    module = importlib.import_module("socket")
    try:
        module.socket(module.AF_INET, module.SOCK_STREAM)
    except Exception as error:
        return type(error).__name__ in {"SocketAccessError", "CaptureBenchmarkNetworkError"}
    return False


def run_capture_report_worker(
    database: Path,
    *,
    session_id: str,
    event_count: int,
) -> dict[str, object]:
    """Measure only snapshot, normalization, report construction, and encoding."""

    if (
        not isinstance(database, Path)
        or not database.is_absolute()
        or type(session_id) is not str
        or type(event_count) is not int
        or not 3 <= event_count <= MAX_CAPTURE_EVENTS_PER_SESSION
        or not _network_denial_is_active()
    ):
        raise CaptureReportBenchmarkError()
    key = _installation_key()
    with CaptureStore.open(
        database,
        installation_key=key,
        busy_timeout_ms=60_000,
        mode=CaptureStoreMode.MAINTENANCE,
    ) as store:
        started = time.perf_counter_ns()
        snapshot = verify_capture_session_snapshot(
            store.snapshot_session(_CONNECTION_ID, session_id),
            installation_key=key,
        )
        normalization = normalize_capture_session_snapshot(snapshot, installation_key=key)
        report = build_capture_session_report(
            snapshot,
            normalization,
            installation_key=key,
            spool=None,
        )
        encoded = encode_capture_session_report(report)
        duration_ms = max((time.perf_counter_ns() - started) / 1_000_000.0, 0.001)
    canonical_report_verified = (
        snapshot.event_count == event_count
        and tuple(item.receipt_ordinal for item in snapshot.events)
        == tuple(range(1, event_count + 1))
        and decode_capture_session_report(encoded) == report
        and canonical_json(report.model_dump(mode="json", warnings=False)) == encoded
    )
    peak_rss_bytes = _peak_rss_bytes()
    loaded_forbidden = sorted(
        root_name for root_name in _FORBIDDEN_RUNTIME_MODULE_ROOTS if root_name in sys.modules
    )
    passed = (
        duration_ms <= CAPTURE_REPORT_DURATION_BUDGET_MS
        and peak_rss_bytes <= CAPTURE_REPORT_PEAK_RSS_BUDGET_BYTES
        and canonical_report_verified
        and not loaded_forbidden
    )
    return {
        "schema_version": CAPTURE_REPORT_BENCHMARK_SCHEMA_VERSION,
        "protocol": {
            "fresh_worker_process": True,
            "event_count": event_count,
            "measured_operations": [
                "authenticated_snapshot",
                "normalization",
                "report_build",
                "canonical_encode",
            ],
            "fixture_preparation_excluded": True,
            "network_access": "socket_and_resolver_denied",
            "provider_credentials": "poisoned",
            "report_canonicality_verified": canonical_report_verified,
        },
        "budgets": {
            "duration_ms": CAPTURE_REPORT_DURATION_BUDGET_MS,
            "peak_rss_bytes": CAPTURE_REPORT_PEAK_RSS_BUDGET_BYTES,
        },
        "measurements": {
            "duration_ms": duration_ms,
            "peak_rss_bytes": peak_rss_bytes,
            "encoded_report_bytes": len(encoded),
        },
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "system": platform.system() or "unknown",
            "machine": platform.machine() or "unknown",
            "forbidden_runtime_modules_loaded": loaded_forbidden,
        },
        "passed": passed,
    }


def _environment_without_provider_credentials(source: Mapping[str, str]) -> dict[str, str]:
    """Copy ambient state without ever reading provider credential values."""

    if not isinstance(source, Mapping):
        raise CaptureReportBenchmarkError()
    environment: dict[str, str] = {}
    try:
        for key in source:
            if type(key) is not str:
                raise CaptureReportBenchmarkError()
            if key.upper() in _PROVIDER_CREDENTIAL_NAMES:
                continue
            value = source[key]
            if type(value) is not str:
                raise CaptureReportBenchmarkError()
            environment[key] = value
    except CaptureReportBenchmarkError:
        raise
    except Exception:
        raise CaptureReportBenchmarkError() from None
    return environment


def _worker_environment(root: Path) -> dict[str, str]:
    environment = _environment_without_provider_credentials(os.environ)
    home = root / "worker-home"
    home.mkdir(mode=0o700)
    environment.update(
        {
            "HOME": str(home),
            "LOCALAPPDATA": str(root / "worker-local-app-data"),
            "PYTHONHASHSEED": "0",
            "XDG_CONFIG_HOME": str(root / "worker-configuration"),
            "XDG_STATE_HOME": str(root / "worker-state"),
            **{name: _POISON_VALUE for name in _PROVIDER_CREDENTIAL_NAMES},
        }
    )
    # The worker boundary receives this owned dict; its seven provider values are
    # synthetic sentinels and no ambient credential value was accessed above.
    return environment


def _run_fresh_worker(
    root: Path,
    database: Path,
    *,
    session_id: str,
    event_count: int,
) -> dict[str, object]:
    repository = Path(__file__).resolve(strict=True).parent.parent
    guard = repository / "scripts" / "run_without_sockets.py"
    command = (
        sys.executable,
        str(guard),
        str(Path(__file__).resolve(strict=True)),
        "--worker",
        "--database",
        str(database),
        "--session-id",
        session_id,
        "--event-count",
        str(event_count),
    )
    try:
        completed = subprocess.run(
            command,
            input=b"",
            capture_output=True,
            check=False,
            timeout=CAPTURE_REPORT_WORKER_TIMEOUT_SECONDS,
            env=_worker_environment(root),
        )
        if (
            completed.returncode != 0
            or completed.stderr
            or not 1 <= len(completed.stdout) <= 1 * 1_024 * 1_024
            or _POISON_VALUE.encode() in completed.stdout
        ):
            raise CaptureReportBenchmarkError()
        decoded = json.loads(completed.stdout)
        if type(decoded) is not dict or canonical_json(decoded) + b"\n" != completed.stdout:
            raise CaptureReportBenchmarkError()
        return decoded
    except CaptureReportBenchmarkError:
        raise
    except Exception:
        raise CaptureReportBenchmarkError() from None


def run_capture_report_benchmark(root: Path) -> dict[str, object]:
    if not isinstance(root, Path) or not root.is_absolute() or not root.is_dir():
        raise CaptureReportBenchmarkError()
    database = root / "capture.sqlite3"
    session_id = prepare_capture_report_fixture(
        database,
        event_count=CAPTURE_REPORT_EVENT_COUNT,
    )
    return _run_fresh_worker(
        root,
        database,
        session_id=session_id,
        event_count=CAPTURE_REPORT_EVENT_COUNT,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark an authenticated 1,000-event capture report.",
        allow_abbrev=False,
    )
    parser.add_argument("--assert-budgets", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--database", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--session-id", help=argparse.SUPPRESS)
    parser.add_argument("--event-count", type=int, help=argparse.SUPPRESS)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    try:
        if parsed.worker:
            if (
                parsed.assert_budgets
                or parsed.database is None
                or parsed.session_id is None
                or parsed.event_count is None
            ):
                raise CaptureReportBenchmarkError()
            report = run_capture_report_worker(
                parsed.database.absolute(),
                session_id=parsed.session_id,
                event_count=parsed.event_count,
            )
        else:
            if any(
                value is not None
                for value in (parsed.database, parsed.session_id, parsed.event_count)
            ):
                raise CaptureReportBenchmarkError()
            with tempfile.TemporaryDirectory(
                prefix="saliencegate-capture-report-benchmark-"
            ) as raw:
                report = run_capture_report_benchmark(Path(raw).resolve(strict=True))
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return 2
    sys.stdout.write(canonical_json(report).decode("utf-8") + "\n")
    return 0 if not parsed.assert_budgets or report["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
