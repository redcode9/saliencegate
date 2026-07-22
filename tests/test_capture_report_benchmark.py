from __future__ import annotations

import ast
import json
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest
from scripts import benchmark_capture_report as benchmark

from saliencegate.domain import canonical_json

SCRIPT = Path("scripts/benchmark_capture_report.py")


def test_report_benchmark_locks_exact_workload_and_budgets() -> None:
    assert benchmark.CAPTURE_REPORT_EVENT_COUNT == 1_000
    assert benchmark.CAPTURE_REPORT_DURATION_BUDGET_MS == 2_000.0
    assert benchmark.CAPTURE_REPORT_PEAK_RSS_BUDGET_BYTES == 128 * 1_024 * 1_024
    assert benchmark.CAPTURE_REPORT_WORKER_TIMEOUT_SECONDS == 60.0


def test_report_worker_environment_never_reads_ambient_provider_credentials() -> None:
    provider_keys = tuple(benchmark._PROVIDER_CREDENTIAL_NAMES)
    mixed_case_keys = tuple(key.swapcase() for key in provider_keys)
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

    copied = benchmark._environment_without_provider_credentials(HostileEnvironment())

    assert copied == {"PATH": "/synthetic/bin"}
    assert observed == ["PATH"]


def test_fresh_socket_denied_worker_builds_a_real_canonical_report(tmp_path: Path) -> None:
    database = tmp_path / "capture.sqlite3"
    session_id = benchmark.prepare_capture_report_fixture(database, event_count=8)

    report = benchmark._run_fresh_worker(
        tmp_path,
        database,
        session_id=session_id,
        event_count=8,
    )

    assert report["schema_version"] == "capture-report-benchmark/v1"
    assert report["protocol"] == {
        "event_count": 8,
        "fixture_preparation_excluded": True,
        "fresh_worker_process": True,
        "measured_operations": [
            "authenticated_snapshot",
            "normalization",
            "report_build",
            "canonical_encode",
        ],
        "network_access": "socket_and_resolver_denied",
        "provider_credentials": "poisoned",
        "report_canonicality_verified": True,
    }
    assert report["budgets"] == {
        "duration_ms": 2_000.0,
        "peak_rss_bytes": 128 * 1_024 * 1_024,
    }
    measurements = report["measurements"]
    assert type(measurements) is dict
    assert 0.0 < measurements["duration_ms"] <= 2_000.0
    assert 0 < measurements["peak_rss_bytes"] <= 128 * 1_024 * 1_024
    assert measurements["encoded_report_bytes"] > 0
    assert report["runtime"]["forbidden_runtime_modules_loaded"] == []
    assert report["passed"] is True


def test_report_benchmark_main_emits_one_canonical_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = {"schema_version": "capture-report-benchmark/v1", "passed": False}
    monkeypatch.setattr(benchmark, "run_capture_report_benchmark", lambda _root: expected)

    assert benchmark.main([]) == 0
    assert benchmark.main(["--assert-budgets"]) == 1

    lines = capsys.readouterr().out.splitlines()
    assert [json.loads(line) for line in lines] == [expected, expected]
    assert all(line.encode() == canonical_json(expected) for line in lines)


def test_report_benchmark_has_no_provider_network_or_model_runtime_imports() -> None:
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
