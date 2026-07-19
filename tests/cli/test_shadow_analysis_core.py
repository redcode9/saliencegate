from __future__ import annotations

import hashlib
from pathlib import Path
from typing import NoReturn
from uuid import UUID

import pytest

import saliencegate.commands.shadow as command_module
import saliencegate.repository.memory as memory_module
import saliencegate.shadow.analyzer as analyzer_module
import saliencegate.shadow.trace as trace_module
from saliencegate.domain import canonical_json
from saliencegate.security import InstallationKey
from saliencegate.shadow import ShadowSession
from saliencegate.shadow.io import PreflightedShadowTrace, decode_shadow_run_report
from saliencegate.shadow.report import ShadowRunReport

RUN_ID = UUID("b35f05f3-555b-4f09-8996-a7b3693bb54a")
ENVIRONMENT_DIGEST = "b" * 64
KEY = InstallationKey(b"g" * 32)


def _row(**values: object) -> bytes:
    return canonical_json(values) + b"\n"


def _small_trace() -> bytes:
    action = _row(
        schema_version="shadow-input/v1",
        kind="action",
        source_event_id="action-1",
        occurred_at="2026-07-16T10:01:00Z",
        argv=["pytest", "-q"],
        working_directory="/project",
        environment_digest=ENVIRONMENT_DIGEST,
    )
    return b"".join(
        (
            _row(
                schema_version="shadow-input/v1",
                kind="run_start",
                source_event_id="start",
                occurred_at="2026-07-16T10:00:00Z",
            ),
            action,
            action,
            _row(
                schema_version="shadow-input/v1",
                kind="tool_result",
                source_event_id="tool-1",
                occurred_at="2026-07-16T10:02:00Z",
                action_source_event_id="action-1",
                status="failed",
                exit_status=1,
                exception_type="AssertionError",
            ),
            _row(
                schema_version="shadow-input/v1",
                kind="run_end",
                source_event_id="finish",
                occurred_at="2026-07-16T10:03:00Z",
            ),
        )
    )


def _trace_with_1001_rows() -> bytes:
    start = _row(
        schema_version="shadow-input/v1",
        kind="run_start",
        source_event_id="start",
        occurred_at="2026-07-16T10:00:00Z",
    )
    action = _row(
        schema_version="shadow-input/v1",
        kind="action",
        source_event_id="action-1",
        occurred_at="2026-07-16T10:01:00Z",
        command="pytest -q",
        working_directory="/project",
        environment_digest=ENVIRONMENT_DIGEST,
    )
    return start + action * 1_000


def _private_trace(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    path.chmod(0o600)
    return path


def _forbid_shadow_trace_factory(*_args: object, **_kwargs: object) -> NoReturn:
    raise AssertionError("legacy NDJSON constructed ShadowTrace")


@pytest.mark.asyncio
async def test_ndjson_command_routes_through_legacy_analyzer_without_shadow_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_path = _private_trace(tmp_path / "events.ndjson", _small_trace())
    output = tmp_path / "report.json"
    real_analyze = analyzer_module._analyze_legacy_preflighted
    calls: list[PreflightedShadowTrace] = []

    async def recording_analyze(
        session: ShadowSession,
        trace: PreflightedShadowTrace,
    ) -> ShadowRunReport:
        calls.append(trace)
        return await real_analyze(session, trace)

    assert command_module._analyze_legacy_preflighted is real_analyze
    monkeypatch.setattr(command_module, "load_or_create_installation_key", lambda: KEY)
    monkeypatch.setattr(command_module, "_analyze_legacy_preflighted", recording_analyze)
    monkeypatch.setattr(trace_module, "_new_shadow_trace", _forbid_shadow_trace_factory)

    result = await command_module.run_shadow_analyze(
        trace_path,
        run_id=RUN_ID,
        output_path=output,
    )

    assert len(calls) == 1
    assert type(calls[0]) is PreflightedShadowTrace
    assert result.input_byte_digest == calls[0].input_byte_digest
    assert output.exists()


@pytest.mark.asyncio
async def test_ndjson_legacy_core_preserves_the_1001_row_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_path = _private_trace(tmp_path / "events.ndjson", _trace_with_1001_rows())
    output = tmp_path / "report.json"

    async def forbid_atomic_batch(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("legacy NDJSON used the public whole-trace batch")

    monkeypatch.setattr(command_module, "load_or_create_installation_key", lambda: KEY)
    monkeypatch.setattr(trace_module, "_new_shadow_trace", _forbid_shadow_trace_factory)
    monkeypatch.setattr(
        memory_module.MemoryRunRepository,
        "append_records_if_head",
        forbid_atomic_batch,
    )

    await command_module.run_shadow_analyze(
        trace_path,
        run_id=RUN_ID,
        output_path=output,
    )

    report = decode_shadow_run_report(output.read_bytes())
    assert report.input_row_count == 1_001
    assert report.unique_input_event_count == 2
    assert report.retry_row_count == 999
    assert len(report.rows) == 1_001
    assert report.rows[-1].input_ordinal == 1_001
    assert report.rows[-1].retry_target_ordinal == 2


@pytest.mark.asyncio
async def test_ndjson_report_bytes_have_a_frozen_deterministic_golden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_path = _private_trace(tmp_path / "events.ndjson", _small_trace())
    output = tmp_path / "report.json"
    monkeypatch.setattr(command_module, "load_or_create_installation_key", lambda: KEY)

    await command_module.run_shadow_analyze(
        trace_path,
        run_id=RUN_ID,
        output_path=output,
    )

    encoded = output.read_bytes()
    assert (len(encoded), hashlib.sha256(encoded).hexdigest()) == (
        22_228,
        "8b13301e1dda9e1f30e0807ed0d7f049edc45e5bd2d810c68947dc3342318482",
    )
