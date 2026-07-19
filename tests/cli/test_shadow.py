from __future__ import annotations

import asyncio
import errno
import json
import os
import socket
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import ValidationError
from tests.cli.conftest import RunCli

import saliencegate.commands.shadow as shadow_module
import saliencegate.security.files as files_module
from saliencegate.commands.shadow import (
    ShadowCommandConfigurationError,
    ShadowCommandInputError,
    ShadowCommandIntegrityError,
    ShadowCommandReport,
    render_shadow_human,
    render_shadow_json,
    run_shadow_analyze,
)
from saliencegate.domain import SignalType, canonical_json
from saliencegate.memory.two_phase import PaperTwoPhaseCycleExecutor
from saliencegate.ports.repository import DigestVerificationError
from saliencegate.repository import MemoryRunRepository, SQLiteRunRepository
from saliencegate.runtime.budget import BudgetGovernor
from saliencegate.runtime.cycles import CycleCoordinator
from saliencegate.runtime.delivery import DeliveryWorker
from saliencegate.security import (
    InstallationKey,
    SecureFileBoundError,
    SecureFileError,
    SecureFileUnsupportedError,
)
from saliencegate.shadow import ShadowInvariantError, ShadowSession
from saliencegate.shadow.evaluation import ShadowHeuristicDisposition
from saliencegate.shadow.io import decode_shadow_run_report

RUN_ID = UUID("b35f05f3-555b-4f09-8996-a7b3693bb54a")
ENVIRONMENT_DIGEST = "b" * 64


def _row(**values: object) -> bytes:
    return canonical_json(values) + b"\n"


def _trace_bytes(*, retry: bool = False, finish: bool = True) -> bytes:
    rows = [
        _row(
            schema_version="shadow-input/v1",
            kind="run_start",
            source_event_id="start",
            occurred_at="2026-07-16T10:00:00Z",
        ),
        _row(
            schema_version="shadow-input/v1",
            kind="action",
            source_event_id="action-1",
            occurred_at="2026-07-16T10:01:00Z",
            argv=["pytest", "-q"],
            working_directory="/project",
            environment_digest=ENVIRONMENT_DIGEST,
        ),
    ]
    if retry:
        rows.append(rows[-1])
    rows.append(
        _row(
            schema_version="shadow-input/v1",
            kind="tool_result",
            source_event_id="tool-1",
            occurred_at="2026-07-16T10:02:00Z",
            action_source_event_id="action-1",
            status="failed",
            exit_status=1,
            exception_type="AssertionError",
        )
    )
    if finish:
        rows.append(
            _row(
                schema_version="shadow-input/v1",
                kind="run_end",
                source_event_id="finish",
                occurred_at="2026-07-16T10:03:00Z",
            )
        )
    return b"".join(rows)


def _private_trace(path: Path, *, retry: bool = False, finish: bool = True) -> Path:
    path.write_bytes(_trace_bytes(retry=retry, finish=finish))
    path.chmod(0o600)
    return path


def _repeated_failure_trace(path: Path) -> Path:
    path.write_bytes(
        b"".join(
            (
                _row(
                    schema_version="shadow-input/v1",
                    kind="run_start",
                    source_event_id="start",
                    occurred_at="2026-07-16T10:00:00Z",
                ),
                _row(
                    schema_version="shadow-input/v1",
                    kind="action",
                    source_event_id="action-1",
                    occurred_at="2026-07-16T10:01:00Z",
                    command="pytest -q",
                    working_directory="/project",
                    environment_digest=ENVIRONMENT_DIGEST,
                ),
                _row(
                    schema_version="shadow-input/v1",
                    kind="tool_result",
                    source_event_id="tool-1",
                    occurred_at="2026-07-16T10:02:00Z",
                    action_source_event_id="action-1",
                    status="failed",
                    exit_status=1,
                    failure_signature="same-failure",
                ),
                _row(
                    schema_version="shadow-input/v1",
                    kind="action",
                    source_event_id="action-2",
                    occurred_at="2026-07-16T10:03:00Z",
                    command="pytest -q",
                    working_directory="/project",
                    environment_digest=ENVIRONMENT_DIGEST,
                ),
                _row(
                    schema_version="shadow-input/v1",
                    kind="tool_result",
                    source_event_id="tool-2",
                    occurred_at="2026-07-16T10:04:00Z",
                    action_source_event_id="action-2",
                    status="failed",
                    exit_status=1,
                    failure_signature="same-failure",
                ),
                _row(
                    schema_version="shadow-input/v1",
                    kind="run_end",
                    source_event_id="finish",
                    occurred_at="2026-07-16T10:05:00Z",
                ),
            )
        )
    )
    path.chmod(0o600)
    return path


def _command(trace: Path, output: Path, *extra: str) -> tuple[str, ...]:
    return (
        "shadow",
        "analyze",
        str(trace),
        "--run-id",
        str(RUN_ID),
        "--output",
        str(output),
        *extra,
    )


def _strict_command_report() -> ShadowCommandReport:
    return ShadowCommandReport(
        run_id=RUN_ID,
        input_byte_digest="1" * 64,
        normalized_input_digest="2" * 64,
        detector_profile_digest="3" * 64,
        supported_signal_types=(
            SignalType.REPEATED_ACTION,
            SignalType.REPEATED_FAILURE,
            SignalType.TEST_FAILURE,
            SignalType.TOOL_ERROR,
        ),
        unsupported_signal_types=(
            SignalType.CONFLICT,
            SignalType.CONTEXT_SHIFT,
            SignalType.IRREVERSIBLE_ACTION,
            SignalType.STAGNATION,
            SignalType.STALE_CONSTRAINT,
        ),
        unique_input_event_count=1,
        retry_row_count=0,
        heuristic_disposition_counts=(
            (ShadowHeuristicDisposition.FLAGGED, 0),
            (ShadowHeuristicDisposition.INDETERMINATE, 0),
            (ShadowHeuristicDisposition.NOT_APPLICABLE, 1),
            (ShadowHeuristicDisposition.NOT_FLAGGED, 0),
        ),
        report_digest="4" * 64,
    )


def test_shadow_json_analyzes_a_private_trace_and_publishes_a_canonical_report(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    trace = _private_trace(tmp_path / "events.ndjson", retry=True)
    output = tmp_path / "shadow-report.json"

    completed = run_cli(*_command(trace, output, "--json"))

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert completed.stdout.endswith("\n")
    command_report = json.loads(completed.stdout)
    assert set(command_report) == {
        "budget_reservations",
        "calibrated",
        "calibration_eligible",
        "confirmatory",
        "counterfactual_effect_supported",
        "cycles_created",
        "decision_authority",
        "deliveries",
        "delivery_authorizations",
        "detector_profile_digest",
        "evidence_level",
        "execution_mode",
        "heuristic_disposition_counts",
        "input_byte_digest",
        "intervention_outcome_evidence",
        "intervention_outcomes",
        "interventions",
        "memory_revisions",
        "model_calls",
        "normalized_input_digest",
        "report_digest",
        "representativeness_supported",
        "retry_row_count",
        "run_id",
        "schema_version",
        "status",
        "supported_signal_types",
        "task_efficacy_supported",
        "task_outcome_evidence",
        "unique_input_event_count",
        "unsupported_signal_types",
    }
    assert command_report["schema_version"] == "shadow-command-report/v1"
    assert command_report["status"] == "ok"
    assert command_report["run_id"] == str(RUN_ID)
    assert command_report["unique_input_event_count"] == 4
    assert command_report["retry_row_count"] == 1
    assert command_report["supported_signal_types"] == [
        "repeated_action",
        "repeated_failure",
        "test_failure",
        "tool_error",
    ]
    assert command_report["unsupported_signal_types"] == [
        "conflict",
        "context_shift",
        "irreversible_action",
        "stagnation",
        "stale_constraint",
    ]
    assert command_report["heuristic_disposition_counts"] == [
        ["flagged", 1],
        ["indeterminate", 0],
        ["not_applicable", 2],
        ["not_flagged", 1],
    ]
    assert command_report["execution_mode"] == "shadow"
    assert command_report["evidence_level"] == "descriptive_observational"
    assert command_report["decision_authority"] is False
    assert command_report["model_calls"] == 0
    assert command_report["memory_revisions"] == 0

    encoded = output.read_bytes()
    report = decode_shadow_run_report(encoded)
    assert encoded == canonical_json(report)
    assert report.run_id == RUN_ID
    assert report.input_row_count == 5
    assert report.unique_input_event_count == 4
    assert report.retry_row_count == 1
    assert report.report_digest == command_report["report_digest"]
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_shadow_human_output_is_bounded_to_sanitized_descriptive_fields(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    trace = _private_trace(tmp_path / "fixture-secret-events.ndjson")
    output = tmp_path / "fixture-secret-report.json"

    completed = run_cli(*_command(trace, output))

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert completed.stdout.startswith("Shadow analysis complete\n")
    assert f"run: {RUN_ID}\n" in completed.stdout
    assert "evaluated events: 4\n" in completed.stdout
    assert "supported detectors: 4 of 9\n" in completed.stdout
    assert "evidence: descriptive observational; no decision authority\n" in completed.stdout
    assert "fixture-secret" not in completed.stdout
    assert "action-1" not in completed.stdout
    assert "pytest" not in completed.stdout


def test_shadow_replacement_is_explicit_bound_and_preserves_corrupt_data(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    trace = _private_trace(tmp_path / "events.ndjson")
    output = tmp_path / "report.json"

    first = run_cli(*_command(trace, output, "--json"))
    assert first.returncode == 0, first.stderr
    original = output.read_bytes()

    refused = run_cli(*_command(trace, output, "--json"))
    assert refused.returncode == 2
    assert refused.stdout == ""
    assert refused.stderr == "error: shadow input or output is invalid\n"
    assert output.read_bytes() == original

    replaced = run_cli(*_command(trace, output, "--replace", "--json"))
    assert replaced.returncode == 0, replaced.stderr
    assert output.read_bytes() == original

    output.write_bytes(b"fixture-secret unrelated output")
    output.chmod(0o600)
    corrupt = run_cli(*_command(trace, output, "--replace", "--json"))
    assert corrupt.returncode == 5
    assert corrupt.stdout == ""
    assert corrupt.stderr == "error: shadow report integrity check failed\n"
    assert output.read_bytes() == b"fixture-secret unrelated output"
    assert "fixture-secret" not in corrupt.stderr


def test_shadow_replacement_rejects_an_extended_input_and_preserves_the_report(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    trace = _private_trace(tmp_path / "events.ndjson", finish=False)
    output = tmp_path / "report.json"
    first = run_cli(*_command(trace, output, "--json"))
    assert first.returncode == 0, first.stderr
    original = output.read_bytes()

    trace.write_bytes(_trace_bytes(finish=True))
    trace.chmod(0o600)
    extended = run_cli(*_command(trace, output, "--replace", "--json"))

    assert extended.returncode == 5
    assert extended.stdout == ""
    assert extended.stderr == "error: shadow report integrity check failed\n"
    assert output.read_bytes() == original


def test_shadow_sqlite_backend_reuses_an_exact_authenticated_prefix(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    trace = _private_trace(tmp_path / "events.ndjson")
    repository = tmp_path / "shadow.sqlite3"
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"

    first = run_cli(*_command(trace, first_output, "--repository", str(repository), "--json"))
    second = run_cli(*_command(trace, second_output, "--repository", str(repository), "--json"))

    assert first.returncode == second.returncode == 0
    first_report = decode_shadow_run_report(first_output.read_bytes())
    second_report = decode_shadow_run_report(second_output.read_bytes())
    assert first_report.appended_event_count == 4
    assert first_report.preexisting_event_count == 0
    assert second_report.appended_event_count == 0
    assert second_report.preexisting_event_count == 4
    assert first_report.observations == second_report.observations

    conflicting_trace = tmp_path / "conflicting.ndjson"
    conflicting_trace.write_bytes(
        _trace_bytes()
        .replace(b'"source_event_id":"action-1"', b'"source_event_id":"action-2"')
        .replace(
            b'"action_source_event_id":"action-1"',
            b'"action_source_event_id":"action-2"',
        )
    )
    conflicting_trace.chmod(0o600)
    conflicting_output = tmp_path / "conflicting.json"
    conflict = run_cli(
        *_command(
            conflicting_trace,
            conflicting_output,
            "--repository",
            str(repository),
            "--json",
        )
    )
    assert conflict.returncode == 3
    assert conflict.stdout == ""
    assert conflict.stderr == "error: shadow configuration is invalid\n"
    assert not conflicting_output.exists()


@pytest.mark.parametrize("alias_kind", ("input", "database", "wal", "shm", "journal"))
def test_shadow_rejects_input_output_and_sqlite_sidecar_aliases_before_connect(
    tmp_path: Path,
    run_cli: RunCli,
    alias_kind: str,
) -> None:
    trace = _private_trace(tmp_path / "events.ndjson")
    repository = tmp_path / "shadow.sqlite3"
    aliases = {
        "input": trace,
        "database": repository,
        "wal": Path(f"{repository}-wal"),
        "shm": Path(f"{repository}-shm"),
        "journal": Path(f"{repository}-journal"),
    }
    output = aliases[alias_kind]
    before = trace.read_bytes()

    completed = run_cli(*_command(trace, output, "--repository", str(repository), "--json"))

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == "error: shadow input or output is invalid\n"
    assert trace.read_bytes() == before
    assert not repository.exists()


def test_shadow_complete_capture_requires_a_terminal_row_and_never_publishes(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    trace = _private_trace(tmp_path / "events.ndjson", finish=False)
    output = tmp_path / "report.json"

    completed = run_cli(
        *_command(
            trace,
            output,
            "--capture-scope",
            "complete_run_declared",
            "--json",
        )
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == "error: shadow input or output is invalid\n"
    assert not output.exists()


def test_shadow_command_report_is_strict_and_renderers_are_canonical() -> None:
    report = _strict_command_report()

    rendered = render_shadow_json(report)

    assert rendered == canonical_json(report).decode("utf-8") + "\n"
    assert json.loads(rendered)["schema_version"] == "shadow-command-report/v1"
    assert render_shadow_human(report).endswith(
        "evidence: descriptive observational; no decision authority\n"
    )
    with pytest.raises(ValidationError):
        ShadowCommandReport.model_validate(
            {**report.model_dump(mode="python"), "path": "/fixture-secret"}
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", 1),
        ("input_byte_digest", "not-a-digest"),
        ("supported_signal_types", [SignalType.REPEATED_ACTION]),
        ("heuristic_disposition_counts", []),
        ("heuristic_disposition_counts", ((ShadowHeuristicDisposition.FLAGGED, True),)),
        (
            "supported_signal_types",
            (
                SignalType.REPEATED_FAILURE,
                SignalType.REPEATED_ACTION,
                SignalType.TEST_FAILURE,
                SignalType.TOOL_ERROR,
            ),
        ),
        (
            "unsupported_signal_types",
            (
                SignalType.CONTEXT_SHIFT,
                SignalType.CONFLICT,
                SignalType.IRREVERSIBLE_ACTION,
                SignalType.STAGNATION,
                SignalType.STALE_CONSTRAINT,
            ),
        ),
        (
            "heuristic_disposition_counts",
            (
                (ShadowHeuristicDisposition.INDETERMINATE, 0),
                (ShadowHeuristicDisposition.FLAGGED, 0),
                (ShadowHeuristicDisposition.NOT_APPLICABLE, 1),
                (ShadowHeuristicDisposition.NOT_FLAGGED, 0),
            ),
        ),
        (
            "heuristic_disposition_counts",
            (
                (ShadowHeuristicDisposition.FLAGGED, 0),
                (ShadowHeuristicDisposition.INDETERMINATE, 0),
                (ShadowHeuristicDisposition.NOT_APPLICABLE, 0),
                (ShadowHeuristicDisposition.NOT_FLAGGED, 0),
            ),
        ),
    ),
)
def test_shadow_command_report_rejects_noncanonical_fields(field: str, value: object) -> None:
    report = _strict_command_report()
    fields = report.model_dump(mode="python")
    fields[field] = value

    with pytest.raises(ValidationError):
        ShadowCommandReport.model_validate(fields)


def test_shadow_renderers_reject_non_report_values() -> None:
    with pytest.raises(ShadowInvariantError):
        render_shadow_json(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "error_type",
    (
        ShadowCommandInputError,
        ShadowCommandConfigurationError,
        ShadowCommandIntegrityError,
    ),
)
def test_shadow_command_errors_are_value_free(error_type: type[Exception]) -> None:
    error = error_type()

    assert "fixture-secret" not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert vars(error) == {}


def test_shadow_command_module_has_no_provider_or_active_runtime_dependencies() -> None:
    source = Path(shadow_module.__file__).read_text(encoding="utf-8")
    forbidden = (
        "saliencegate.models",
        "saliencegate.memory",
        "saliencegate.runtime.cycles",
        "saliencegate.runtime.delivery",
        "saliencegate.runtime.engine",
        "saliencegate.commands.pilot",
        "httpx",
        "openai_harmony",
    )

    assert all(name not in source for name in forbidden)
    secret = os.environ.get("OPENAI_API_KEY")
    assert secret is None or secret not in source


@pytest.mark.asyncio
async def test_shadow_invalid_installation_key_configuration_is_sanitized_and_publishes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _private_trace(tmp_path / "events.ndjson")
    output = tmp_path / "report.json"

    def fail_key() -> InstallationKey:
        raise ValueError("fixture-secret-key-configuration")

    monkeypatch.setattr(shadow_module, "load_or_create_installation_key", fail_key)

    with pytest.raises(ShadowCommandConfigurationError) as captured:
        await run_shadow_analyze(trace, run_id=RUN_ID, output_path=output)

    assert str(captured.value) == "shadow configuration is invalid"
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert "fixture-secret" not in repr(captured.value)
    assert not output.exists()


def test_shadow_rejects_a_hardlinked_input_output_without_mutation(
    tmp_path: Path,
    run_cli: RunCli,
) -> None:
    trace = _private_trace(tmp_path / "events.ndjson")
    output = tmp_path / "report.json"
    os.link(trace, output)
    before = trace.read_bytes()

    completed = run_cli(*_command(trace, output, "--json"))

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == "error: shadow input or output is invalid\n"
    assert trace.read_bytes() == output.read_bytes() == before
    assert trace.stat().st_ino == output.stat().st_ino


@pytest.mark.asyncio
async def test_shadow_cancel_after_committed_event_leaves_recoverable_sqlite_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _private_trace(tmp_path / "events.ndjson")
    repository = tmp_path / "shadow.sqlite3"
    output = tmp_path / "report.json"
    key = InstallationKey(b"e" * 32)
    monkeypatch.setattr(shadow_module, "load_or_create_installation_key", lambda: key)
    original = SQLiteRunRepository.append_event_if_head
    cancelled = False

    async def commit_then_cancel(
        self: SQLiteRunRepository,
        *args: object,
        **kwargs: object,
    ) -> object:
        nonlocal cancelled
        receipt = await original(self, *args, **kwargs)  # type: ignore[arg-type]
        if not cancelled and receipt.event.source_event_id == "tool-1":
            cancelled = True
            raise asyncio.CancelledError
        return receipt

    monkeypatch.setattr(SQLiteRunRepository, "append_event_if_head", commit_then_cancel)

    with pytest.raises(asyncio.CancelledError):
        await run_shadow_analyze(
            trace,
            run_id=RUN_ID,
            output_path=output,
            repository_path=repository,
        )

    assert cancelled is True
    assert not output.exists()
    monkeypatch.setattr(SQLiteRunRepository, "append_event_if_head", original)

    recovered = await run_shadow_analyze(
        trace,
        run_id=RUN_ID,
        output_path=output,
        repository_path=repository,
    )
    report = decode_shadow_run_report(output.read_bytes())
    assert recovered.report_digest == report.report_digest
    assert report.preexisting_event_count == 3
    assert report.appended_event_count == 1


@pytest.mark.parametrize("fault_after", (1, 2, 3, 4))
@pytest.mark.asyncio
async def test_shadow_cancel_after_each_committed_signal_recovers_without_duplicate_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_after: int,
) -> None:
    trace = _repeated_failure_trace(tmp_path / "events.ndjson")
    repository = tmp_path / "shadow.sqlite3"
    output = tmp_path / "report.json"
    key = InstallationKey(bytes([fault_after]) * 32)
    monkeypatch.setattr(shadow_module, "load_or_create_installation_key", lambda: key)
    original = SQLiteRunRepository.record_signal_if_head
    committed = 0

    async def commit_then_cancel(
        self: SQLiteRunRepository,
        *args: object,
        **kwargs: object,
    ) -> object:
        nonlocal committed
        receipt = await original(self, *args, **kwargs)  # type: ignore[arg-type]
        committed += 1
        if committed == fault_after:
            raise asyncio.CancelledError
        return receipt

    monkeypatch.setattr(SQLiteRunRepository, "record_signal_if_head", commit_then_cancel)

    with pytest.raises(asyncio.CancelledError):
        await run_shadow_analyze(
            trace,
            run_id=RUN_ID,
            output_path=output,
            repository_path=repository,
        )

    assert committed == fault_after
    assert not output.exists()
    monkeypatch.setattr(SQLiteRunRepository, "record_signal_if_head", original)

    recovered = await run_shadow_analyze(
        trace,
        run_id=RUN_ID,
        output_path=output,
        repository_path=repository,
    )
    report = decode_shadow_run_report(output.read_bytes())
    assert recovered.unique_input_event_count == 6
    assert report.unique_input_event_count == 6
    assert report.observation_count == 6
    assert sum(count for _disposition, count in report.heuristic_disposition_counts) == 6


@pytest.mark.parametrize("suffix", ("", "-wal", "-shm", "-journal"))
@pytest.mark.asyncio
async def test_shadow_rejects_sqlite_slot_changes_between_preflight_and_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    trace = _private_trace(tmp_path / "events.ndjson")
    repository = tmp_path / "shadow.sqlite3"
    output = tmp_path / "report.json"
    raced = Path(f"{repository}{suffix}")
    original = shadow_module._authorize_output

    def race_after_output_authorization(*args: object, **kwargs: object) -> object:
        publication = original(*args, **kwargs)  # type: ignore[arg-type]
        raced.write_bytes(b"raced-private-slot")
        raced.chmod(0o600)
        return publication

    monkeypatch.setattr(shadow_module, "_authorize_output", race_after_output_authorization)
    monkeypatch.setattr(
        shadow_module,
        "load_or_create_installation_key",
        lambda: InstallationKey(b"r" * 32),
    )

    with pytest.raises(ShadowCommandInputError):
        await run_shadow_analyze(
            trace,
            run_id=RUN_ID,
            output_path=output,
            repository_path=repository,
        )

    assert raced.read_bytes() == b"raced-private-slot"
    assert not output.exists()


@pytest.mark.parametrize("mutated", ("input", "output"))
@pytest.mark.asyncio
async def test_shadow_revalidates_input_and_output_after_session_close_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutated: str,
) -> None:
    trace = _private_trace(tmp_path / "events.ndjson")
    output = tmp_path / "report.json"
    original_close = ShadowSession.aclose

    async def close_then_mutate(self: ShadowSession) -> None:
        await original_close(self)
        target = trace if mutated == "input" else output
        target.write_bytes(b"mutated-after-session-close")
        target.chmod(0o600)

    monkeypatch.setattr(ShadowSession, "aclose", close_then_mutate)
    monkeypatch.setattr(
        shadow_module,
        "load_or_create_installation_key",
        lambda: InstallationKey(b"m" * 32),
    )

    with pytest.raises(ShadowCommandInputError):
        await run_shadow_analyze(trace, run_id=RUN_ID, output_path=output)

    if mutated == "input":
        assert trace.read_bytes() == b"mutated-after-session-close"
        assert not output.exists()
    else:
        assert output.read_bytes() == b"mutated-after-session-close"


@pytest.mark.parametrize(
    ("failure", "expected"),
    (
        (SecureFileUnsupportedError(), ShadowCommandConfigurationError),
        (SecureFileError(), ShadowCommandInputError),
    ),
)
@pytest.mark.asyncio
async def test_shadow_classifies_publication_platform_and_namespace_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    expected: type[Exception],
) -> None:
    trace = _private_trace(tmp_path / "events.ndjson")
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        shadow_module,
        "load_or_create_installation_key",
        lambda: InstallationKey(b"p" * 32),
    )

    def fail_publish(*_args: object, **_kwargs: object) -> object:
        raise failure

    monkeypatch.setattr(shadow_module.AtomicFilePublication, "publish", fail_publish)

    with pytest.raises(expected):
        await run_shadow_analyze(trace, run_id=RUN_ID, output_path=output)

    assert not output.exists()


@pytest.mark.asyncio
async def test_shadow_classifies_failed_postpublication_validation_as_integrity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _private_trace(tmp_path / "events.ndjson")
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        shadow_module,
        "load_or_create_installation_key",
        lambda: InstallationKey(b"v" * 32),
    )

    def corrupt_publish(
        _self: object,
        _data: bytes,
        *,
        validate_published: object,
    ) -> object:
        assert callable(validate_published)
        assert validate_published(b"corrupt") is False
        raise SecureFileError()

    monkeypatch.setattr(shadow_module.AtomicFilePublication, "publish", corrupt_publish)

    with pytest.raises(ShadowCommandIntegrityError):
        await run_shadow_analyze(trace, run_id=RUN_ID, output_path=output)

    assert not output.exists()


@pytest.mark.asyncio
async def test_shadow_classifies_a_corrupt_reopened_result_as_integrity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _private_trace(tmp_path / "events.ndjson")
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        shadow_module,
        "load_or_create_installation_key",
        lambda: InstallationKey(b"w" * 32),
    )

    def corrupt_reopen(
        _self: object,
        data: bytes,
        *,
        validate_published: object,
    ) -> object:
        assert callable(validate_published)
        assert validate_published(data) is True
        return SimpleNamespace(data=b"corrupt-reopened-data")

    monkeypatch.setattr(shadow_module.AtomicFilePublication, "publish", corrupt_reopen)

    with pytest.raises(ShadowCommandIntegrityError):
        await run_shadow_analyze(trace, run_id=RUN_ID, output_path=output)

    assert not output.exists()


@pytest.mark.asyncio
async def test_shadow_command_never_opens_network_or_imports_optional_provider_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _private_trace(tmp_path / "events.ndjson")
    output = tmp_path / "report.json"

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(PaperTwoPhaseCycleExecutor, "execute", forbidden)
    monkeypatch.setattr(PaperTwoPhaseCycleExecutor, "execute_phase_one", forbidden)
    monkeypatch.setattr(CycleCoordinator, "begin", forbidden)
    monkeypatch.setattr(CycleCoordinator, "reserve", forbidden)
    monkeypatch.setattr(BudgetGovernor, "reserve", forbidden)
    monkeypatch.setattr(DeliveryWorker, "deliver", forbidden)
    monkeypatch.setattr(MemoryRunRepository, "begin_cycle", forbidden)
    monkeypatch.setattr(MemoryRunRepository, "reserve_cycle", forbidden)
    monkeypatch.setattr(MemoryRunRepository, "record_outcome", forbidden)
    monkeypatch.setattr(
        shadow_module,
        "load_or_create_installation_key",
        lambda: InstallationKey(b"n" * 32),
    )
    for module in ("httpx", "openai_harmony"):
        monkeypatch.delitem(sys.modules, module, raising=False)

    report = await run_shadow_analyze(trace, run_id=RUN_ID, output_path=output)

    assert report.model_calls == 0
    assert report.memory_revisions == 0
    assert report.cycles_created == 0
    assert report.deliveries == 0
    assert "httpx" not in sys.modules
    assert "openai_harmony" not in sys.modules


class _BrokenPath:
    def __fspath__(self) -> str:
        raise OSError("fixture-secret-path")


@pytest.mark.parametrize("value", (_BrokenPath(), b"bytes-path", ""))
def test_shadow_path_copy_rejects_unsupported_or_empty_values(value: object) -> None:
    with pytest.raises(ShadowCommandInputError):
        shadow_module._copy_path(value)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_shadow_maps_output_and_repository_inspection_failures_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _private_trace(tmp_path / "events.ndjson")
    monkeypatch.setattr(
        shadow_module,
        "load_or_create_installation_key",
        lambda: InstallationKey(b"i" * 32),
    )

    with pytest.raises(ShadowCommandInputError):
        await run_shadow_analyze(
            trace,
            run_id=RUN_ID,
            output_path=tmp_path / "missing-output-parent" / "report.json",
        )
    with pytest.raises(ShadowCommandConfigurationError):
        await run_shadow_analyze(
            trace,
            run_id=RUN_ID,
            output_path=tmp_path / "report.json",
            repository_path=tmp_path / "missing-repository-parent" / "shadow.sqlite3",
        )
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize(
    ("failure", "expected"),
    (
        (SecureFileUnsupportedError(), ShadowCommandConfigurationError),
        (SecureFileError(), ShadowCommandInputError),
    ),
)
@pytest.mark.asyncio
async def test_shadow_classifies_output_authorization_failures_before_session_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    expected: type[Exception],
) -> None:
    trace = _private_trace(tmp_path / "events.ndjson")
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        shadow_module,
        "load_or_create_installation_key",
        lambda: InstallationKey(b"o" * 32),
    )

    def fail_authorization(*_args: object, **_kwargs: object) -> object:
        raise failure

    monkeypatch.setattr(
        shadow_module,
        "authorize_shadow_report_publication",
        fail_authorization,
    )

    with pytest.raises(expected):
        await run_shadow_analyze(trace, run_id=RUN_ID, output_path=output)

    assert not output.exists()


@pytest.mark.parametrize(
    ("failure", "expected"),
    (
        (SecureFileBoundError(), ShadowCommandIntegrityError),
        (SecureFileError(), ShadowCommandInputError),
    ),
)
@pytest.mark.asyncio
async def test_shadow_classifies_existing_replacement_read_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    expected: type[Exception],
) -> None:
    trace = _private_trace(tmp_path / "events.ndjson")
    output = tmp_path / "report.json"
    key = InstallationKey(b"x" * 32)
    monkeypatch.setattr(shadow_module, "load_or_create_installation_key", lambda: key)
    await run_shadow_analyze(trace, run_id=RUN_ID, output_path=output)
    original = output.read_bytes()

    def fail_read(*_args: object, **_kwargs: object) -> object:
        raise failure

    monkeypatch.setattr(shadow_module, "read_stable_file", fail_read)

    with pytest.raises(expected):
        await run_shadow_analyze(
            trace,
            run_id=RUN_ID,
            output_path=output,
            replace=True,
        )

    assert output.read_bytes() == original


@pytest.mark.asyncio
async def test_shadow_maps_sqlite_authorization_failure_to_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _private_trace(tmp_path / "events.ndjson")
    repository = tmp_path / "shadow.sqlite3"
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        shadow_module,
        "load_or_create_installation_key",
        lambda: InstallationKey(b"z" * 32),
    )

    def fail_sqlite(*_args: object, **_kwargs: object) -> object:
        raise SecureFileError()

    monkeypatch.setattr(shadow_module, "authorize_private_sqlite_path", fail_sqlite)

    with pytest.raises(ShadowCommandConfigurationError):
        await run_shadow_analyze(
            trace,
            run_id=RUN_ID,
            output_path=output,
            repository_path=repository,
        )

    assert not output.exists()


@pytest.mark.asyncio
async def test_shadow_maps_unsupported_output_inspection_to_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _private_trace(tmp_path / "events.ndjson")
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        shadow_module,
        "load_or_create_installation_key",
        lambda: InstallationKey(b"z" * 32),
    )

    real_open_parent = files_module._open_parent
    output_inspections = 0

    def fail_nested_output_inspection(path: Path) -> tuple[int, object]:
        nonlocal output_inspections
        if path == output.resolve():
            output_inspections += 1
            if output_inspections == 2:
                raise OSError(
                    getattr(errno, "EOPNOTSUPP", errno.ENOSYS),
                    "unsupported",
                )
        return real_open_parent(path)

    monkeypatch.setattr(files_module, "_open_parent", fail_nested_output_inspection)

    with pytest.raises(ShadowCommandConfigurationError):
        await run_shadow_analyze(trace, run_id=RUN_ID, output_path=output)

    assert output_inspections == 2
    assert not output.exists()


@pytest.mark.parametrize(
    ("run_id", "replace"),
    (
        (UUID("b35f05f3-555b-1f09-8996-a7b3693bb54a"), False),
        (RUN_ID, 1),
    ),
)
@pytest.mark.asyncio
async def test_shadow_service_rejects_non_v4_run_and_non_boolean_replace(
    tmp_path: Path,
    run_id: UUID,
    replace: object,
) -> None:
    trace = _private_trace(tmp_path / "events.ndjson")

    with pytest.raises(ShadowCommandInputError):
        await run_shadow_analyze(
            trace,
            run_id=run_id,
            output_path=tmp_path / "report.json",
            replace=replace,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_shadow_rejects_session_result_and_final_prefix_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _private_trace(tmp_path / "events.ndjson")
    key = InstallationKey(b"q" * 32)
    monkeypatch.setattr(shadow_module, "load_or_create_installation_key", lambda: key)
    original_submit = ShadowSession._submit

    async def mismatched_result(
        self: ShadowSession,
        input_record: object,
        *,
        cli_input_ordinal: int | None,
    ) -> object:
        result = await original_submit(
            self,
            input_record,  # type: ignore[arg-type]
            cli_input_ordinal=cli_input_ordinal,
        )
        if result.ref.sequence == 2:
            return result.model_copy(update={"ref": result.ref.model_copy(update={"sequence": 3})})
        return result

    monkeypatch.setattr(ShadowSession, "_submit", mismatched_result)
    first_output = tmp_path / "first.json"
    with pytest.raises(ShadowCommandConfigurationError):
        await run_shadow_analyze(trace, run_id=RUN_ID, output_path=first_output)
    assert not first_output.exists()

    monkeypatch.setattr(ShadowSession, "_submit", original_submit)
    original_snapshot = ShadowSession._snapshot_for_cli
    snapshot_calls = 0

    async def extra_final_event(self: ShadowSession) -> object:
        nonlocal snapshot_calls
        snapshot_calls += 1
        head, events = await original_snapshot(self)
        if snapshot_calls == 2:
            return head, (*events, events[-1])
        return head, events

    monkeypatch.setattr(ShadowSession, "_snapshot_for_cli", extra_final_event)
    second_output = tmp_path / "second.json"
    with pytest.raises(ShadowCommandConfigurationError):
        await run_shadow_analyze(trace, run_id=RUN_ID, output_path=second_output)
    assert not second_output.exists()


@pytest.mark.asyncio
async def test_shadow_rejects_report_binding_drift_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _private_trace(tmp_path / "events.ndjson")
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        shadow_module,
        "load_or_create_installation_key",
        lambda: InstallationKey(b"d" * 32),
    )
    monkeypatch.setattr(shadow_module, "shadow_report_binding", lambda _report: object())

    with pytest.raises(ShadowInvariantError):
        await run_shadow_analyze(trace, run_id=RUN_ID, output_path=output)

    assert not output.exists()


@pytest.mark.asyncio
async def test_shadow_requires_publication_to_invoke_postvalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _private_trace(tmp_path / "events.ndjson")
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        shadow_module,
        "load_or_create_installation_key",
        lambda: InstallationKey(b"j" * 32),
    )

    def skip_validation(_self: object, data: bytes, **_kwargs: object) -> object:
        return SimpleNamespace(data=data)

    monkeypatch.setattr(shadow_module.AtomicFilePublication, "publish", skip_validation)

    with pytest.raises(ShadowCommandIntegrityError):
        await run_shadow_analyze(trace, run_id=RUN_ID, output_path=output)

    assert not output.exists()


@pytest.mark.parametrize(
    ("failure", "expected"),
    (
        (ShadowInvariantError(), ShadowInvariantError),
        (DigestVerificationError("fixture-secret-repository"), ShadowCommandConfigurationError),
        (SecureFileError(), ShadowCommandConfigurationError),
        (OSError("fixture-secret-os"), ShadowCommandConfigurationError),
        (ValueError("fixture-secret-unexpected"), ShadowInvariantError),
    ),
)
@pytest.mark.asyncio
async def test_shadow_service_sanitizes_underlying_failure_families(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    expected: type[Exception],
) -> None:
    async def fail(*_args: object, **_kwargs: object) -> object:
        raise failure

    monkeypatch.setattr(shadow_module, "_run_shadow_analyze", fail)

    with pytest.raises(expected) as captured:
        await run_shadow_analyze("trace", run_id=RUN_ID, output_path="output")

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert "fixture-secret" not in repr(captured.value)
