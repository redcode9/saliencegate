from __future__ import annotations

import _socket
import contextlib
import importlib.util
import io
import json
import os
import socket
import stat
import sys
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from unittest import mock
from uuid import UUID

_MAX_COMMAND_REPORT_BYTES = 1 << 20
_OPTIONAL_MODULES = ("httpx", "openai_harmony")
_PROVIDER_MODULES = ("harbor", "anthropic", "openai")
_IMPORTED_MODULE_EXCLUSIONS = (*_PROVIDER_MODULES, *_OPTIONAL_MODULES)
_PROVIDER_CREDENTIAL_KEYS = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_ORGANIZATION",
        "OPENAI_ORG_ID",
        "OPENAI_PROJECT",
        "OPENAI_PROJECT_ID",
    }
)
_LOCAL_KEY_ENVIRONMENT_KEYS = frozenset({"APPDATA", "HOME", "XDG_CONFIG_HOME"})
_ZERO_OPERATION_FIELDS = (
    "model_calls",
    "budget_reservations",
    "cycles_created",
    "memory_revisions",
    "interventions",
    "delivery_authorizations",
    "deliveries",
    "intervention_outcomes",
)
_ATIF_ENVIRONMENT_DIGEST = "e" * 64
_ATIF_WORKING_DIRECTORY = "/synthetic/installed-shadow"


@dataclass(frozen=True, slots=True)
class _ATIFValidationExpectation:
    profile_id: str
    source_schema_version: str
    agent_name: str
    mapped_record_count: int
    mapped_action_count: int
    mapped_structured_outcome_count: int
    flagged_count: int
    outcome_evidence_authority: str


_PUBLIC_ATIF_EXPECTATIONS = {
    "harbor-codex/v1": _ATIFValidationExpectation(
        profile_id="harbor-codex/v1",
        source_schema_version="ATIF-v1.7",
        agent_name="codex",
        mapped_record_count=6,
        mapped_action_count=2,
        mapped_structured_outcome_count=2,
        flagged_count=3,
        outcome_evidence_authority="producer_claimed_structured",
    ),
    "harbor-terminus-2/v1": _ATIFValidationExpectation(
        profile_id="harbor-terminus-2/v1",
        source_schema_version="ATIF-v1.7",
        agent_name="terminus-2",
        mapped_record_count=4,
        mapped_action_count=2,
        mapped_structured_outcome_count=0,
        flagged_count=1,
        outcome_evidence_authority="none",
    ),
}
_ATIF_CASES = (
    (
        "codex",
        "harbor-codex-v1",
        UUID("65555555-5555-4555-8555-555555555555"),
        "SMOKE_CODEX_COMMAND_MUST_NOT_PERSIST",
        _ATIFValidationExpectation(
            profile_id="harbor-codex/v1",
            source_schema_version="ATIF-v1.7",
            agent_name="codex",
            mapped_record_count=4,
            mapped_action_count=1,
            mapped_structured_outcome_count=1,
            flagged_count=0,
            outcome_evidence_authority="producer_claimed_structured",
        ),
    ),
    (
        "terminus",
        "harbor-terminus-2-v1",
        UUID("75555555-5555-4555-8555-555555555555"),
        "SMOKE_TERMINUS_COMMAND_MUST_NOT_PERSIST",
        _ATIFValidationExpectation(
            profile_id="harbor-terminus-2/v1",
            source_schema_version="ATIF-v1.6",
            agent_name="terminus-2",
            mapped_record_count=3,
            mapped_action_count=1,
            mapped_structured_outcome_count=0,
            flagged_count=0,
            outcome_evidence_authority="none",
        ),
    ),
)
_TRACE_ROWS = (
    {
        "schema_version": "shadow-input/v1",
        "kind": "run_start",
        "source_event_id": "start",
        "occurred_at": "2026-07-16T10:00:00Z",
    },
    {
        "schema_version": "shadow-input/v1",
        "kind": "action",
        "source_event_id": "action-1",
        "occurred_at": "2026-07-16T10:01:00Z",
        "argv": ["example-tool", "--check"],
        "working_directory": "/example",
        "environment_digest": "b" * 64,
    },
    {
        "schema_version": "shadow-input/v1",
        "kind": "tool_result",
        "source_event_id": "tool-result-1",
        "occurred_at": "2026-07-16T10:02:00Z",
        "action_source_event_id": "action-1",
        "status": "failed",
        "exit_status": 1,
        "exception_type": "ExampleToolFailure",
    },
    {
        "schema_version": "shadow-input/v1",
        "kind": "run_end",
        "source_event_id": "finish",
        "occurred_at": "2026-07-16T10:03:00Z",
    },
)


class _EnvironmentReadGuard(MutableMapping[str, str]):
    """Delegate normal environment access but fail on provider credential reads."""

    __slots__ = ("_environment", "_reads")

    def __init__(self, environment: MutableMapping[str, str]) -> None:
        self._environment = environment
        self._reads: set[str] = set()

    @staticmethod
    def _normalized(key: object) -> str:
        if type(key) is not str:
            raise TypeError("environment key must be text")
        return key.upper()

    def _record(self, key: object) -> str:
        normalized = self._normalized(key)
        if normalized in _PROVIDER_CREDENTIAL_KEYS:
            raise RuntimeError("provider credential environment access is forbidden")
        self._reads.add(normalized)
        return key  # type: ignore[return-value]

    @property
    def reads(self) -> frozenset[str]:
        return frozenset(self._reads)

    def __getitem__(self, key: str) -> str:
        return self._environment[self._record(key)]

    def __setitem__(self, key: str, value: str) -> None:
        self._environment[key] = value

    def __delitem__(self, key: str) -> None:
        del self._environment[key]

    def __iter__(self) -> Iterator[str]:
        for key in self._environment:
            self._record(key)
            yield key

    def __len__(self) -> int:
        return len(self._environment)

    def __contains__(self, key: object) -> bool:
        return self._record(key) in self._environment

    def copy(self) -> dict[str, str]:
        return {key: self[key] for key in self}

    def __repr__(self) -> str:
        return "_EnvironmentReadGuard(<redacted>)"


@contextlib.contextmanager
def _guard_provider_environment_reads() -> Iterator[_EnvironmentReadGuard]:
    original = os.environ
    guard = _EnvironmentReadGuard(original)
    with mock.patch.object(os, "environ", guard):
        yield guard


def _assert_core_only() -> None:
    for optional_module in _IMPORTED_MODULE_EXCLUSIONS:
        if importlib.util.find_spec(optional_module) is not None:
            raise RuntimeError("provider or optional model runtime is present in core")
    _assert_import_exclusions()


def _assert_import_exclusions() -> None:
    for optional_module in _IMPORTED_MODULE_EXCLUSIONS:
        if optional_module in sys.modules or any(
            imported.startswith(f"{optional_module}.") for imported in sys.modules
        ):
            raise RuntimeError("provider or optional model runtime was imported")


def _assert_network_is_denied() -> None:
    for constructor in (socket.socket, _socket.socket):
        opened = None
        try:
            opened = constructor(socket.AF_INET, socket.SOCK_STREAM)
        except Exception:
            continue
        finally:
            if opened is not None:
                opened.close()
        raise RuntimeError("installed ATIF smoke requires the socket guard")


def _require_private_parent(path: Path) -> None:
    parent = path.parent
    metadata = parent.stat(follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("shadow smoke parent is invalid")
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise RuntimeError("shadow smoke parent is not private")
    if "tests/fixtures" in parent.as_posix():
        raise RuntimeError("shadow smoke input must not use checkout fixtures")


def _write_all(descriptor: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(descriptor, data[offset:])
        if written <= 0:
            raise OSError("private trace write failed")
        offset += written


def _write_private(path: Path, data: bytes) -> None:
    _require_private_parent(path)
    if type(data) is not bytes or not data:
        raise RuntimeError("private smoke payload is invalid")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    complete = False
    try:
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, data)
        os.fsync(descriptor)
        complete = True
    finally:
        os.close(descriptor)
        if not complete:
            path.unlink(missing_ok=True)
    metadata = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
    ):
        raise RuntimeError("private trace publication failed")


def _write_trace(path: Path) -> None:
    from saliencegate.domain import canonical_json

    encoded = b"".join(canonical_json(row) + b"\n" for row in _TRACE_ROWS)
    _write_private(path, encoded)


def _read_private(path: Path, *, maximum_bytes: int) -> bytes:
    from saliencegate.security import StableReadPolicy, read_stable_file

    _require_private_parent(path)
    return read_stable_file(
        path,
        maximum_bytes=maximum_bytes,
        policy=StableReadPolicy.PRIVATE_OWNER,
    ).data


def _validate_report(report_path: Path, command_path: Path, run_id_text: str) -> None:
    from saliencegate.domain import canonical_json
    from saliencegate.shadow.io import MAX_SHADOW_REPORT_BYTES, decode_shadow_run_report

    run_id = UUID(run_id_text)
    report_bytes = _read_private(report_path, maximum_bytes=MAX_SHADOW_REPORT_BYTES)
    command_bytes = _read_private(command_path, maximum_bytes=_MAX_COMMAND_REPORT_BYTES)
    report = decode_shadow_run_report(report_bytes)
    command_report = json.loads(command_bytes)
    if type(command_report) is not dict:
        raise RuntimeError("shadow command report is invalid")
    if report_bytes != canonical_json(report.model_dump(mode="json", warnings=False)):
        raise RuntimeError("shadow report is not canonical")
    if command_bytes != canonical_json(command_report) + b"\n":
        raise RuntimeError("shadow command report is not canonical")
    if report.run_id != run_id or command_report.get("run_id") != str(run_id):
        raise RuntimeError("shadow report run binding is invalid")
    if command_report.get("report_digest") != report.report_digest:
        raise RuntimeError("shadow report digest binding is invalid")
    if report.execution_mode != "shadow" or command_report.get("execution_mode") != "shadow":
        raise RuntimeError("shadow execution mode is invalid")
    if report.decision_authority or command_report.get("decision_authority") is not False:
        raise RuntimeError("shadow report granted decision authority")
    if report.input_row_count != 4 or report.unique_input_event_count != 4:
        raise RuntimeError("shadow report input counts are invalid")
    for field in _ZERO_OPERATION_FIELDS:
        if getattr(report, field) != 0 or command_report.get(field) != 0:
            raise RuntimeError("shadow smoke performed a forbidden operation")


def _atif_payload(case_name: str, command: str) -> dict[str, object]:
    if case_name == "codex":
        return {
            "agent": {
                "extra": {"fixture_generator": "installed-shadow-smoke/v1"},
                "model_name": "synthetic-openai-compatible-model",
                "name": "codex",
                "version": "synthetic",
            },
            "final_metrics": {
                "total_cached_tokens": 0,
                "total_completion_tokens": 0,
                "total_cost_usd": 0,
                "total_prompt_tokens": 0,
                "total_steps": 2,
            },
            "schema_version": "ATIF-v1.7",
            "session_id": "installed-codex-smoke-session",
            "steps": [
                {
                    "message": "SYNTHETIC_INSTALLED_CODEX_TASK",
                    "source": "user",
                    "step_id": 1,
                },
                {
                    "extra": {
                        "tool_call_details": {
                            "installed-codex-call": {
                                "metadata": {"exit_code": 0},
                                "status": "completed",
                            }
                        }
                    },
                    "message": "SYNTHETIC_INSTALLED_CODEX_ACTION",
                    "model_name": "synthetic-openai-compatible-model",
                    "observation": {
                        "results": [
                            {
                                "content": "SYNTHETIC_INSTALLED_CODEX_OUTPUT",
                                "source_call_id": "installed-codex-call",
                            }
                        ]
                    },
                    "source": "agent",
                    "step_id": 2,
                    "timestamp": "2026-07-17T00:00:00.123Z",
                    "tool_calls": [
                        {
                            "arguments": {
                                "cmd": command,
                                "login": False,
                                "sandbox_permissions": "workspace-write",
                                "shell": "/bin/sh",
                                "tty": False,
                                "workdir": _ATIF_WORKING_DIRECTORY,
                                "yield_time_ms": 1000,
                            },
                            "function_name": "exec_command",
                            "tool_call_id": "installed-codex-call",
                        }
                    ],
                },
            ],
        }
    if case_name == "terminus":
        return {
            "agent": {
                "extra": {"parser": "installed-shadow-smoke/v1"},
                "model_name": "synthetic-model",
                "name": "terminus-2",
                "version": "synthetic",
            },
            "final_metrics": {
                "total_cached_tokens": 0,
                "total_completion_tokens": 0,
                "total_cost_usd": 0,
                "total_prompt_tokens": 0,
            },
            "schema_version": "ATIF-v1.6",
            "session_id": "installed-terminus-smoke-session",
            "steps": [
                {
                    "message": "SYNTHETIC_INSTALLED_TERMINUS_TASK",
                    "source": "user",
                    "step_id": 1,
                },
                {
                    "message": "SYNTHETIC_INSTALLED_TERMINUS_ACTION",
                    "metrics": {
                        "completion_token_ids": [],
                        "completion_tokens": 0,
                        "cost_usd": 0,
                        "logprobs": [],
                        "prompt_token_ids": [],
                        "prompt_tokens": 0,
                    },
                    "model_name": "synthetic-model",
                    "observation": {
                        "results": [{"content": "SYNTHETIC_INSTALLED_TERMINUS_OUTPUT"}]
                    },
                    "source": "agent",
                    "step_id": 2,
                    "tool_calls": [
                        {
                            "arguments": {
                                "duration": 0.1,
                                "keystrokes": f"{command}\n",
                            },
                            "function_name": "bash_command",
                            "tool_call_id": "installed-terminus-call",
                        }
                    ],
                },
            ],
        }
    raise RuntimeError("installed ATIF smoke case is invalid")


def _validate_atif_report(
    *,
    source_path: Path,
    report_path: Path,
    command_path: Path,
    run_id: UUID,
    expectation: _ATIFValidationExpectation,
) -> None:
    from saliencegate.commands.shadow import _atif_command_report, render_shadow_atif_json
    from saliencegate.shadow import decode_shadow_trace_report, encode_shadow_trace_report
    from saliencegate.shadow.trace import ATIFShadowDiagnostics
    from saliencegate.shadow.trace_report import MAX_SHADOW_TRACE_REPORT_BYTES

    source_bytes = _read_private(source_path, maximum_bytes=64 << 20)
    report_bytes = _read_private(report_path, maximum_bytes=MAX_SHADOW_TRACE_REPORT_BYTES)
    command_bytes = _read_private(command_path, maximum_bytes=_MAX_COMMAND_REPORT_BYTES)
    report = decode_shadow_trace_report(report_bytes)
    try:
        source = json.loads(source_bytes)
        parsed_command_report = json.loads(command_bytes)
    except Exception:
        raise RuntimeError("ATIF source or command report is invalid") from None
    diagnostics = report.diagnostics
    if (
        type(source) is not dict
        or type(parsed_command_report) is not dict
        or type(diagnostics) is not ATIFShadowDiagnostics
    ):
        raise RuntimeError("ATIF command report is invalid")
    if report_bytes != encode_shadow_trace_report(report):
        raise RuntimeError("ATIF trace report is not canonical")
    expected_command_report = _atif_command_report(report)
    if command_bytes != render_shadow_atif_json(expected_command_report).encode("utf-8"):
        raise RuntimeError("ATIF command report does not match the trace report")
    command_report = expected_command_report
    agent = source.get("agent")
    steps = source.get("steps")
    if (
        type(agent) is not dict
        or type(steps) is not list
        or source.get("schema_version") != expectation.source_schema_version
        or agent.get("name") != expectation.agent_name
    ):
        raise RuntimeError("ATIF source identity is invalid")
    total_tool_call_count = 0
    total_observation_result_count = 0
    for step in steps:
        if type(step) is not dict:
            raise RuntimeError("ATIF source steps are invalid")
        tool_calls = step.get("tool_calls", [])
        if tool_calls is None:
            tool_calls = []
        observation = step.get("observation")
        if observation is None:
            results = []
        elif type(observation) is dict:
            results = observation.get("results", [])
        else:
            raise RuntimeError("ATIF source observations are invalid")
        if type(tool_calls) is not list or type(results) is not list:
            raise RuntimeError("ATIF source evidence is invalid")
        total_tool_call_count += len(tool_calls)
        total_observation_result_count += len(results)
    if (
        report.run_id != run_id
        or command_report.run_id != run_id
        or report.binding.adapter_profile_id != expectation.profile_id
        or report.binding.source_schema_version != expectation.source_schema_version
        or command_report.adapter_profile_id != expectation.profile_id
        or command_report.report_digest != report.report_digest
    ):
        raise RuntimeError("ATIF report identity is invalid")
    if (
        report.binding.source_format != "atif"
        or report.binding.capture_scope != "selected_events"
        or report.binding.source_byte_count != len(source_bytes)
        or report.binding.source_byte_digest != sha256(source_bytes).hexdigest()
        or report.shadow_report.input_byte_digest != report.binding.source_byte_digest
    ):
        raise RuntimeError("ATIF source binding is invalid")
    if (
        diagnostics.root_segment_only is not True
        or diagnostics.complete_execution_session_coverage is not False
        or diagnostics.continued_trajectory_ref_present
        != (source.get("continued_trajectory_ref") is not None)
        or diagnostics.embedded_subagent_trajectory_count
        != len(source.get("subagent_trajectories") or [])
        or command_report.root_segment_only is not True
        or command_report.complete_execution_session_coverage is not False
        or command_report.decision_authority is not False
        or report.shadow_report.decision_authority
    ):
        raise RuntimeError("ATIF observational scope is invalid")
    tool_dispositions = dict(diagnostics.tool_call_disposition_counts)
    result_dispositions = dict(diagnostics.result_disposition_counts)
    heuristic_dispositions = {
        disposition.value: count
        for disposition, count in report.shadow_report.heuristic_disposition_counts
    }
    if (
        diagnostics.producer_authentication != "none"
        or diagnostics.outcome_evidence_authority != expectation.outcome_evidence_authority
        or diagnostics.total_step_count != len(steps)
        or diagnostics.total_tool_call_count != total_tool_call_count
        or diagnostics.total_observation_result_count != total_observation_result_count
        or diagnostics.mapped_shadow_record_count != expectation.mapped_record_count
        or tool_dispositions["mapped_action"] != expectation.mapped_action_count
        or result_dispositions["mapped_structured_outcome"]
        != expectation.mapped_structured_outcome_count
        or report.shadow_report.input_row_count != expectation.mapped_record_count
        or report.shadow_report.unique_input_event_count != expectation.mapped_record_count
        or report.shadow_report.evaluated_unique_event_count != expectation.mapped_record_count
        or heuristic_dispositions["flagged"] != expectation.flagged_count
    ):
        raise RuntimeError("ATIF report does not match the expected fixture semantics")
    for field in _ZERO_OPERATION_FIELDS:
        if getattr(report.shadow_report, field) != 0 or getattr(command_report, field) != 0:
            raise RuntimeError("ATIF smoke performed a forbidden operation")
    sensitive_keys = frozenset(
        {
            "argv",
            "cmd",
            "command",
            "content",
            "keystrokes",
            "message",
            "session_id",
            "source_call_id",
            "tool_call_id",
            "workdir",
            "working_directory",
        }
    )
    sensitive_source_values: set[bytes] = set()

    def collect_sensitive(value: object, *, sensitive: bool = False) -> None:
        if type(value) is str:
            encoded = value.rstrip("\n").encode("utf-8")
            if sensitive and len(encoded) >= 8:
                sensitive_source_values.add(encoded)
            return
        if type(value) is list:
            for item in value:
                collect_sensitive(item, sensitive=sensitive)
            return
        if type(value) is dict:
            for key, item in value.items():
                collect_sensitive(item, sensitive=key in sensitive_keys)

    collect_sensitive(source)
    sensitive_values = (
        *sensitive_source_values,
        os.fspath(source_path).encode("utf-8"),
        os.fspath(report_path).encode("utf-8"),
        os.fspath(command_path).encode("utf-8"),
    )
    if any(value in report_bytes or value in command_bytes for value in sensitive_values):
        raise RuntimeError("ATIF smoke report disclosed source content or paths")


def _validate_local_key(environment_guard: _EnvironmentReadGuard) -> None:
    from saliencegate.security import default_installation_key_path

    key_path = default_installation_key_path()
    metadata = key_path.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
        or metadata.st_size != 32
    ):
        raise RuntimeError("installed ATIF smoke key boundary is invalid")
    if not environment_guard.reads.intersection(_LOCAL_KEY_ENVIRONMENT_KEYS):
        raise RuntimeError("installed ATIF smoke did not exercise the local key boundary")


def _exercise_atif_profiles(root: Path, environment_guard: _EnvironmentReadGuard) -> None:
    from saliencegate.cli import main as cli_main
    from saliencegate.domain import canonical_json

    _require_private_parent(root / "private-boundary-probe")
    _assert_network_is_denied()
    _assert_import_exclusions()
    for case_name, profile_alias, run_id, secret_command, expectation in _ATIF_CASES:
        source_path = root / f"{case_name}.trajectory.json"
        report_path = root / f"{case_name}.report.json"
        command_path = root / f"{case_name}.command.json"
        _write_private(source_path, canonical_json(_atif_payload(case_name, secret_command)))
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = cli_main(
                (
                    "shadow",
                    "analyze-atif",
                    os.fspath(source_path),
                    "--profile",
                    profile_alias,
                    "--run-id",
                    str(run_id),
                    "--working-directory",
                    _ATIF_WORKING_DIRECTORY,
                    "--environment-digest",
                    _ATIF_ENVIRONMENT_DIGEST,
                    "--output",
                    os.fspath(report_path),
                    "--json",
                )
            )
        if exit_code != 0 or stderr.getvalue():
            raise RuntimeError("installed ATIF CLI smoke failed")
        command_bytes = stdout.getvalue().encode("utf-8")
        _write_private(command_path, command_bytes)
        _validate_atif_report(
            source_path=source_path,
            report_path=report_path,
            command_path=command_path,
            run_id=run_id,
            expectation=expectation,
        )
        _assert_import_exclusions()
    _validate_local_key(environment_guard)


def main() -> None:
    message: str | None = None
    with _guard_provider_environment_reads() as environment_guard:
        if len(sys.argv) == 3 and sys.argv[1] == "write-trace":
            _assert_core_only()
            _write_trace(Path(sys.argv[2]))
            message = "shadow-trace-ok"
        elif len(sys.argv) == 5 and sys.argv[1] == "validate-report":
            _assert_core_only()
            _validate_report(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
            _assert_core_only()
            message = "shadow-installed-ok"
        elif len(sys.argv) == 7 and sys.argv[1] == "validate-public-atif":
            expectation = _PUBLIC_ATIF_EXPECTATIONS.get(sys.argv[6])
            if expectation is None:
                raise SystemExit(2)
            _assert_core_only()
            _validate_atif_report(
                source_path=Path(sys.argv[2]),
                report_path=Path(sys.argv[3]),
                command_path=Path(sys.argv[4]),
                run_id=UUID(sys.argv[5]),
                expectation=expectation,
            )
            _assert_core_only()
            message = "shadow-atif-public-report-ok"
        elif len(sys.argv) == 3 and sys.argv[1] == "exercise-atif":
            _exercise_atif_profiles(Path(sys.argv[2]), environment_guard)
            _assert_import_exclusions()
            message = "shadow-atif-installed-ok"
        else:
            raise SystemExit(2)
    if message is not None:
        print(message)
        return
    raise SystemExit(2)  # pragma: no cover - every dispatch assigns or raises


if __name__ == "__main__":
    main()
