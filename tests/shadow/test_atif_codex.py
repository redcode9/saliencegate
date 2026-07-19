from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from saliencegate.domain import canonical_json
from saliencegate.shadow import ShadowAnalyzer, ShadowSession
from saliencegate.shadow.atif import ATIFProfile, ATIFShadowAdapter, ShadowEnvironmentBinding
from saliencegate.shadow.errors import ShadowTraceInputError
from saliencegate.shadow.trace import ATIFShadowDiagnostics, ShadowTrace
from saliencegate.shadow.trace_report import (
    decode_shadow_trace_report,
    encode_shadow_trace_report,
)

RUN_ID = UUID("44444444-4444-4444-8444-444444444444")
ENVIRONMENT_DIGEST = "c" * 64
DEFAULT_WORKING_DIRECTORY = "/synthetic/codex-default"
FIXTURES = Path("tests/fixtures/shadow/atif")
CODEX_PROFILE = ATIFProfile.HARBOR_CODEX_V1
TOOL_DISPOSITIONS = (
    "mapped_action",
    "ignored_unsupported_function",
    "ignored_continuation",
    "ignored_non_command_wait",
    "ignored_unsubmitted_keystrokes",
    "ignored_unresolved_terminal_submission",
    "ignored_copied_context",
)
RESULT_DISPOSITIONS = (
    "mapped_structured_outcome",
    "ignored_evidence_absent",
    "ignored_ambiguous_parent",
    "ignored_no_parent",
    "ignored_unsupported_parent",
    "ignored_copied_context",
)
_ABSENT = object()


def adapter() -> ATIFShadowAdapter:
    return ATIFShadowAdapter(
        profile=CODEX_PROFILE,
        environment=ShadowEnvironmentBinding(
            default_working_directory=DEFAULT_WORKING_DIRECTORY,
            environment_digest=ENVIRONMENT_DIGEST,
        ),
    )


def call(
    ordinal: int,
    *,
    function_name: str = "exec_command",
    arguments: object = _ABSENT,
    tool_call_id: str | None = None,
) -> dict[str, object]:
    if arguments is _ABSENT:
        arguments = {"cmd": f"printf synthetic-codex-{ordinal}"}
    return {
        "tool_call_id": tool_call_id or f"synthetic-codex-call-{ordinal}",
        "function_name": function_name,
        "arguments": arguments,
    }


def result(
    source_call_id: str | None = None,
    *,
    content: str = "synthetic-codex-output",
    **overrides: object,
) -> dict[str, object]:
    value: dict[str, object] = {"content": content, **overrides}
    if source_call_id is not None:
        value["source_call_id"] = source_call_id
    return value


def step(
    ordinal: int,
    *,
    calls: list[dict[str, object]] | None = None,
    results: list[dict[str, object]] | None = None,
    extra: object = _ABSENT,
    timestamp: object = _ABSENT,
    copied: bool = False,
) -> dict[str, object]:
    value: dict[str, object] = {"step_id": ordinal, "source": "agent"}
    if calls is not None:
        value["tool_calls"] = calls
    if results is not None:
        value["observation"] = {"results": results}
    if extra is not _ABSENT:
        value["extra"] = extra
    if timestamp is not _ABSENT:
        value["timestamp"] = timestamp
    if copied:
        value["is_copied_context"] = True
    return value


def source_bytes(
    steps: list[dict[str, object]],
    *,
    schema_version: str = "ATIF-v1.7",
    agent_name: str = "codex",
) -> bytes:
    return canonical_json(
        {
            "schema_version": schema_version,
            "session_id": "synthetic-codex-session",
            "agent": {"name": agent_name, "version": "synthetic"},
            "steps": steps,
            "continued_trajectory_ref": None,
            "subagent_trajectories": None,
        }
    )


def adapt(source: bytes) -> ShadowTrace:
    return adapter().adapt_bytes(source, run_id=RUN_ID)


def diagnostics(trace: ShadowTrace) -> ATIFShadowDiagnostics:
    value = trace.diagnostics
    assert type(value) is ATIFShadowDiagnostics
    assert tuple(name for name, _count in value.tool_call_disposition_counts) == TOOL_DISPOSITIONS
    assert tuple(name for name, _count in value.result_disposition_counts) == RESULT_DISPOSITIONS
    return value


def counts(items: tuple[tuple[str, int], ...]) -> dict[str, int]:
    return dict(items)


def assert_error(
    source: bytes,
    reason_code: str,
    *,
    step_ordinal: int | None = None,
    call_ordinal: int | None = None,
    result_ordinal: int | None = None,
    secret: str = "synthetic-secret-codex-value",
) -> None:
    with pytest.raises(ShadowTraceInputError) as captured:
        adapt(source)
    error = captured.value
    assert error.reason_code == reason_code
    assert error.step_ordinal == step_ordinal
    assert error.call_ordinal == call_ordinal
    assert error.result_ordinal == result_ordinal
    assert error.args == ("shadow input is invalid",)
    assert secret not in str(error)
    assert secret not in repr(error)
    assert error.__cause__ is None
    assert error.__context__ is None


def single_call_source(
    exit_code: object = _ABSENT,
    *,
    arguments: object = _ABSENT,
    extra: object = _ABSENT,
    timestamp: object = _ABSENT,
) -> bytes:
    tool_call_id = "synthetic-codex-selected"
    if extra is _ABSENT:
        extra = (
            {}
            if exit_code is _ABSENT
            else {"tool_metadata": {"exit_code": exit_code, "status": "completed"}}
        )
    return source_bytes(
        [
            step(
                1,
                calls=[call(1, arguments=arguments, tool_call_id=tool_call_id)],
                results=[
                    result(
                        tool_call_id,
                        content="synthetic-secret-codex-value exit_code=0 completed",
                    )
                ],
                extra=extra,
                timestamp=timestamp,
            )
        ]
    )


def test_synthetic_bundled_fixture_maps_only_dispatch_and_structured_exit() -> None:
    source = (FIXTURES / "codex-bundled-synthetic.trajectory.json").read_bytes()

    trace = adapt(source)
    mapped = trace.records
    report = diagnostics(trace)

    assert trace.binding.source_schema_version == "ATIF-v1.7"
    assert trace.binding.adapter_profile_id == CODEX_PROFILE.value
    assert trace.binding.timestamp_mode == "source_utc"
    assert [record["kind"] for record in mapped] == [
        "run_start",
        "action",
        "tool_result",
        "run_end",
    ]
    assert [record["source_event_id"] for record in mapped] == [
        "atif-run-start",
        "atif-s00000002-c0001-action",
        "atif-s00000002-c0001-result",
        "atif-run-end",
    ]
    assert all(record["occurred_at"] == "2026-07-17T00:00:00.123000Z" for record in mapped)
    assert mapped[1]["command"] == "printf synthetic-codex-command"
    assert mapped[1]["working_directory"] == "/synthetic/workspace"
    assert mapped[1]["environment_digest"] != ENVIRONMENT_DIGEST
    assert mapped[2]["action_source_event_id"] == mapped[1]["source_event_id"]
    assert mapped[2]["status"] == "succeeded"
    assert mapped[2]["exit_status"] == 0
    assert "synthetic-codex-exec-1" not in repr(mapped)
    assert "SYNTHETIC_CODEX_EXEC_OUTPUT" not in repr(mapped)

    assert report.total_step_count == 2
    assert report.ignored_message_step_count == 1
    assert report.total_tool_call_count == 2
    assert counts(report.tool_call_disposition_counts) == {
        "mapped_action": 1,
        "ignored_unsupported_function": 0,
        "ignored_continuation": 1,
        "ignored_non_command_wait": 0,
        "ignored_unsubmitted_keystrokes": 0,
        "ignored_unresolved_terminal_submission": 0,
        "ignored_copied_context": 0,
    }
    assert report.total_observation_result_count == 2
    assert counts(report.result_disposition_counts) == {
        "mapped_structured_outcome": 1,
        "ignored_evidence_absent": 0,
        "ignored_ambiguous_parent": 0,
        "ignored_no_parent": 0,
        "ignored_unsupported_parent": 1,
        "ignored_copied_context": 0,
    }
    assert report.mapped_shadow_record_count == 4
    assert report.complete_execution_session_coverage is False
    assert report.producer_authentication == "none"
    assert report.outcome_evidence_authority == "producer_claimed_structured"


def test_unsupported_converter_call_may_have_an_empty_identifier() -> None:
    unsupported = call(1, function_name="web_search_call", arguments={})
    unsupported["tool_call_id"] = ""
    trace = adapt(source_bytes([step(1, calls=[unsupported, call(2)])]))

    assert [record["kind"] for record in trace.records] == [
        "run_start",
        "action",
        "run_end",
    ]
    assert (
        counts(diagnostics(trace).tool_call_disposition_counts)["ignored_unsupported_function"] == 1
    )


@pytest.mark.asyncio
async def test_codex_trace_analyzes_into_a_self_verifying_atif_report() -> None:
    trace = adapt(single_call_source(exit_code=0))
    session = ShadowSession.in_memory_for_trace(
        run_id=trace.run_id,
        trace_binding=trace.binding,
    )

    async with session:
        report = await ShadowAnalyzer(session).analyze(trace)
    decoded = decode_shadow_trace_report(encode_shadow_trace_report(report))

    assert decoded == report
    assert decoded.binding.adapter_profile_id == CODEX_PROFILE.value
    assert type(decoded.diagnostics) is ATIFShadowDiagnostics
    row_kinds = tuple(row.input_kind.value for row in decoded.shadow_report.rows)
    assert row_kinds == ("start", "action", "tool_result", "finish")
    assert counts(decoded.diagnostics.tool_call_disposition_counts)["mapped_action"] == 1
    assert counts(decoded.diagnostics.result_disposition_counts)["mapped_structured_outcome"] == 1


def test_codex_profile_requires_exact_v17_schema_and_agent_name() -> None:
    selected = step(1, calls=[call(1)])

    assert_error(source_bytes([selected], schema_version="ATIF-v1.6"), "unsupported_schema")
    assert_error(source_bytes([selected], agent_name="Codex"), "profile_mismatch")


def test_workdir_overrides_the_caller_default_without_changing_context_digest() -> None:
    implicit = adapt(single_call_source(arguments={"cmd": "printf same"}))
    explicit = adapt(
        single_call_source(arguments={"cmd": "printf same", "workdir": "/explicit/workdir"})
    )

    assert implicit.records[1]["working_directory"] == DEFAULT_WORKING_DIRECTORY
    assert explicit.records[1]["working_directory"] == "/explicit/workdir"
    assert implicit.records[1]["environment_digest"] == explicit.records[1]["environment_digest"]
    assert implicit.mapped_record_digest != explicit.mapped_record_digest


def test_source_only_exec_options_do_not_change_mapped_action_or_configuration_identity() -> None:
    baseline = adapt(
        single_call_source(
            arguments={
                "cmd": "printf stable",
                "workdir": "/stable/workdir",
                "shell": "/bin/sh",
                "login": False,
                "tty": False,
                "sandbox_permissions": "workspace-write",
            }
        )
    )
    source_only = adapt(
        single_call_source(
            arguments={
                "cmd": "printf stable",
                "workdir": "/stable/workdir",
                "shell": "/bin/sh",
                "login": False,
                "tty": False,
                "sandbox_permissions": "workspace-write",
                "yield_time_ms": 30_000,
                "max_output_tokens": 2_000,
                "justification": "synthetic source-only justification",
                "prefix_rule": ["git", "status"],
            }
        )
    )

    assert baseline.records == source_only.records
    assert baseline.mapped_record_digest == source_only.mapped_record_digest
    assert (
        baseline.binding.adapter_configuration_digest
        == source_only.binding.adapter_configuration_digest
    )
    assert baseline.binding.source_byte_digest != source_only.binding.source_byte_digest


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("shell", "/bin/bash"),
        ("login", True),
        ("tty", True),
        ("sandbox_permissions", "danger-full-access"),
    ),
)
def test_every_execution_semantics_field_changes_the_action_context_digest(
    field: str,
    value: object,
) -> None:
    baseline_arguments: dict[str, object] = {
        "cmd": "printf stable",
        "shell": "/bin/sh",
        "login": False,
        "tty": False,
        "sandbox_permissions": "workspace-write",
    }
    changed_arguments = dict(baseline_arguments)
    changed_arguments[field] = value

    baseline = adapt(single_call_source(arguments=baseline_arguments))
    changed = adapt(single_call_source(arguments=changed_arguments))

    assert baseline.records[1]["command"] == changed.records[1]["command"]
    assert baseline.records[1]["working_directory"] == changed.records[1]["working_directory"]
    assert baseline.records[1]["environment_digest"] != changed.records[1]["environment_digest"]


@pytest.mark.parametrize(
    "arguments",
    (
        None,
        {},
        {"cmd": ""},
        {"cmd": 1},
        {"cmd": "printf valid", "workdir": ""},
        {"cmd": "printf valid", "unknown_option": True},
        {"cmd": "printf valid", "yield_time_ms": True},
        {"cmd": "printf valid", "max_output_tokens": -1},
        {"cmd": "printf valid", "prefix_rule": "git status"},
    ),
)
def test_codex_exec_arguments_are_closed_and_exact(arguments: object) -> None:
    source = single_call_source(arguments=arguments)

    assert_error(source, "invalid_tool_call", step_ordinal=1, call_ordinal=1)


def test_write_stdin_is_a_continuation_and_other_functions_are_unsupported() -> None:
    selected_id = "selected"
    continuation_id = "continuation"
    patch_id = "patch"
    web_id = "web"
    calls = [
        call(1, tool_call_id=selected_id),
        call(
            2,
            function_name="write_stdin",
            arguments={"session_id": 1, "chars": "\n"},
            tool_call_id=continuation_id,
        ),
        call(3, function_name="apply_patch", arguments={}, tool_call_id=patch_id),
        call(4, function_name="web_search", arguments={}, tool_call_id=web_id),
    ]
    extra = {
        "tool_call_details": {selected_id: {"metadata": {"exit_code": 0}, "status": "completed"}}
    }
    source = source_bytes(
        [
            step(
                1,
                calls=calls,
                results=[
                    result(identifier)
                    for identifier in (selected_id, continuation_id, patch_id, web_id)
                ],
                extra=extra,
            )
        ]
    )

    trace = adapt(source)
    report = diagnostics(trace)

    assert [record["kind"] for record in trace.records] == [
        "run_start",
        "action",
        "tool_result",
        "run_end",
    ]
    assert counts(report.tool_call_disposition_counts) == {
        "mapped_action": 1,
        "ignored_unsupported_function": 2,
        "ignored_continuation": 1,
        "ignored_non_command_wait": 0,
        "ignored_unsubmitted_keystrokes": 0,
        "ignored_unresolved_terminal_submission": 0,
        "ignored_copied_context": 0,
    }
    assert counts(report.result_disposition_counts) == {
        "mapped_structured_outcome": 1,
        "ignored_evidence_absent": 0,
        "ignored_ambiguous_parent": 0,
        "ignored_no_parent": 0,
        "ignored_unsupported_parent": 3,
        "ignored_copied_context": 0,
    }


@pytest.mark.parametrize(
    ("exit_code", "status"),
    (
        (0, "succeeded"),
        (1, "failed"),
        (-1, "failed"),
        ((1 << 31) - 1, "failed"),
        (-(1 << 31), "failed"),
    ),
)
def test_single_call_exact_int32_exit_metadata_maps_producer_claimed_outcome(
    exit_code: int,
    status: str,
) -> None:
    trace = adapt(single_call_source(exit_code))
    report = diagnostics(trace)
    outcome = trace.records[2]

    assert [record["kind"] for record in trace.records] == [
        "run_start",
        "action",
        "tool_result",
        "run_end",
    ]
    assert outcome["status"] == status
    assert outcome["exit_status"] == exit_code
    assert report.outcome_evidence_authority == "producer_claimed_structured"
    assert counts(report.result_disposition_counts)["mapped_structured_outcome"] == 1


@pytest.mark.parametrize(
    "exit_code",
    (
        True,
        "0",
        0.0,
        1 << 31,
        -(1 << 31) - 1,
        None,
    ),
)
def test_non_exact_or_out_of_range_exit_metadata_is_ignored_not_coerced(
    exit_code: object,
) -> None:
    trace = adapt(single_call_source(exit_code))
    report = diagnostics(trace)

    assert [record["kind"] for record in trace.records] == ["run_start", "action", "run_end"]
    assert counts(report.result_disposition_counts)["mapped_structured_outcome"] == 0
    assert counts(report.result_disposition_counts)["ignored_evidence_absent"] == 1


def test_bundled_exact_exit_metadata_is_keyed_by_the_same_step_call_id() -> None:
    tool_call_id = "bundled-selected"
    source = source_bytes(
        [
            step(
                1,
                calls=[call(1, tool_call_id=tool_call_id)],
                results=[result(tool_call_id)],
                extra={
                    "tool_call_details": {
                        tool_call_id: {"metadata": {"exit_code": 17}, "status": "completed"}
                    }
                },
            )
        ]
    )

    trace = adapt(source)

    assert trace.records[2]["status"] == "failed"
    assert trace.records[2]["exit_status"] == 17


def test_matching_single_and_bundled_metadata_paths_map_one_outcome() -> None:
    tool_call_id = "synthetic-codex-selected"
    extra = {
        "tool_metadata": {"exit_code": 0},
        "tool_call_details": {tool_call_id: {"metadata": {"exit_code": 0}}},
    }

    trace = adapt(single_call_source(extra=extra))
    report = diagnostics(trace)

    assert trace.records[2]["exit_status"] == 0
    assert counts(report.result_disposition_counts)["mapped_structured_outcome"] == 1


def test_conflicting_recognized_metadata_paths_fail_closed() -> None:
    tool_call_id = "synthetic-codex-selected"
    extra = {
        "tool_metadata": {"exit_code": 0},
        "tool_call_details": {tool_call_id: {"metadata": {"exit_code": 1}}},
    }

    assert_error(
        single_call_source(extra=extra),
        "invalid_outcome_metadata",
        step_ordinal=1,
        call_ordinal=1,
    )


@pytest.mark.parametrize(
    "extra",
    (
        {"tool_metadata": []},
        {"tool_call_details": []},
        {"tool_call_details": {"synthetic-codex-selected": {"metadata": []}}},
    ),
)
def test_malformed_metadata_containers_are_not_treated_as_admissible_exit_paths(
    extra: object,
) -> None:
    trace = adapt(single_call_source(extra=extra))
    report = diagnostics(trace)

    assert [record["kind"] for record in trace.records] == ["run_start", "action", "run_end"]
    assert counts(report.result_disposition_counts)["ignored_evidence_absent"] == 1


def test_lifecycle_status_output_and_alternate_exit_paths_never_become_outcome_evidence() -> None:
    tool_call_id = "synthetic-codex-selected"
    extra = {
        "status": "completed",
        "exit_code": 0,
        "tool_metadata": {"exit_status": 0},
        "tool_call_details": {tool_call_id: {"status": "completed", "output_exit_code": 0}},
    }
    source = source_bytes(
        [
            step(
                1,
                calls=[call(1, tool_call_id=tool_call_id)],
                results=[
                    result(
                        tool_call_id,
                        content="synthetic-secret-codex-value exit_code 0 succeeded completed",
                        exit_code=0,
                        status="completed",
                    )
                ],
                extra=extra,
            )
        ]
    )

    trace = adapt(source)
    report = diagnostics(trace)

    assert [record["kind"] for record in trace.records] == ["run_start", "action", "run_end"]
    assert counts(report.result_disposition_counts)["ignored_evidence_absent"] == 1
    assert "synthetic-secret-codex-value" not in repr(trace)
    assert "synthetic-secret-codex-value" not in repr(report)


def test_single_call_metadata_is_not_applied_to_a_bundled_step() -> None:
    selected_id = "selected"
    continuation_id = "continuation"
    source = source_bytes(
        [
            step(
                1,
                calls=[
                    call(1, tool_call_id=selected_id),
                    call(
                        2,
                        function_name="write_stdin",
                        arguments={"session_id": 1, "chars": ""},
                        tool_call_id=continuation_id,
                    ),
                ],
                results=[result(selected_id), result(continuation_id)],
                extra={"tool_metadata": {"exit_code": 0}},
            )
        ]
    )

    trace = adapt(source)
    report = diagnostics(trace)

    assert [record["kind"] for record in trace.records] == ["run_start", "action", "run_end"]
    assert counts(report.result_disposition_counts)["ignored_evidence_absent"] == 1
    assert counts(report.result_disposition_counts)["ignored_unsupported_parent"] == 1


def test_copied_codex_calls_and_results_are_counted_without_execution_mapping() -> None:
    copied_id = "copied"
    selected_id = "selected"
    source = source_bytes(
        [
            step(
                1,
                calls=[call(1, tool_call_id=copied_id)],
                results=[result(copied_id)],
                extra={"tool_metadata": {"exit_code": 0}},
                copied=True,
            ),
            step(2, calls=[call(2, tool_call_id=selected_id)]),
        ]
    )

    trace = adapt(source)
    report = diagnostics(trace)

    assert len(trace.records) == 3
    assert counts(report.tool_call_disposition_counts)["ignored_copied_context"] == 1
    assert counts(report.result_disposition_counts)["ignored_copied_context"] == 1
    assert counts(report.tool_call_disposition_counts)["mapped_action"] == 1
