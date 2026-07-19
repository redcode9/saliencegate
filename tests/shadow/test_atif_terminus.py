from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import UUID

import pytest

from saliencegate.domain import canonical_json
from saliencegate.shadow.atif import ATIFProfile, ATIFShadowAdapter, ShadowEnvironmentBinding
from saliencegate.shadow.errors import ShadowTraceInputError
from saliencegate.shadow.trace import ATIFShadowDiagnostics, ShadowTrace

RUN_ID = UUID("33333333-3333-4333-8333-333333333333")
ENVIRONMENT_DIGEST = "b" * 64
DEFAULT_WORKING_DIRECTORY = "/synthetic/terminus-workspace"
FIXTURES = Path("tests/fixtures/shadow/atif")
TERMINUS_PROFILE = ATIFProfile.HARBOR_TERMINUS_2_V1
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
        profile=TERMINUS_PROFILE,
        environment=ShadowEnvironmentBinding(
            default_working_directory=DEFAULT_WORKING_DIRECTORY,
            environment_digest=ENVIRONMENT_DIGEST,
        ),
    )


def call(
    ordinal: int,
    *,
    function_name: str = "bash_command",
    arguments: object = _ABSENT,
    tool_call_id: str | None = None,
) -> dict[str, object]:
    if arguments is _ABSENT:
        arguments = {"keystrokes": f"printf synthetic-terminus-{ordinal}\n", "duration": 0.1}
    return {
        "tool_call_id": tool_call_id or f"synthetic-terminus-call-{ordinal}",
        "function_name": function_name,
        "arguments": arguments,
    }


def result(
    *, source_call_id: object = _ABSENT, content: str = "synthetic-output"
) -> dict[str, object]:
    value: dict[str, object] = {"content": content}
    if source_call_id is not _ABSENT:
        value["source_call_id"] = source_call_id
    return value


def step(
    ordinal: int,
    *,
    calls: list[dict[str, object]] | None = None,
    results: list[dict[str, object]] | None = None,
    timestamp: object = _ABSENT,
    copied: bool = False,
) -> dict[str, object]:
    value: dict[str, object] = {"step_id": ordinal, "source": "agent"}
    if calls is not None:
        value["tool_calls"] = calls
    if results is not None:
        value["observation"] = {"results": results}
    if timestamp is not _ABSENT:
        value["timestamp"] = timestamp
    if copied:
        value["is_copied_context"] = True
    return value


def source_bytes(
    steps: list[dict[str, object]],
    *,
    schema_version: str = "ATIF-v1.7",
    **root_overrides: object,
) -> bytes:
    root: dict[str, object] = {
        "schema_version": schema_version,
        "session_id": "synthetic-terminus-session",
        "agent": {"name": "terminus-2", "version": "synthetic"},
        "steps": steps,
        "continued_trajectory_ref": None,
        "subagent_trajectories": None,
    }
    root.update(root_overrides)
    return canonical_json(root)


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
    secret: str = "synthetic-secret-terminal-output",
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


def test_sanitized_timeout_fixture_maps_actions_but_never_terminal_text_outcomes() -> None:
    source = (FIXTURES / "terminus-timeout-sanitized.trajectory.json").read_bytes()

    trace = adapt(source)
    mapped = trace.records
    report = diagnostics(trace)

    assert trace.binding.source_format == "atif"
    assert trace.binding.source_schema_version == "ATIF-v1.6"
    assert trace.binding.source_digest_kind == "original_bytes"
    assert trace.binding.source_byte_count == len(source)
    assert trace.binding.source_byte_digest == hashlib.sha256(source).hexdigest()
    assert trace.binding.adapter_profile_id == TERMINUS_PROFILE.value
    assert trace.binding.timestamp_mode == "logical_order"
    assert trace.binding.capture_scope == "selected_events"
    assert [record["kind"] for record in mapped] == [
        "run_start",
        "action",
        "action",
        "action",
        "run_end",
    ]
    assert [record["command"] for record in mapped[1:-1]] == [
        "printf synthetic-timeout-one",
        "printf synthetic-timeout-two",
        "printf synthetic-timeout-three",
    ]
    assert [record["occurred_at"] for record in mapped] == [
        "2000-01-01T00:00:00.000000Z",
        "2000-01-01T00:00:00.000001Z",
        "2000-01-01T00:00:00.000002Z",
        "2000-01-01T00:00:00.000003Z",
        "2000-01-01T00:00:00.000004Z",
    ]
    assert all(record["working_directory"] == DEFAULT_WORKING_DIRECTORY for record in mapped[1:-1])
    action_environment_digests = {record["environment_digest"] for record in mapped[1:-1]}
    assert len(action_environment_digests) == 1
    assert ENVIRONMENT_DIGEST not in action_environment_digests

    assert report.total_step_count == 4
    assert report.ignored_message_step_count == 1
    assert report.total_tool_call_count == 3
    assert counts(report.tool_call_disposition_counts) == {
        "mapped_action": 3,
        "ignored_unsupported_function": 0,
        "ignored_continuation": 0,
        "ignored_non_command_wait": 0,
        "ignored_unsubmitted_keystrokes": 0,
        "ignored_unresolved_terminal_submission": 0,
        "ignored_copied_context": 0,
    }
    assert report.total_observation_result_count == 3
    assert counts(report.result_disposition_counts) == {
        "mapped_structured_outcome": 0,
        "ignored_evidence_absent": 3,
        "ignored_ambiguous_parent": 0,
        "ignored_no_parent": 0,
        "ignored_unsupported_parent": 0,
        "ignored_copied_context": 0,
    }
    assert report.mapped_shadow_record_count == 5
    assert report.outcome_evidence_authority == "none"


def test_sanitized_context_fixture_counts_root_coverage_without_opening_nested_refs() -> None:
    source = (FIXTURES / "terminus-context-sanitized.trajectory.json").read_bytes()

    trace = adapt(source)
    report = diagnostics(trace)

    assert [record["kind"] for record in trace.records] == [
        "run_start",
        "action",
        "action",
        "action",
        "action",
        "action",
        "run_end",
    ]
    assert report.root_segment_only is True
    assert report.continued_trajectory_ref_present is False
    assert report.embedded_subagent_trajectory_count == 0
    assert report.total_step_count == 10
    assert report.ignored_message_step_count == 3
    assert report.total_tool_call_count == 7
    assert counts(report.tool_call_disposition_counts) == {
        "mapped_action": 5,
        "ignored_unsupported_function": 2,
        "ignored_continuation": 0,
        "ignored_non_command_wait": 0,
        "ignored_unsubmitted_keystrokes": 0,
        "ignored_unresolved_terminal_submission": 0,
        "ignored_copied_context": 0,
    }
    assert report.total_observation_result_count == 8
    assert counts(report.result_disposition_counts) == {
        "mapped_structured_outcome": 0,
        "ignored_evidence_absent": 5,
        "ignored_ambiguous_parent": 0,
        "ignored_no_parent": 1,
        "ignored_unsupported_parent": 2,
        "ignored_copied_context": 0,
    }
    assert report.mapped_shadow_record_count == 7


@pytest.mark.parametrize("schema_version", ("ATIF-v1.6", "ATIF-v1.7"))
def test_terminus_profile_accepts_all_pinned_schema_versions_with_identical_mapping(
    schema_version: str,
) -> None:
    selected_step = step(1, calls=[call(1)])
    baseline = adapt(source_bytes([selected_step], schema_version="ATIF-v1.6"))

    candidate = adapt(source_bytes([selected_step], schema_version=schema_version))

    assert candidate.binding.source_schema_version == schema_version
    assert candidate.records == baseline.records
    assert candidate.mapped_record_digest == baseline.mapped_record_digest


def test_terminus_profile_rejects_unfixtureed_v15_schema() -> None:
    source = source_bytes([step(1, calls=[call(1)])], schema_version="ATIF-v1.5")

    assert_error(source, "unsupported_schema")


def test_terminus_classifies_waits_unsubmitted_and_unresolved_terminal_state() -> None:
    calls = [
        call(1, function_name="mark_task_complete", arguments={}),
        call(2, arguments={"keystrokes": "", "duration": 0}),
        call(3, arguments={"keystrokes": "typed-but-not-submitted", "duration": 1}),
        call(4, arguments={"keystrokes": "\n", "duration": 1}),
        call(5, arguments={"keystrokes": " \t\n", "duration": 1}),
        call(6, arguments={"keystrokes": "printf one\nprintf two\n", "duration": 1}),
    ]

    trace = adapt(source_bytes([step(1, calls=calls)]))
    report = diagnostics(trace)

    assert [record["kind"] for record in trace.records] == ["run_start", "action", "run_end"]
    assert trace.records[1]["source_event_id"] == "atif-s00000001-c0006-action"
    assert trace.records[1]["command"] == "printf one\nprintf two"
    assert counts(report.tool_call_disposition_counts) == {
        "mapped_action": 1,
        "ignored_unsupported_function": 1,
        "ignored_continuation": 0,
        "ignored_non_command_wait": 1,
        "ignored_unsubmitted_keystrokes": 1,
        "ignored_unresolved_terminal_submission": 2,
        "ignored_copied_context": 0,
    }


@pytest.mark.parametrize(
    "arguments",
    (
        None,
        {},
        {"keystrokes": 1},
        {"keystrokes": "echo submitted\n", "duration": True},
        {"keystrokes": "echo submitted\n", "duration": None},
        {"keystrokes": "echo submitted\n", "duration": -0.1},
        {"keystrokes": "echo submitted\n", "duration": "1"},
    ),
)
def test_malformed_selected_terminus_calls_fail_with_coordinates(arguments: object) -> None:
    source = source_bytes(
        [
            step(
                1,
                calls=[
                    call(
                        1,
                        arguments=arguments,
                        tool_call_id="synthetic-secret-terminal-output",
                    )
                ],
            )
        ]
    )

    assert_error(source, "invalid_tool_call", step_ordinal=1, call_ordinal=1)


def test_duplicate_selected_call_ids_fail_without_echoing_the_identifier() -> None:
    duplicate = "synthetic-secret-terminal-output"
    source = source_bytes(
        [
            step(
                1,
                calls=[
                    call(1, tool_call_id=duplicate),
                    call(2, tool_call_id=duplicate),
                ],
            )
        ]
    )

    assert_error(
        source,
        "duplicate_tool_call_id",
        step_ordinal=1,
        call_ordinal=2,
        secret=duplicate,
    )


def test_terminus_result_linking_is_step_local_exhaustive_and_conservative() -> None:
    source = source_bytes(
        [
            step(
                1,
                calls=[call(1, tool_call_id="mapped-1")],
                results=[result(source_call_id="mapped-1")],
            ),
            step(
                2,
                calls=[
                    call(2, tool_call_id="mapped-2"),
                    call(3, function_name="unknown", arguments={}, tool_call_id="unsupported-2"),
                ],
                results=[result()],
            ),
            step(3, results=[result()]),
            step(
                4,
                calls=[
                    call(4, function_name="unknown", arguments={}, tool_call_id="unsupported-4")
                ],
                results=[result(source_call_id="unsupported-4")],
            ),
        ]
    )

    trace = adapt(source)
    report = diagnostics(trace)

    assert [record["kind"] for record in trace.records] == [
        "run_start",
        "action",
        "action",
        "run_end",
    ]
    assert report.total_tool_call_count == 4
    assert report.total_observation_result_count == 4
    assert counts(report.result_disposition_counts) == {
        "mapped_structured_outcome": 0,
        "ignored_evidence_absent": 1,
        "ignored_ambiguous_parent": 1,
        "ignored_no_parent": 1,
        "ignored_unsupported_parent": 1,
        "ignored_copied_context": 0,
    }


def test_results_linked_to_nonmapped_terminal_operations_are_unsupported_parent() -> None:
    source = source_bytes(
        [
            step(1, calls=[call(1, tool_call_id="selected")]),
            step(
                2,
                calls=[
                    call(
                        2,
                        arguments={"keystrokes": "", "duration": 0},
                        tool_call_id="wait",
                    )
                ],
                results=[result(source_call_id="wait")],
            ),
            step(
                3,
                calls=[
                    call(
                        3,
                        arguments={"keystrokes": "typed", "duration": 0},
                        tool_call_id="unsubmitted",
                    )
                ],
                results=[result(source_call_id="unsubmitted")],
            ),
            step(
                4,
                calls=[
                    call(
                        4,
                        arguments={"keystrokes": "\n", "duration": 0},
                        tool_call_id="unresolved",
                    )
                ],
                results=[result(source_call_id="unresolved")],
            ),
        ]
    )

    report = diagnostics(adapt(source))

    assert counts(report.result_disposition_counts)["ignored_unsupported_parent"] == 3
    assert report.mapped_shadow_record_count == 3


@pytest.mark.parametrize("source_call_id", (_ABSENT, None))
def test_absent_and_null_result_parent_use_the_exact_one_call_one_result_rule(
    source_call_id: object,
) -> None:
    selected_id = "selected"
    source = source_bytes(
        [
            step(
                1,
                calls=[call(1, tool_call_id=selected_id)],
                results=[result(source_call_id=source_call_id)],
            )
        ]
    )

    report = diagnostics(adapt(source))

    assert counts(report.result_disposition_counts)["ignored_evidence_absent"] == 1


@pytest.mark.parametrize("duration", (_ABSENT, 0, 1, 0.25))
def test_admissible_duration_is_source_only_and_does_not_change_action_mapping(
    duration: object,
) -> None:
    baseline = adapt(
        source_bytes(
            [
                step(
                    1,
                    calls=[call(1, arguments={"keystrokes": "printf stable\n"})],
                )
            ]
        )
    )
    arguments: dict[str, object] = {"keystrokes": "printf stable\n"}
    if duration is not _ABSENT:
        arguments["duration"] = duration

    trace = adapt(source_bytes([step(1, calls=[call(1, arguments=arguments)])]))

    assert trace.records == baseline.records
    assert trace.mapped_record_digest == baseline.mapped_record_digest


def test_caller_attested_environment_changes_terminal_context_digest() -> None:
    source = source_bytes([step(1, calls=[call(1)])])
    baseline = adapt(source)
    changed_adapter = ATIFShadowAdapter(
        profile=TERMINUS_PROFILE,
        environment=ShadowEnvironmentBinding(
            default_working_directory=DEFAULT_WORKING_DIRECTORY,
            environment_digest="d" * 64,
        ),
    )

    changed = changed_adapter.adapt_bytes(source, run_id=RUN_ID)

    assert baseline.records[1]["command"] == changed.records[1]["command"]
    assert baseline.records[1]["working_directory"] == changed.records[1]["working_directory"]
    assert baseline.records[1]["environment_digest"] != changed.records[1]["environment_digest"]


def test_nonempty_orphan_result_reference_is_an_error_with_result_coordinates() -> None:
    orphan = "synthetic-secret-terminal-output"
    source = source_bytes(
        [
            step(
                1,
                calls=[call(1, tool_call_id="real-call")],
                results=[result(source_call_id=orphan, content=orphan)],
            )
        ]
    )

    assert_error(
        source,
        "orphan_result",
        step_ordinal=1,
        result_ordinal=1,
        secret=orphan,
    )


def test_copied_context_is_counted_but_never_mapped() -> None:
    source = source_bytes(
        [
            step(
                1,
                calls=[call(1, tool_call_id="copied")],
                results=[result(source_call_id="copied")],
                timestamp="2026-07-17T10:00:00Z",
                copied=True,
            ),
            step(2, calls=[call(2, tool_call_id="selected")]),
        ]
    )

    trace = adapt(source)
    report = diagnostics(trace)

    assert trace.binding.timestamp_mode == "logical_order"
    assert counts(report.tool_call_disposition_counts)["ignored_copied_context"] == 1
    assert counts(report.result_disposition_counts)["ignored_copied_context"] == 1
    assert counts(report.tool_call_disposition_counts)["mapped_action"] == 1
    assert len(trace.records) == 3


@pytest.mark.parametrize(
    ("calls", "results", "reason_code", "call_ordinal", "result_ordinal"),
    (
        ([None], [], "invalid_tool_call", 1, None),
        ([], [1], "invalid_step", None, 1),
    ),
)
def test_copied_context_still_requires_atif_container_elements(
    calls: list[object],
    results: list[object],
    reason_code: str,
    call_ordinal: int | None,
    result_ordinal: int | None,
) -> None:
    source = source_bytes(
        [
            step(1, calls=calls, results=results, copied=True),
            step(2, calls=[call(2)]),
        ]
    )

    assert_error(
        source,
        reason_code,
        step_ordinal=1,
        call_ordinal=call_ordinal,
        result_ordinal=result_ordinal,
    )


def test_root_continuation_and_immediate_subagents_are_reported_but_not_traversed() -> None:
    embedded = {
        "schema_version": "ATIF-v1.7",
        "agent": {"name": "terminus-2"},
        "steps": [
            step(
                1,
                calls=[call(10, tool_call_id="embedded-call")],
                results=[result(source_call_id="embedded-call")],
            )
        ],
    }
    source = source_bytes(
        [step(1, calls=[call(1, tool_call_id="root-call")])],
        continued_trajectory_ref={"trajectory_path": "private-continuation.json"},
        subagent_trajectories=[embedded, {"opaque": True}],
    )

    report = diagnostics(adapt(source))

    assert report.continued_trajectory_ref_present is True
    assert report.embedded_subagent_trajectory_count == 2
    assert report.total_step_count == 1
    assert report.total_tool_call_count == 1
    assert report.total_observation_result_count == 0


def test_complete_selected_timestamps_use_normalized_source_utc() -> None:
    source = source_bytes(
        [
            step(1, calls=[call(1)], timestamp="2026-07-17T10:00:00.123Z"),
            step(2, calls=[call(2)], timestamp="2026-07-17T10:00:01Z"),
        ]
    )

    trace = adapt(source)

    assert trace.binding.timestamp_mode == "source_utc"
    assert [record["occurred_at"] for record in trace.records] == [
        "2026-07-17T10:00:00.123000Z",
        "2026-07-17T10:00:00.123000Z",
        "2026-07-17T10:00:01Z",
        "2026-07-17T10:00:01Z",
    ]


@pytest.mark.parametrize("timestamp", ("2026-07-17T10:00:00.0Z", "2026-07-17T10:00:00.000000Z"))
def test_zero_fraction_source_timestamps_are_accepted_and_padded(timestamp: str) -> None:
    trace = adapt(source_bytes([step(1, calls=[call(1)], timestamp=timestamp)]))

    assert trace.binding.timestamp_mode == "source_utc"
    assert [record["occurred_at"] for record in trace.records] == [
        "2026-07-17T10:00:00.000000Z",
        "2026-07-17T10:00:00.000000Z",
        "2026-07-17T10:00:00.000000Z",
    ]


def test_absent_selected_timestamps_use_strict_logical_record_order() -> None:
    trace = adapt(source_bytes([step(1, calls=[call(1)]), step(2, calls=[call(2)])]))

    assert trace.binding.timestamp_mode == "logical_order"
    assert [record["occurred_at"] for record in trace.records] == [
        "2000-01-01T00:00:00.000000Z",
        "2000-01-01T00:00:00.000001Z",
        "2000-01-01T00:00:00.000002Z",
        "2000-01-01T00:00:00.000003Z",
    ]


def test_partial_selected_timestamps_fail_without_requiring_an_optional_coordinate() -> None:
    source = source_bytes(
        [
            step(1, calls=[call(1)], timestamp="2026-07-17T10:00:00Z"),
            step(2, calls=[call(2)]),
        ]
    )

    assert_error(source, "partial_timestamps")


@pytest.mark.parametrize(
    "timestamp",
    (
        "2026-07-17T10:00:00.1234567Z",
        "2026-07-17T10:00:60Z",
    ),
)
def test_invalid_selected_timestamp_spelling_fails(timestamp: object) -> None:
    source = source_bytes([step(1, calls=[call(1)], timestamp=timestamp)])

    assert_error(source, "invalid_timestamp", step_ordinal=1)


def test_null_selected_timestamps_are_treated_as_absent() -> None:
    trace = adapt(
        source_bytes(
            [
                step(1, calls=[call(1)], timestamp=None),
                step(2, calls=[call(2)], timestamp=None),
            ]
        )
    )

    assert trace.binding.timestamp_mode == "logical_order"
    assert [record["occurred_at"] for record in trace.records] == [
        "2000-01-01T00:00:00.000000Z",
        "2000-01-01T00:00:00.000001Z",
        "2000-01-01T00:00:00.000002Z",
        "2000-01-01T00:00:00.000003Z",
    ]


def test_decreasing_selected_timestamps_fail_at_the_reversed_step() -> None:
    source = source_bytes(
        [
            step(1, calls=[call(1)], timestamp="2026-07-17T10:00:01Z"),
            step(2, calls=[call(2)], timestamp="2026-07-17T10:00:00Z"),
        ]
    )

    assert_error(source, "invalid_timestamp", step_ordinal=2)


def test_unsupported_step_timestamps_do_not_select_or_poison_timestamp_mode() -> None:
    source = source_bytes(
        [
            step(1, calls=[call(1)]),
            step(
                2,
                calls=[call(2, function_name="mark_task_complete", arguments={})],
                timestamp="not-a-timestamp",
            ),
        ]
    )

    trace = adapt(source)

    assert trace.binding.timestamp_mode == "logical_order"
    assert len(trace.records) == 3
