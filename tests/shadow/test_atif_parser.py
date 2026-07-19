from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from uuid import UUID

import pytest
from pydantic import ValidationError

import saliencegate.shadow.atif as atif_module
from saliencegate.domain import canonical_json, length_prefixed_sha256
from saliencegate.shadow.atif import ATIFProfile, ATIFShadowAdapter, ShadowEnvironmentBinding
from saliencegate.shadow.errors import ShadowTraceInputError
from saliencegate.shadow.trace import MAX_SHADOW_TRACE_BYTES, ATIFShadowDiagnostics, ShadowTrace
from saliencegate.shadow.trace_report import _sealed_atif_report_contract

RUN_ID = UUID("22222222-2222-4222-8222-222222222222")
ENVIRONMENT_DIGEST = "a" * 64
DEFAULT_WORKING_DIRECTORY = "/synthetic/workspace"
MAX_ATIF_STRING_BYTES = 2 * 1_024 * 1_024
EXPECTED_PROFILE_DIGESTS = {
    ATIFProfile.HARBOR_TERMINUS_2_V1: (
        "a590e2232ec7957b31234c7ab6c9392e371285cad0e28a4c751d0c78833e70df"
    ),
    ATIFProfile.HARBOR_CODEX_V1: (
        "64f8404359b3630b48780e53f2855d8e692bcfb140e577e2af707c84832150f6"
    ),
}
EXPECTED_MANIFEST_DIGEST = "b12cdabc7a25644efb5da81aec3f8036f280e5d638e6b7aa07bb6593893967f2"
EXPECTED_ATIF_FIXTURE_SHA256 = {
    "tests/fixtures/shadow/atif/codex-bundled-synthetic.trajectory.json": (
        "9ee5263186de96695f3f80c50caf9073dc9a4010f10227119fc22ed4ded58b81"
    ),
    "tests/fixtures/shadow/atif/terminus-context-sanitized.trajectory.json": (
        "fc626621b45e91a3d7e4b0463d52d0cabd83b0f08e03305d28ebdca369e5ec99"
    ),
    "tests/fixtures/shadow/atif/terminus-timeout-sanitized.trajectory.json": (
        "ed5b96a240458160f5ddd66c933c77752ea83cbf445c061f1626e86d9c89881c"
    ),
}


def environment() -> ShadowEnvironmentBinding:
    return ShadowEnvironmentBinding(
        default_working_directory=DEFAULT_WORKING_DIRECTORY,
        environment_digest=ENVIRONMENT_DIGEST,
    )


def adapter(profile: ATIFProfile = ATIFProfile.HARBOR_CODEX_V1) -> ATIFShadowAdapter:
    return ATIFShadowAdapter(profile=profile, environment=environment())


def trajectory_bytes(
    *,
    profile: ATIFProfile = ATIFProfile.HARBOR_CODEX_V1,
    steps: list[dict[str, object]] | None = None,
    schema_version: str | None = None,
    agent_name: str | None = None,
    **root_overrides: object,
) -> bytes:
    is_codex = profile is ATIFProfile.HARBOR_CODEX_V1
    if steps is None:
        arguments: dict[str, object]
        function_name: str
        if is_codex:
            function_name = "exec_command"
            arguments = {"cmd": "printf synthetic-parser-command"}
        else:
            function_name = "bash_command"
            arguments = {"keystrokes": "printf synthetic-parser-command\n", "duration": 0.1}
        steps = [
            {
                "step_id": 1,
                "source": "agent",
                "tool_calls": [
                    {
                        "tool_call_id": "synthetic-parser-call",
                        "function_name": function_name,
                        "arguments": arguments,
                    }
                ],
            }
        ]
    root: dict[str, object] = {
        "schema_version": schema_version or ("ATIF-v1.7" if is_codex else "ATIF-v1.6"),
        "session_id": "synthetic-parser-session",
        "agent": {"name": agent_name or ("codex" if is_codex else "terminus-2")},
        "steps": steps,
        "continued_trajectory_ref": None,
        "subagent_trajectories": None,
    }
    root.update(root_overrides)
    return canonical_json(root)


def adapt(
    source: bytes,
    *,
    profile: ATIFProfile = ATIFProfile.HARBOR_CODEX_V1,
) -> ShadowTrace:
    return adapter(profile).adapt_bytes(source, run_id=RUN_ID)


def assert_trace_error(
    source: object,
    reason_code: str,
    *,
    profile: ATIFProfile = ATIFProfile.HARBOR_CODEX_V1,
    step_ordinal: int | None = None,
    call_ordinal: int | None = None,
    result_ordinal: int | None = None,
    secret: str = "synthetic-parser-secret",
) -> ShadowTraceInputError:
    with pytest.raises(ShadowTraceInputError) as captured:
        adapter(profile).adapt_bytes(source, run_id=RUN_ID)  # type: ignore[arg-type]
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
    return error


def _replace_atif_configuration_and_rehash_binding(
    trace: ShadowTrace,
    configuration: dict[str, object],
) -> None:
    configuration_bytes = canonical_json(configuration)
    configuration_digest = length_prefixed_sha256(
        configuration_bytes,
        domain="saliencegate:shadow:adapter-configuration:v1",
    )
    binding_body = trace.binding.model_dump(
        mode="json",
        exclude={"binding_digest"},
        warnings=False,
    )
    binding_body["adapter_configuration_digest"] = configuration_digest
    binding_body["source_adapter"] = (
        f"{binding_body['source_format']}.{binding_body['adapter_profile_id']}"
        f"+p.{binding_body['adapter_profile_digest']}+c.{configuration_digest}"
    )
    binding_digest = length_prefixed_sha256(
        canonical_json(binding_body),
        domain="saliencegate:shadow:trace-binding:v1",
    )
    trace.binding.__dict__.update(
        binding_body,
        binding_digest=binding_digest,
    )
    object.__setattr__(trace, "_adapter_configuration_bytes", configuration_bytes)
    object.__setattr__(trace, "_binding_bytes", canonical_json(trace.binding))


def _replace_records_and_rehash(
    trace: ShadowTrace,
    records: list[dict[str, object]],
) -> None:
    record_bytes = tuple(canonical_json(record) for record in records)
    object.__setattr__(
        trace,
        "_records",
        tuple(MappingProxyType(record) for record in records),
    )
    object.__setattr__(trace, "_record_bytes", record_bytes)
    object.__setattr__(
        trace,
        "mapped_record_digest",
        length_prefixed_sha256(
            *record_bytes,
            domain="saliencegate:shadow:mapped-records:v1",
        ),
    )


def _nested_value(depth: int) -> object:
    value: object = "leaf"
    for _ in range(depth):
        value = [value]
    return value


def _selected_steps(count: int) -> list[dict[str, object]]:
    return [
        {
            "step_id": ordinal,
            "source": "agent",
            "tool_calls": [
                {
                    "tool_call_id": f"synthetic-call-{ordinal}",
                    "function_name": "exec_command",
                    "arguments": {"cmd": "true"},
                }
            ],
        }
        for ordinal in range(1, count + 1)
    ]


def test_atif_module_exports_only_the_supported_surface() -> None:
    assert atif_module.__all__ == [
        "ATIFProfile",
        "ATIFShadowAdapter",
        "ShadowEnvironmentBinding",
    ]
    assert tuple(ATIFProfile) == (
        ATIFProfile.HARBOR_TERMINUS_2_V1,
        ATIFProfile.HARBOR_CODEX_V1,
    )
    assert tuple(profile.value for profile in ATIFProfile) == (
        "harbor-terminus-2/v1",
        "harbor-codex/v1",
    )


def test_environment_and_adapter_are_exact_frozen_content_free_contracts() -> None:
    binding = environment()
    selected = ATIFShadowAdapter(
        profile=ATIFProfile.HARBOR_CODEX_V1,
        environment=binding,
    )

    assert binding.schema_version == "shadow-environment-binding/v1"
    assert repr(binding) == "ShadowEnvironmentBinding(<redacted>)"
    assert repr(selected) == "ATIFShadowAdapter(<configured>)"
    assert DEFAULT_WORKING_DIRECTORY not in repr(binding)
    assert DEFAULT_WORKING_DIRECTORY not in repr(selected)
    assert ENVIRONMENT_DIGEST not in repr(binding)
    assert ENVIRONMENT_DIGEST not in repr(selected)

    with pytest.raises((TypeError, ValidationError)):
        binding.default_working_directory = "/changed"  # type: ignore[misc]

    with pytest.raises(TypeError):

        class SubclassedAdapter(ATIFShadowAdapter):
            pass

    with pytest.raises(TypeError):

        class SubclassedEnvironment(ShadowEnvironmentBinding):
            pass

    with pytest.raises((TypeError, ValidationError)):
        ATIFShadowAdapter(
            profile=ATIFProfile.HARBOR_CODEX_V1.value,  # type: ignore[arg-type]
            environment=binding,
        )


@pytest.mark.parametrize(
    ("overrides", "secret"),
    (
        ({"default_working_directory": ""}, ""),
        ({"default_working_directory": "synthetic-secret\x00path"}, "synthetic-secret"),
        ({"default_working_directory": b"/synthetic/bytes"}, "synthetic"),
        ({"environment_digest": "A" * 64}, "A" * 16),
        ({"environment_digest": "0" * 63}, "0" * 16),
    ),
)
def test_environment_rejects_invalid_exact_values_without_echoing_them(
    overrides: dict[str, object],
    secret: str,
) -> None:
    values: dict[str, object] = {
        "default_working_directory": DEFAULT_WORKING_DIRECTORY,
        "environment_digest": ENVIRONMENT_DIGEST,
    }
    values.update(overrides)

    with pytest.raises(ValidationError) as captured:
        ShadowEnvironmentBinding(**values)  # type: ignore[arg-type]

    if secret:
        assert secret not in str(captured.value)


def test_adapter_snapshots_the_environment_at_construction() -> None:
    supplied = environment()
    selected = ATIFShadowAdapter(
        profile=ATIFProfile.HARBOR_CODEX_V1,
        environment=supplied,
    )
    object.__setattr__(supplied, "default_working_directory", "/mutated-after-construction")

    trace = selected.adapt_bytes(trajectory_bytes(), run_id=RUN_ID)
    action = trace.records[1]

    assert action["working_directory"] == DEFAULT_WORKING_DIRECTORY
    assert action["working_directory"] != supplied.default_working_directory


@pytest.mark.parametrize(
    "source",
    (
        bytearray(b"{}"),
        memoryview(b"{}"),
        "{}",
        None,
    ),
)
def test_parser_accepts_only_exact_bytes(source: object) -> None:
    assert_trace_error(source, "invalid_json")


def test_parser_rejects_empty_and_over_limit_sources_before_decoding() -> None:
    assert_trace_error(b"", "input_limit_exceeded")
    assert_trace_error(b" " * (MAX_SHADOW_TRACE_BYTES + 1), "input_limit_exceeded")


def test_parser_accepts_a_valid_source_at_the_exact_byte_limit() -> None:
    compact = trajectory_bytes()
    source = compact + (b" " * (MAX_SHADOW_TRACE_BYTES - len(compact)))
    baseline = adapt(compact)

    trace = adapt(source)

    assert trace.binding.source_byte_count == MAX_SHADOW_TRACE_BYTES
    assert trace.binding.source_byte_digest != baseline.binding.source_byte_digest
    assert trace.mapped_record_digest == baseline.mapped_record_digest


@pytest.mark.parametrize(
    "source",
    (
        b"\xef\xbb\xbf{}",
        b"\xff",
        b'{"schema_version":"ATIF-v1.7","schema_version":"ATIF-v1.7"}',
        b'{"schema_version":"ATIF-v1.7","agent":{"name":"codex","name":"codex"}}',
        b'{"schema_version":"ATIF-v1.7","agent":{"name":"codex"},"steps":NaN}',
        b'{"schema_version":"ATIF-v1.7","agent":{"name":"codex"},"steps":Infinity}',
        b'{"schema_version":"ATIF-v1.7","agent":{"name":"codex"},"steps":-Infinity}',
        (
            b'{"schema_version":"ATIF-v1.7","agent":{"name":"codex"},"steps":'
            b'[{"step_id":1,"tool_calls":[{"tool_call_id":"x",'
            b'"function_name":"exec_command","arguments":{"cmd":"true","cmd":"true"}}]}]}'
        ),
        (
            b'{"schema_version":"ATIF-v1.7","agent":{"name":"codex"},"steps":'
            b'[{"step_id":1,"extra":{"nested":{"x":1,"x":2}}}]}'
        ),
        b'{"schema_version":"ATIF-v1.7","agent":{"name":"codex"},"steps":',
    ),
)
def test_parser_rejects_non_strict_json(source: bytes) -> None:
    assert_trace_error(source, "invalid_json")


@pytest.mark.parametrize(
    ("source", "reason_code", "step_ordinal"),
    (
        (canonical_json([]), "invalid_json", None),
        (
            trajectory_bytes(schema_version="ATIF-v1.6"),
            "unsupported_schema",
            None,
        ),
        (
            trajectory_bytes(agent_name="Codex"),
            "profile_mismatch",
            None,
        ),
        (
            trajectory_bytes(steps=[]),
            "invalid_step",
            None,
        ),
        (
            b'{"schema_version":"ATIF-v1.7","agent":{"name":"codex"},"steps":1e309}',
            "invalid_step",
            None,
        ),
        (
            trajectory_bytes(steps=[{"step_id": True, "source": "user"}]),
            "invalid_step",
            1,
        ),
        (
            trajectory_bytes(steps=[{"step_id": 2, "source": "user"}]),
            "invalid_step",
            1,
        ),
    ),
)
def test_parser_rejects_invalid_root_profile_and_step_shapes(
    source: bytes,
    reason_code: str,
    step_ordinal: int | None,
) -> None:
    assert_trace_error(source, reason_code, step_ordinal=step_ordinal)


@pytest.mark.parametrize(
    "root_overrides",
    (
        {"unconsumed": _nested_value(40)},
        {"unconsumed": "x" * (MAX_ATIF_STRING_BYTES + 1)},
        {"unconsumed": list(range(100_001))},
        {"unconsumed": {f"k{index}": None for index in range(4_097)}},
    ),
)
def test_parser_enforces_generic_structural_limits(root_overrides: dict[str, object]) -> None:
    assert_trace_error(
        trajectory_bytes(**root_overrides),
        "input_limit_exceeded",
    )


@pytest.mark.parametrize(
    "root_overrides",
    (
        {"unconsumed": _nested_value(31)},
        {"unconsumed": "x" * MAX_ATIF_STRING_BYTES},
        {"unconsumed": list(range(100_000))},
        {"unconsumed": {f"k{index}": None for index in range(4_096)}},
    ),
)
def test_parser_accepts_exact_generic_structural_boundaries(
    root_overrides: dict[str, object],
) -> None:
    trace = adapt(trajectory_bytes(**root_overrides))

    assert len(trace.records) == 3


def test_parser_enforces_the_total_decoded_node_limit() -> None:
    million_scalar_nodes = [[None] * 100_000 for _ in range(10)]

    assert_trace_error(
        trajectory_bytes(unconsumed=million_scalar_nodes),
        "input_limit_exceeded",
    )


def test_parser_enforces_step_and_per_step_tool_call_limits() -> None:
    too_many_steps = [{"step_id": ordinal, "source": "user"} for ordinal in range(1, 10_002)]
    assert_trace_error(trajectory_bytes(steps=too_many_steps), "input_limit_exceeded")

    calls = [
        {
            "tool_call_id": f"unsupported-{ordinal}",
            "function_name": "unsupported",
            "arguments": {},
        }
        for ordinal in range(1, 1_026)
    ]
    step = {"step_id": 1, "source": "agent", "tool_calls": calls}
    assert_trace_error(
        trajectory_bytes(steps=[step]),
        "input_limit_exceeded",
        step_ordinal=1,
    )

    results = [{"content": None} for _ in range(1_025)]
    result_step = {
        "step_id": 1,
        "source": "agent",
        "observation": {"results": results},
    }
    assert_trace_error(
        trajectory_bytes(steps=[result_step]),
        "input_limit_exceeded",
        step_ordinal=1,
    )


def test_parser_accepts_exact_per_step_call_and_result_boundaries() -> None:
    calls = [
        {
            "tool_call_id": f"call-{ordinal}",
            "function_name": "exec_command" if ordinal == 1 else "unsupported",
            "arguments": {"cmd": "true"} if ordinal == 1 else {},
        }
        for ordinal in range(1, 1_025)
    ]
    call_trace = adapt(
        trajectory_bytes(steps=[{"step_id": 1, "source": "agent", "tool_calls": calls}])
    )
    call_report = call_trace.diagnostics
    assert type(call_report) is ATIFShadowDiagnostics
    assert call_report.total_tool_call_count == 1_024
    assert dict(call_report.tool_call_disposition_counts)["mapped_action"] == 1
    assert dict(call_report.tool_call_disposition_counts)["ignored_unsupported_function"] == 1_023

    results = [{"content": None} for _ in range(1_024)]
    result_trace = adapt(
        trajectory_bytes(
            steps=[
                {
                    "step_id": 1,
                    "source": "agent",
                    "tool_calls": [
                        {
                            "tool_call_id": "selected",
                            "function_name": "exec_command",
                            "arguments": {"cmd": "true"},
                        }
                    ],
                    "observation": {"results": results},
                }
            ]
        )
    )
    result_report = result_trace.diagnostics
    assert type(result_report) is ATIFShadowDiagnostics
    assert result_report.total_observation_result_count == 1_024
    assert dict(result_report.result_disposition_counts)["ignored_ambiguous_parent"] == 1_024


@pytest.mark.parametrize("kind", ("calls", "results"))
@pytest.mark.parametrize(("total", "should_succeed"), ((10_000, True), (10_001, False)))
def test_parser_enforces_root_tool_call_and_result_totals(
    kind: str,
    total: int,
    should_succeed: bool,
) -> None:
    steps: list[dict[str, object]] = []
    item_ordinal = 0
    step_ordinal = 0
    while item_ordinal < total:
        step_ordinal += 1
        item_count = min(1_000, total - item_ordinal)
        items: list[dict[str, object]] = []
        for _ in range(item_count):
            item_ordinal += 1
            if kind == "calls":
                items.append(
                    {
                        "tool_call_id": f"unsupported-{item_ordinal}",
                        "function_name": ("exec_command" if item_ordinal == 1 else "unsupported"),
                        "arguments": {"cmd": "true"} if item_ordinal == 1 else {},
                    }
                )
            else:
                items.append({"content": None})
        current: dict[str, object] = {"step_id": step_ordinal, "source": "agent"}
        if kind == "calls":
            current["tool_calls"] = items
        else:
            if step_ordinal == 1:
                current["tool_calls"] = [
                    {
                        "tool_call_id": "selected",
                        "function_name": "exec_command",
                        "arguments": {"cmd": "true"},
                    }
                ]
            current["observation"] = {"results": items}
        steps.append(current)

    source = trajectory_bytes(steps=steps)
    if should_succeed:
        trace = adapt(source)
        report = trace.diagnostics
        assert type(report) is ATIFShadowDiagnostics
        expected_call_count = total if kind == "calls" else 1
        assert report.total_tool_call_count == expected_call_count
        assert report.total_observation_result_count == (total if kind == "results" else 0)
    else:
        assert_trace_error(
            source,
            "input_limit_exceeded",
            step_ordinal=11,
        )


@pytest.mark.parametrize(
    ("selected_count", "should_succeed"),
    ((998, True), (999, False)),
)
def test_parser_enforces_the_mapped_expansion_limit(
    selected_count: int,
    should_succeed: bool,
) -> None:
    source = trajectory_bytes(steps=_selected_steps(selected_count))

    if should_succeed:
        trace = adapt(source)
        assert len(trace.records) == 1_000
    else:
        assert_trace_error(source, "input_limit_exceeded")


def test_parser_requires_at_least_one_supported_action() -> None:
    source = trajectory_bytes(steps=[{"step_id": 1, "source": "user", "message": "ignored"}])

    assert_trace_error(source, "no_supported_action")


def test_copied_context_marker_requires_an_exact_boolean() -> None:
    steps = _selected_steps(1)
    steps[0]["is_copied_context"] = 1

    assert_trace_error(
        trajectory_bytes(steps=steps),
        "invalid_step",
        step_ordinal=1,
    )


def test_absent_null_and_bounded_session_ids_do_not_change_mapping_semantics() -> None:
    base = json.loads(trajectory_bytes())
    assert type(base) is dict
    absent = dict(base)
    absent.pop("session_id")
    explicit_null = {**base, "session_id": None}
    bounded = {**base, "session_id": "another-synthetic-session"}

    traces = tuple(adapt(canonical_json(value)) for value in (absent, explicit_null, bounded))

    assert len({trace.mapped_record_digest for trace in traces}) == 1
    assert len({trace.binding.adapter_configuration_digest for trace in traces}) == 1
    assert len({trace.binding.source_byte_digest for trace in traces}) == 3

    invalid = {**base, "session_id": 7}
    assert_trace_error(canonical_json(invalid), "invalid_step")


def test_unconsumed_standard_fields_change_only_source_identity() -> None:
    baseline_source = trajectory_bytes()
    baseline = adapt(baseline_source)
    parsed = json.loads(baseline_source)
    assert type(parsed) is dict
    parsed.update(
        {
            "final_answer": "ignored final answer",
            "final_metrics": {"reward": 1.0},
            "artifacts": [{"path": "ignored-artifact"}],
        }
    )
    steps = parsed["steps"]
    assert type(steps) is list
    selected_step = steps[0]
    assert type(selected_step) is dict
    selected_step.update(
        {
            "message": "ignored message",
            "reasoning_content": "ignored reasoning",
            "model_name": "ignored model",
            "metrics": {"tokens": 1},
        }
    )

    changed = adapt(canonical_json(parsed))

    assert changed.records == baseline.records
    assert changed.diagnostics == baseline.diagnostics
    assert changed.mapped_record_digest == baseline.mapped_record_digest
    assert (
        changed.binding.adapter_configuration_digest
        == baseline.binding.adapter_configuration_digest
    )
    assert changed.binding.source_byte_digest != baseline.binding.source_byte_digest


def test_parser_accepts_a_large_finite_json_number_in_an_unconsumed_field() -> None:
    baseline = adapt(trajectory_bytes())
    source = trajectory_bytes(unconsumed=0)
    marker = b'"unconsumed":0'
    assert source.count(marker) == 1
    source = source.replace(marker, b'"unconsumed":1e309', 1)

    trace = adapt(source)

    assert trace.records == baseline.records
    assert trace.mapped_record_digest == baseline.mapped_record_digest
    assert (
        trace.binding.adapter_configuration_digest == baseline.binding.adapter_configuration_digest
    )
    assert trace.binding.source_byte_digest != baseline.binding.source_byte_digest


@pytest.mark.parametrize(
    "profile",
    (ATIFProfile.HARBOR_TERMINUS_2_V1, ATIFProfile.HARBOR_CODEX_V1),
)
def test_profiles_normalize_exact_utc_offset_timestamps_to_z(
    profile: ATIFProfile,
) -> None:
    source = json.loads(trajectory_bytes(profile=profile))
    source["steps"][0]["timestamp"] = "2026-07-17T10:00:00.123+00:00"
    z_source = json.loads(trajectory_bytes(profile=profile))
    z_source["steps"][0]["timestamp"] = "2026-07-17T10:00:00.123Z"

    trace = adapt(canonical_json(source), profile=profile)
    z_trace = adapt(canonical_json(z_source), profile=profile)

    assert trace.binding.timestamp_mode == "source_utc"
    assert [record["occurred_at"] for record in trace.records] == [
        "2026-07-17T10:00:00.123000Z",
        "2026-07-17T10:00:00.123000Z",
        "2026-07-17T10:00:00.123000Z",
    ]
    assert trace.records == z_trace.records
    assert trace.mapped_record_digest == z_trace.mapped_record_digest
    assert (
        trace.binding.adapter_configuration_digest == z_trace.binding.adapter_configuration_digest
    )
    assert trace.binding.source_byte_digest != z_trace.binding.source_byte_digest


@pytest.mark.parametrize(
    "profile",
    (ATIFProfile.HARBOR_TERMINUS_2_V1, ATIFProfile.HARBOR_CODEX_V1),
)
@pytest.mark.parametrize("offset", ("+01:00", "-04:30", "-00:00"))
def test_profiles_reject_non_utc_timestamp_offsets(
    profile: ATIFProfile,
    offset: str,
) -> None:
    source = json.loads(trajectory_bytes(profile=profile))
    source["steps"][0]["timestamp"] = f"2026-07-17T10:00:00{offset}"

    assert_trace_error(
        canonical_json(source),
        "invalid_timestamp",
        profile=profile,
        step_ordinal=1,
    )


def test_raw_and_unicode_call_identifiers_never_change_coordinate_mapping() -> None:
    baseline_source = trajectory_bytes()
    parsed = json.loads(baseline_source)
    assert type(parsed) is dict
    steps = parsed["steps"]
    assert type(steps) is list
    selected_step = steps[0]
    assert type(selected_step) is dict
    tool_calls = selected_step["tool_calls"]
    assert type(tool_calls) is list
    selected_call = tool_calls[0]
    assert type(selected_call) is dict

    traces: list[ShadowTrace] = []
    for raw_id in ("raw-identifier-one", "caf\u00e9", "cafe\u0301"):
        changed = json.loads(baseline_source)
        changed_steps = changed["steps"]
        changed_steps[0]["tool_calls"][0]["tool_call_id"] = raw_id
        traces.append(adapt(canonical_json(changed)))

    assert {trace.records[1]["source_event_id"] for trace in traces} == {
        "atif-s00000001-c0001-action"
    }
    assert len({trace.mapped_record_digest for trace in traces}) == 1
    assert len({trace.binding.source_byte_digest for trace in traces}) == 3


@pytest.mark.parametrize(
    "profile",
    (ATIFProfile.HARBOR_TERMINUS_2_V1, ATIFProfile.HARBOR_CODEX_V1),
)
def test_profile_descriptor_registry_and_report_contract_are_self_consistent(
    profile: ATIFProfile,
) -> None:
    selected = adapter(profile)
    trace = selected.adapt_bytes(trajectory_bytes(profile=profile), run_id=RUN_ID)
    diagnostics = trace.diagnostics
    assert type(diagnostics) is ATIFShadowDiagnostics
    descriptor_bytes = trace._descriptor_preimage()
    computed_digest = length_prefixed_sha256(
        descriptor_bytes,
        domain="saliencegate:shadow:adapter-profile:v1",
    )

    assert selected.profile_id == profile.value
    assert selected.profile_digest == computed_digest == EXPECTED_PROFILE_DIGESTS[profile]
    assert trace.binding.adapter_profile_id == profile.value
    assert trace.binding.adapter_profile_digest == computed_digest
    assert diagnostics.profile_audit_manifest_digest.encode("ascii") in descriptor_bytes
    assert trace._is_exact()
    assert _sealed_atif_report_contract(profile.value) == (
        computed_digest,
        diagnostics.profile_audit_manifest_digest,
        diagnostics.outcome_evidence_authority,
    )


def test_exact_trace_rejects_a_coherently_rehashed_caller_environment_change() -> None:
    trace = adapt(trajectory_bytes())
    records = trace.records
    record_bytes = trace._wire_record_bytes()
    mapped_record_digest = trace.mapped_record_digest
    configuration = json.loads(trace._configuration_preimage())
    assert type(configuration) is dict
    configured_environment = configuration["environment"]
    assert type(configured_environment) is dict
    configured_environment["environment_digest"] = "b" * 64

    _replace_atif_configuration_and_rehash_binding(trace, configuration)

    assert trace._configuration_preimage() == canonical_json(configuration)
    assert type(trace.binding).model_validate(trace.binding) == trace.binding
    assert trace.records is records
    assert trace._wire_record_bytes() == record_bytes
    assert trace.mapped_record_digest == mapped_record_digest
    assert not trace._is_exact()


def test_exact_trace_rejects_coherently_rehashed_mapped_execution_semantics() -> None:
    steps = _selected_steps(1)
    calls = steps[0]["tool_calls"]
    assert type(calls) is list
    selected_call = calls[0]
    assert type(selected_call) is dict
    arguments = selected_call["arguments"]
    assert type(arguments) is dict
    arguments["login"] = False
    trace = adapt(trajectory_bytes(steps=steps))
    records = trace.records
    record_bytes = trace._wire_record_bytes()
    mapped_record_digest = trace.mapped_record_digest
    configuration = json.loads(trace._configuration_preimage())
    assert type(configuration) is dict
    contexts = configuration["mapped_action_contexts"]
    assert type(contexts) is list
    context = contexts[0]
    assert type(context) is dict
    semantics = context["execution_semantics"]
    assert type(semantics) is dict
    assert semantics["login"] is False
    semantics["login"] = True

    _replace_atif_configuration_and_rehash_binding(trace, configuration)

    assert trace._configuration_preimage() == canonical_json(configuration)
    assert type(trace.binding).model_validate(trace.binding) == trace.binding
    assert trace.records is records
    assert trace._wire_record_bytes() == record_bytes
    assert trace.mapped_record_digest == mapped_record_digest
    assert not trace._is_exact()


@pytest.mark.parametrize(
    "changed_id",
    (
        "atif-s00000000-c0000-action",
        "atif-s00000002-c0001-action",
        "atif-s00000001-c0002-action",
    ),
)
def test_exact_trace_rejects_a_coherently_rehashed_impossible_coordinate(
    changed_id: str,
) -> None:
    trace = adapt(trajectory_bytes())
    changed_records = [dict(record) for record in trace.records]
    changed_records[1]["source_event_id"] = changed_id
    _replace_records_and_rehash(trace, changed_records)
    configuration = json.loads(trace._configuration_preimage())
    contexts = configuration["mapped_action_contexts"]
    assert type(contexts) is list and type(contexts[0]) is dict
    contexts[0]["source_event_id"] = changed_id

    _replace_atif_configuration_and_rehash_binding(trace, configuration)

    assert not trace._is_exact()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("command", 123),
        ("command", "x" * (128 * 1_024 + 1)),
        ("schema_version", "shadow-input/v2"),
    ),
)
def test_exact_trace_revalidates_coherently_rehashed_wire_records(
    field: str,
    value: object,
) -> None:
    trace = adapt(trajectory_bytes())
    changed_records = [dict(record) for record in trace.records]
    changed_records[1][field] = value

    _replace_records_and_rehash(trace, changed_records)

    assert not trace._is_exact()


def test_exact_trace_cross_links_diagnostic_mapping_counts_to_record_kinds() -> None:
    call_id = "synthetic-diagnostic-call"
    trace = adapt(
        trajectory_bytes(
            steps=[
                {
                    "step_id": 1,
                    "source": "agent",
                    "tool_calls": [
                        {
                            "tool_call_id": call_id,
                            "function_name": "exec_command",
                            "arguments": {"cmd": "true"},
                        }
                    ],
                    "observation": {"results": [{"source_call_id": call_id}]},
                    "extra": {"tool_metadata": {"exit_code": 0}},
                }
            ]
        )
    )
    body = trace.diagnostics.model_dump(mode="json", warnings=False)
    body["total_tool_call_count"] = 2
    body["tool_call_disposition_counts"][0][1] = 2
    body["result_disposition_counts"][0][1] = 0
    body["result_disposition_counts"][1][1] = 1
    body.pop("diagnostics_digest")
    body["diagnostics_digest"] = length_prefixed_sha256(
        canonical_json(body),
        domain="saliencegate:shadow:atif-diagnostics:v1",
    )
    changed = ATIFShadowDiagnostics.model_validate(body)
    object.__setattr__(trace, "diagnostics", changed)
    object.__setattr__(trace, "_diagnostics_bytes", canonical_json(changed))

    assert changed.mapped_shadow_record_count == len(trace.records) == 4
    assert not trace._is_exact()


def test_exact_terminus_trace_rejects_an_impossible_continuation_disposition() -> None:
    trace = adapt(
        trajectory_bytes(profile=ATIFProfile.HARBOR_TERMINUS_2_V1),
        profile=ATIFProfile.HARBOR_TERMINUS_2_V1,
    )
    body = trace.diagnostics.model_dump(mode="json", warnings=False)
    body["total_tool_call_count"] = 2
    body["tool_call_disposition_counts"][2][1] = 1
    body.pop("diagnostics_digest")
    body["diagnostics_digest"] = length_prefixed_sha256(
        canonical_json(body),
        domain="saliencegate:shadow:atif-diagnostics:v1",
    )
    changed = ATIFShadowDiagnostics.model_validate(body)
    object.__setattr__(trace, "diagnostics", changed)
    object.__setattr__(trace, "_diagnostics_bytes", canonical_json(changed))

    assert dict(changed.tool_call_disposition_counts)["ignored_continuation"] == 1
    assert not trace._is_exact()


def test_compatibility_manifest_is_installed_canonical_and_shared_by_both_profiles() -> None:
    manifest_bytes = (
        resources.files("saliencegate.shadow")
        .joinpath("atif_profile_compatibility.json")
        .read_bytes()
    )
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    assert type(manifest) is dict
    assert canonical_json(manifest) == manifest_bytes

    traces = tuple(
        adapter(profile).adapt_bytes(trajectory_bytes(profile=profile), run_id=RUN_ID)
        for profile in ATIFProfile
    )
    manifest_digests = {
        trace.diagnostics.profile_audit_manifest_digest
        for trace in traces
        if type(trace.diagnostics) is ATIFShadowDiagnostics
    }

    assert len(manifest_digests) == 1
    manifest_digest = next(iter(manifest_digests))
    assert manifest_digest == EXPECTED_MANIFEST_DIGEST
    assert len(manifest_digest) == 64
    assert all(character in "0123456789abcdef" for character in manifest_digest)
    assert all(manifest_digest.encode("ascii") in trace._descriptor_preimage() for trace in traces)


def test_compatibility_manifest_references_are_closed_and_fixture_bytes_are_frozen() -> None:
    manifest_bytes = (
        resources.files("saliencegate.shadow")
        .joinpath("atif_profile_compatibility.json")
        .read_bytes()
    )
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    assert type(manifest) is dict
    assert canonical_json(manifest) == manifest_bytes

    artifacts = manifest["artifacts"]
    transforms = manifest["transforms"]
    profiles = manifest["profiles"]
    fixtures = manifest["fixtures"]
    assert all(
        type(collection) is list for collection in (artifacts, transforms, profiles, fixtures)
    )
    assert all(
        type(item) is dict
        for collection in (artifacts, transforms, profiles, fixtures)
        for item in collection
    )

    artifact_ids = tuple(item["artifact_id"] for item in artifacts)
    transform_ids = tuple(item["transform_id"] for item in transforms)
    profile_ids = tuple(item["profile_id"] for item in profiles)
    fixture_ids = tuple(item["fixture_id"] for item in fixtures)
    assert len(set(artifact_ids)) == len(artifact_ids)
    assert len(set(transform_ids)) == len(transform_ids)
    assert len(set(profile_ids)) == len(profile_ids)
    assert len(set(fixture_ids)) == len(fixture_ids)
    assert set(profile_ids) == {profile.value for profile in ATIFProfile}
    profile_by_id = {item["profile_id"]: item for item in profiles}
    assert profile_by_id[ATIFProfile.HARBOR_TERMINUS_2_V1.value][
        "accepted_source_schema_versions"
    ] == ["ATIF-v1.6", "ATIF-v1.7"]
    assert (
        profile_by_id[ATIFProfile.HARBOR_CODEX_V1.value]["compatibility_claim"]
        == "pinned_harbor_converter_field_shape_only_no_codex_cli_version_guarantee"
    )
    assert manifest["limitations"] == [
        "paper_and_proactive_repository_are_research_inspiration_not_direct_runtime_compatibility_evidence",
        "codex_cli_runtime_version_is_not_pinned_and_the_codex_fixture_is_fully_synthetic",
        "terminus_sanitization_preserves_consumed_field_structure_not_sensitive_metric_array_cardinality",
    ]

    known_artifacts = set(artifact_ids)
    known_transforms = set(transform_ids)
    known_profiles = set(profile_ids)
    artifact_profiles: dict[str, set[str]] = {}
    referenced_transforms: set[str] = set()
    for artifact in artifacts:
        applies_to = artifact["applies_to_profiles"]
        assert type(applies_to) is list
        assert applies_to
        assert len(set(applies_to)) == len(applies_to)
        assert set(applies_to) <= known_profiles
        artifact_profiles[artifact["artifact_id"]] = set(applies_to)

    artifact_by_id = {item["artifact_id"]: item for item in artifacts}
    assert artifact_by_id["remember-when-it-matters-paper"]["evidence_role"] == (
        "research_inspiration_not_runtime_compatibility"
    )
    assert artifact_by_id["proactive-memory-agent-repository"]["evidence_role"] == (
        "research_inspiration_not_runtime_compatibility"
    )
    assert artifact_by_id["harbor-codex-converter"]["runtime_package_identity"] == (
        "not_pinned_converter_defaults_to_latest_when_unspecified"
    )

    for transform in transforms:
        definition = transform["definition"]
        assert type(definition) is dict
        assert (
            hashlib.sha256(canonical_json(definition)).hexdigest() == transform["definition_sha256"]
        )
    assert transforms[0]["definition"]["metric_array_policy"] == (
        "replace_sensitive_array_contents_with_empty_arrays"
    )

    manifest_fixture_paths: set[str] = set()
    for fixture in fixtures:
        applies_to = fixture["applies_to_profiles"]
        provenance = fixture["provenance_artifact_ids"]
        retained_fields = fixture["retained_field_allowlist"]
        transform_id = fixture["transform_id"]
        fixture_path = fixture["path"]
        assert type(applies_to) is list and applies_to
        assert type(provenance) is list and provenance
        assert type(retained_fields) is list
        assert type(transform_id) is str
        assert type(fixture_path) is str
        assert len(set(applies_to)) == len(applies_to)
        assert len(set(provenance)) == len(provenance)
        assert len(set(retained_fields)) == len(retained_fields)
        assert set(applies_to) <= known_profiles
        assert set(provenance) <= known_artifacts
        assert all(set(applies_to) <= artifact_profiles[item] for item in provenance)
        assert transform_id in known_transforms
        referenced_transforms.add(transform_id)
        manifest_fixture_paths.add(fixture_path)

        expected_sha256 = EXPECTED_ATIF_FIXTURE_SHA256[fixture_path]
        fixture_bytes = Path(fixture_path).read_bytes()
        assert fixture["sha256"] == expected_sha256
        assert hashlib.sha256(fixture_bytes).hexdigest() == expected_sha256
        assert canonical_json(json.loads(fixture_bytes.decode("utf-8"))) == fixture_bytes

    assert manifest_fixture_paths == set(EXPECTED_ATIF_FIXTURE_SHA256)
    assert {path.as_posix() for path in Path("tests/fixtures/shadow/atif").glob("*.json")} == set(
        EXPECTED_ATIF_FIXTURE_SHA256
    )
    assert referenced_transforms == known_transforms


def test_atif_records_and_diagnostics_are_immutable_exact_snapshots() -> None:
    trace = adapt(trajectory_bytes())
    diagnostics = trace.diagnostics
    assert type(diagnostics) is ATIFShadowDiagnostics
    assert type(trace.records) is tuple
    assert all(type(record) is MappingProxyType for record in trace.records)
    assert repr(trace) == "ShadowTrace(<validated>)"
    assert repr(diagnostics) == "ATIFShadowDiagnostics(<counts>)"

    with pytest.raises(TypeError):
        trace.records[1]["command"] = "mutated"  # type: ignore[index]
