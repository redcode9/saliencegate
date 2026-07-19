from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from uuid import UUID

import pytest
from pydantic import TypeAdapter, ValidationError

import saliencegate.shadow.trace as trace_module
from saliencegate.domain import canonical_json, length_prefixed_sha256
from saliencegate.shadow.errors import ShadowInputError, ShadowTraceInputError
from saliencegate.shadow.inputs import ShadowInputKind
from saliencegate.shadow.trace import (
    MAX_SHADOW_TRACE_BYTES,
    MAX_SHADOW_TRACE_ROWS,
    ATIFShadowDiagnostics,
    ShadowRecordDiagnostics,
    ShadowTrace,
    ShadowTraceBinding,
    ShadowTraceDiagnostics,
    _build_atif_diagnostics,
    _build_binding,
)

RUN_ID = UUID("11111111-1111-4111-8111-111111111111")
ENVIRONMENT_DIGEST = "a" * 64
OPAQUE_ACTION_DIGEST = "b" * 64
OPAQUE_WORKSPACE_DIGEST = "c" * 64
OPAQUE_ENVIRONMENT_DIGEST = "d" * 64
DESCRIPTOR: dict[str, object] = {
    "schema_version": "example-shadow-adapter/v1",
    "mapping": {"mode": "structured", "selected": ["tool", "test"]},
}

LEGACY_RECORD_BYTES = (
    b'{"kind":"run_start","occurred_at":"2026-07-17T09:00:00Z",'
    b'"schema_version":"shadow-input/v1","source_event_id":"start-1"}',
    b'{"command":"pytest -q","environment_digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
    b'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","kind":"action","occurred_at":'
    b'"2026-07-17T09:00:01Z","schema_version":"shadow-input/v1",'
    b'"source_event_id":"action-1","working_directory":"/private/project"}',
    b'{"action_source_event_id":"action-1","error_code":"TEST_FAILURE",'
    b'"exit_status":1,"kind":"tool_result","occurred_at":"2026-07-17T09:00:01Z",'
    b'"schema_version":"shadow-input/v1","source_event_id":"tool-1",'
    b'"status":"failed"}',
    b'{"action_source_event_id":"action-1","failures":[{"failure_type":'
    b'"AssertionError","schema_version":"1.0","signature":"expected-one-got-two",'
    b'"test_id":"tests/test_unit.py::test_example"}],"framework":"pytest",'
    b'"kind":"test_result","occurred_at":"2026-07-17T09:00:02Z",'
    b'"schema_version":"shadow-input/v1","source_event_id":"test-1",'
    b'"status":"failed"}',
    b'{"kind":"observation","occurred_at":"2026-07-17T09:00:03Z",'
    b'"payload":{"request":{"kind":"unit","labels":["a","b"]}},'
    b'"schema_version":"shadow-input/v1","source":"task_input",'
    b'"source_event_id":"observation-1"}',
    b'{"error_code":"controller_timeout","kind":"controller_error",'
    b'"occurred_at":"2026-07-17T09:00:04Z","schema_version":"shadow-input/v1",'
    b'"source_event_id":"controller-1"}',
    b'{"kind":"run_end","occurred_at":"2026-07-17T09:00:05Z",'
    b'"schema_version":"shadow-input/v1","source_event_id":"finish-1"}',
)


def complete_records() -> list[dict[str, object]]:
    return [
        {
            "schema_version": "shadow-input/v1",
            "kind": "run_start",
            "source_event_id": "start-1",
            "occurred_at": "2026-07-17T09:00:00Z",
        },
        {
            "schema_version": "shadow-input/v1",
            "kind": "action",
            "source_event_id": "action-1",
            "occurred_at": "2026-07-17T09:00:01Z",
            "command": "pytest -q",
            "working_directory": "/private/project",
            "environment_digest": ENVIRONMENT_DIGEST,
        },
        {
            "schema_version": "shadow-input/v1",
            "kind": "tool_result",
            "source_event_id": "tool-1",
            "occurred_at": "2026-07-17T09:00:01Z",
            "action_source_event_id": "action-1",
            "status": "failed",
            "exit_status": 1,
            "error_code": "TEST_FAILURE",
        },
        {
            "schema_version": "shadow-input/v1",
            "kind": "test_result",
            "source_event_id": "test-1",
            "occurred_at": "2026-07-17T09:00:02Z",
            "action_source_event_id": "action-1",
            "framework": "pytest",
            "status": "failed",
            "failures": [
                {
                    "schema_version": "1.0",
                    "test_id": "tests/test_unit.py::test_example",
                    "failure_type": "AssertionError",
                    "signature": "expected-one-got-two",
                }
            ],
        },
        {
            "schema_version": "shadow-input/v1",
            "kind": "observation",
            "source_event_id": "observation-1",
            "occurred_at": "2026-07-17T09:00:03Z",
            "source": "task_input",
            "payload": {"request": {"kind": "unit", "labels": ["a", "b"]}},
        },
        {
            "schema_version": "shadow-input/v1",
            "kind": "controller_error",
            "source_event_id": "controller-1",
            "occurred_at": "2026-07-17T09:00:04Z",
            "error_code": "controller_timeout",
        },
        {
            "schema_version": "shadow-input/v1",
            "kind": "run_end",
            "source_event_id": "finish-1",
            "occurred_at": "2026-07-17T09:00:05Z",
        },
    ]


def identity_records() -> list[dict[str, object]]:
    records = complete_records()
    records[1] = {
        "schema_version": "shadow-input/v1",
        "kind": "action_identity",
        "source_event_id": "identity-exact",
        "occurred_at": "2026-07-17T09:00:01Z",
        "action_digest": OPAQUE_ACTION_DIGEST,
        "workspace_digest": OPAQUE_WORKSPACE_DIGEST,
        "environment_digest": OPAQUE_ENVIRONMENT_DIGEST,
        "identity_authority": "exact",
    }
    records[2]["action_source_event_id"] = "identity-exact"
    records[3]["action_source_event_id"] = "identity-exact"
    records[4] = {
        "schema_version": "shadow-input/v1",
        "kind": "action_identity",
        "source_event_id": "identity-coarse",
        "occurred_at": "2026-07-17T09:00:03Z",
        "action_digest": "e" * 64,
        "workspace_digest": OPAQUE_WORKSPACE_DIGEST,
        "environment_digest": OPAQUE_ENVIRONMENT_DIGEST,
        "identity_authority": "coarse",
    }
    records[5] = {
        "schema_version": "shadow-input/v1",
        "kind": "action_identity",
        "source_event_id": "identity-unavailable",
        "occurred_at": "2026-07-17T09:00:04Z",
        "action_digest": "f" * 64,
        "workspace_digest": OPAQUE_WORKSPACE_DIGEST,
        "environment_digest": OPAQUE_ENVIRONMENT_DIGEST,
        "identity_authority": "unavailable",
    }
    return records


def build_trace(
    records: list[dict[str, object]] | Iterator[dict[str, object]] | None = None,
    **overrides: object,
) -> ShadowTrace:
    arguments: dict[str, object] = {
        "run_id": RUN_ID,
        "adapter_profile_id": "example-agent/v1",
        "adapter_descriptor": DESCRIPTOR,
        "capture_scope": "complete_run_declared",
    }
    arguments.update(overrides)
    return ShadowTrace.from_records(
        complete_records() if records is None else records,
        **arguments,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("profile_id", ("harbor-terminus-2/v1", "harbor-codex/v1"))
def test_direct_factory_reserves_built_in_atif_profile_ids(profile_id: str) -> None:
    records_consumed = False

    def records() -> Iterator[dict[str, object]]:
        nonlocal records_consumed
        records_consumed = True
        yield from complete_records()

    with pytest.raises(ShadowTraceInputError) as captured:
        build_trace(records(), adapter_profile_id=profile_id)

    assert captured.value.reason_code == "profile_mismatch"
    assert not records_consumed


def direct_diagnostics(trace: ShadowTrace) -> ShadowRecordDiagnostics:
    diagnostics = trace.diagnostics
    assert type(diagnostics) is ShadowRecordDiagnostics
    return diagnostics


def test_from_records_builds_a_complete_content_addressed_trace() -> None:
    records = complete_records()

    trace = build_trace(records)

    record_bytes = tuple(canonical_json(record) for record in records)
    assert record_bytes == LEGACY_RECORD_BYTES
    source_bytes = b"[" + b",".join(record_bytes) + b"]"
    descriptor_bytes = canonical_json(DESCRIPTOR)
    profile_digest = length_prefixed_sha256(
        descriptor_bytes,
        domain="saliencegate:shadow:adapter-profile:v1",
    )
    configuration_bytes = canonical_json(
        {
            "schema_version": "shadow-adapter-configuration/v1",
            "adapter_profile_id": "example-agent/v1",
            "adapter_profile_digest": profile_digest,
            "source_format": "shadow-records",
            "source_schema_version": "shadow-input/v1",
            "timestamp_mode": "record_declared",
            "capture_scope": "complete_run_declared",
        }
    )
    configuration_digest = length_prefixed_sha256(
        configuration_bytes,
        domain="saliencegate:shadow:adapter-configuration:v1",
    )

    assert trace.schema_version == "shadow-trace/v1"
    assert trace.run_id == RUN_ID
    assert trace.run_id is not RUN_ID
    assert trace.binding.source_digest_kind == "canonical_records"
    assert trace.binding.source_byte_count == len(source_bytes)
    assert trace.binding.source_byte_digest == hashlib.sha256(source_bytes).hexdigest()
    assert trace.binding.adapter_profile_digest == profile_digest
    assert trace.binding.adapter_configuration_digest == configuration_digest
    assert trace.binding.source_adapter == (
        f"shadow-records.example-agent/v1+p.{profile_digest}+c.{configuration_digest}"
    )
    assert trace.binding.identity_mode == "profile_content_addressed"
    assert trace.binding.timestamp_mode == "record_declared"
    assert trace.binding.capture_scope == "complete_run_declared"
    assert trace.mapped_record_digest == length_prefixed_sha256(
        *record_bytes,
        domain="saliencegate:shadow:mapped-records:v1",
    )
    diagnostics = direct_diagnostics(trace)
    assert diagnostics.source_record_count == 7
    assert diagnostics.input_kind_counts == (
        (ShadowInputKind.START, 1),
        (ShadowInputKind.ACTION, 1),
        (ShadowInputKind.TOOL_RESULT, 1),
        (ShadowInputKind.TEST_RESULT, 1),
        (ShadowInputKind.OBSERVATION, 1),
        (ShadowInputKind.CONTROLLER_ERROR, 1),
        (ShadowInputKind.FINISH, 1),
    )
    assert diagnostics.repeated_source_identifier_count == 0
    assert diagnostics.mapped_shadow_record_count == 7
    assert trace._descriptor_preimage() == descriptor_bytes
    assert trace._configuration_preimage() == configuration_bytes
    assert trace._wire_record_bytes() == record_bytes
    assert trace.binding.adapter_profile_digest == (
        "53237e674322879affb31be768c9da3ce88b9d6f46376431ea29fc1ee9bae6bf"
    )
    assert trace.binding.adapter_configuration_digest == (
        "b3dfcc7a2010950a5a94395521878458ae27d8943d6ab365f8018d948bf66490"
    )
    assert trace.binding.source_byte_digest == (
        "ccef797561e5948d3a84a4e0ceff3ccdb1234c4a540d8d957029813443c2aa15"
    )
    assert trace.binding.binding_digest == (
        "3c49803df1ff3baaef2d113710c27239254cfdb62e6271489e900127a049b885"
    )
    assert trace.diagnostics.diagnostics_digest == (
        "e55046bcccc1a50fc6e1d2e13feb1be8ff01c05a458a8baaa46b2d034637ee7d"
    )
    assert trace.mapped_record_digest == (
        "957401f3820d57146026cb5814ee99798afdf959b9a87632c6eb9438b31c560b"
    )


def test_identity_records_round_trip_with_canonical_diagnostics() -> None:
    records = identity_records()

    trace = build_trace(records)

    assert trace._wire_record_bytes() == tuple(canonical_json(record) for record in records)
    diagnostics = direct_diagnostics(trace)
    expected_counts = (
        (ShadowInputKind.START, 1),
        (ShadowInputKind.ACTION, 0),
        (ShadowInputKind.TOOL_RESULT, 1),
        (ShadowInputKind.TEST_RESULT, 1),
        (ShadowInputKind.OBSERVATION, 0),
        (ShadowInputKind.CONTROLLER_ERROR, 0),
        (ShadowInputKind.FINISH, 1),
        (ShadowInputKind.ACTION_IDENTITY, 3),
    )
    expected_body = {
        "schema_version": "shadow-record-diagnostics/v1",
        "source_record_count": 7,
        "input_kind_counts": expected_counts,
        "repeated_source_identifier_count": 0,
        "mapped_shadow_record_count": 7,
    }
    assert diagnostics.input_kind_counts == expected_counts
    assert diagnostics.model_dump(mode="python", exclude={"diagnostics_digest"}) == expected_body
    assert diagnostics.diagnostics_digest == length_prefixed_sha256(
        canonical_json(expected_body),
        domain="saliencegate:shadow:record-diagnostics:v1",
    )

    zero_identity = diagnostics.model_copy(
        update={
            "input_kind_counts": (
                *diagnostics.input_kind_counts[:-1],
                (ShadowInputKind.ACTION_IDENTITY, 0),
            )
        }
    )
    with pytest.raises(ValueError, match="identity count is not canonical"):
        ShadowRecordDiagnostics.model_validate(zero_identity)


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    (
        ("action_digest", "A" * 64),
        ("workspace_digest", "c" * 63),
        ("environment_digest", "d" * 65),
        ("identity_authority", "unknown"),
        ("working_directory", "/synthetic-placeholder"),
    ),
)
def test_identity_wire_records_reject_noncanonical_digests_and_shell_placeholders(
    changed_field: str,
    changed_value: str,
) -> None:
    rows = identity_records()
    build_trace(rows)
    rows[1][changed_field] = changed_value

    with pytest.raises(ShadowTraceInputError) as raised:
        build_trace(rows)

    assert raised.value.reason_code == "invalid_step"
    assert raised.value.step_ordinal == 2
    assert changed_value not in repr(raised.value)


@pytest.mark.parametrize(
    ("result_index", "parent_source_event_id"),
    (
        (2, "missing-action"),
        (2, "start-1"),
        (2, "identity-coarse"),
        (3, "tool-1"),
    ),
)
def test_identity_result_parents_must_reference_one_prior_action_exactly(
    result_index: int,
    parent_source_event_id: str,
) -> None:
    rows = identity_records()
    build_trace(rows)
    rows[result_index]["action_source_event_id"] = parent_source_event_id

    with pytest.raises(ShadowTraceInputError) as raised:
        build_trace(rows)

    assert raised.value.reason_code == "invalid_step"
    assert raised.value.step_ordinal == result_index + 1
    assert parent_source_event_id not in repr(raised.value)


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    (
        ("action_digest", "9" * 64),
        ("workspace_digest", "8" * 64),
        ("environment_digest", "7" * 64),
        ("identity_authority", "coarse"),
    ),
)
def test_identity_retries_require_the_whole_canonical_record_to_match(
    changed_field: str,
    changed_value: str,
) -> None:
    rows = identity_records()
    retry = dict(rows[1])
    rows.insert(2, retry)

    accepted = build_trace(rows)
    assert direct_diagnostics(accepted).repeated_source_identifier_count == 1

    retry[changed_field] = changed_value
    with pytest.raises(ShadowTraceInputError) as raised:
        build_trace(rows)

    assert raised.value.reason_code == "invalid_step"
    assert raised.value.step_ordinal == 3
    assert changed_value not in repr(raised.value)


def test_original_source_bytes_have_an_exact_plain_sha256_identity() -> None:
    source = b"native\x00trajectory\n"

    trace = build_trace(source_bytes=source, source_format="custom")

    assert trace.binding.source_digest_kind == "original_bytes"
    assert trace.binding.source_byte_count == len(source)
    assert trace.binding.source_byte_digest == hashlib.sha256(source).hexdigest()


def test_input_and_descriptor_are_deeply_snapshotted() -> None:
    records = complete_records()
    descriptor = {
        "schema_version": "example-shadow-adapter/v1",
        "mapping": {"selected": ["tool"]},
    }
    trace = build_trace(records, adapter_descriptor=descriptor)

    records[1]["command"] = "caller changed this"
    observation_payload = records[4]["payload"]
    assert isinstance(observation_payload, dict)
    observation_payload["request"] = {"kind": "changed"}
    mapping = descriptor["mapping"]
    assert isinstance(mapping, dict)
    mapping["selected"] = ["changed"]

    assert trace.records[1]["command"] == "pytest -q"
    assert trace.records[4]["payload"] == MappingProxyType(
        {"request": MappingProxyType({"kind": "unit", "labels": ("a", "b")})}
    )
    with pytest.raises(TypeError):
        trace.records[1]["command"] = "mutation"  # type: ignore[index]
    assert b"changed" not in trace._descriptor_preimage()


def test_trace_and_models_have_safe_repr_and_reject_public_construction() -> None:
    trace = build_trace()

    assert repr(trace) == "ShadowTrace(<validated>)"
    assert repr(trace.binding) == "ShadowTraceBinding(<redacted>)"
    assert repr(trace.diagnostics) == "ShadowRecordDiagnostics(<counts>)"
    assert "pytest" not in repr(trace)
    assert "private" not in repr(trace)
    with pytest.raises(TypeError):
        ShadowTrace()
    with pytest.raises(TypeError):

        class DerivedTrace(ShadowTrace):  # type: ignore[misc]
            pass


def test_binding_and_diagnostics_revalidate_their_self_digests() -> None:
    trace = build_trace()
    changed_binding = trace.binding.model_copy(update={"source_byte_count": 1})
    changed_diagnostics = trace.diagnostics.model_copy(
        update={"repeated_source_identifier_count": 1}
    )

    with pytest.raises(ValidationError, match="binding digest does not match"):
        ShadowTraceBinding.model_validate(changed_binding)
    with pytest.raises(ValidationError, match="diagnostics digest does not match"):
        TypeAdapter(ShadowTraceDiagnostics).validate_python(changed_diagnostics)


def test_diagnostics_union_validates_both_discriminated_branches() -> None:
    direct = build_trace().diagnostics
    atif = _build_atif_diagnostics(
        continued_trajectory_ref_present=True,
        embedded_subagent_trajectory_count=2,
        outcome_evidence_authority="producer_claimed_structured",
        profile_audit_manifest_digest="b" * 64,
        total_step_count=3,
        ignored_message_step_count=1,
        total_tool_call_count=2,
        tool_call_disposition_counts=(
            ("mapped_action", 1),
            ("ignored_unsupported_function", 1),
            ("ignored_continuation", 0),
            ("ignored_non_command_wait", 0),
            ("ignored_unsubmitted_keystrokes", 0),
            ("ignored_unresolved_terminal_submission", 0),
            ("ignored_copied_context", 0),
        ),
        total_observation_result_count=2,
        result_disposition_counts=(
            ("mapped_structured_outcome", 1),
            ("ignored_evidence_absent", 1),
            ("ignored_ambiguous_parent", 0),
            ("ignored_no_parent", 0),
            ("ignored_unsupported_parent", 0),
            ("ignored_copied_context", 0),
        ),
        mapped_shadow_record_count=4,
    )
    adapter: TypeAdapter[ShadowTraceDiagnostics] = TypeAdapter(ShadowTraceDiagnostics)

    assert type(adapter.validate_python(direct)) is ShadowRecordDiagnostics
    validated_atif = adapter.validate_python(atif)
    assert type(validated_atif) is ATIFShadowDiagnostics
    assert repr(validated_atif) == "ATIFShadowDiagnostics(<counts>)"
    assert validated_atif.root_segment_only is True
    assert validated_atif.complete_execution_session_coverage is False
    assert validated_atif.producer_authentication == "none"
    assert validated_atif.diagnostics_digest == (
        "f92d1a03c223589811928b1b5c3b328e36488240d6eb17e82267bf27bbd548d8"
    )

    changed = atif.model_copy(update={"total_tool_call_count": 3})
    with pytest.raises(ValidationError, match="tool disposition equation"):
        adapter.validate_python(changed)

    class TextSubclass(str):
        pass

    direct_body = direct.model_dump(mode="python")
    direct_body["schema_version"] = TextSubclass("shadow-record-diagnostics/v1")
    with pytest.raises(ValidationError):
        adapter.validate_python(direct_body)

    atif_bodies = []
    for field_name, invalid_value in (
        ("schema_version", TextSubclass("atif-shadow-diagnostics/v1")),
        ("root_segment_only", 1),
        ("complete_execution_session_coverage", 0),
    ):
        body = atif.model_dump(mode="python")
        body[field_name] = invalid_value
        atif_bodies.append(body)
    disposition_body = atif.model_dump(mode="python")
    disposition_counts = list(disposition_body["tool_call_disposition_counts"])
    disposition_counts[0] = (TextSubclass("mapped_action"), 1)
    disposition_body["tool_call_disposition_counts"] = tuple(disposition_counts)
    atif_bodies.append(disposition_body)
    for body in atif_bodies:
        with pytest.raises(ValidationError):
            adapter.validate_python(body)


def test_public_binding_rejects_legacy_identity_and_model_subclasses() -> None:
    binding = build_trace().binding
    legacy_body = binding.model_dump(mode="json", exclude={"binding_digest"})
    legacy_body.update(identity_mode="legacy_explicit", source_adapter="legacy/v1")
    legacy_digest = length_prefixed_sha256(
        canonical_json(legacy_body),
        domain="saliencegate:shadow:trace-binding:v1",
    )

    with pytest.raises(ValidationError, match="not publicly constructible"):
        ShadowTraceBinding.model_validate({**legacy_body, "binding_digest": legacy_digest})
    with pytest.raises(TypeError):

        class DerivedBinding(ShadowTraceBinding):
            extra: str


def test_binding_rejects_forged_configuration_and_invalid_canonical_source_identity() -> None:
    binding = build_trace().binding
    forged = binding.model_dump(mode="json", exclude={"binding_digest"})
    forged["adapter_configuration_digest"] = "f" * 64
    forged["source_adapter"] = (
        f"{binding.source_format}.{binding.adapter_profile_id}"
        f"+p.{binding.adapter_profile_digest}+c.{'f' * 64}"
    )
    forged_digest = length_prefixed_sha256(
        canonical_json(forged),
        domain="saliencegate:shadow:trace-binding:v1",
    )

    with pytest.raises(ValidationError, match="configuration identity"):
        ShadowTraceBinding.model_validate({**forged, "binding_digest": forged_digest})

    invalid_canonical = binding.model_dump(mode="json", exclude={"binding_digest"})
    invalid_canonical.update(
        source_format="example",
        source_schema_version="example/v1",
        source_digest_kind="canonical_records",
    )
    configuration = canonical_json(
        {
            "schema_version": "shadow-adapter-configuration/v1",
            "adapter_profile_id": binding.adapter_profile_id,
            "adapter_profile_digest": binding.adapter_profile_digest,
            "source_format": "example",
            "source_schema_version": "example/v1",
            "timestamp_mode": binding.timestamp_mode,
            "capture_scope": binding.capture_scope,
        }
    )
    configuration_digest = length_prefixed_sha256(
        configuration,
        domain="saliencegate:shadow:adapter-configuration:v1",
    )
    invalid_canonical["adapter_configuration_digest"] = configuration_digest
    invalid_canonical["source_adapter"] = (
        f"example.{binding.adapter_profile_id}"
        f"+p.{binding.adapter_profile_digest}+c.{configuration_digest}"
    )
    invalid_digest = length_prefixed_sha256(
        canonical_json(invalid_canonical),
        domain="saliencegate:shadow:trace-binding:v1",
    )

    with pytest.raises(ValidationError, match="canonical record source"):
        ShadowTraceBinding.model_validate({**invalid_canonical, "binding_digest": invalid_digest})

    atif_binding = _build_binding(
        source_format="atif",
        source_schema_version="ATIF-v1.7",
        source_digest_kind="original_bytes",
        source_bytes=b"{}",
        adapter_profile_id="harbor-codex/v1",
        adapter_profile_digest="6" * 64,
        adapter_configuration_digest="7" * 64,
        timestamp_mode="logical_order",
        capture_scope="selected_events",
        task_scope_digest=None,
        lineage_scope_digest=None,
        capture_manifest_digest=None,
    )
    assert atif_binding.adapter_configuration_digest == "7" * 64


def test_exact_trace_check_detects_public_and_private_tampering() -> None:
    trace = build_trace()
    assert trace._is_exact()

    object.__setattr__(trace, "mapped_record_digest", "0" * 64)
    assert not trace._is_exact()

    trace = build_trace()
    trace.binding.__dict__["source_byte_count"] = 1
    assert not trace._is_exact()

    trace = build_trace()
    object.__setattr__(
        trace,
        "run_id",
        UUID("22222222-2222-4222-8222-222222222222"),
    )
    assert not trace._is_exact()

    partial = object.__new__(ShadowTrace)
    assert not partial._is_exact()


def test_exact_trace_copy_is_revalidated_and_deeply_detached() -> None:
    trace = build_trace()

    copied = trace._copy_exact()

    assert copied is not trace
    assert copied._is_exact()
    assert copied.run_id == trace.run_id
    assert copied.run_id is not trace.run_id
    assert copied.binding == trace.binding
    assert copied.binding is not trace.binding
    assert copied.diagnostics == trace.diagnostics
    assert copied.diagnostics is not trace.diagnostics
    assert copied.records == trace.records
    assert copied.records is not trace.records
    assert copied.records[4] is not trace.records[4]
    assert copied.records[4]["payload"] is not trace.records[4]["payload"]
    assert copied._wire_record_bytes() == trace._wire_record_bytes()
    assert copied._descriptor_preimage() == trace._descriptor_preimage()
    assert copied._configuration_preimage() == trace._configuration_preimage()

    trace.binding.__dict__["source_byte_count"] = 1
    object.__setattr__(trace, "_records", ())

    assert not trace._is_exact()
    assert copied._is_exact()
    assert len(copied.records) == 7


def test_exact_trace_copy_rejects_tampering_and_hides_internal_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tampered = build_trace()
    object.__setattr__(tampered, "mapped_record_digest", "0" * 64)

    with pytest.raises(ShadowTraceInputError) as tampered_error:
        tampered._copy_exact()

    assert tampered_error.value.reason_code == "invalid_step"
    assert tampered_error.value.__cause__ is None
    assert tampered_error.value.__context__ is None

    trace = build_trace()

    def fail_copy(_value: object) -> object:
        raise RuntimeError("private record content")

    monkeypatch.setattr(trace_module, "_copy_frozen_json", fail_copy)
    with pytest.raises(ShadowTraceInputError) as internal_error:
        trace._copy_exact()

    assert internal_error.value.reason_code == "invalid_step"
    assert "private" not in repr(internal_error.value)
    assert internal_error.value.__cause__ is None
    assert internal_error.value.__context__ is None


class _CountingIterator(Iterator[dict[str, object]]):
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = iter(rows)
        self.consumed = 0

    def __next__(self) -> dict[str, object]:
        item = next(self._rows)
        self.consumed += 1
        return item


def test_from_records_consumes_a_one_shot_iterator_once() -> None:
    supplied = _CountingIterator(complete_records())

    trace = build_trace(supplied)

    assert supplied.consumed == 7
    assert direct_diagnostics(trace).source_record_count == 7


def test_row_overflow_reads_only_the_limit_plus_one() -> None:
    action = complete_records()[1]
    rows = [complete_records()[0], *[dict(action) for _ in range(MAX_SHADOW_TRACE_ROWS)]]
    supplied = _CountingIterator(rows)

    with pytest.raises(ShadowTraceInputError) as raised:
        build_trace(supplied, capture_scope="unknown")

    assert raised.value.reason_code == "input_limit_exceeded"
    assert raised.value.step_ordinal == MAX_SHADOW_TRACE_ROWS + 1
    assert supplied.consumed == MAX_SHADOW_TRACE_ROWS + 1


def test_exact_row_limit_is_accepted_without_truncation() -> None:
    base = complete_records()
    rows = [base[0], *[dict(base[1]) for _ in range(MAX_SHADOW_TRACE_ROWS - 2)], base[-1]]

    trace = build_trace(rows)

    assert direct_diagnostics(trace).source_record_count == MAX_SHADOW_TRACE_ROWS
    assert len(trace.records) == MAX_SHADOW_TRACE_ROWS


def test_original_source_overflow_is_rejected_before_record_consumption() -> None:
    supplied = _CountingIterator(complete_records())

    with pytest.raises(ShadowTraceInputError) as raised:
        build_trace(supplied, source_bytes=b"x" * (MAX_SHADOW_TRACE_BYTES + 1))

    assert raised.value.reason_code == "input_limit_exceeded"
    assert supplied.consumed == 0


def test_exact_original_source_limit_is_accepted() -> None:
    source = b"x" * MAX_SHADOW_TRACE_BYTES

    trace = build_trace(source_bytes=source)

    assert trace.binding.source_byte_count == MAX_SHADOW_TRACE_BYTES
    assert trace.binding.source_byte_digest == hashlib.sha256(source).hexdigest()


def test_canonical_byte_limit_is_checked_before_full_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_canonical_json = canonical_json
    calls: list[object] = []

    def recording_canonical_json(value: object) -> bytes:
        calls.append(value)
        return real_canonical_json(value)

    rows = complete_records()
    rows[1]["command"] = "x" * 1_000
    monkeypatch.setattr(trace_module, "MAX_SHADOW_TRACE_BYTES", 512)
    monkeypatch.setattr(trace_module, "canonical_json", recording_canonical_json)

    with pytest.raises(ShadowTraceInputError) as raised:
        build_trace(rows)

    assert raised.value.reason_code == "input_limit_exceeded"
    assert raised.value.step_ordinal == 2
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("mutate", "reason", "ordinal"),
    (
        (lambda rows: rows.pop(0), "invalid_step", None),
        (lambda rows: rows.pop(), "invalid_step", None),
        (
            lambda rows: rows[2].update(action_source_event_id="missing-action"),
            "invalid_step",
            3,
        ),
        (
            lambda rows: rows[3].update(occurred_at="2026-07-17T08:59:59Z"),
            "invalid_timestamp",
            4,
        ),
        (
            lambda rows: rows[1].update(occurred_at="2026-07-17T09:00:01+00:00"),
            "invalid_timestamp",
            2,
        ),
    ),
)
def test_lifecycle_parent_and_timestamp_failures_are_structural(
    mutate: object,
    reason: str,
    ordinal: int | None,
) -> None:
    rows = complete_records()
    assert callable(mutate)
    mutate(rows)

    with pytest.raises(ShadowTraceInputError) as raised:
        build_trace(rows)

    assert str(raised.value) == "shadow input is invalid"
    assert raised.value.reason_code == reason
    assert raised.value.step_ordinal == ordinal
    assert "missing-action" not in repr(raised.value)


def test_retry_metadata_must_be_stable_and_is_counted() -> None:
    rows = complete_records()
    rows.insert(2, dict(rows[1]))

    trace = build_trace(rows)

    diagnostics = direct_diagnostics(trace)
    assert diagnostics.source_record_count == 8
    assert diagnostics.repeated_source_identifier_count == 1
    assert dict(diagnostics.input_kind_counts)[ShadowInputKind.ACTION] == 2

    rows[2]["occurred_at"] = "2026-07-17T09:00:02Z"
    with pytest.raises(ShadowTraceInputError) as raised:
        build_trace(rows)
    assert raised.value.reason_code == "invalid_step"
    assert raised.value.step_ordinal == 3


def test_logical_timestamps_are_strictly_increasing_for_unique_events() -> None:
    rows = complete_records()

    with pytest.raises(ShadowTraceInputError) as raised:
        build_trace(rows, timestamp_mode="logical_order")

    assert raised.value.reason_code == "invalid_timestamp"
    assert raised.value.step_ordinal == 3

    minimal = [rows[0], rows[1], rows[-1]]
    trace = build_trace(minimal, timestamp_mode="logical_order")
    assert trace.binding.timestamp_mode == "logical_order"

    retry = [rows[0], rows[1], dict(rows[1]), rows[-1]]
    with pytest.raises(ShadowTraceInputError) as retry_error:
        build_trace(retry, timestamp_mode="logical_order")
    assert retry_error.value.reason_code == "invalid_timestamp"
    assert retry_error.value.step_ordinal == 3


class _StringSubclass(str):
    pass


class _ExplosiveMapping(Mapping[str, object]):
    def __init__(self) -> None:
        self.executed = False

    def __getitem__(self, key: str) -> object:
        self.executed = True
        raise AssertionError(key)

    def __iter__(self) -> Iterator[str]:
        self.executed = True
        raise AssertionError("custom mapping was evaluated")

    def __len__(self) -> int:
        self.executed = True
        raise AssertionError("custom mapping was evaluated")


@pytest.mark.parametrize(
    "invalid_value",
    (
        _StringSubclass("shadow-input/v1"),
        ("pytest", "-q"),
        object(),
    ),
)
def test_python_values_must_already_be_exact_json_types(invalid_value: object) -> None:
    rows = complete_records()
    if isinstance(invalid_value, _StringSubclass):
        rows[0]["schema_version"] = invalid_value
    elif type(invalid_value) is tuple:
        rows[1]["argv"] = invalid_value
        rows[1].pop("command")
    else:
        rows[4]["payload"] = {"value": invalid_value}

    with pytest.raises(ShadowTraceInputError) as raised:
        build_trace(rows)

    assert raised.value.reason_code == "invalid_json"


def test_custom_mapping_is_rejected_without_executing_its_callbacks() -> None:
    rows = complete_records()
    supplied = _ExplosiveMapping()
    rows[4]["payload"] = {"value": supplied}

    with pytest.raises(ShadowTraceInputError) as raised:
        build_trace(rows)

    assert raised.value.reason_code == "invalid_json"
    assert not supplied.executed


def test_profile_descriptor_limits_are_exact_and_have_no_hidden_node_cap() -> None:
    valid_many_nodes: dict[str, object] = {"matrix": [[[] for _ in range(64)] for _ in range(64)]}
    assert len(canonical_json(valid_many_nodes)) < 16 * 1_024
    assert len(ShadowTrace.adapter_profile_digest(valid_many_nodes)) == 64

    exact_byte_limit: dict[str, object] = {"v": [*["x" * 1_024 for _ in range(15)], "x" * 969]}
    assert len(canonical_json(exact_byte_limit)) == 16 * 1_024
    assert len(ShadowTrace.adapter_profile_digest(exact_byte_limit)) == 64

    unicode_and_controls: dict[str, object] = {"value": 'quote" slash\\ newline\n é 😀 \b'}
    assert len(ShadowTrace.adapter_profile_digest(unicode_and_controls)) == 64

    invalid_descriptors: tuple[dict[str, object], ...] = (
        {"items": [None] * 65},
        {"value": "x" * 1_025},
        {"value": float("nan")},
        {"v": [*["x" * 1_024 for _ in range(15)], "x" * 970]},
    )
    for descriptor in invalid_descriptors:
        with pytest.raises(ShadowTraceInputError) as raised:
            ShadowTrace.adapter_profile_digest(descriptor)
        assert raised.value.reason_code == "invalid_step"
        assert raised.value.__cause__ is None
        assert raised.value.__context__ is None


def test_profile_and_configuration_identity_change_with_mapping_rules() -> None:
    first = build_trace(adapter_descriptor={"schema_version": "example/v1", "mapping": "a"})
    second = build_trace(adapter_descriptor={"schema_version": "example/v1", "mapping": "b"})

    assert first.binding.adapter_profile_digest != second.binding.adapter_profile_digest
    assert first.binding.adapter_configuration_digest != second.binding.adapter_configuration_digest


def test_direct_factory_reserves_the_built_in_atif_source_format() -> None:
    with pytest.raises(ShadowTraceInputError) as raised:
        build_trace(source_bytes=b"{}", source_format="atif", source_schema_version="ATIF-v1.7")

    assert raised.value.reason_code == "profile_mismatch"


def test_canonical_record_source_cannot_be_mislabeled_as_a_native_format() -> None:
    with pytest.raises(ShadowTraceInputError) as format_error:
        build_trace(source_format="custom")
    with pytest.raises(ShadowTraceInputError) as schema_error:
        build_trace(source_schema_version="custom/v1")

    assert format_error.value.reason_code == "profile_mismatch"
    assert schema_error.value.reason_code == "unsupported_schema"


def test_invalid_descriptor_and_caller_digests_fail_without_echoing_values() -> None:
    descriptor: dict[str, object] = {"secret": "credential-value"}
    descriptor["cycle"] = descriptor

    with pytest.raises(ShadowTraceInputError) as descriptor_error:
        build_trace(adapter_descriptor=descriptor)
    with pytest.raises(ShadowTraceInputError) as digest_error:
        build_trace(task_scope_digest="credential-value")

    for error in (descriptor_error.value, digest_error.value):
        assert str(error) == "shadow input is invalid"
        assert "credential" not in repr(error)


def test_wrapped_validation_and_iterator_errors_have_no_exception_context() -> None:
    rows = complete_records()
    rows[1]["working_directory"] = "/sentinel/secret\x00path"

    with pytest.raises(ShadowTraceInputError) as validation_error:
        build_trace(rows)

    class FailingIterator(Iterator[dict[str, object]]):
        def __next__(self) -> dict[str, object]:
            raise RuntimeError("sentinel iterator secret")

    with pytest.raises(ShadowTraceInputError) as iterator_error:
        build_trace(FailingIterator())

    for error in (validation_error.value, iterator_error.value):
        assert str(error) == "shadow input is invalid"
        assert "sentinel" not in repr(error)
        assert error.__cause__ is None
        assert error.__context__ is None


def test_trace_error_exposes_only_stable_reason_and_ordinals() -> None:
    error = ShadowTraceInputError(
        "orphan_result",
        step_ordinal=2,
        call_ordinal=3,
        result_ordinal=4,
    )

    assert isinstance(error, ShadowInputError)
    assert str(error) == "shadow input is invalid"
    assert repr(error) == "ShadowTraceInputError(reason_code='orphan_result')"
    assert error.reason_code == "orphan_result"
    assert error.step_ordinal == 2
    assert error.call_ordinal == 3
    assert error.result_ordinal == 4
    with pytest.raises(TypeError):
        ShadowTraceInputError("caller-secret")
    with pytest.raises(TypeError):
        ShadowTraceInputError("orphan_result", step_ordinal=0)
