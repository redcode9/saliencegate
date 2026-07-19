from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import pytest
from tests.shadow.conftest import NOW, OTHER_RUN_ID, RUN_ID, TraceEventFactory
from tests.shadow.test_report import (
    CAPTURE_MANIFEST_DIGEST,
    LINEAGE_SCOPE_DIGEST,
    NORMALIZED_INPUT_DIGEST,
    TASK_SCOPE_DIGEST,
    _make_case,
)

import saliencegate.shadow.trace_report as trace_report_module
from saliencegate.domain import canonical_json, length_prefixed_sha256
from saliencegate.security import SecureFileError
from saliencegate.shadow.errors import ShadowInvariantError, ShadowTraceInputError
from saliencegate.shadow.io import (
    authorize_shadow_trace_report_publication,
    decode_shadow_run_report,
    encode_shadow_run_report,
    shadow_trace_report_binding,
    validate_published_shadow_trace_report,
    validate_shadow_trace_report_binding,
    validate_shadow_trace_report_replacement,
)
from saliencegate.shadow.report import ShadowRunReport, build_shadow_run_report
from saliencegate.shadow.trace import (
    ATIFShadowDiagnostics,
    CaptureScope,
    ShadowTrace,
    ShadowTraceBinding,
    _build_atif_diagnostics,
    _build_binding,
)
from saliencegate.shadow.trace_report import (
    MAX_SHADOW_TRACE_REPORT_BYTES,
    ShadowTraceReport,
    _build_shadow_trace_report,
    decode_shadow_trace_report,
    encode_shadow_trace_report,
    verify_shadow_trace_source,
)

_REPORT_DIGEST_DOMAIN = "saliencegate:shadow:trace-report:v1"
_CONFIGURATION_DIGEST_DOMAIN = "saliencegate:shadow:adapter-configuration:v1"
_SOURCE = b'{"native":"private-command /private/project"}'
_DESCRIPTOR: dict[str, object] = {
    "schema_version": "example-shadow-adapter/v1",
    "mapping": {"mode": "documented"},
}


@dataclass(frozen=True)
class TraceReportCase:
    source: bytes
    trace: ShadowTrace
    shadow_report: ShadowRunReport
    report: ShadowTraceReport


def _timestamp(offset: int) -> str:
    return (NOW + timedelta(seconds=offset)).isoformat().replace("+00:00", "Z")


def _records(*, command: str = "private-command") -> list[dict[str, object]]:
    tool_result: dict[str, object] = {
        "schema_version": "shadow-input/v1",
        "kind": "tool_result",
        "source_event_id": "shadow-source-3",
        "occurred_at": _timestamp(3),
        "action_source_event_id": "shadow-source-2",
        "status": "failed",
        "exit_status": 1,
        "error_code": "TEST_FAILURE",
    }
    return [
        {
            "schema_version": "shadow-input/v1",
            "kind": "run_start",
            "source_event_id": "shadow-source-1",
            "occurred_at": _timestamp(1),
        },
        {
            "schema_version": "shadow-input/v1",
            "kind": "action",
            "source_event_id": "shadow-source-2",
            "occurred_at": _timestamp(2),
            "command": command,
            "working_directory": "/private/project",
            "environment_digest": "a" * 64,
        },
        tool_result,
        dict(tool_result),
        {
            "schema_version": "shadow-input/v1",
            "kind": "controller_error",
            "source_event_id": "shadow-source-4",
            "occurred_at": _timestamp(4),
            "error_code": "controller_timeout",
        },
        {
            "schema_version": "shadow-input/v1",
            "kind": "run_end",
            "source_event_id": "shadow-source-5",
            "occurred_at": _timestamp(5),
        },
    ]


def _trace(
    *,
    records: list[dict[str, object]] | None = None,
    run_id: UUID = RUN_ID,
    source: bytes = _SOURCE,
    adapter_descriptor: dict[str, object] | None = None,
    capture_scope: CaptureScope = "complete_run_declared",
    task_scope_digest: str | None = TASK_SCOPE_DIGEST,
    lineage_scope_digest: str | None = LINEAGE_SCOPE_DIGEST,
    capture_manifest_digest: str | None = CAPTURE_MANIFEST_DIGEST,
) -> ShadowTrace:
    return ShadowTrace.from_records(
        _records() if records is None else records,
        run_id=run_id,
        adapter_profile_id="example/v1",
        adapter_descriptor=_DESCRIPTOR if adapter_descriptor is None else adapter_descriptor,
        source_bytes=source,
        source_format="example",
        source_schema_version="example/v1",
        capture_scope=capture_scope,
        task_scope_digest=task_scope_digest,
        lineage_scope_digest=lineage_scope_digest,
        capture_manifest_digest=capture_manifest_digest,
    )


def _matching_shadow_report(
    trace_event_factory: TraceEventFactory,
    trace: ShadowTrace,
    *,
    capture_scope: CaptureScope | None = None,
) -> ShadowRunReport:
    base = _make_case(trace_event_factory, run_id=trace.run_id)
    selected_capture_scope = trace.binding.capture_scope if capture_scope is None else capture_scope
    return build_shadow_run_report(
        run_id=trace.run_id,
        initial_ledger_entry_count=base.report.initial_ledger_entry_count,
        initial_ledger_chain_tag=base.report.initial_ledger_chain_tag,
        initial_ledger_projection_tag=base.report.initial_ledger_projection_tag,
        initial_ledger_head_tag=base.report.initial_ledger_head_tag,
        input_byte_digest=trace.binding.source_byte_digest,
        normalized_input_digest=NORMALIZED_INPUT_DIGEST,
        redaction_policy_tag=base.report.redaction_policy_tag,
        detector_profile_digest=base.report.detector_profile_digest,
        capture_scope=selected_capture_scope,
        task_scope_digest=trace.binding.task_scope_digest,
        lineage_scope_digest=trace.binding.lineage_scope_digest,
        capture_manifest_digest=trace.binding.capture_manifest_digest,
        rows=base.rows,
        observations=base.observations,
    )


def _build_report(trace: ShadowTrace, shadow_report: ShadowRunReport) -> ShadowTraceReport:
    return _build_shadow_trace_report(
        trace=trace,
        shadow_report=shadow_report,
        session_binding=trace.binding,
        authenticated_start_source_adapter=trace.binding.source_adapter,
    )


@pytest.fixture
def trace_report_case(trace_event_factory: TraceEventFactory) -> TraceReportCase:
    trace = _trace()
    shadow_report = _matching_shadow_report(trace_event_factory, trace)
    return TraceReportCase(
        source=_SOURCE,
        trace=trace,
        shadow_report=shadow_report,
        report=_build_report(trace, shadow_report),
    )


def _report_body(report: ShadowTraceReport) -> dict[str, Any]:
    body = report.model_dump(mode="json", warnings=False)
    assert type(body) is dict
    return body


def _resign(body: dict[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result.pop("report_digest", None)
    result["report_digest"] = length_prefixed_sha256(
        canonical_json(result),
        domain=_REPORT_DIGEST_DOMAIN,
    )
    return result


def _adapter_configuration_digest(
    *,
    profile_id: str,
    profile_digest: str,
    source_format: str,
    source_schema_version: str,
    timestamp_mode: str,
    capture_scope: str,
) -> str:
    return length_prefixed_sha256(
        canonical_json(
            {
                "schema_version": "shadow-adapter-configuration/v1",
                "adapter_profile_id": profile_id,
                "adapter_profile_digest": profile_digest,
                "source_format": source_format,
                "source_schema_version": source_schema_version,
                "timestamp_mode": timestamp_mode,
                "capture_scope": capture_scope,
            }
        ),
        domain=_CONFIGURATION_DIGEST_DOMAIN,
    )


def _decode_body(body: dict[str, Any]) -> ShadowTraceReport:
    return decode_shadow_trace_report(canonical_json(body))


def test_builder_and_codec_round_trip_exact_canonical_bytes(
    trace_report_case: TraceReportCase,
) -> None:
    trace = trace_report_case.trace
    shadow_report = trace_report_case.shadow_report
    report = trace_report_case.report

    assert set(ShadowTraceReport.model_fields) == {
        "schema_version",
        "run_id",
        "binding",
        "binding_digest",
        "diagnostics",
        "diagnostics_digest",
        "mapped_record_digest",
        "normalized_input_digest",
        "shadow_report",
        "report_digest",
    }
    assert report.schema_version == "shadow-trace-report/v1"
    assert report.run_id == trace.run_id == shadow_report.run_id
    assert report.binding == trace.binding
    assert report.binding is not trace.binding
    assert report.binding_digest == trace.binding.binding_digest
    assert report.diagnostics == trace.diagnostics
    assert report.diagnostics is not trace.diagnostics
    assert report.diagnostics_digest == trace.diagnostics.diagnostics_digest
    assert report.mapped_record_digest == trace.mapped_record_digest
    assert report.normalized_input_digest == shadow_report.normalized_input_digest
    assert report.shadow_report == shadow_report
    assert report.shadow_report is not shadow_report
    assert report.report_digest == _resign(_report_body(report))["report_digest"]
    assert report.report_digest == (
        "a81f70be1b75bb4ffe801602b06f83543a74e75de21ce4a9d7fa5af80a74d9ae"
    )

    encoded = encode_shadow_trace_report(report)
    decoded = decode_shadow_trace_report(encoded)

    assert MAX_SHADOW_TRACE_REPORT_BYTES == 130 * 1_024 * 1_024
    assert encoded == canonical_json(report.model_dump(mode="json", warnings=False))
    assert len(encoded) == 29_238
    assert hashlib.sha256(encoded).hexdigest() == (
        "35e27bb5985f1f685e8e308bb89cc4272e75fccdb0e6b72d2d052c9ac5b71196"
    )
    assert not encoded.endswith(b"\n")
    assert decoded == report
    assert decoded is not report
    assert encode_shadow_trace_report(decoded) == encoded
    assert "private-command" not in repr(report)
    assert "/private/project" not in repr(report)


def test_builder_is_keyword_only_and_rejects_live_invariant_mismatches(
    trace_event_factory: TraceEventFactory,
    trace_report_case: TraceReportCase,
) -> None:
    trace = trace_report_case.trace
    shadow_report = trace_report_case.shadow_report

    with pytest.raises(TypeError):
        _build_shadow_trace_report(  # type: ignore[misc]
            trace,
            shadow_report,
            trace.binding,
            trace.binding.source_adapter,
        )

    changed_descriptor = {
        "schema_version": "example-shadow-adapter/v1",
        "mapping": {"mode": "changed"},
    }
    other_binding = _trace(adapter_descriptor=changed_descriptor).binding
    with pytest.raises(ShadowInvariantError):
        _build_shadow_trace_report(
            trace=trace,
            shadow_report=shadow_report,
            session_binding=other_binding,
            authenticated_start_source_adapter=trace.binding.source_adapter,
        )
    with pytest.raises(ShadowInvariantError):
        _build_shadow_trace_report(
            trace=trace,
            shadow_report=shadow_report,
            session_binding=trace.binding,
            authenticated_start_source_adapter="changed-adapter/v1",
        )

    other_run_trace = _trace(run_id=OTHER_RUN_ID)
    with pytest.raises(ShadowInvariantError):
        _build_report(other_run_trace, shadow_report)

    changed_source_trace = _trace(source=b"changed exact source")
    with pytest.raises(ShadowInvariantError):
        _build_report(changed_source_trace, shadow_report)

    short_trace = _trace(records=[_records()[0], _records()[-1]])
    with pytest.raises(ShadowInvariantError):
        _build_report(short_trace, shadow_report)

    changed_identity_records = _records()
    changed_identity_records[0]["source_event_id"] = "other-source-1"
    changed_identity_records[1]["source_event_id"] = "other-source-2"
    changed_identity_records[2]["source_event_id"] = "other-source-3"
    changed_identity_records[2]["action_source_event_id"] = "other-source-2"
    changed_identity_records[3]["source_event_id"] = "other-source-3"
    changed_identity_records[3]["action_source_event_id"] = "other-source-2"
    changed_identity_records[4]["source_event_id"] = "other-source-4"
    changed_identity_records[5]["source_event_id"] = "other-source-5"
    changed_identity_trace = _trace(records=changed_identity_records)
    with pytest.raises(ShadowInvariantError):
        _build_report(changed_identity_trace, shadow_report)

    damaged_trace = _trace()
    object.__setattr__(damaged_trace, "mapped_record_digest", "0" * 64)
    with pytest.raises(ShadowInvariantError):
        _build_report(damaged_trace, shadow_report)

    damaged_diagnostics_trace = _trace()
    object.__setattr__(
        damaged_diagnostics_trace.diagnostics,
        "mapped_shadow_record_count",
        len(damaged_diagnostics_trace.records) - 1,
    )
    with pytest.raises(ShadowInvariantError):
        _build_report(damaged_diagnostics_trace, shadow_report)

    mismatched_nested_report = _matching_shadow_report(
        trace_event_factory,
        _trace(source=b"another source"),
    )
    with pytest.raises(ShadowInvariantError):
        _build_report(trace, mismatched_nested_report)


def test_decoder_rejects_every_unresigned_top_level_mutation(
    trace_report_case: TraceReportCase,
) -> None:
    original = _report_body(trace_report_case.report)
    mutations: tuple[tuple[str, object], ...] = (
        ("schema_version", "shadow-trace-report/v2"),
        ("run_id", str(OTHER_RUN_ID)),
        ("binding_digest", "0" * 64),
        ("diagnostics_digest", "0" * 64),
        ("mapped_record_digest", "0" * 64),
        ("normalized_input_digest", "0" * 64),
        ("report_digest", "0" * 64),
    )
    candidates: list[dict[str, Any]] = []
    for field_name, changed_value in mutations:
        candidate = json.loads(json.dumps(original))
        candidate[field_name] = changed_value
        candidates.append(candidate)

    changed_binding = json.loads(json.dumps(original))
    changed_binding["binding"]["source_byte_count"] += 1
    candidates.append(changed_binding)
    changed_diagnostics = json.loads(json.dumps(original))
    changed_diagnostics["diagnostics"]["source_record_count"] -= 1
    candidates.append(changed_diagnostics)
    changed_nested = json.loads(json.dumps(original))
    changed_nested["shadow_report"]["report_digest"] = "0" * 64
    candidates.append(changed_nested)

    for candidate in candidates:
        with pytest.raises(ShadowInvariantError):
            _decode_body(candidate)


def test_decoder_rechecks_self_contained_crosslinks_after_valid_outer_resigning(
    trace_report_case: TraceReportCase,
) -> None:
    original = _report_body(trace_report_case.report)
    candidates: list[dict[str, Any]] = []
    for field_name, value in (
        ("run_id", str(OTHER_RUN_ID)),
        ("binding_digest", "0" * 64),
        ("diagnostics_digest", "0" * 64),
        ("normalized_input_digest", "0" * 64),
    ):
        candidate = json.loads(json.dumps(original))
        candidate[field_name] = value
        candidates.append(_resign(candidate))

    direct_with_atif_diagnostics = json.loads(json.dumps(original))
    atif_diagnostics = _atif_diagnostics(mapped_shadow_record_count=6)
    direct_with_atif_diagnostics["diagnostics"] = atif_diagnostics.model_dump(
        mode="json", warnings=False
    )
    direct_with_atif_diagnostics["diagnostics_digest"] = atif_diagnostics.diagnostics_digest
    candidates.append(_resign(direct_with_atif_diagnostics))

    for changed_trace in (
        _trace(source=b"different exact source"),
        _trace(capture_scope="selected_events"),
        _trace(task_scope_digest="6" * 64),
        _trace(lineage_scope_digest="7" * 64),
        _trace(capture_manifest_digest="8" * 64),
    ):
        candidate = json.loads(json.dumps(original))
        candidate["binding"] = changed_trace.binding.model_dump(mode="json", warnings=False)
        candidate["binding_digest"] = changed_trace.binding.binding_digest
        candidates.append(_resign(candidate))

    for candidate in candidates:
        with pytest.raises(ShadowInvariantError):
            _decode_body(candidate)


def test_decoder_does_not_claim_builder_only_mapped_record_authentication(
    trace_report_case: TraceReportCase,
) -> None:
    body = _report_body(trace_report_case.report)
    body["mapped_record_digest"] = "f" * 64
    self_consistent_bytes = canonical_json(_resign(body))

    decoded = decode_shadow_trace_report(self_consistent_bytes)

    assert decoded.mapped_record_digest == "f" * 64
    assert decoded.binding == trace_report_case.report.binding
    assert decoded.shadow_report == trace_report_case.report.shadow_report
    assert encode_shadow_trace_report(decoded) == self_consistent_bytes


def _atif_diagnostics(
    *,
    mapped_shadow_record_count: int,
    outcome_evidence_authority: Literal[
        "none", "producer_claimed_structured"
    ] = "producer_claimed_structured",
    profile_audit_manifest_digest: str = "9" * 64,
) -> ATIFShadowDiagnostics:
    mapped_actions = mapped_shadow_record_count - 3
    return _build_atif_diagnostics(
        continued_trajectory_ref_present=False,
        embedded_subagent_trajectory_count=0,
        outcome_evidence_authority=outcome_evidence_authority,
        profile_audit_manifest_digest=profile_audit_manifest_digest,
        total_step_count=3,
        ignored_message_step_count=0,
        total_tool_call_count=mapped_actions,
        tool_call_disposition_counts=(
            ("mapped_action", mapped_actions),
            ("ignored_unsupported_function", 0),
            ("ignored_continuation", 0),
            ("ignored_non_command_wait", 0),
            ("ignored_unsubmitted_keystrokes", 0),
            ("ignored_unresolved_terminal_submission", 0),
            ("ignored_copied_context", 0),
        ),
        total_observation_result_count=1,
        result_disposition_counts=(
            ("mapped_structured_outcome", 1),
            ("ignored_evidence_absent", 0),
            ("ignored_ambiguous_parent", 0),
            ("ignored_no_parent", 0),
            ("ignored_unsupported_parent", 0),
            ("ignored_copied_context", 0),
        ),
        mapped_shadow_record_count=mapped_shadow_record_count,
    )


def _atif_binding(*, profile_digest: str = "6" * 64) -> ShadowTraceBinding:
    configuration_digest = _adapter_configuration_digest(
        profile_id="harbor-codex/v1",
        profile_digest=profile_digest,
        source_format="atif",
        source_schema_version="ATIF-v1.7",
        timestamp_mode="logical_order",
        capture_scope="selected_events",
    )
    return _build_binding(
        source_format="atif",
        source_schema_version="ATIF-v1.7",
        source_digest_kind="original_bytes",
        source_bytes=_SOURCE,
        adapter_profile_id="harbor-codex/v1",
        adapter_profile_digest=profile_digest,
        adapter_configuration_digest=configuration_digest,
        timestamp_mode="logical_order",
        capture_scope="selected_events",
        task_scope_digest=TASK_SCOPE_DIGEST,
        lineage_scope_digest=LINEAGE_SCOPE_DIGEST,
        capture_manifest_digest=CAPTURE_MANIFEST_DIGEST,
    )


def _atif_report_body(
    trace_event_factory: TraceEventFactory,
) -> dict[str, Any]:
    binding = _atif_binding()
    direct_trace = _trace(capture_scope="selected_events")
    shadow_report = _matching_shadow_report(
        trace_event_factory,
        direct_trace,
        capture_scope="selected_events",
    )
    diagnostics = _atif_diagnostics(mapped_shadow_record_count=shadow_report.input_row_count)
    return {
        "schema_version": "shadow-trace-report/v1",
        "run_id": str(shadow_report.run_id),
        "binding": binding.model_dump(mode="json", warnings=False),
        "binding_digest": binding.binding_digest,
        "diagnostics": diagnostics.model_dump(mode="json", warnings=False),
        "diagnostics_digest": diagnostics.diagnostics_digest,
        "mapped_record_digest": "8" * 64,
        "normalized_input_digest": shadow_report.normalized_input_digest,
        "shadow_report": shadow_report.model_dump(mode="json", warnings=False),
    }


def test_decoder_fails_closed_for_atif_until_sealed_profiles_are_supported(
    trace_event_factory: TraceEventFactory,
) -> None:
    with pytest.raises(ShadowInvariantError):
        _decode_body(_resign(_atif_report_body(trace_event_factory)))


def test_decoder_rejects_an_impossible_atif_row_topology_even_with_mocked_profile_links(
    monkeypatch: pytest.MonkeyPatch,
    trace_event_factory: TraceEventFactory,
) -> None:
    monkeypatch.setattr(
        trace_report_module,
        "_sealed_atif_report_contract",
        lambda profile_id: (
            ("6" * 64, "9" * 64, "producer_claimed_structured")
            if profile_id == "harbor-codex/v1"
            else None
        ),
    )
    body = _atif_report_body(trace_event_factory)

    with pytest.raises(ShadowInvariantError):
        _decode_body(_resign(body))


@pytest.mark.parametrize(
    "invalid_data",
    (
        b"",
        b"[]",
        b"not-json",
        b"\xff",
        b'{"value":NaN}',
    ),
)
def test_codec_rejects_invalid_json_boundaries(invalid_data: bytes) -> None:
    with pytest.raises(ShadowInvariantError):
        decode_shadow_trace_report(invalid_data)


def test_codec_is_strict_about_duplicates_canonical_form_unknown_fields_and_bounds(
    monkeypatch: pytest.MonkeyPatch,
    trace_report_case: TraceReportCase,
) -> None:
    report = trace_report_case.report
    encoded = encode_shadow_trace_report(report)
    duplicate = b'{"schema_version":"shadow-trace-report/v1",' + encoded[1:]
    unknown = _report_body(report)
    unknown["unknown"] = True
    unknown_bytes = canonical_json(_resign(unknown))

    for invalid in (
        encoded + b"\n",
        b" " + encoded,
        encoded.replace(b":", b": ", 1),
        duplicate,
        unknown_bytes,
    ):
        with pytest.raises(ShadowInvariantError):
            decode_shadow_trace_report(invalid)
    with pytest.raises(ShadowInvariantError):
        decode_shadow_trace_report(bytearray(encoded))  # type: ignore[arg-type]
    with pytest.raises(ShadowInvariantError):
        encode_shadow_trace_report(object())  # type: ignore[arg-type]

    monkeypatch.setattr(
        trace_report_module,
        "MAX_SHADOW_TRACE_REPORT_BYTES",
        len(encoded) - 1,
    )
    with pytest.raises(ShadowInvariantError):
        encode_shadow_trace_report(report)
    with pytest.raises(ShadowInvariantError):
        decode_shadow_trace_report(encoded)


def test_codec_preflights_structure_before_materializing_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser_called = False
    parser = json.loads

    def recording_parser(*args: object, **kwargs: object) -> object:
        nonlocal parser_called
        parser_called = True
        return parser(*args, **kwargs)

    monkeypatch.setattr(trace_report_module.json, "loads", recording_parser)
    monkeypatch.setattr(trace_report_module, "_MAX_JSON_STRUCTURAL_TOKENS", 0)

    with pytest.raises(ShadowInvariantError):
        decode_shadow_trace_report(b"{}")

    assert parser_called is False


def test_codec_surfaces_are_sanitized_and_not_cross_compatible(
    trace_report_case: TraceReportCase,
) -> None:
    encoded = encode_shadow_trace_report(trace_report_case.report)
    nested = encode_shadow_run_report(trace_report_case.shadow_report)

    with pytest.raises(ShadowInvariantError):
        decode_shadow_run_report(encoded)
    with pytest.raises(ShadowInvariantError):
        decode_shadow_trace_report(nested)

    sentinel = "private-source-sentinel"
    with pytest.raises(ShadowInvariantError) as raised:
        decode_shadow_trace_report(canonical_json({"unknown": sentinel}))
    error = raised.value
    assert sentinel not in str(error)
    assert sentinel not in repr(error)
    assert error.__cause__ is None
    assert error.__context__ is None


class _RecordingAdapter:
    def __init__(
        self,
        trace: ShadowTrace,
        *,
        profile_id: str | None = None,
        profile_digest: str | None = None,
    ) -> None:
        self.trace = trace
        self._profile_id = profile_id
        self._profile_digest = profile_digest
        self.calls: list[tuple[bytes, UUID, str | None, str | None, str | None]] = []

    @property
    def profile_id(self) -> str:
        return (
            self.trace.binding.adapter_profile_id if self._profile_id is None else self._profile_id
        )

    @property
    def profile_digest(self) -> str:
        return (
            self.trace.binding.adapter_profile_digest
            if self._profile_digest is None
            else self._profile_digest
        )

    def adapt_bytes(
        self,
        source: bytes,
        *,
        run_id: UUID,
        task_scope_digest: str | None = None,
        lineage_scope_digest: str | None = None,
        capture_manifest_digest: str | None = None,
    ) -> ShadowTrace:
        self.calls.append(
            (
                source,
                run_id,
                task_scope_digest,
                lineage_scope_digest,
                capture_manifest_digest,
            )
        )
        return self.trace


def test_verify_shadow_trace_source_readapts_exact_bytes_with_report_provenance(
    trace_report_case: TraceReportCase,
) -> None:
    adapter = _RecordingAdapter(trace_report_case.trace)

    verified = verify_shadow_trace_source(
        trace_report_case.report,
        trace_report_case.source,
        adapter=adapter,
    )

    assert verified is trace_report_case.trace
    assert adapter.calls == [
        (
            trace_report_case.source,
            trace_report_case.report.run_id,
            TASK_SCOPE_DIGEST,
            LINEAGE_SCOPE_DIGEST,
            CAPTURE_MANIFEST_DIGEST,
        )
    ]


def test_verify_shadow_trace_source_detects_source_mapping_and_configuration_changes(
    trace_report_case: TraceReportCase,
) -> None:
    report = trace_report_case.report

    source_adapter = _RecordingAdapter(trace_report_case.trace)
    with pytest.raises(ShadowTraceInputError) as changed_source:
        verify_shadow_trace_source(report, b"changed source", adapter=source_adapter)
    assert changed_source.value.reason_code == "digest_mismatch"
    assert source_adapter.calls == []

    changed_records = _records(command="changed-command")
    mapping_adapter = _RecordingAdapter(_trace(records=changed_records))
    with pytest.raises(ShadowTraceInputError) as changed_mapping:
        verify_shadow_trace_source(report, _SOURCE, adapter=mapping_adapter)
    assert changed_mapping.value.reason_code == "digest_mismatch"

    configuration_adapter = _RecordingAdapter(_trace(capture_scope="selected_events"))
    with pytest.raises(ShadowTraceInputError) as changed_configuration:
        verify_shadow_trace_source(report, _SOURCE, adapter=configuration_adapter)
    assert changed_configuration.value.reason_code == "digest_mismatch"

    for mismatched_adapter in (
        _RecordingAdapter(trace_report_case.trace, profile_id="other/v1"),
        _RecordingAdapter(trace_report_case.trace, profile_digest="f" * 64),
    ):
        with pytest.raises(ShadowTraceInputError) as profile_mismatch:
            verify_shadow_trace_source(report, _SOURCE, adapter=mismatched_adapter)
        assert profile_mismatch.value.reason_code == "profile_mismatch"
        assert mismatched_adapter.calls == []


def test_verify_shadow_trace_source_fails_closed_for_invalid_boundary_types(
    trace_report_case: TraceReportCase,
) -> None:
    adapter = _RecordingAdapter(trace_report_case.trace)

    with pytest.raises(ShadowInvariantError):
        verify_shadow_trace_source(  # type: ignore[arg-type]
            object(),
            trace_report_case.source,
            adapter=adapter,
        )
    with pytest.raises(ShadowTraceInputError):
        verify_shadow_trace_source(
            trace_report_case.report,
            bytearray(trace_report_case.source),  # type: ignore[arg-type]
            adapter=adapter,
        )
    assert hashlib.sha256(trace_report_case.source).hexdigest() == (
        trace_report_case.report.binding.source_byte_digest
    )


def test_trace_report_publication_validators_require_exact_canonical_identity(
    trace_report_case: TraceReportCase,
) -> None:
    report = trace_report_case.report
    encoded = encode_shadow_trace_report(report)

    assert validate_published_shadow_trace_report(encoded, report)
    assert validate_shadow_trace_report_replacement(encoded, report)
    assert not validate_published_shadow_trace_report(encoded + b"\n", report)
    assert not validate_shadow_trace_report_replacement(encoded[:-1], report)
    assert not validate_published_shadow_trace_report(encoded, object())  # type: ignore[arg-type]


def test_trace_report_pre_repository_binding_rejects_a_different_valid_trace(
    trace_report_case: TraceReportCase,
    trace_event_factory: TraceEventFactory,
) -> None:
    report = trace_report_case.report
    binding = shadow_trace_report_binding(report)
    changed_trace = _trace(source=b'{"native":"different"}')
    changed_report = _build_report(
        changed_trace,
        _matching_shadow_report(trace_event_factory, changed_trace),
    )

    assert validate_shadow_trace_report_binding(
        encode_shadow_trace_report(report),
        binding,
    )
    assert not validate_shadow_trace_report_binding(
        encode_shadow_trace_report(changed_report),
        binding,
    )
    assert "private-command" not in repr(binding)


def test_trace_report_atomic_publication_creates_or_replaces_only_the_exact_report(
    trace_report_case: TraceReportCase,
    tmp_path: Path,
) -> None:
    report = trace_report_case.report
    encoded = encode_shadow_trace_report(report)
    output = tmp_path / "trace-report.json"

    publication = authorize_shadow_trace_report_publication(output)
    created = publication.publish(
        encoded,
        validate_published=lambda data: validate_published_shadow_trace_report(data, report),
    )
    assert created.data == encoded
    assert output.stat().st_mode & 0o777 == 0o600

    replacement = authorize_shadow_trace_report_publication(
        output,
        replacement_report=report,
    )
    reopened = replacement.publish(
        encoded,
        validate_published=lambda data: validate_published_shadow_trace_report(data, report),
    )
    assert reopened.data == encoded

    bound_replacement = authorize_shadow_trace_report_publication(
        output,
        replacement_binding=shadow_trace_report_binding(report),
    )
    bound_reopened = bound_replacement.publish(
        encoded,
        validate_published=lambda data: validate_published_shadow_trace_report(data, report),
    )
    assert bound_reopened.data == encoded

    output.write_bytes(b"{}")
    output.chmod(0o600)
    with pytest.raises(SecureFileError):
        authorize_shadow_trace_report_publication(
            output,
            replacement_report=report,
        )
