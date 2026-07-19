"""Canonical provenance commitments for complete Shadow traces."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from types import MappingProxyType
from typing import Literal, Never, Self, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from saliencegate.domain import canonical_json, length_prefixed_sha256
from saliencegate.domain.records import UUID4, Sha256Digest
from saliencegate.shadow.adapters import ShadowTraceAdapter
from saliencegate.shadow.errors import ShadowInvariantError, ShadowTraceInputError
from saliencegate.shadow.inputs import (
    ShadowInputKind,
    derive_shadow_event_id,
    derive_shadow_source_event_digest,
)
from saliencegate.shadow.report import (
    ShadowRunReport,
    _copy_exact_model,
    _model_state_is_exact,
    _require_trusted_shadow_run_report,
    _TrustedShadowRunReport,
)
from saliencegate.shadow.trace import (
    MAX_SHADOW_TRACE_BYTES,
    ATIFShadowDiagnostics,
    ShadowRecordDiagnostics,
    ShadowTrace,
    ShadowTraceBinding,
    ShadowTraceDiagnostics,
)

SHADOW_TRACE_REPORT_SCHEMA_VERSION: Literal["shadow-trace-report/v1"] = "shadow-trace-report/v1"
MAX_SHADOW_TRACE_REPORT_BYTES = 130 * 1_024 * 1_024

_TRACE_REPORT_DIGEST_DOMAIN = "saliencegate:shadow:trace-report:v1"
_MAPPED_RECORD_DIGEST_DOMAIN = "saliencegate:shadow:mapped-records:v1"
_MAX_JSON_DEPTH = 96
_MAX_JSON_NODES = 3_000_000
_MAX_JSON_CONTAINER_ITEMS = 100_000
_MAX_JSON_STRING_BYTES = 2 * 1_024 * 1_024
_MAX_JSON_SCALAR_BYTES = 128
_MAX_JSON_STRUCTURAL_TOKENS = 4_000_000
_BUILT_IN_ATIF_PROFILE_IDS = frozenset({"harbor-terminus-2/v1", "harbor-codex/v1"})
_WIRE_INPUT_KINDS = MappingProxyType(
    {
        "run_start": ShadowInputKind.START,
        "action": ShadowInputKind.ACTION,
        "tool_result": ShadowInputKind.TOOL_RESULT,
        "test_result": ShadowInputKind.TEST_RESULT,
        "observation": ShadowInputKind.OBSERVATION,
        "controller_error": ShadowInputKind.CONTROLLER_ERROR,
        "run_end": ShadowInputKind.FINISH,
    }
)


class _TraceReportModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


def _is_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_digest(value: object) -> object:
    if not _is_digest(value):
        raise ValueError("trace report digest is invalid")
    return value


def _models_match_exactly(left: BaseModel, right: BaseModel) -> bool:
    try:
        return type(left) is type(right) and hmac.compare_digest(
            canonical_json(left),
            canonical_json(right),
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return False


def _sealed_atif_report_contract(profile_id: str) -> tuple[str, str, str] | None:
    """Resolve one immutable built-in contract from the sealed ATIF registry."""

    if profile_id not in _BUILT_IN_ATIF_PROFILE_IDS:
        return None
    from saliencegate.shadow.atif import _sealed_report_contract

    contract = _sealed_report_contract(profile_id)
    if contract is None:
        return None
    if (
        type(contract) is not tuple
        or len(contract) != 3
        or not _is_digest(contract[0])
        or not _is_digest(contract[1])
        or type(contract[2]) is not str
        or contract[2] not in {"none", "producer_claimed_structured"}
    ):
        return None
    return contract


def _require_profile_diagnostic_links(
    binding: ShadowTraceBinding,
    diagnostics: ShadowTraceDiagnostics,
) -> None:
    is_atif = binding.source_format == "atif"
    if is_atif != (type(diagnostics) is ATIFShadowDiagnostics):
        raise ValueError("trace report diagnostic branch is invalid")
    if not is_atif:
        if type(diagnostics) is not ShadowRecordDiagnostics:
            raise ValueError("trace report diagnostic branch is invalid")
        if binding.adapter_profile_id in _BUILT_IN_ATIF_PROFILE_IDS:
            raise ValueError("built-in ATIF profile source is invalid")
        return

    if type(diagnostics) is not ATIFShadowDiagnostics:
        raise ValueError("ATIF report diagnostics are invalid")
    contract = _sealed_atif_report_contract(binding.adapter_profile_id)
    if contract is None:
        raise ValueError("ATIF report profile is not sealed")
    profile_digest, manifest_digest, outcome_authority = contract
    from saliencegate.shadow.atif import _matches_sealed_report_claims

    if (
        binding.source_digest_kind != "original_bytes"
        or binding.capture_scope != "selected_events"
        or not hmac.compare_digest(binding.adapter_profile_digest, profile_digest)
        or diagnostics.root_segment_only is not True
        or diagnostics.complete_execution_session_coverage is not False
        or diagnostics.producer_authentication != "none"
        or diagnostics.outcome_evidence_authority != outcome_authority
        or not hmac.compare_digest(
            diagnostics.profile_audit_manifest_digest,
            manifest_digest,
        )
        or not _matches_sealed_report_claims(
            profile_id=binding.adapter_profile_id,
            source_schema_version=binding.source_schema_version,
            timestamp_mode=binding.timestamp_mode,
            capture_scope=binding.capture_scope,
            diagnostics=diagnostics,
        )
    ):
        raise ValueError("ATIF report contract does not match its sealed profile")


class _ShadowTraceReportBody(_TraceReportModel):
    schema_version: Literal["shadow-trace-report/v1"] = SHADOW_TRACE_REPORT_SCHEMA_VERSION
    run_id: UUID4 = Field(repr=False)
    binding: ShadowTraceBinding
    binding_digest: Sha256Digest
    diagnostics: ShadowTraceDiagnostics
    diagnostics_digest: Sha256Digest
    mapped_record_digest: Sha256Digest
    normalized_input_digest: Sha256Digest
    shadow_report: ShadowRunReport

    @field_validator("schema_version", mode="before")
    @classmethod
    def require_exact_schema_version(cls, value: object) -> object:
        if type(value) is not str:
            raise ValueError("trace report schema version is invalid")
        return value

    @field_validator(
        "binding_digest",
        "diagnostics_digest",
        "mapped_record_digest",
        "normalized_input_digest",
        mode="before",
    )
    @classmethod
    def require_exact_digest_strings(cls, value: object) -> object:
        return _require_digest(value)

    @model_validator(mode="after")
    def validate_self_contained_commitments(self) -> Self:
        if (
            type(self.binding) is not ShadowTraceBinding
            or type(self.diagnostics) not in (ShadowRecordDiagnostics, ATIFShadowDiagnostics)
            or type(self.shadow_report) is not ShadowRunReport
            or not _model_state_is_exact(ShadowTraceBinding, self.binding)
            or not _model_state_is_exact(type(self.diagnostics), self.diagnostics)
            or not _model_state_is_exact(ShadowRunReport, self.shadow_report)
        ):
            raise ValueError("trace report nested model is invalid")
        if self.run_id != self.shadow_report.run_id:
            raise ValueError("trace report run identity does not match")
        if not hmac.compare_digest(self.binding_digest, self.binding.binding_digest):
            raise ValueError("trace report binding digest does not match")
        if not hmac.compare_digest(
            self.diagnostics_digest,
            self.diagnostics.diagnostics_digest,
        ):
            raise ValueError("trace report diagnostics digest does not match")
        _require_profile_diagnostic_links(self.binding, self.diagnostics)
        if type(self.diagnostics) is ATIFShadowDiagnostics:
            tool_counts = dict(self.diagnostics.tool_call_disposition_counts)
            result_counts = dict(self.diagnostics.result_disposition_counts)
            row_kinds = tuple(row.input_kind for row in self.shadow_report.rows)
            if (
                self.diagnostics.mapped_shadow_record_count != len(row_kinds)
                or any(
                    kind
                    not in {
                        ShadowInputKind.START,
                        ShadowInputKind.ACTION,
                        ShadowInputKind.TOOL_RESULT,
                        ShadowInputKind.FINISH,
                    }
                    for kind in row_kinds
                )
                or tool_counts["mapped_action"]
                != sum(kind is ShadowInputKind.ACTION for kind in row_kinds)
                or result_counts["mapped_structured_outcome"]
                != sum(kind is ShadowInputKind.TOOL_RESULT for kind in row_kinds)
            ):
                raise ValueError("ATIF report rows do not match mapping diagnostics")
        if not hmac.compare_digest(
            self.binding.source_byte_digest,
            self.shadow_report.input_byte_digest,
        ):
            raise ValueError("trace report source identity does not match")
        if (
            self.binding.capture_scope != self.shadow_report.capture_scope
            or self.binding.task_scope_digest != self.shadow_report.task_scope_digest
            or self.binding.lineage_scope_digest != self.shadow_report.lineage_scope_digest
            or self.binding.capture_manifest_digest != self.shadow_report.capture_manifest_digest
        ):
            raise ValueError("trace report capture provenance does not match")
        if not hmac.compare_digest(
            self.normalized_input_digest,
            self.shadow_report.normalized_input_digest,
        ):
            raise ValueError("trace report normalized identity does not match")
        return self


def _trace_report_body_digest(value: _ShadowTraceReportBody) -> str:
    fields = type(value).__pydantic_serializer__.to_python(
        value,
        mode="json",
        exclude={"report_digest"},
        warnings=False,
    )
    return length_prefixed_sha256(
        canonical_json(fields),
        domain=_TRACE_REPORT_DIGEST_DOMAIN,
    )


class ShadowTraceReport(_ShadowTraceReportBody):
    """A content-addressed provenance commitment, not a signature or truth proof."""

    report_digest: Sha256Digest

    @field_validator("report_digest", mode="before")
    @classmethod
    def require_exact_report_digest(cls, value: object) -> object:
        return _require_digest(value)

    @model_validator(mode="after")
    def validate_outer_digest(self) -> Self:
        if not hmac.compare_digest(self.report_digest, _trace_report_body_digest(self)):
            raise ValueError("trace report outer digest does not match")
        return self

    def __repr__(self) -> str:
        return "ShadowTraceReport(<provenance-commitment>)"


def _copy_trace_binding(value: object) -> ShadowTraceBinding:
    return _copy_exact_model(ShadowTraceBinding, value)


def _copy_trace_diagnostics(value: object) -> ShadowTraceDiagnostics:
    model_type = type(value)
    if model_type not in (ShadowRecordDiagnostics, ATIFShadowDiagnostics):
        raise ValueError("trace diagnostics are invalid")
    return cast(ShadowTraceDiagnostics, _copy_exact_model(model_type, value))


def _require_builder_evidence_links(
    trace: ShadowTrace,
    shadow_report: ShadowRunReport,
) -> None:
    records = trace.records
    rows = shadow_report.rows
    observations = shadow_report.observations
    if (
        len(records) != shadow_report.input_row_count
        or len(rows) != len(records)
        or trace.diagnostics.mapped_shadow_record_count != len(records)
    ):
        raise ValueError("trace report mapped record count does not match")

    first_positions: dict[str, int] = {}
    observation_index = 0
    for ordinal, (record, row) in enumerate(zip(records, rows, strict=True), start=1):
        source_event_id = record.get("source_event_id")
        wire_kind = record.get("kind")
        if type(source_event_id) is not str or type(wire_kind) is not str:
            raise ValueError("trace report mapped record identity is invalid")
        input_kind = _WIRE_INPUT_KINDS.get(wire_kind)
        if input_kind is None:
            raise ValueError("trace report mapped record kind is invalid")
        source_digest = derive_shadow_source_event_digest(trace.run_id, source_event_id)
        if (
            row.input_ordinal != ordinal
            or row.input_kind is not input_kind
            or not hmac.compare_digest(row.source_event_digest, source_digest)
        ):
            raise ValueError("trace report row does not match its mapped record")

        first_ordinal = first_positions.get(source_event_id)
        if first_ordinal is None:
            first_positions[source_event_id] = ordinal
            if (
                row.first_occurrence_ordinal != ordinal
                or row.retry_target_ordinal is not None
                or observation_index >= len(observations)
            ):
                raise ValueError("trace report first occurrence does not match")
            observation = observations[observation_index]
            observation_index += 1
            if (
                observation.event_id != derive_shadow_event_id(trace.run_id, source_event_id)
                or not hmac.compare_digest(observation.source_event_digest, source_digest)
                or observation.cli_input_ordinal != ordinal
                or not hmac.compare_digest(
                    observation.observation_digest,
                    row.observation_digest,
                )
            ):
                raise ValueError("trace report observation does not match its mapped record")
        elif row.first_occurrence_ordinal is not None or row.retry_target_ordinal != first_ordinal:
            raise ValueError("trace report retry does not match its mapped record")

    if observation_index != len(observations):
        raise ValueError("trace report observation count does not match")


def _build_shadow_trace_report(
    *,
    trace: ShadowTrace,
    shadow_report: ShadowRunReport,
    session_binding: ShadowTraceBinding,
    authenticated_start_source_adapter: str,
) -> ShadowTraceReport:
    """Build a report while the analyzer still owns trace and ledger evidence."""

    result: ShadowTraceReport | None = None
    try:
        if type(trace) is not ShadowTrace or not trace._is_exact():
            raise ValueError("trace is invalid")
        copied_binding = _copy_trace_binding(trace.binding)
        copied_session_binding = _copy_trace_binding(session_binding)
        copied_diagnostics = _copy_trace_diagnostics(trace.diagnostics)
        copied_report = _copy_exact_model(ShadowRunReport, shadow_report)
        if not _models_match_exactly(copied_binding, copied_session_binding):
            raise ValueError("session binding does not match")
        if trace.run_id != copied_report.run_id:
            raise ValueError("trace run identity does not match")
        _require_builder_evidence_links(trace, copied_report)
        expected_mapped_digest = length_prefixed_sha256(
            *trace._wire_record_bytes(),
            domain=_MAPPED_RECORD_DIGEST_DOMAIN,
        )
        if not hmac.compare_digest(trace.mapped_record_digest, expected_mapped_digest):
            raise ValueError("trace mapped identity does not match")
        if type(authenticated_start_source_adapter) is not str or not hmac.compare_digest(
            authenticated_start_source_adapter,
            copied_binding.source_adapter,
        ):
            raise ValueError("authenticated start adapter does not match")
        body = _ShadowTraceReportBody(
            run_id=trace.run_id,
            binding=copied_binding,
            binding_digest=copied_binding.binding_digest,
            diagnostics=copied_diagnostics,
            diagnostics_digest=copied_diagnostics.diagnostics_digest,
            mapped_record_digest=trace.mapped_record_digest,
            normalized_input_digest=copied_report.normalized_input_digest,
            shadow_report=copied_report,
        )
        candidate = ShadowTraceReport(
            **body.model_dump(mode="python", warnings=False),
            report_digest=_trace_report_body_digest(body),
        )
        result = _copy_exact_model(ShadowTraceReport, candidate)
    except Exception:
        result = None
    if result is None:
        raise ShadowInvariantError()
    return result


def _build_shadow_trace_report_trusted(
    *,
    trace: ShadowTrace,
    shadow_report: _TrustedShadowRunReport,
    session_binding: ShadowTraceBinding,
    authenticated_start_source_adapter: str,
) -> ShadowTraceReport:
    """Finish an analyzer-owned report without recursively recopying sealed evidence."""

    result: ShadowTraceReport | None = None
    try:
        if type(trace) is not ShadowTrace or not trace._is_exact():
            raise ValueError("trace is invalid")
        copied_binding = _copy_trace_binding(trace.binding)
        copied_session_binding = _copy_trace_binding(session_binding)
        copied_diagnostics = _copy_trace_diagnostics(trace.diagnostics)
        report = _require_trusted_shadow_run_report(shadow_report)
        if not _models_match_exactly(copied_binding, copied_session_binding):
            raise ValueError("session binding does not match")
        if trace.run_id != report.run_id:
            raise ValueError("trace run identity does not match")
        _require_builder_evidence_links(trace, report)
        _require_profile_diagnostic_links(copied_binding, copied_diagnostics)
        if (
            not hmac.compare_digest(
                copied_binding.source_byte_digest,
                report.input_byte_digest,
            )
            or copied_binding.capture_scope != report.capture_scope
            or copied_binding.task_scope_digest != report.task_scope_digest
            or copied_binding.lineage_scope_digest != report.lineage_scope_digest
            or copied_binding.capture_manifest_digest != report.capture_manifest_digest
        ):
            raise ValueError("trace capture provenance does not match")
        expected_mapped_digest = length_prefixed_sha256(
            *trace._wire_record_bytes(),
            domain=_MAPPED_RECORD_DIGEST_DOMAIN,
        )
        if not hmac.compare_digest(trace.mapped_record_digest, expected_mapped_digest):
            raise ValueError("trace mapped identity does not match")
        if type(authenticated_start_source_adapter) is not str or not hmac.compare_digest(
            authenticated_start_source_adapter,
            copied_binding.source_adapter,
        ):
            raise ValueError("authenticated start adapter does not match")
        body = _ShadowTraceReportBody.model_construct(
            run_id=trace.run_id,
            binding=copied_binding,
            binding_digest=copied_binding.binding_digest,
            diagnostics=copied_diagnostics,
            diagnostics_digest=copied_diagnostics.diagnostics_digest,
            mapped_record_digest=trace.mapped_record_digest,
            normalized_input_digest=report.normalized_input_digest,
            shadow_report=report,
        )
        candidate = ShadowTraceReport.model_construct(
            **body.__dict__,
            report_digest=_trace_report_body_digest(body),
        )
        if not _model_state_is_exact(
            _ShadowTraceReportBody,
            body,
        ) or not _model_state_is_exact(ShadowTraceReport, candidate):
            raise ValueError("trusted trace report construction failed")
        result = _copy_exact_model(ShadowTraceReport, candidate)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        result = None
    if result is None:
        raise ShadowInvariantError()
    return result


def _reject_json_constant(_token: str) -> Never:
    raise ValueError("non-finite JSON number")


def _finite_json_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    return value


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _preflight_canonical_json_structure(data: bytes) -> None:
    """Bound canonical JSON structure before allocating a decoded object tree."""

    stack: list[list[int]] = []
    in_string = False
    escaped = False
    string_bytes = 0
    scalar_bytes = 0
    structural_tokens = 0
    for byte in data:
        if in_string:
            string_bytes += 1
            if string_bytes > _MAX_JSON_STRING_BYTES:
                raise ValueError("trace report JSON string is too large")
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                in_string = False
                string_bytes = 0
            continue

        if byte == 0x22:
            in_string = True
            escaped = False
            string_bytes = 0
            scalar_bytes = 0
        elif byte in (0x20, 0x09, 0x0A, 0x0D):
            raise ValueError("trace report JSON is not canonical")
        elif byte in (0x7B, 0x5B):
            structural_tokens += 1
            stack.append([byte, 0])
            if len(stack) > _MAX_JSON_DEPTH:
                raise ValueError("trace report JSON is too deep")
            scalar_bytes = 0
        elif byte in (0x7D, 0x5D):
            expected_opener = 0x7B if byte == 0x7D else 0x5B
            if not stack or stack[-1][0] != expected_opener:
                raise ValueError("trace report JSON structure is invalid")
            stack.pop()
            scalar_bytes = 0
        elif byte == 0x2C:
            structural_tokens += 1
            if stack:
                stack[-1][1] += 1
                if stack[-1][1] >= _MAX_JSON_CONTAINER_ITEMS:
                    raise ValueError("trace report JSON container is too large")
            scalar_bytes = 0
        elif byte == 0x3A:
            structural_tokens += 1
            scalar_bytes = 0
        else:
            scalar_bytes += 1
            if scalar_bytes > _MAX_JSON_SCALAR_BYTES:
                raise ValueError("trace report JSON scalar is too large")

        if structural_tokens > _MAX_JSON_STRUCTURAL_TOKENS:
            raise ValueError("trace report JSON structure is too large")


def _bounded_json_shape(root: object) -> None:
    nodes = 0
    stack: list[tuple[object, int]] = [(root, 1)]
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            raise ValueError("report JSON structure is too large")
        if type(value) is dict:
            assert isinstance(value, dict)
            if len(value) > _MAX_JSON_CONTAINER_ITEMS:
                raise ValueError("report JSON object is too large")
            for key, item in value.items():
                if type(key) is not str or len(key.encode("utf-8")) > _MAX_JSON_STRING_BYTES:
                    raise ValueError("report JSON key is too large")
                stack.append((item, depth + 1))
        elif type(value) is list:
            assert isinstance(value, list)
            if len(value) > _MAX_JSON_CONTAINER_ITEMS:
                raise ValueError("report JSON array is too large")
            stack.extend((item, depth + 1) for item in value)
        elif type(value) is str:
            assert isinstance(value, str)
            if len(value.encode("utf-8")) > _MAX_JSON_STRING_BYTES:
                raise ValueError("report JSON string is too large")
        elif value is not None and type(value) not in (bool, int, float):
            raise ValueError("report JSON value is invalid")


def _decode_shadow_trace_report(data: bytes) -> ShadowTraceReport:
    if type(data) is not bytes or not 1 <= len(data) <= MAX_SHADOW_TRACE_REPORT_BYTES:
        raise ValueError("trace report byte bound is invalid")
    _preflight_canonical_json_structure(data)
    parsed = json.loads(
        data.decode("utf-8", errors="strict"),
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
        parse_float=_finite_json_float,
    )
    if type(parsed) is not dict:
        raise ValueError("trace report is not an object")
    _bounded_json_shape(parsed)
    report = ShadowTraceReport.model_validate_json(data)
    if not _model_state_is_exact(ShadowTraceReport, report):
        raise ValueError("trace report model is invalid")
    canonical = canonical_json(report.model_dump(mode="json", warnings=False))
    if not hmac.compare_digest(canonical, data):
        raise ValueError("trace report is not canonical")
    return report


def encode_shadow_trace_report(report: ShadowTraceReport) -> bytes:
    """Return the one bounded canonical representation of a trace report."""

    result: bytes | None = None
    try:
        if not _model_state_is_exact(ShadowTraceReport, report):
            raise ValueError("trace report type is invalid")
        candidate = canonical_json(report.model_dump(mode="json", warnings=False))
        if _decode_shadow_trace_report(candidate) != report:
            raise ValueError("trace report defensive copy differs")
        result = candidate
    except Exception:
        result = None
    if result is None:
        raise ShadowInvariantError()
    return result


def decode_shadow_trace_report(data: bytes) -> ShadowTraceReport:
    """Decode only bounded, canonical, self-verifying trace-report bytes."""

    result: ShadowTraceReport | None = None
    try:
        result = _decode_shadow_trace_report(data)
    except Exception:
        result = None
    if result is None:
        raise ShadowInvariantError()
    return result


def _copy_adapter_error(error: ShadowTraceInputError) -> ShadowTraceInputError:
    if type(error) is not ShadowTraceInputError:
        return ShadowTraceInputError("invalid_step")
    try:
        return ShadowTraceInputError(
            error.reason_code,
            step_ordinal=error.step_ordinal,
            call_ordinal=error.call_ordinal,
            result_ordinal=error.result_ordinal,
        )
    except Exception:
        return ShadowTraceInputError("invalid_step")


def verify_shadow_trace_source(
    report: ShadowTraceReport,
    source: bytes,
    *,
    adapter: ShadowTraceAdapter,
) -> ShadowTrace:
    """Re-adapt exact source bytes and return the matching validated trace.

    For ``canonical_records`` bindings, ``source`` is the exact canonical record-array
    representation committed by the report. This verifies source-side reproducibility only;
    authenticated-ledger verification remains a separate operation.
    """

    checked_report = decode_shadow_trace_report(encode_shadow_trace_report(report))
    failure: ShadowTraceInputError | None = None
    verified: ShadowTrace | None = None
    try:
        binding = checked_report.binding
        if type(source) is not bytes:
            raise ShadowTraceInputError("invalid_json")
        if not 1 <= len(source) <= MAX_SHADOW_TRACE_BYTES:
            raise ShadowTraceInputError("input_limit_exceeded")
        if len(source) != binding.source_byte_count or not hmac.compare_digest(
            hashlib.sha256(source).hexdigest(),
            binding.source_byte_digest,
        ):
            raise ShadowTraceInputError("digest_mismatch")
        profile_id = adapter.profile_id
        profile_digest = adapter.profile_digest
        if (
            type(profile_id) is not str
            or type(profile_digest) is not str
            or profile_id != binding.adapter_profile_id
            or not hmac.compare_digest(profile_digest, binding.adapter_profile_digest)
        ):
            raise ShadowTraceInputError("profile_mismatch")
        candidate = adapter.adapt_bytes(
            bytes(source),
            run_id=checked_report.run_id,
            task_scope_digest=binding.task_scope_digest,
            lineage_scope_digest=binding.lineage_scope_digest,
            capture_manifest_digest=binding.capture_manifest_digest,
        )
        if (
            type(candidate) is not ShadowTrace
            or not candidate._is_exact()
            or candidate.run_id != checked_report.run_id
            or not _models_match_exactly(candidate.binding, binding)
            or not _models_match_exactly(candidate.diagnostics, checked_report.diagnostics)
            or not hmac.compare_digest(
                candidate.mapped_record_digest,
                checked_report.mapped_record_digest,
            )
        ):
            raise ShadowTraceInputError("digest_mismatch")
        verified = candidate
    except ShadowTraceInputError as error:
        failure = _copy_adapter_error(error)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        failure = ShadowTraceInputError("invalid_step")
    if failure is not None:
        raise failure
    if verified is None:
        raise ShadowTraceInputError("invalid_step")
    return verified


__all__ = [
    "ShadowTraceReport",
    "decode_shadow_trace_report",
    "encode_shadow_trace_report",
    "verify_shadow_trace_source",
]
