"""Strict, bounded adapters for Shadow NDJSON input and canonical reports.

This module owns only typed file adaptation.  Reading and publication delegate to the
schema-neutral secure filesystem boundary; repository construction and command orchestration
remain outside this module.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import unicodedata
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Never, TypeAlias, cast
from uuid import UUID

from saliencegate.domain import (
    PayloadDigest,
    PayloadDigestAlgorithm,
    ReasonCode,
    Signal,
    TraceEvent,
    canonical_json,
    length_prefixed_sha256,
    trace_event_payload_is_bounded,
)
from saliencegate.security import (
    AtomicFilePublication,
    InstallationKey,
    RedactionPolicy,
    Redactor,
    StableFileAuthorization,
    StableReadPolicy,
    authorize_atomic_file_publication,
    read_stable_file,
)
from saliencegate.shadow.config import ShadowConfig, validate_shadow_config
from saliencegate.shadow.errors import ShadowInputError, ShadowInvariantError
from saliencegate.shadow.inputs import (
    SHADOW_PROJECTION_MATRIX,
    ShadowActionInput,
    ShadowControllerErrorInput,
    ShadowEventRef,
    ShadowFinishInput,
    ShadowInputKind,
    ShadowInputRecord,
    ShadowObservationInput,
    ShadowObservationSource,
    ShadowStartInput,
    ShadowTestResultInput,
    ShadowToolResultInput,
    derive_shadow_event_id,
    derive_shadow_source_event_digest,
    project_shadow_input,
)
from saliencegate.shadow.report import ShadowRunReport
from saliencegate.shadow.trace import ShadowTraceBinding
from saliencegate.shadow.trace_report import (
    MAX_SHADOW_TRACE_REPORT_BYTES,
    ShadowTraceReport,
    decode_shadow_trace_report,
    encode_shadow_trace_report,
)
from saliencegate.signals import (
    ShellActionEvidence,
    TestFailureEvidence,
    TestReportEvidence,
    ToolOutcomeEvidence,
)

MAX_SHADOW_INPUT_ROWS = 10_000
MAX_SHADOW_LINE_BYTES = 2 * 1024 * 1024
MAX_SHADOW_INPUT_BYTES = 64 * 1024 * 1024
MAX_SHADOW_REPORT_BYTES = 128 * 1024 * 1024

_MAX_REPORT_BYTES = MAX_SHADOW_REPORT_BYTES
_NORMALIZED_INPUT_DOMAIN = "saliencegate:shadow:normalized-input:v1"
_REDACTION_POLICY_DOMAIN = b"saliencegate:shadow:redaction-policy:v1"
_MARKER_PREFLIGHT_EVENT_ID = UUID("00000000-0000-4000-8000-000000000001")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMPONENT_IDENTIFIER = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._:/+\-]{0,255}$")
_CANONICAL_TIMESTAMP_PATTERN = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{6})?Z$"
)
_CAPTURE_SCOPES = frozenset(
    {"unknown", "selected_events", "bounded_window", "complete_run_declared"}
)
_RESERVED_OBSERVATION_KEYS = frozenset(
    {
        "shadow_run",
        "shadow_run_end",
        "action",
        "tool_outcome",
        "test_report",
        "controller_error",
    }
)

CaptureScope: TypeAlias = Literal[
    "unknown",
    "selected_events",
    "bounded_window",
    "complete_run_declared",
]

_WIRE_TO_INPUT_KIND: Mapping[str, ShadowInputKind] = {
    "run_start": ShadowInputKind.START,
    "action": ShadowInputKind.ACTION,
    "tool_result": ShadowInputKind.TOOL_RESULT,
    "test_result": ShadowInputKind.TEST_RESULT,
    "observation": ShadowInputKind.OBSERVATION,
    "controller_error": ShadowInputKind.CONTROLLER_ERROR,
    "run_end": ShadowInputKind.FINISH,
}

_COMMON_FIELDS = frozenset({"schema_version", "kind", "source_event_id", "occurred_at"})
_WIRE_FIELDS: Mapping[ShadowInputKind, tuple[frozenset[str], frozenset[str]]] = {
    ShadowInputKind.START: (_COMMON_FIELDS, frozenset()),
    ShadowInputKind.ACTION: (
        _COMMON_FIELDS | frozenset({"working_directory", "environment_digest"}),
        frozenset({"command", "argv"}),
    ),
    ShadowInputKind.TOOL_RESULT: (
        _COMMON_FIELDS | frozenset({"action_source_event_id"}),
        frozenset(
            {
                "status",
                "exit_status",
                "exception_type",
                "error_code",
                "failure_signature",
            }
        ),
    ),
    ShadowInputKind.TEST_RESULT: (
        _COMMON_FIELDS | frozenset({"action_source_event_id", "framework", "status", "failures"}),
        frozenset(),
    ),
    ShadowInputKind.OBSERVATION: (
        _COMMON_FIELDS | frozenset({"source", "payload"}),
        frozenset(),
    ),
    ShadowInputKind.CONTROLLER_ERROR: (
        _COMMON_FIELDS | frozenset({"error_code"}),
        frozenset(),
    ),
    ShadowInputKind.FINISH: (_COMMON_FIELDS, frozenset()),
}


def _is_uuid4(value: object) -> bool:
    return type(value) is UUID and value.version == 4


def _is_digest(value: object) -> bool:
    return type(value) is str and _SHA256_PATTERN.fullmatch(value) is not None


def _copy_optional_digest(value: object) -> str | None:
    if value is None:
        return None
    if not _is_digest(value):
        raise ValueError("capture digest is invalid")
    assert isinstance(value, str)
    return value


def _copy_redaction_policy(value: object) -> RedactionPolicy:
    if (
        type(value) is not RedactionPolicy
        or type(value.literal_secrets) is not tuple
        or type(value.structured_field_names) is not tuple
        or any(type(item) is not str for item in value.literal_secrets)
        or any(type(item) is not str for item in value.structured_field_names)
    ):
        raise ValueError("redaction policy is invalid")
    copied = RedactionPolicy(
        literal_secrets=value.literal_secrets,
        structured_field_names=value.structured_field_names,
    )
    if copied != value:
        raise ValueError("redaction policy is invalid")
    return copied


def _copy_installation_key(value: object) -> InstallationKey:
    if type(value) is not InstallationKey:
        raise ValueError("installation key is invalid")
    return value._copy()


def _copy_redaction_policy_tag(value: object) -> PayloadDigest:
    if type(value) is not PayloadDigest:
        raise ValueError("redaction policy tag is invalid")
    copied = PayloadDigest.model_validate(
        PayloadDigest.__pydantic_serializer__.to_python(
            value,
            mode="python",
            warnings=False,
        )
    )
    if copied != value or copied.algorithm is not PayloadDigestAlgorithm.HMAC_SHA256:
        raise ValueError("redaction policy tag is invalid")
    return copied


def _expected_redaction_policy_tag(
    key: InstallationKey,
    policy: RedactionPolicy,
) -> PayloadDigest:
    configuration = canonical_json(
        {
            "literal_secrets": policy.literal_secrets,
            "structured_field_names": policy.structured_field_names,
        }
    )
    return PayloadDigest(
        algorithm=PayloadDigestAlgorithm.HMAC_SHA256,
        value=key._hmac_sha256(configuration, domain=_REDACTION_POLICY_DOMAIN),
    )


def _copy_capture_scope(value: object) -> CaptureScope:
    if type(value) is not str or value not in _CAPTURE_SCOPES:
        raise ValueError("capture scope is invalid")
    return cast(CaptureScope, value)


@dataclass(frozen=True, slots=True, repr=False)
class PreflightedShadowRow:
    """One fully resolved row; its repr deliberately omits all caller content."""

    input_ordinal: int
    input_kind: ShadowInputKind
    input_record: ShadowInputRecord
    source_event_digest: str
    event_ref: ShadowEventRef
    first_occurrence_ordinal: int
    retry_target_ordinal: int | None

    @property
    def event_sequence(self) -> int:
        return self.event_ref.sequence

    @property
    def is_retry(self) -> bool:
        return self.retry_target_ordinal is not None

    def __repr__(self) -> str:
        return (
            "PreflightedShadowRow("
            f"input_ordinal={self.input_ordinal}, input_kind={self.input_kind.value!r}, "
            f"event_sequence={self.event_sequence}, is_retry={self.is_retry})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class PreflightedShadowTrace:
    """A stable private input snapshot and its complete side-effect-free preflight."""

    run_id: UUID
    authorization: StableFileAuthorization
    input_bytes: bytes
    input_byte_digest: str
    normalized_input_digest: str
    rows: tuple[PreflightedShadowRow, ...]

    @property
    def unique_input_event_count(self) -> int:
        return sum(not row.is_retry for row in self.rows)

    @property
    def retry_row_count(self) -> int:
        return sum(row.is_retry for row in self.rows)

    def __repr__(self) -> str:
        return (
            "PreflightedShadowTrace("
            f"row_count={len(self.rows)}, unique_input_event_count="
            f"{self.unique_input_event_count}, retry_row_count={self.retry_row_count})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ShadowReportBinding:
    """Exact provenance an existing report must bind before replacement is allowed."""

    run_id: UUID
    input_byte_digest: str
    normalized_input_digest: str
    redaction_policy_tag: PayloadDigest
    detector_profile_digest: str
    capture_scope: CaptureScope
    task_scope_digest: str | None = None
    lineage_scope_digest: str | None = None
    capture_manifest_digest: str | None = None

    def __post_init__(self) -> None:
        valid = False
        try:
            if not _is_uuid4(self.run_id):
                raise ValueError("run identity is invalid")
            run_id = UUID(int=self.run_id.int)
            if not _is_digest(self.input_byte_digest) or not _is_digest(
                self.normalized_input_digest
            ):
                raise ValueError("input identity is invalid")
            tag = _copy_redaction_policy_tag(self.redaction_policy_tag)
            if not _is_digest(self.detector_profile_digest):
                raise ValueError("detector identity is invalid")
            capture_scope = _copy_capture_scope(self.capture_scope)
            task = _copy_optional_digest(self.task_scope_digest)
            lineage = _copy_optional_digest(self.lineage_scope_digest)
            manifest = _copy_optional_digest(self.capture_manifest_digest)
            object.__setattr__(self, "run_id", run_id)
            object.__setattr__(self, "redaction_policy_tag", tag)
            object.__setattr__(self, "capture_scope", capture_scope)
            object.__setattr__(self, "task_scope_digest", task)
            object.__setattr__(self, "lineage_scope_digest", lineage)
            object.__setattr__(self, "capture_manifest_digest", manifest)
            valid = True
        except Exception:
            pass
        if not valid:
            raise ShadowInvariantError()

    def __repr__(self) -> str:
        return "ShadowReportBinding(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class ShadowTraceReportBinding:
    """Pure identity available before a trace repository is opened."""

    run_id: UUID
    trace_binding: ShadowTraceBinding
    diagnostics_digest: str
    mapped_record_digest: str
    normalized_input_digest: str
    redaction_policy_tag: PayloadDigest
    detector_profile_digest: str

    def __post_init__(self) -> None:
        valid = False
        try:
            if not _is_uuid4(self.run_id):
                raise ValueError("run identity is invalid")
            if (
                type(self.trace_binding) is not ShadowTraceBinding
                or type(self.trace_binding.__dict__) is not dict
                or set(self.trace_binding.__dict__) != set(ShadowTraceBinding.model_fields)
                or self.trace_binding.__pydantic_extra__ is not None
                or self.trace_binding.__pydantic_private__ is not None
            ):
                raise ValueError("trace binding is invalid")
            serialized = ShadowTraceBinding.__pydantic_serializer__.to_json(
                self.trace_binding,
                warnings=False,
            )
            trace_binding = ShadowTraceBinding.model_validate_json(serialized)
            copied = ShadowTraceBinding.__pydantic_serializer__.to_json(
                trace_binding,
                warnings=False,
            )
            if trace_binding != self.trace_binding or not hmac.compare_digest(
                copied,
                serialized,
            ):
                raise ValueError("trace binding copy differs")
            for digest in (
                self.diagnostics_digest,
                self.mapped_record_digest,
                self.normalized_input_digest,
                self.detector_profile_digest,
            ):
                if not _is_digest(digest):
                    raise ValueError("trace report identity is invalid")
            object.__setattr__(self, "run_id", UUID(int=self.run_id.int))
            object.__setattr__(self, "trace_binding", trace_binding)
            object.__setattr__(
                self,
                "redaction_policy_tag",
                _copy_redaction_policy_tag(self.redaction_policy_tag),
            )
            valid = True
        except Exception:
            pass
        if not valid:
            raise ShadowInvariantError()

    def __repr__(self) -> str:
        return "ShadowTraceReportBinding(<redacted>)"


@dataclass(frozen=True, slots=True)
class _PreflightOptions:
    run_id: UUID
    config: ShadowConfig
    installation_key: InstallationKey
    redaction_policy: RedactionPolicy
    redaction_policy_tag: PayloadDigest
    capture_scope: CaptureScope
    task_scope_digest: str | None
    lineage_scope_digest: str | None
    capture_manifest_digest: str | None
    source_adapter: str


@dataclass(frozen=True, slots=True)
class _FirstOccurrence:
    ordinal: int
    kind: ShadowInputKind
    redacted_event: object
    row: PreflightedShadowRow


def _payload_is_redaction_identity(
    policy: RedactionPolicy,
    payload: Mapping[str, object],
) -> bool:
    try:
        redactor = Redactor(
            literal_secrets=policy.literal_secrets,
            structured_field_names=policy.structured_field_names,
        )
        redacted = redactor.redact_payload(payload)
        return canonical_json(redacted.payload.root) == canonical_json(payload)
    except Exception:
        return False


def _literal_can_match_a_uuid(value: str) -> bool:
    without_controls = "".join(
        character for character in value if unicodedata.category(character) != "Cf"
    )
    normalized = unicodedata.normalize("NFKC", without_controls)
    return bool(normalized) and all(character in "0123456789abcdef-" for character in normalized)


def _signal_probe(
    options: _PreflightOptions,
    *,
    detector_index: int,
    created_at: datetime,
    evidence_event_id: UUID,
) -> Signal:
    detector = options.config.detectors[detector_index]
    return Signal(
        signal_id=UUID(int=_MARKER_PREFLIGHT_EVENT_ID.int + detector_index),
        run_id=options.run_id,
        created_at=created_at,
        signal_type=detector.signal_type,
        strength=1.0,
        evidence_event_ids=(evidence_event_id,),
        detector_version=detector.detector_version,
        reason_code=ReasonCode(detector.signal_type.value),
    )


def _require_static_policy_compatibility(options: _PreflightOptions) -> None:
    if not _payload_is_redaction_identity(
        options.redaction_policy,
        _start_payload(options),
    ) or not _payload_is_redaction_identity(
        options.redaction_policy,
        _finish_payload(options, _MARKER_PREFLIGHT_EVENT_ID),
    ):
        raise ValueError("redaction policy conflicts with shadow markers")
    if any(_literal_can_match_a_uuid(value) for value in options.redaction_policy.literal_secrets):
        raise ValueError("redaction policy conflicts with generated identities")
    metadata = {
        "source_event_id": "shadow-preflight",
        "source_adapter": options.source_adapter,
    }
    if not _payload_is_redaction_identity(options.redaction_policy, metadata):
        raise ValueError("redaction policy conflicts with shadow metadata")
    for detector_index in range(len(options.config.detectors)):
        probe = _signal_probe(
            options,
            detector_index=detector_index,
            created_at=datetime(2000, 1, 1, tzinfo=UTC),
            evidence_event_id=_MARKER_PREFLIGHT_EVENT_ID,
        )
        if not _payload_is_redaction_identity(
            options.redaction_policy,
            probe.model_dump(mode="json", warnings=False),
        ):
            raise ValueError("redaction policy conflicts with shadow signals")


def _prepare_options(
    *,
    run_id: object,
    config: object,
    installation_key: object,
    redaction_policy: object,
    redaction_policy_tag: object,
    capture_scope: object,
    task_scope_digest: object,
    lineage_scope_digest: object,
    capture_manifest_digest: object,
    source_adapter: object,
) -> _PreflightOptions:
    if not _is_uuid4(run_id) or type(source_adapter) is not str:
        raise ValueError("preflight identity is invalid")
    assert isinstance(run_id, UUID)
    copied_config = validate_shadow_config(config)
    copied_key = _copy_installation_key(installation_key)
    copied_policy = _copy_redaction_policy(redaction_policy)
    copied_tag = _copy_redaction_policy_tag(redaction_policy_tag)
    expected_tag = _expected_redaction_policy_tag(copied_key, copied_policy)
    if copied_tag != expected_tag:
        raise ValueError("redaction policy tag does not match policy")
    if (
        _COMPONENT_IDENTIFIER.fullmatch(source_adapter) is None
        or source_adapter.casefold() == "saliencegate.repository"
    ):
        raise ValueError("source adapter is invalid")
    options = _PreflightOptions(
        run_id=UUID(int=run_id.int),
        config=copied_config,
        installation_key=copied_key,
        redaction_policy=copied_policy,
        redaction_policy_tag=copied_tag,
        capture_scope=_copy_capture_scope(capture_scope),
        task_scope_digest=_copy_optional_digest(task_scope_digest),
        lineage_scope_digest=_copy_optional_digest(lineage_scope_digest),
        capture_manifest_digest=_copy_optional_digest(capture_manifest_digest),
        source_adapter=source_adapter,
    )
    _require_static_policy_compatibility(options)
    return options


def _reject_constant(_token: str) -> Never:
    raise ValueError("non-finite JSON number")


def _finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _parse_json_object(line: bytes) -> dict[str, object]:
    if not line:
        raise ValueError("blank NDJSON row")
    text = line.decode("utf-8", errors="strict")
    value = json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
        parse_float=_finite_float,
    )
    if type(value) is not dict:
        raise ValueError("NDJSON row is not an object")
    return value


def _parse_canonical_timestamp(value: object) -> datetime:
    if type(value) is not str or _CANONICAL_TIMESTAMP_PATTERN.fullmatch(value) is None:
        raise ValueError("timestamp is not canonical")
    parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp is not UTC")
    parsed = parsed.astimezone(UTC)
    canonical = parsed.isoformat(
        timespec="microseconds" if parsed.microsecond else "seconds"
    ).replace("+00:00", "Z")
    explicit_zero_fraction = (
        parsed.microsecond == 0 and value.endswith(".000000Z") and f"{value[:-8]}Z" == canonical
    )
    if canonical != value and not explicit_zero_fraction:
        raise ValueError("timestamp is not canonical")
    return parsed


def _require_wire_fields(value: dict[str, object], kind: ShadowInputKind) -> None:
    required, optional = _WIRE_FIELDS[kind]
    fields = frozenset(value)
    if not required.issubset(fields) or not fields.issubset(required | optional):
        raise ValueError("wire fields are invalid")


def _action_parent(
    value: object,
    known_sources: Mapping[str, _FirstOccurrence],
) -> ShadowEventRef:
    if type(value) is not str:
        raise ValueError("action parent is invalid")
    known = known_sources.get(value)
    if known is None or known.kind is not ShadowInputKind.ACTION:
        raise ValueError("action parent is missing or not an action")
    return known.row.event_ref


def _parse_input_record(
    value: dict[str, object],
    *,
    known_sources: Mapping[str, _FirstOccurrence],
) -> tuple[ShadowInputKind, ShadowInputRecord]:
    if (
        value.get("schema_version") != "shadow-input/v1"
        or type(value.get("schema_version")) is not str
    ):
        raise ValueError("wire schema is unsupported")
    wire_kind = value.get("kind")
    if type(wire_kind) is not str or wire_kind not in _WIRE_TO_INPUT_KIND:
        raise ValueError("wire kind is unsupported")
    kind = _WIRE_TO_INPUT_KIND[wire_kind]
    _require_wire_fields(value, kind)
    source_event_id = value["source_event_id"]
    if type(source_event_id) is not str:
        raise ValueError("source identity is invalid")
    common: dict[str, object] = {
        "schema_version": "shadow-input/v1",
        "source_event_id": source_event_id,
        "occurred_at": _parse_canonical_timestamp(value["occurred_at"]),
    }
    record: ShadowInputRecord
    if kind is ShadowInputKind.START:
        record = ShadowStartInput.model_validate(common)
    elif kind is ShadowInputKind.ACTION:
        argv = value.get("argv")
        if argv is not None:
            if type(argv) is not list or any(type(item) is not str for item in argv):
                raise ValueError("action arguments are invalid")
            argv = tuple(argv)
        record = ShadowActionInput.model_validate(
            {
                **common,
                "command": value.get("command"),
                "argv": argv,
                "working_directory": value["working_directory"],
                "environment_digest": value["environment_digest"],
            }
        )
    elif kind is ShadowInputKind.TOOL_RESULT:
        record = ShadowToolResultInput.model_validate(
            {
                **common,
                "action": _action_parent(
                    value["action_source_event_id"],
                    known_sources,
                ),
                "status": value.get("status"),
                "exit_status": value.get("exit_status"),
                "exception_type": value.get("exception_type"),
                "error_code": value.get("error_code"),
                "failure_signature": value.get("failure_signature"),
            }
        )
    elif kind is ShadowInputKind.TEST_RESULT:
        failures = value["failures"]
        if type(failures) is not list:
            raise ValueError("test failures are invalid")
        record = ShadowTestResultInput.model_validate(
            {
                **common,
                "action": _action_parent(
                    value["action_source_event_id"],
                    known_sources,
                ),
                "framework": value["framework"],
                "status": value["status"],
                "failures": tuple(failures),
            }
        )
    elif kind is ShadowInputKind.OBSERVATION:
        source = value["source"]
        if type(source) is not str:
            raise ValueError("observation source is invalid")
        record = ShadowObservationInput.model_validate(
            {
                **common,
                "source": ShadowObservationSource(source),
                "payload": value["payload"],
            }
        )
    elif kind is ShadowInputKind.CONTROLLER_ERROR:
        record = ShadowControllerErrorInput.model_validate(
            {
                **common,
                "error_code": value["error_code"],
            }
        )
    else:
        record = ShadowFinishInput.model_validate(common)
    return kind, record


def _start_payload(options: _PreflightOptions) -> dict[str, object]:
    return {
        "schema_version": "shadow-run/v1",
        "detector_profile_digest": options.config.detector_profile_digest,
        "evaluator_configuration_digest": options.config.evaluator_configuration_digest,
        "redaction_policy_tag": options.redaction_policy_tag.model_dump(mode="json"),
        "source_adapter": options.source_adapter,
        "capture_scope": options.capture_scope,
        "task_scope_digest": options.task_scope_digest,
        "lineage_scope_digest": options.lineage_scope_digest,
        "capture_manifest_digest": options.capture_manifest_digest,
        "split_metadata_complete": options.capture_manifest_digest is not None
        or (options.task_scope_digest is not None and options.lineage_scope_digest is not None),
    }


def _finish_payload(options: _PreflightOptions, start_event_id: UUID) -> dict[str, object]:
    payload = _start_payload(options)
    payload["schema_version"] = "shadow-run-end/v1"
    payload["start_event_id"] = str(start_event_id)
    return payload


def _event_payload_is_valid(
    options: _PreflightOptions,
    event: TraceEvent,
    kind: ShadowInputKind,
) -> bool:
    try:
        namespace = SHADOW_PROJECTION_MATRIX[kind].payload_namespace
        value = event.payload[namespace]
        if kind is ShadowInputKind.START:
            return canonical_json(value) == canonical_json(_start_payload(options))
        if kind is ShadowInputKind.FINISH:
            return isinstance(value, Mapping)
        if kind is ShadowInputKind.ACTION:
            action_evidence = ShellActionEvidence.model_validate_json(canonical_json(value))
            return canonical_json(action_evidence) == canonical_json(value)
        if kind is ShadowInputKind.TOOL_RESULT:
            tool_evidence = ToolOutcomeEvidence.model_validate_json(canonical_json(value))
            return canonical_json(tool_evidence) == canonical_json(value)
        if kind is ShadowInputKind.TEST_RESULT:
            if not isinstance(value, Mapping) or set(value) != {
                "schema_version",
                "framework",
                "status",
                "failures",
            }:
                return False
            raw_failures = value["failures"]
            if type(raw_failures) is not tuple:
                return False
            failures = tuple(
                TestFailureEvidence.model_validate_json(canonical_json(item))
                for item in raw_failures
            )
            test_evidence = TestReportEvidence(
                schema_version=cast(Literal["1.0"], value["schema_version"]),
                framework=cast(str, value["framework"]),
                status=cast(Literal["passed", "failed"], value["status"]),
                failures=failures,
            )
            return canonical_json(test_evidence) == canonical_json(value)
        if kind is ShadowInputKind.OBSERVATION:
            return (
                isinstance(value, Mapping)
                and trace_event_payload_is_bounded(value)
                and not any(key in _RESERVED_OBSERVATION_KEYS for key in value)
            )
        if kind is ShadowInputKind.CONTROLLER_ERROR:
            return (
                isinstance(value, Mapping)
                and set(value) == {"schema_version", "error_code"}
                and value.get("schema_version") == "controller_error/v1"
                and type(value.get("error_code")) is str
                and _COMPONENT_IDENTIFIER.fullmatch(value["error_code"]) is not None
            )
    except Exception:
        return False
    return False


def _preflight_redacted_event(
    options: _PreflightOptions,
    redactor: Redactor,
    record: ShadowInputRecord,
    *,
    kind: ShadowInputKind,
    event_id: UUID,
    sequence: int,
    start_payload: Mapping[str, object] | None,
    finish_payload: Mapping[str, object] | None,
) -> TraceEvent:
    if kind is ShadowInputKind.START:
        if (
            start_payload is None
            or not _payload_is_redaction_identity(
                options.redaction_policy,
                start_payload,
            )
            or not _payload_is_redaction_identity(
                options.redaction_policy,
                _finish_payload(options, event_id),
            )
        ):
            raise ValueError("start marker is not redaction-stable")
    elif kind is ShadowInputKind.FINISH and (
        finish_payload is None
        or not _payload_is_redaction_identity(
            options.redaction_policy,
            finish_payload,
        )
    ):
        raise ValueError("finish marker is not redaction-stable")

    draft = project_shadow_input(
        record,
        run_id=options.run_id,
        source_adapter=options.source_adapter,
        start_payload=start_payload,
        finish_payload=finish_payload,
    )
    metadata = {
        "source_event_id": draft.source_event_id,
        "source_adapter": draft.source_adapter,
    }
    redacted_metadata = redactor.redact_payload(metadata)
    if canonical_json(redacted_metadata.payload.root) != canonical_json(metadata):
        raise ValueError("shadow metadata is not redaction-stable")
    redacted = redactor.redact_event(draft, key=options.installation_key)
    values = redacted.event.model_dump(mode="python", warnings=False)
    values.update(
        record_type="trace_event",
        event_id=event_id,
        sequence=sequence,
    )
    candidate = TraceEvent.model_validate(values)
    if not _event_payload_is_valid(options, candidate, kind):
        raise ValueError("redacted event payload is not canonical")

    applicability = next(item for item in options.config.applicability if item.input_kind is kind)
    for detector_index, detector in enumerate(options.config.detectors):
        if detector.signal_type not in applicability.applicable_signal_types:
            continue
        probe = _signal_probe(
            options,
            detector_index=detector_index,
            created_at=candidate.timestamp,
            evidence_event_id=candidate.event_id,
        )
        if not _payload_is_redaction_identity(
            options.redaction_policy,
            probe.model_dump(mode="json", warnings=False),
        ):
            raise ValueError("dynamic signal is not redaction-stable")
    return candidate


def _normalized_input_digest(
    options: _PreflightOptions,
    records: tuple[ShadowInputRecord, ...],
) -> str:
    provenance = canonical_json(
        {
            "schema_version": "shadow-normalized-input-provenance/v1",
            "run_id": str(options.run_id),
            "source_adapter": options.source_adapter,
            "capture_scope": options.capture_scope,
            "task_scope_digest": options.task_scope_digest,
            "lineage_scope_digest": options.lineage_scope_digest,
            "capture_manifest_digest": options.capture_manifest_digest,
            "redaction_policy_tag": options.redaction_policy_tag.model_dump(mode="json"),
            "detector_profile_digest": options.config.detector_profile_digest,
            "evaluator_configuration_digest": options.config.evaluator_configuration_digest,
        }
    )
    record_bytes = tuple(
        canonical_json(record.model_dump(mode="json", warnings=False)) for record in records
    )
    return length_prefixed_sha256(
        provenance,
        *record_bytes,
        domain=_NORMALIZED_INPUT_DOMAIN,
    )


def _preflight_rows(
    lines: tuple[bytes, ...],
    options: _PreflightOptions,
) -> tuple[PreflightedShadowRow, ...]:
    records = tuple(_parse_json_object(line) for line in lines)
    return _preflight_record_values(records, options)


def _preflight_record_values(
    records: tuple[dict[str, object], ...],
    options: _PreflightOptions,
) -> tuple[PreflightedShadowRow, ...]:
    """Completely preflight already-decoded exact Shadow wire records."""

    if not records:
        raise ValueError("trace is empty")
    if records[0].get("kind") != "run_start":
        raise ValueError("trace does not start with run_start")
    run_start_positions = tuple(
        ordinal
        for ordinal, value in enumerate(records, start=1)
        if value.get("kind") == "run_start"
    )
    run_end_positions = tuple(
        ordinal for ordinal, value in enumerate(records, start=1) if value.get("kind") == "run_end"
    )
    if run_start_positions != (1,):
        raise ValueError("trace has an invalid run_start lifecycle")
    if len(run_end_positions) > 1 or (run_end_positions and run_end_positions[0] != len(records)):
        raise ValueError("trace has an invalid run_end lifecycle")
    if options.capture_scope == "complete_run_declared" and run_end_positions != (len(records),):
        raise ValueError("complete capture is missing run_end")

    redactor = Redactor(
        literal_secrets=options.redaction_policy.literal_secrets,
        structured_field_names=options.redaction_policy.structured_field_names,
    )
    first_by_source: dict[str, _FirstOccurrence] = {}
    rows: list[PreflightedShadowRow] = []
    unique_count = 0
    latest_timestamp: datetime | None = None
    start_event_id: UUID | None = None
    for ordinal, wire in enumerate(records, start=1):
        kind, record = _parse_input_record(wire, known_sources=first_by_source)
        source_event_id = record.source_event_id
        event_id = derive_shadow_event_id(options.run_id, source_event_id)
        if kind is ShadowInputKind.START:
            marker_payload = _start_payload(options)
            start_event_id = event_id
            finish_payload = None
        elif kind is ShadowInputKind.FINISH:
            if start_event_id is None:
                raise ValueError("run_end has no run_start")
            marker_payload = None
            finish_payload = _finish_payload(options, start_event_id)
        else:
            marker_payload = None
            finish_payload = None
        existing = first_by_source.get(source_event_id)
        sequence = existing.row.event_sequence if existing is not None else unique_count + 1
        candidate = _preflight_redacted_event(
            options,
            redactor,
            record,
            kind=kind,
            event_id=event_id,
            sequence=sequence,
            start_payload=marker_payload,
            finish_payload=finish_payload,
        )
        source_digest = derive_shadow_source_event_digest(
            options.run_id,
            source_event_id,
        )
        if existing is not None:
            if candidate != existing.redacted_event:
                raise ValueError("source event collides after redaction")
            row = PreflightedShadowRow(
                input_ordinal=ordinal,
                input_kind=kind,
                input_record=record,
                source_event_digest=source_digest,
                event_ref=existing.row.event_ref,
                first_occurrence_ordinal=existing.ordinal,
                retry_target_ordinal=existing.ordinal,
            )
            rows.append(row)
            continue
        if latest_timestamp is not None and record.occurred_at < latest_timestamp:
            raise ValueError("timestamps decrease")
        unique_count += 1
        latest_timestamp = record.occurred_at
        event_ref = ShadowEventRef(
            run_id=options.run_id,
            event_id=event_id,
            sequence=unique_count,
        )
        row = PreflightedShadowRow(
            input_ordinal=ordinal,
            input_kind=kind,
            input_record=record,
            source_event_digest=source_digest,
            event_ref=event_ref,
            first_occurrence_ordinal=ordinal,
            retry_target_ordinal=None,
        )
        rows.append(row)
        first_by_source[source_event_id] = _FirstOccurrence(
            ordinal=ordinal,
            kind=kind,
            redacted_event=candidate,
            row=row,
        )
    return tuple(rows)


def read_shadow_trace(
    path: str | os.PathLike[str],
    *,
    run_id: UUID,
    config: ShadowConfig,
    installation_key: InstallationKey,
    redaction_policy: RedactionPolicy,
    redaction_policy_tag: PayloadDigest,
    capture_scope: CaptureScope,
    source_adapter: str,
    task_scope_digest: str | None = None,
    lineage_scope_digest: str | None = None,
    capture_manifest_digest: str | None = None,
) -> PreflightedShadowTrace:
    """Read and completely preflight one private Shadow NDJSON trace.

    This function opens only the input.  It never inspects or mutates a repository or output.
    """

    result: PreflightedShadowTrace | None = None
    interrupted: BaseException | None = None
    try:
        options = _prepare_options(
            run_id=run_id,
            config=config,
            installation_key=installation_key,
            redaction_policy=redaction_policy,
            redaction_policy_tag=redaction_policy_tag,
            capture_scope=capture_scope,
            task_scope_digest=task_scope_digest,
            lineage_scope_digest=lineage_scope_digest,
            capture_manifest_digest=capture_manifest_digest,
            source_adapter=source_adapter,
        )
        stable = read_stable_file(
            path,
            maximum_bytes=MAX_SHADOW_INPUT_BYTES,
            policy=StableReadPolicy.PRIVATE_OWNER,
        )
        lines = tuple(
            stable.iter_lines(
                maximum_line_bytes=MAX_SHADOW_LINE_BYTES,
                maximum_lines=MAX_SHADOW_INPUT_ROWS,
            )
        )
        rows = _preflight_rows(lines, options)
        records = tuple(row.input_record for row in rows)
        result = PreflightedShadowTrace(
            run_id=options.run_id,
            authorization=stable.authorization,
            input_bytes=stable.data,
            input_byte_digest=hashlib.sha256(stable.data).hexdigest(),
            normalized_input_digest=_normalized_input_digest(options, records),
            rows=rows,
        )
    except (KeyboardInterrupt, SystemExit) as error:
        interrupted = error
    except Exception:
        pass
    if interrupted is not None:
        raise interrupted
    if result is None:
        raise ShadowInputError()
    return result


def _parse_unique_json(data: bytes) -> object:
    text = data.decode("utf-8", errors="strict")
    return json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
        parse_float=_finite_float,
    )


def _decode_report(data: bytes) -> ShadowRunReport:
    if type(data) is not bytes or not 1 <= len(data) <= _MAX_REPORT_BYTES:
        raise ValueError("report byte bound is invalid")
    parsed = _parse_unique_json(data)
    if type(parsed) is not dict:
        raise ValueError("report is not an object")
    report = ShadowRunReport.model_validate_json(data)
    if canonical_json(report.model_dump(mode="json", warnings=False)) != data:
        raise ValueError("report is not canonical")
    return report


def encode_shadow_run_report(report: ShadowRunReport) -> bytes:
    """Return the one bounded canonical byte representation of a valid report."""

    result: bytes | None = None
    try:
        if type(report) is not ShadowRunReport:
            raise ValueError("report type is invalid")
        candidate = canonical_json(report.model_dump(mode="json", warnings=False))
        if _decode_report(candidate) != report:
            raise ValueError("report copy differs")
        result = candidate
    except Exception:
        pass
    if result is None:
        raise ShadowInvariantError()
    return result


def decode_shadow_run_report(data: bytes) -> ShadowRunReport:
    """Decode only bounded, canonical, self-verifying Shadow report bytes."""

    result: ShadowRunReport | None = None
    with suppress(Exception):
        result = _decode_report(data)
    if result is None:
        raise ShadowInvariantError()
    return result


def shadow_report_binding(report: ShadowRunReport) -> ShadowReportBinding:
    """Copy the exact replacement provenance from one validated report."""

    validated = decode_shadow_run_report(encode_shadow_run_report(report))
    return ShadowReportBinding(
        run_id=validated.run_id,
        input_byte_digest=validated.input_byte_digest,
        normalized_input_digest=validated.normalized_input_digest,
        redaction_policy_tag=validated.redaction_policy_tag,
        detector_profile_digest=validated.detector_profile_digest,
        capture_scope=validated.capture_scope,
        task_scope_digest=validated.task_scope_digest,
        lineage_scope_digest=validated.lineage_scope_digest,
        capture_manifest_digest=validated.capture_manifest_digest,
    )


def _report_matches_binding(
    report: ShadowRunReport,
    binding: ShadowReportBinding,
) -> bool:
    return (
        report.run_id == binding.run_id
        and hmac.compare_digest(report.input_byte_digest, binding.input_byte_digest)
        and hmac.compare_digest(
            report.normalized_input_digest,
            binding.normalized_input_digest,
        )
        and report.redaction_policy_tag == binding.redaction_policy_tag
        and hmac.compare_digest(
            report.detector_profile_digest,
            binding.detector_profile_digest,
        )
        and report.capture_scope == binding.capture_scope
        and report.task_scope_digest == binding.task_scope_digest
        and report.lineage_scope_digest == binding.lineage_scope_digest
        and report.capture_manifest_digest == binding.capture_manifest_digest
    )


def validate_shadow_report_replacement(
    data: bytes,
    binding: ShadowReportBinding,
) -> bool:
    """Return true only for canonical report bytes with the exact replacement binding."""

    try:
        if type(binding) is not ShadowReportBinding:
            return False
        report = _decode_report(data)
        return _report_matches_binding(report, binding)
    except Exception:
        return False


def validate_published_shadow_report(
    data: bytes,
    report: ShadowRunReport,
) -> bool:
    """Return true only when reopened bytes are the exact expected canonical report."""

    try:
        if type(report) is not ShadowRunReport:
            return False
        decoded = _decode_report(data)
        return decoded == report and data == canonical_json(
            report.model_dump(mode="json", warnings=False)
        )
    except Exception:
        return False


def authorize_shadow_report_publication(
    path: str | os.PathLike[str],
    *,
    replacement_binding: ShadowReportBinding | None = None,
) -> AtomicFilePublication:
    """Authorize a bounded no-clobber output or an exactly bound report replacement."""

    validator: Callable[[bytes], bool] | None
    if replacement_binding is None:
        validator = None
    elif type(replacement_binding) is ShadowReportBinding:
        binding = replacement_binding

        def replacement_validator(data: bytes) -> bool:
            return validate_shadow_report_replacement(data, binding)

        validator = replacement_validator
    else:
        raise ShadowInvariantError()
    return authorize_atomic_file_publication(
        path,
        maximum_bytes=MAX_SHADOW_REPORT_BYTES,
        validate_replacement=validator,
    )


def validate_published_shadow_trace_report(
    data: bytes,
    report: ShadowTraceReport,
) -> bool:
    """Accept only the exact canonical bytes of one complete trace report."""

    try:
        if type(report) is not ShadowTraceReport:
            return False
        expected = encode_shadow_trace_report(report)
        decoded = decode_shadow_trace_report(data)
        return decoded == report and hmac.compare_digest(data, expected)
    except Exception:
        return False


def validate_shadow_trace_report_replacement(
    data: bytes,
    report: ShadowTraceReport,
) -> bool:
    """Require an existing trace report to be the exact expected report."""

    return validate_published_shadow_trace_report(data, report)


def shadow_trace_report_binding(report: ShadowTraceReport) -> ShadowTraceReportBinding:
    """Copy the complete pre-repository identity of one outer trace report."""

    validated = decode_shadow_trace_report(encode_shadow_trace_report(report))
    nested = validated.shadow_report
    return ShadowTraceReportBinding(
        run_id=validated.run_id,
        trace_binding=validated.binding,
        diagnostics_digest=validated.diagnostics_digest,
        mapped_record_digest=validated.mapped_record_digest,
        normalized_input_digest=validated.normalized_input_digest,
        redaction_policy_tag=nested.redaction_policy_tag,
        detector_profile_digest=nested.detector_profile_digest,
    )


def validate_shadow_trace_report_binding(
    data: bytes,
    binding: ShadowTraceReportBinding,
) -> bool:
    """Validate every trace and nested binding known before repository access."""

    try:
        if type(binding) is not ShadowTraceReportBinding:
            return False
        report = decode_shadow_trace_report(data)
        nested = report.shadow_report
        return (
            report.run_id == binding.run_id
            and hmac.compare_digest(
                canonical_json(report.binding),
                canonical_json(binding.trace_binding),
            )
            and hmac.compare_digest(
                report.diagnostics_digest,
                binding.diagnostics_digest,
            )
            and hmac.compare_digest(
                report.mapped_record_digest,
                binding.mapped_record_digest,
            )
            and hmac.compare_digest(
                report.normalized_input_digest,
                binding.normalized_input_digest,
            )
            and nested.redaction_policy_tag == binding.redaction_policy_tag
            and hmac.compare_digest(
                nested.detector_profile_digest,
                binding.detector_profile_digest,
            )
        )
    except Exception:
        return False


def authorize_shadow_trace_report_publication(
    path: str | os.PathLike[str],
    *,
    replacement_binding: ShadowTraceReportBinding | None = None,
    replacement_report: ShadowTraceReport | None = None,
) -> AtomicFilePublication:
    """Authorize one absent outer report or one exact canonical replacement."""

    validator: Callable[[bytes], bool] | None
    if replacement_binding is not None and replacement_report is not None:
        raise ShadowInvariantError()
    if replacement_binding is not None:
        if type(replacement_binding) is not ShadowTraceReportBinding:
            raise ShadowInvariantError()
        expected_binding = replacement_binding

        def binding_validator(data: bytes) -> bool:
            return validate_shadow_trace_report_binding(data, expected_binding)

        validator = binding_validator
    elif replacement_report is None:
        validator = None
    elif type(replacement_report) is ShadowTraceReport:
        expected = decode_shadow_trace_report(encode_shadow_trace_report(replacement_report))

        def replacement_validator(data: bytes) -> bool:
            return validate_shadow_trace_report_replacement(data, expected)

        validator = replacement_validator
    else:
        raise ShadowInvariantError()
    return authorize_atomic_file_publication(
        path,
        maximum_bytes=MAX_SHADOW_TRACE_REPORT_BYTES,
        validate_replacement=validator,
    )


__all__ = [
    "MAX_SHADOW_INPUT_BYTES",
    "MAX_SHADOW_INPUT_ROWS",
    "MAX_SHADOW_LINE_BYTES",
    "MAX_SHADOW_REPORT_BYTES",
    "PreflightedShadowRow",
    "PreflightedShadowTrace",
    "ShadowReportBinding",
    "ShadowTraceReportBinding",
    "authorize_shadow_report_publication",
    "authorize_shadow_trace_report_publication",
    "decode_shadow_run_report",
    "encode_shadow_run_report",
    "read_shadow_trace",
    "shadow_report_binding",
    "shadow_trace_report_binding",
    "validate_published_shadow_report",
    "validate_published_shadow_trace_report",
    "validate_shadow_report_replacement",
    "validate_shadow_trace_report_binding",
    "validate_shadow_trace_report_replacement",
]
