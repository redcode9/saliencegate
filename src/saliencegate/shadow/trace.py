"""Immutable, provider-free whole-trace contracts for Shadow analysis."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Annotated, Literal, TypeAlias, cast, final
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    field_validator,
    model_validator,
)

from saliencegate.domain import canonical_json, length_prefixed_sha256
from saliencegate.domain.records import Sha256Digest
from saliencegate.shadow.errors import ShadowTraceInputError
from saliencegate.shadow.inputs import (
    ShadowActionIdentityInput,
    ShadowActionInput,
    ShadowControllerErrorInput,
    ShadowEventRef,
    ShadowFinishInput,
    ShadowInputKind,
    ShadowObservationInput,
    ShadowObservationSource,
    ShadowStartInput,
    ShadowTestResultInput,
    ShadowToolResultInput,
    derive_shadow_event_id,
)

MAX_SHADOW_TRACE_ROWS = 1_000
MAX_SHADOW_TRACE_BYTES = 64 * 1_024 * 1_024
MAX_ADAPTER_DESCRIPTOR_BYTES = 16 * 1_024

_MAX_DESCRIPTOR_DEPTH = 8
_MAX_DESCRIPTOR_ITEMS = 64
_MAX_DESCRIPTOR_STRING_BYTES = 1_024
_MAX_RECORD_DEPTH = 68
_MAX_RECORD_ITEMS = 50_000
_MAX_RECORD_NODES = 64_000
_PROFILE_DIGEST_DOMAIN = "saliencegate:shadow:adapter-profile:v1"
_CONFIGURATION_DIGEST_DOMAIN = "saliencegate:shadow:adapter-configuration:v1"
_BINDING_DIGEST_DOMAIN = "saliencegate:shadow:trace-binding:v1"
_DIRECT_DIAGNOSTICS_DIGEST_DOMAIN = "saliencegate:shadow:record-diagnostics:v1"
_ATIF_DIAGNOSTICS_DIGEST_DOMAIN = "saliencegate:shadow:atif-diagnostics:v1"
_MAPPED_RECORD_DIGEST_DOMAIN = "saliencegate:shadow:mapped-records:v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_FORMAT_PATTERN = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
_PROFILE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._:/-]{0,79}$")
_SOURCE_SCHEMA_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._:/+-]{0,127}$")
_SOURCE_ADAPTER_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._:/+\-]{0,255}$")
_BUILT_IN_ATIF_PROFILE_IDS = frozenset({"harbor-terminus-2/v1", "harbor-codex/v1"})
_CANONICAL_TIMESTAMP_PATTERN = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{6})?Z$"
)
_CAPTURE_SCOPES = frozenset(
    {"unknown", "selected_events", "bounded_window", "complete_run_declared"}
)
_TIMESTAMP_MODES = frozenset({"source_utc", "logical_order", "record_declared"})
_SOURCE_DIGEST_KINDS = frozenset({"original_bytes", "canonical_records"})
_IDENTITY_MODES = frozenset({"profile_content_addressed", "legacy_explicit"})
_WIRE_TO_INPUT_KIND: Mapping[str, ShadowInputKind] = MappingProxyType(
    {
        "run_start": ShadowInputKind.START,
        "action": ShadowInputKind.ACTION,
        "action_identity": ShadowInputKind.ACTION_IDENTITY,
        "tool_result": ShadowInputKind.TOOL_RESULT,
        "test_result": ShadowInputKind.TEST_RESULT,
        "observation": ShadowInputKind.OBSERVATION,
        "controller_error": ShadowInputKind.CONTROLLER_ERROR,
        "run_end": ShadowInputKind.FINISH,
    }
)
_COMMON_WIRE_FIELDS = frozenset({"schema_version", "kind", "source_event_id", "occurred_at"})
_WIRE_FIELDS: Mapping[ShadowInputKind, tuple[frozenset[str], frozenset[str]]] = MappingProxyType(
    {
        ShadowInputKind.START: (_COMMON_WIRE_FIELDS, frozenset()),
        ShadowInputKind.ACTION: (
            _COMMON_WIRE_FIELDS | frozenset({"working_directory", "environment_digest"}),
            frozenset({"command", "argv"}),
        ),
        ShadowInputKind.ACTION_IDENTITY: (
            _COMMON_WIRE_FIELDS
            | frozenset(
                {
                    "action_digest",
                    "workspace_digest",
                    "environment_digest",
                    "identity_authority",
                }
            ),
            frozenset(),
        ),
        ShadowInputKind.TOOL_RESULT: (
            _COMMON_WIRE_FIELDS | frozenset({"action_source_event_id"}),
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
            _COMMON_WIRE_FIELDS
            | frozenset({"action_source_event_id", "framework", "status", "failures"}),
            frozenset(),
        ),
        ShadowInputKind.OBSERVATION: (
            _COMMON_WIRE_FIELDS | frozenset({"source", "payload"}),
            frozenset(),
        ),
        ShadowInputKind.CONTROLLER_ERROR: (
            _COMMON_WIRE_FIELDS | frozenset({"error_code"}),
            frozenset(),
        ),
        ShadowInputKind.FINISH: (_COMMON_WIRE_FIELDS, frozenset()),
    }
)
_TRACE_FACTORY_TOKEN = object()

_LEGACY_INPUT_KINDS = (
    ShadowInputKind.START,
    ShadowInputKind.ACTION,
    ShadowInputKind.TOOL_RESULT,
    ShadowInputKind.TEST_RESULT,
    ShadowInputKind.OBSERVATION,
    ShadowInputKind.CONTROLLER_ERROR,
    ShadowInputKind.FINISH,
)
_EXTENDED_INPUT_KINDS = (
    *_LEGACY_INPUT_KINDS,
    ShadowInputKind.ACTION_IDENTITY,
)

CaptureScope: TypeAlias = Literal[
    "unknown",
    "selected_events",
    "bounded_window",
    "complete_run_declared",
]
TimestampMode: TypeAlias = Literal["source_utc", "logical_order", "record_declared"]
SourceDigestKind: TypeAlias = Literal["original_bytes", "canonical_records"]
IdentityMode: TypeAlias = Literal["profile_content_addressed", "legacy_explicit"]
ToolCallDisposition: TypeAlias = Literal[
    "mapped_action",
    "ignored_unsupported_function",
    "ignored_continuation",
    "ignored_non_command_wait",
    "ignored_unsubmitted_keystrokes",
    "ignored_unresolved_terminal_submission",
    "ignored_copied_context",
]
ResultDisposition: TypeAlias = Literal[
    "mapped_structured_outcome",
    "ignored_evidence_absent",
    "ignored_ambiguous_parent",
    "ignored_no_parent",
    "ignored_unsupported_parent",
    "ignored_copied_context",
]

_TOOL_CALL_DISPOSITIONS: tuple[ToolCallDisposition, ...] = (
    "mapped_action",
    "ignored_unsupported_function",
    "ignored_continuation",
    "ignored_non_command_wait",
    "ignored_unsubmitted_keystrokes",
    "ignored_unresolved_terminal_submission",
    "ignored_copied_context",
)
_RESULT_DISPOSITIONS: tuple[ResultDisposition, ...] = (
    "mapped_structured_outcome",
    "ignored_evidence_absent",
    "ignored_ambiguous_parent",
    "ignored_no_parent",
    "ignored_unsupported_parent",
    "ignored_copied_context",
)


class _TraceModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


def _exact_digest(value: object) -> bool:
    return type(value) is str and _SHA256_PATTERN.fullmatch(value) is not None


def _body_digest(value: BaseModel, *, excluded: str, domain: str) -> str:
    serializer = type(value).__pydantic_serializer__
    body = serializer.to_python(value, mode="json", exclude={excluded}, warnings=False)
    return length_prefixed_sha256(canonical_json(body), domain=domain)


class ShadowTraceBinding(_TraceModel):
    """Content-addressed source, mapping, and capture provenance."""

    schema_version: Literal["shadow-trace-binding/v1"] = "shadow-trace-binding/v1"
    source_format: Annotated[str, Field(min_length=1, max_length=32)]
    source_schema_version: Annotated[str, Field(min_length=1, max_length=128)]
    source_digest_kind: SourceDigestKind
    source_byte_count: Annotated[int, Field(ge=1, le=MAX_SHADOW_TRACE_BYTES)]
    source_byte_digest: Sha256Digest
    adapter_profile_id: Annotated[str, Field(min_length=1, max_length=80)]
    adapter_profile_digest: Sha256Digest
    adapter_configuration_digest: Sha256Digest
    source_adapter: Annotated[str, Field(min_length=1, max_length=256)]
    identity_mode: IdentityMode
    timestamp_mode: TimestampMode
    capture_scope: CaptureScope
    task_scope_digest: Sha256Digest | None = None
    lineage_scope_digest: Sha256Digest | None = None
    capture_manifest_digest: Sha256Digest | None = None
    binding_digest: Sha256Digest

    def __init_subclass__(cls, **_kwargs: object) -> None:
        raise TypeError("ShadowTraceBinding cannot be subclassed")

    @field_validator(
        "schema_version",
        "source_format",
        "source_schema_version",
        "source_digest_kind",
        "source_byte_digest",
        "adapter_profile_id",
        "adapter_profile_digest",
        "adapter_configuration_digest",
        "source_adapter",
        "identity_mode",
        "timestamp_mode",
        "capture_scope",
        "task_scope_digest",
        "lineage_scope_digest",
        "capture_manifest_digest",
        "binding_digest",
        mode="before",
    )
    @classmethod
    def require_exact_text(cls, value: object) -> object:
        if value is not None and type(value) is not str:
            raise ValueError("trace binding text is invalid")
        return value

    @field_validator("source_byte_count", mode="before")
    @classmethod
    def require_exact_byte_count(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("trace binding byte count is invalid")
        return value

    @model_validator(mode="after")
    def validate_identity_and_digest(self) -> ShadowTraceBinding:
        if _SOURCE_FORMAT_PATTERN.fullmatch(self.source_format) is None:
            raise ValueError("source format is invalid")
        if _SOURCE_SCHEMA_PATTERN.fullmatch(self.source_schema_version) is None:
            raise ValueError("source schema version is invalid")
        if _PROFILE_ID_PATTERN.fullmatch(self.adapter_profile_id) is None:
            raise ValueError("adapter profile identifier is invalid")
        if _SOURCE_ADAPTER_PATTERN.fullmatch(self.source_adapter) is None:
            raise ValueError("source adapter is invalid")
        if self.source_digest_kind not in _SOURCE_DIGEST_KINDS:
            raise ValueError("source digest kind is invalid")
        if self.identity_mode not in _IDENTITY_MODES:
            raise ValueError("identity mode is invalid")
        if self.identity_mode != "profile_content_addressed":
            raise ValueError("legacy identity is not publicly constructible")
        if self.source_digest_kind == "canonical_records" and (
            self.source_format != "shadow-records"
            or self.source_schema_version != "shadow-input/v1"
        ):
            raise ValueError("canonical record source identity is invalid")
        if self.timestamp_mode not in _TIMESTAMP_MODES:
            raise ValueError("timestamp mode is invalid")
        if self.capture_scope not in _CAPTURE_SCOPES:
            raise ValueError("capture scope is invalid")
        digests = (
            self.source_byte_digest,
            self.adapter_profile_digest,
            self.adapter_configuration_digest,
            self.task_scope_digest,
            self.lineage_scope_digest,
            self.capture_manifest_digest,
            self.binding_digest,
        )
        if any(value is not None and not _exact_digest(value) for value in digests):
            raise ValueError("trace binding digest is invalid")
        # Direct-record configuration is fully derivable here. ATIF additionally commits
        # caller-attested environment fields retained only in the exact trace preimage.
        if self.identity_mode == "profile_content_addressed" and self.source_format != "atif":
            configuration_bytes = canonical_json(
                {
                    "schema_version": "shadow-adapter-configuration/v1",
                    "adapter_profile_id": self.adapter_profile_id,
                    "adapter_profile_digest": self.adapter_profile_digest,
                    "source_format": self.source_format,
                    "source_schema_version": self.source_schema_version,
                    "timestamp_mode": self.timestamp_mode,
                    "capture_scope": self.capture_scope,
                }
            )
            expected_configuration_digest = length_prefixed_sha256(
                configuration_bytes,
                domain=_CONFIGURATION_DIGEST_DOMAIN,
            )
            if not hmac.compare_digest(
                self.adapter_configuration_digest,
                expected_configuration_digest,
            ):
                raise ValueError("adapter configuration identity is invalid")
        if self.identity_mode == "profile_content_addressed":
            expected_adapter = _source_adapter(
                self.source_format,
                self.adapter_profile_id,
                self.adapter_profile_digest,
                self.adapter_configuration_digest,
            )
            if self.source_adapter != expected_adapter:
                raise ValueError("source adapter identity is invalid")
        expected = _body_digest(
            self,
            excluded="binding_digest",
            domain=_BINDING_DIGEST_DOMAIN,
        )
        if not hmac.compare_digest(self.binding_digest, expected):
            raise ValueError("trace binding digest does not match")
        return self

    def __repr__(self) -> str:
        return "ShadowTraceBinding(<redacted>)"


class ShadowRecordDiagnostics(_TraceModel):
    """Exhaustive counts for direct Shadow wire records."""

    schema_version: Literal["shadow-record-diagnostics/v1"] = "shadow-record-diagnostics/v1"
    source_record_count: Annotated[int, Field(ge=1, le=MAX_SHADOW_TRACE_ROWS)]
    input_kind_counts: tuple[tuple[ShadowInputKind, Annotated[int, Field(ge=0)]], ...]
    repeated_source_identifier_count: Annotated[int, Field(ge=0)]
    mapped_shadow_record_count: Annotated[int, Field(ge=1, le=MAX_SHADOW_TRACE_ROWS)]
    diagnostics_digest: Sha256Digest

    def __init_subclass__(cls, **_kwargs: object) -> None:
        raise TypeError("ShadowRecordDiagnostics cannot be subclassed")

    @field_validator("schema_version", "diagnostics_digest", mode="before")
    @classmethod
    def require_exact_text(cls, value: object) -> object:
        if type(value) is not str:
            raise ValueError("trace diagnostics text is invalid")
        return value

    @field_validator(
        "source_record_count",
        "repeated_source_identifier_count",
        "mapped_shadow_record_count",
        mode="before",
    )
    @classmethod
    def require_exact_count(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("trace diagnostics count is invalid")
        return value

    @field_validator("input_kind_counts", mode="before")
    @classmethod
    def require_exact_nested_counts(cls, value: object) -> object:
        if type(value) not in (tuple, list):
            raise ValueError("trace diagnostics kind counts are invalid")
        assert isinstance(value, (tuple, list))
        copied: list[tuple[object, object]] = []
        for item in value:
            if type(item) not in (tuple, list):
                raise ValueError("trace diagnostics kind count is invalid")
            assert isinstance(item, (tuple, list))
            if (
                len(item) != 2
                or type(item[0]) not in (str, ShadowInputKind)
                or type(item[1]) is not int
            ):
                raise ValueError("trace diagnostics kind count is invalid")
            copied.append((item[0], item[1]))
        return tuple(copied)

    @model_validator(mode="after")
    def validate_counts_and_digest(self) -> ShadowRecordDiagnostics:
        observed_kinds = tuple(item[0] for item in self.input_kind_counts)
        if len(observed_kinds) not in (
            len(_LEGACY_INPUT_KINDS),
            len(_EXTENDED_INPUT_KINDS),
        ):
            raise ValueError("trace diagnostics kinds are incomplete")
        if observed_kinds not in (_LEGACY_INPUT_KINDS, _EXTENDED_INPUT_KINDS):
            raise ValueError("trace diagnostics kinds are not canonical")
        if observed_kinds == _EXTENDED_INPUT_KINDS and self.input_kind_counts[-1][1] == 0:
            raise ValueError("trace diagnostics identity count is not canonical")
        if sum(item[1] for item in self.input_kind_counts) != self.source_record_count:
            raise ValueError("trace diagnostics count equation is invalid")
        if self.repeated_source_identifier_count > self.source_record_count:
            raise ValueError("trace retry count is invalid")
        if self.mapped_shadow_record_count != self.source_record_count:
            raise ValueError("direct trace mapped count is invalid")
        if not _exact_digest(self.diagnostics_digest):
            raise ValueError("trace diagnostics digest is invalid")
        expected = _body_digest(
            self,
            excluded="diagnostics_digest",
            domain=_DIRECT_DIAGNOSTICS_DIGEST_DOMAIN,
        )
        if not hmac.compare_digest(self.diagnostics_digest, expected):
            raise ValueError("trace diagnostics digest does not match")
        return self

    def __repr__(self) -> str:
        return "ShadowRecordDiagnostics(<counts>)"


class ATIFShadowDiagnostics(_TraceModel):
    """Bounded, exhaustive mapping counts for one root ATIF segment."""

    schema_version: Literal["atif-shadow-diagnostics/v1"] = "atif-shadow-diagnostics/v1"
    root_segment_only: Literal[True] = True
    continued_trajectory_ref_present: bool
    embedded_subagent_trajectory_count: Annotated[int, Field(ge=0, le=100_000)]
    complete_execution_session_coverage: Literal[False] = False
    producer_authentication: Literal["none"] = "none"
    outcome_evidence_authority: Literal["none", "producer_claimed_structured"]
    profile_audit_manifest_digest: Sha256Digest
    total_step_count: Annotated[int, Field(ge=1, le=10_000)]
    ignored_message_step_count: Annotated[int, Field(ge=0, le=10_000)]
    total_tool_call_count: Annotated[int, Field(ge=0, le=10_000)]
    tool_call_disposition_counts: tuple[
        tuple[ToolCallDisposition, Annotated[int, Field(ge=0)]], ...
    ]
    total_observation_result_count: Annotated[int, Field(ge=0, le=10_000)]
    result_disposition_counts: tuple[tuple[ResultDisposition, Annotated[int, Field(ge=0)]], ...]
    mapped_shadow_record_count: Annotated[int, Field(ge=2, le=MAX_SHADOW_TRACE_ROWS)]
    diagnostics_digest: Sha256Digest

    def __init_subclass__(cls, **_kwargs: object) -> None:
        raise TypeError("ATIFShadowDiagnostics cannot be subclassed")

    @field_validator(
        "schema_version",
        "producer_authentication",
        "outcome_evidence_authority",
        "profile_audit_manifest_digest",
        "diagnostics_digest",
        mode="before",
    )
    @classmethod
    def require_exact_text(cls, value: object) -> object:
        if type(value) is not str:
            raise ValueError("ATIF diagnostics text is invalid")
        return value

    @field_validator(
        "root_segment_only",
        "continued_trajectory_ref_present",
        "complete_execution_session_coverage",
        mode="before",
    )
    @classmethod
    def require_exact_boolean(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("ATIF diagnostics boolean is invalid")
        return value

    @field_validator(
        "embedded_subagent_trajectory_count",
        "total_step_count",
        "ignored_message_step_count",
        "total_tool_call_count",
        "total_observation_result_count",
        "mapped_shadow_record_count",
        mode="before",
    )
    @classmethod
    def require_exact_count(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("ATIF diagnostics count is invalid")
        return value

    @field_validator(
        "tool_call_disposition_counts",
        "result_disposition_counts",
        mode="before",
    )
    @classmethod
    def require_exact_nested_counts(cls, value: object) -> object:
        if type(value) not in (tuple, list):
            raise ValueError("ATIF diagnostics disposition counts are invalid")
        assert isinstance(value, (tuple, list))
        copied: list[tuple[str, int]] = []
        for item in value:
            if type(item) not in (tuple, list):
                raise ValueError("ATIF diagnostics disposition count is invalid")
            assert isinstance(item, (tuple, list))
            if len(item) != 2 or type(item[0]) is not str or type(item[1]) is not int:
                raise ValueError("ATIF diagnostics disposition count is invalid")
            copied.append((item[0], item[1]))
        return tuple(copied)

    @model_validator(mode="after")
    def validate_equations_and_digest(self) -> ATIFShadowDiagnostics:
        tool_kinds = tuple(item[0] for item in self.tool_call_disposition_counts)
        result_kinds = tuple(item[0] for item in self.result_disposition_counts)
        if tool_kinds != _TOOL_CALL_DISPOSITIONS:
            raise ValueError("ATIF tool dispositions are not canonical")
        if result_kinds != _RESULT_DISPOSITIONS:
            raise ValueError("ATIF result dispositions are not canonical")
        if sum(item[1] for item in self.tool_call_disposition_counts) != self.total_tool_call_count:
            raise ValueError("ATIF tool disposition equation is invalid")
        if (
            sum(item[1] for item in self.result_disposition_counts)
            != self.total_observation_result_count
        ):
            raise ValueError("ATIF result disposition equation is invalid")
        if self.ignored_message_step_count > self.total_step_count:
            raise ValueError("ATIF ignored step count is invalid")
        mapped_actions = dict(self.tool_call_disposition_counts)["mapped_action"]
        mapped_outcomes = dict(self.result_disposition_counts)["mapped_structured_outcome"]
        if self.mapped_shadow_record_count != 2 + mapped_actions + mapped_outcomes:
            raise ValueError("ATIF mapped record equation is invalid")
        if not _exact_digest(self.profile_audit_manifest_digest) or not _exact_digest(
            self.diagnostics_digest
        ):
            raise ValueError("ATIF diagnostics digest is invalid")
        expected = _body_digest(
            self,
            excluded="diagnostics_digest",
            domain=_ATIF_DIAGNOSTICS_DIGEST_DOMAIN,
        )
        if not hmac.compare_digest(self.diagnostics_digest, expected):
            raise ValueError("ATIF diagnostics digest does not match")
        return self

    def __repr__(self) -> str:
        return "ATIFShadowDiagnostics(<counts>)"


def _diagnostics_discriminator(value: object) -> str | None:
    schema_version: object
    if type(value) is dict:
        schema_version = value.get("schema_version")
    elif type(value) in (ShadowRecordDiagnostics, ATIFShadowDiagnostics):
        assert isinstance(value, (ShadowRecordDiagnostics, ATIFShadowDiagnostics))
        schema_version = value.schema_version
    else:
        return None
    return schema_version if type(schema_version) is str else None


ShadowTraceDiagnostics: TypeAlias = Annotated[
    Annotated[ShadowRecordDiagnostics, Tag("shadow-record-diagnostics/v1")]
    | Annotated[ATIFShadowDiagnostics, Tag("atif-shadow-diagnostics/v1")],
    Discriminator(_diagnostics_discriminator),
]


@dataclass(frozen=True, slots=True)
class _Occurrence:
    kind: ShadowInputKind
    occurred_at: datetime
    event_ref: ShadowEventRef
    parent_source_event_id: str | None
    record_bytes: bytes


def _source_adapter(
    source_format: str,
    profile_id: str,
    profile_digest: str,
    configuration_digest: str,
) -> str:
    value = f"{source_format}.{profile_id}+p.{profile_digest}+c.{configuration_digest}"
    if len(value) > 256 or _SOURCE_ADAPTER_PATTERN.fullmatch(value) is None:
        raise ValueError("generated source adapter is invalid")
    return value


class _ExactJSONValueError(ValueError):
    pass


class _ExactJSONLimitError(ValueError):
    pass


def _utf8_size(value: str, *, limit: int | None) -> int:
    size = 0
    for character in value:
        codepoint = ord(character)
        if 0xD800 <= codepoint <= 0xDFFF:
            raise _ExactJSONValueError
        if codepoint <= 0x7F:
            size += 1
        elif codepoint <= 0x7FF:
            size += 2
        elif codepoint <= 0xFFFF:
            size += 3
        else:
            size += 4
        if limit is not None and size > limit:
            raise _ExactJSONLimitError
    return size


def _json_string_size(value: str, *, limit: int) -> int:
    size = 2
    for character in value:
        codepoint = ord(character)
        if character in ('"', "\\"):
            size += 2
        elif codepoint <= 0x1F:
            size += 2 if character in ("\b", "\f", "\n", "\r", "\t") else 6
        elif 0xD800 <= codepoint <= 0xDFFF:
            raise _ExactJSONValueError
        elif codepoint <= 0x7F:
            size += 1
        elif codepoint <= 0x7FF:
            size += 2
        elif codepoint <= 0xFFFF:
            size += 3
        else:
            size += 4
        if size > limit:
            raise _ExactJSONLimitError
    return size


def _copy_exact_json(
    value: object,
    *,
    max_bytes: int,
    max_depth: int,
    max_container_items: int,
    max_nodes: int | None,
    max_string_bytes: int | None,
) -> tuple[object, int]:
    nodes = 0
    active: set[int] = set()

    def add_size(current: int, extra: int, limit: int) -> int:
        result = current + extra
        if result > limit:
            raise _ExactJSONLimitError
        return result

    def copy(item: object, depth: int, budget: int) -> tuple[object, int]:
        nonlocal nodes
        nodes += 1
        if depth > max_depth or (max_nodes is not None and nodes > max_nodes):
            raise _ExactJSONLimitError
        if type(item) is dict:
            if budget < 2 or len(item) > max_container_items or id(item) in active:
                raise _ExactJSONLimitError
            active.add(id(item))
            result: dict[str, object] = {}
            size = 2
            try:
                for index, (key, child) in enumerate(item.items()):
                    if type(key) is not str:
                        raise _ExactJSONValueError
                    if max_string_bytes is not None:
                        _utf8_size(key, limit=max_string_bytes)
                    if index:
                        size = add_size(size, 1, budget)
                    key_size = _json_string_size(key, limit=budget - size)
                    size = add_size(size, key_size + 1, budget)
                    copied, child_size = copy(child, depth + 1, budget - size)
                    result[key] = copied
                    size = add_size(size, child_size, budget)
            finally:
                active.remove(id(item))
            return result, size
        if type(item) is list:
            if budget < 2 or len(item) > max_container_items or id(item) in active:
                raise _ExactJSONLimitError
            active.add(id(item))
            result_list: list[object] = []
            size = 2
            try:
                for index, child in enumerate(item):
                    if index:
                        size = add_size(size, 1, budget)
                    copied, child_size = copy(child, depth + 1, budget - size)
                    result_list.append(copied)
                    size = add_size(size, child_size, budget)
            finally:
                active.remove(id(item))
            return result_list, size
        if type(item) is str:
            if max_string_bytes is not None:
                _utf8_size(item, limit=max_string_bytes)
            return item, _json_string_size(item, limit=budget)
        if item is None:
            if budget < 4:
                raise _ExactJSONLimitError
            return None, 4
        if type(item) is bool:
            size = 4 if item else 5
            if size > budget:
                raise _ExactJSONLimitError
            return item, size
        if type(item) is int:
            try:
                size = len(str(item))
            except ValueError:
                raise _ExactJSONLimitError from None
            if size > budget:
                raise _ExactJSONLimitError
            return item, size
        if type(item) is float:
            if not math.isfinite(item):
                raise _ExactJSONValueError
            size = len(json.dumps(item, allow_nan=False, separators=(",", ":")))
            if size > budget:
                raise _ExactJSONLimitError
            return item, size
        raise _ExactJSONValueError

    return copy(value, 0, max_bytes)


def _copy_descriptor(value: object) -> bytes:
    if type(value) is not dict:
        raise _ExactJSONValueError
    copied, estimated_size = _copy_exact_json(
        value,
        max_bytes=MAX_ADAPTER_DESCRIPTOR_BYTES,
        max_depth=_MAX_DESCRIPTOR_DEPTH,
        max_container_items=_MAX_DESCRIPTOR_ITEMS,
        max_nodes=None,
        max_string_bytes=_MAX_DESCRIPTOR_STRING_BYTES,
    )
    encoded = canonical_json(copied)
    if len(encoded) != estimated_size:
        raise _ExactJSONValueError
    return encoded


def _freeze_json(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    return value


def _frozen_json_is_exact(value: object) -> bool:
    nodes = 0
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_RECORD_NODES or depth > _MAX_RECORD_DEPTH:
            return False
        if type(item) is MappingProxyType:
            if len(item) > _MAX_RECORD_ITEMS:
                return False
            for key, child in item.items():
                if type(key) is not str:
                    return False
                stack.append((child, depth + 1))
        elif type(item) is tuple:
            if len(item) > _MAX_RECORD_ITEMS:
                return False
            stack.extend((child, depth + 1) for child in item)
        elif item is None or type(item) in (str, bool, int):
            continue
        elif type(item) is float:
            if not math.isfinite(item):
                return False
        else:
            return False
    return True


def _copy_frozen_json(value: object) -> object:
    if type(value) is MappingProxyType:
        return MappingProxyType({key: _copy_frozen_json(item) for key, item in value.items()})
    if type(value) is tuple:
        return tuple(_copy_frozen_json(item) for item in value)
    return value


def _parse_timestamp(value: object, *, ordinal: int) -> datetime:
    if type(value) is not str or _CANONICAL_TIMESTAMP_PATTERN.fullmatch(value) is None:
        raise ShadowTraceInputError("invalid_timestamp", step_ordinal=ordinal)
    failed = False
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00").astimezone(UTC)
    except (OverflowError, ValueError):
        failed = True
        parsed = datetime.min.replace(tzinfo=UTC)
    if failed:
        raise ShadowTraceInputError("invalid_timestamp", step_ordinal=ordinal)
    canonical = parsed.isoformat(
        timespec="microseconds" if parsed.microsecond else "seconds"
    ).replace("+00:00", "Z")
    explicit_zero_fraction = (
        parsed.microsecond == 0 and value.endswith(".000000Z") and f"{value[:-8]}Z" == canonical
    )
    if canonical != value and not explicit_zero_fraction:
        raise ShadowTraceInputError("invalid_timestamp", step_ordinal=ordinal)
    return parsed


def _require_fields(value: dict[str, object], kind: ShadowInputKind) -> None:
    required, optional = _WIRE_FIELDS[kind]
    fields = frozenset(value)
    if not required.issubset(fields) or not fields.issubset(required | optional):
        raise ValueError("wire fields are invalid")


def _parent_ref(
    value: object,
    known: Mapping[str, _Occurrence],
) -> tuple[str, ShadowEventRef]:
    if type(value) is not str:
        raise ValueError("action parent is invalid")
    occurrence = known.get(value)
    if occurrence is None or occurrence.kind not in (
        ShadowInputKind.ACTION,
        ShadowInputKind.ACTION_IDENTITY,
    ):
        raise ValueError("action parent is missing")
    return value, occurrence.event_ref


def _validate_wire_record(
    value: dict[str, object],
    *,
    run_id: UUID,
    known: Mapping[str, _Occurrence],
    ordinal: int,
) -> tuple[ShadowInputKind, str, datetime, str | None]:
    if (
        value.get("schema_version") != "shadow-input/v1"
        or type(value.get("schema_version")) is not str
    ):
        raise ShadowTraceInputError("unsupported_schema", step_ordinal=ordinal)
    wire_kind = value.get("kind")
    if type(wire_kind) is not str or wire_kind not in _WIRE_TO_INPUT_KIND:
        raise ValueError("wire kind is invalid")
    kind = _WIRE_TO_INPUT_KIND[wire_kind]
    _require_fields(value, kind)
    source_event_id = value["source_event_id"]
    if type(source_event_id) is not str:
        raise ValueError("source event identifier is invalid")
    occurred_at = _parse_timestamp(value["occurred_at"], ordinal=ordinal)
    common: dict[str, object] = {
        "source_event_id": source_event_id,
        "occurred_at": occurred_at,
    }
    parent_source_event_id: str | None = None
    if kind is ShadowInputKind.START:
        ShadowStartInput.model_validate(common)
    elif kind is ShadowInputKind.ACTION:
        argv = value.get("argv")
        if argv is not None:
            if type(argv) is not list or any(type(item) is not str for item in argv):
                raise ValueError("action arguments are invalid")
            argv = tuple(argv)
        ShadowActionInput.model_validate(
            {
                **common,
                "command": value.get("command"),
                "argv": argv,
                "working_directory": value["working_directory"],
                "environment_digest": value["environment_digest"],
            }
        )
    elif kind is ShadowInputKind.ACTION_IDENTITY:
        ShadowActionIdentityInput.model_validate(
            {
                **common,
                "action_digest": value["action_digest"],
                "workspace_digest": value["workspace_digest"],
                "environment_digest": value["environment_digest"],
                "identity_authority": value["identity_authority"],
            }
        )
    elif kind is ShadowInputKind.TOOL_RESULT:
        parent_source_event_id, parent = _parent_ref(value["action_source_event_id"], known)
        ShadowToolResultInput.model_validate(
            {
                **common,
                "action": parent,
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
        parent_source_event_id, parent = _parent_ref(value["action_source_event_id"], known)
        ShadowTestResultInput.model_validate(
            {
                **common,
                "action": parent,
                "framework": value["framework"],
                "status": value["status"],
                "failures": tuple(failures),
            }
        )
    elif kind is ShadowInputKind.OBSERVATION:
        source = value["source"]
        if type(source) is not str:
            raise ValueError("observation source is invalid")
        ShadowObservationInput.model_validate(
            {
                **common,
                "source": ShadowObservationSource(source),
                "payload": value["payload"],
            }
        )
    elif kind is ShadowInputKind.CONTROLLER_ERROR:
        ShadowControllerErrorInput.model_validate({**common, "error_code": value["error_code"]})
    else:
        ShadowFinishInput.model_validate(common)
    return kind, source_event_id, occurred_at, parent_source_event_id


def _snapshot_wire_record(
    supplied: object,
    *,
    ordinal: int,
    byte_budget: int,
) -> tuple[dict[str, object], bytes]:
    if type(supplied) is not dict:
        raise ShadowTraceInputError("invalid_step", step_ordinal=ordinal)
    limit_failure = False
    value_failure = False
    unexpected_failure = False
    try:
        copied, estimated_size = _copy_exact_json(
            supplied,
            max_bytes=byte_budget,
            max_depth=_MAX_RECORD_DEPTH,
            max_container_items=_MAX_RECORD_ITEMS,
            max_nodes=_MAX_RECORD_NODES,
            max_string_bytes=None,
        )
    except _ExactJSONLimitError:
        limit_failure = True
        copied, estimated_size = {}, 0
    except _ExactJSONValueError:
        value_failure = True
        copied, estimated_size = {}, 0
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        unexpected_failure = True
        copied, estimated_size = {}, 0
    if limit_failure:
        raise ShadowTraceInputError("input_limit_exceeded", step_ordinal=ordinal)
    if value_failure or unexpected_failure or type(copied) is not dict:
        raise ShadowTraceInputError("invalid_json", step_ordinal=ordinal)

    encoding_failure = False
    try:
        encoded = canonical_json(copied)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        encoding_failure = True
        encoded = b""
    if encoding_failure:
        raise ShadowTraceInputError("invalid_json", step_ordinal=ordinal)
    if len(encoded) != estimated_size:
        raise ShadowTraceInputError("invalid_json", step_ordinal=ordinal)
    return copied, encoded


def _canonical_records(
    records: Iterable[Mapping[str, object]],
    *,
    run_id: UUID,
    capture_scope: CaptureScope,
    timestamp_mode: TimestampMode,
) -> tuple[
    tuple[Mapping[str, object], ...],
    tuple[bytes, ...],
    tuple[tuple[ShadowInputKind, int], ...],
    int,
]:
    snapshots: list[Mapping[str, object]] = []
    encoded_records: list[bytes] = []
    kinds: list[ShadowInputKind] = []
    known: dict[str, _Occurrence] = {}
    latest_timestamp: datetime | None = None
    repeated = 0
    aggregate_size = 2
    iterator = iter(records)
    for ordinal in range(1, MAX_SHADOW_TRACE_ROWS + 2):
        iterator_failure = False
        try:
            supplied = next(iterator)
        except StopIteration:
            break
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            iterator_failure = True
            supplied = None
        if iterator_failure:
            raise ShadowTraceInputError("invalid_step", step_ordinal=ordinal)
        if ordinal > MAX_SHADOW_TRACE_ROWS:
            raise ShadowTraceInputError("input_limit_exceeded", step_ordinal=ordinal)
        separator_size = 1 if encoded_records else 0
        remaining = MAX_SHADOW_TRACE_BYTES - aggregate_size - separator_size
        decoded, encoded = _snapshot_wire_record(
            supplied,
            ordinal=ordinal,
            byte_budget=max(0, remaining),
        )
        aggregate_size += len(encoded) + separator_size
        trace_failure: ShadowTraceInputError | None = None
        validation_failure = False
        try:
            kind, source_event_id, occurred_at, parent_source_event_id = _validate_wire_record(
                decoded,
                run_id=run_id,
                known=known,
                ordinal=ordinal,
            )
        except ShadowTraceInputError as error:
            trace_failure = error
            kind = ShadowInputKind.START
            source_event_id = ""
            occurred_at = datetime.min.replace(tzinfo=UTC)
            parent_source_event_id = None
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            validation_failure = True
            kind = ShadowInputKind.START
            source_event_id = ""
            occurred_at = datetime.min.replace(tzinfo=UTC)
            parent_source_event_id = None
        if trace_failure is not None:
            raise trace_failure
        if validation_failure:
            raise ShadowTraceInputError("invalid_step", step_ordinal=ordinal)
        existing = known.get(source_event_id)
        if (
            timestamp_mode == "logical_order"
            and latest_timestamp is not None
            and occurred_at <= latest_timestamp
        ):
            raise ShadowTraceInputError("invalid_timestamp", step_ordinal=ordinal)
        if existing is not None:
            if (
                existing.kind is not kind
                or existing.occurred_at != occurred_at
                or existing.parent_source_event_id != parent_source_event_id
                or (kind is ShadowInputKind.ACTION_IDENTITY and existing.record_bytes != encoded)
            ):
                raise ShadowTraceInputError("invalid_step", step_ordinal=ordinal)
            repeated += 1
        else:
            if latest_timestamp is not None:
                timestamp_reversed = occurred_at < latest_timestamp
                if timestamp_reversed:
                    raise ShadowTraceInputError("invalid_timestamp", step_ordinal=ordinal)
            sequence = len(known) + 1
            event_ref = ShadowEventRef(
                run_id=run_id,
                event_id=derive_shadow_event_id(run_id, source_event_id),
                sequence=sequence,
            )
            known[source_event_id] = _Occurrence(
                kind=kind,
                occurred_at=occurred_at,
                event_ref=event_ref,
                parent_source_event_id=parent_source_event_id,
                record_bytes=encoded,
            )
            latest_timestamp = occurred_at
        kinds.append(kind)
        encoded_records.append(encoded)
        frozen = _freeze_json(decoded)
        if not isinstance(frozen, Mapping):
            raise ShadowTraceInputError("invalid_step", step_ordinal=ordinal)
        snapshots.append(frozen)
    if not snapshots:
        raise ShadowTraceInputError("invalid_step")
    start_positions = tuple(
        index for index, kind in enumerate(kinds, start=1) if kind is ShadowInputKind.START
    )
    finish_positions = tuple(
        index for index, kind in enumerate(kinds, start=1) if kind is ShadowInputKind.FINISH
    )
    if start_positions != (1,):
        raise ShadowTraceInputError("invalid_step")
    if len(finish_positions) > 1 or (finish_positions and finish_positions != (len(kinds),)):
        raise ShadowTraceInputError("invalid_step")
    if capture_scope == "complete_run_declared" and finish_positions != (len(kinds),):
        raise ShadowTraceInputError("invalid_step")
    identity_count = kinds.count(ShadowInputKind.ACTION_IDENTITY)
    diagnostic_kinds = _EXTENDED_INPUT_KINDS if identity_count else _LEGACY_INPUT_KINDS
    counts = tuple((kind, kinds.count(kind)) for kind in diagnostic_kinds)
    return tuple(snapshots), tuple(encoded_records), counts, repeated


def _build_binding(
    *,
    source_format: str,
    source_schema_version: str,
    source_digest_kind: SourceDigestKind,
    source_bytes: bytes,
    adapter_profile_id: str,
    adapter_profile_digest: str,
    adapter_configuration_digest: str,
    timestamp_mode: TimestampMode,
    capture_scope: CaptureScope,
    task_scope_digest: str | None,
    lineage_scope_digest: str | None,
    capture_manifest_digest: str | None,
) -> ShadowTraceBinding:
    source_adapter = _source_adapter(
        source_format,
        adapter_profile_id,
        adapter_profile_digest,
        adapter_configuration_digest,
    )
    body: dict[str, object] = {
        "schema_version": "shadow-trace-binding/v1",
        "source_format": source_format,
        "source_schema_version": source_schema_version,
        "source_digest_kind": source_digest_kind,
        "source_byte_count": len(source_bytes),
        "source_byte_digest": hashlib.sha256(source_bytes).hexdigest(),
        "adapter_profile_id": adapter_profile_id,
        "adapter_profile_digest": adapter_profile_digest,
        "adapter_configuration_digest": adapter_configuration_digest,
        "source_adapter": source_adapter,
        "identity_mode": "profile_content_addressed",
        "timestamp_mode": timestamp_mode,
        "capture_scope": capture_scope,
        "task_scope_digest": task_scope_digest,
        "lineage_scope_digest": lineage_scope_digest,
        "capture_manifest_digest": capture_manifest_digest,
    }
    binding_digest = length_prefixed_sha256(
        canonical_json(body),
        domain=_BINDING_DIGEST_DOMAIN,
    )
    return ShadowTraceBinding.model_validate({**body, "binding_digest": binding_digest})


def _build_diagnostics(
    *,
    source_record_count: int,
    input_kind_counts: tuple[tuple[ShadowInputKind, int], ...],
    repeated_source_identifier_count: int,
) -> ShadowRecordDiagnostics:
    body: dict[str, object] = {
        "schema_version": "shadow-record-diagnostics/v1",
        "source_record_count": source_record_count,
        "input_kind_counts": input_kind_counts,
        "repeated_source_identifier_count": repeated_source_identifier_count,
        "mapped_shadow_record_count": source_record_count,
    }
    digest = length_prefixed_sha256(
        canonical_json(body),
        domain=_DIRECT_DIAGNOSTICS_DIGEST_DOMAIN,
    )
    return ShadowRecordDiagnostics.model_validate({**body, "diagnostics_digest": digest})


def _build_atif_diagnostics(
    *,
    continued_trajectory_ref_present: bool,
    embedded_subagent_trajectory_count: int,
    outcome_evidence_authority: Literal["none", "producer_claimed_structured"],
    profile_audit_manifest_digest: str,
    total_step_count: int,
    ignored_message_step_count: int,
    total_tool_call_count: int,
    tool_call_disposition_counts: tuple[tuple[ToolCallDisposition, int], ...],
    total_observation_result_count: int,
    result_disposition_counts: tuple[tuple[ResultDisposition, int], ...],
    mapped_shadow_record_count: int,
) -> ATIFShadowDiagnostics:
    body: dict[str, object] = {
        "schema_version": "atif-shadow-diagnostics/v1",
        "root_segment_only": True,
        "continued_trajectory_ref_present": continued_trajectory_ref_present,
        "embedded_subagent_trajectory_count": embedded_subagent_trajectory_count,
        "complete_execution_session_coverage": False,
        "producer_authentication": "none",
        "outcome_evidence_authority": outcome_evidence_authority,
        "profile_audit_manifest_digest": profile_audit_manifest_digest,
        "total_step_count": total_step_count,
        "ignored_message_step_count": ignored_message_step_count,
        "total_tool_call_count": total_tool_call_count,
        "tool_call_disposition_counts": tool_call_disposition_counts,
        "total_observation_result_count": total_observation_result_count,
        "result_disposition_counts": result_disposition_counts,
        "mapped_shadow_record_count": mapped_shadow_record_count,
    }
    digest = length_prefixed_sha256(
        canonical_json(body),
        domain=_ATIF_DIAGNOSTICS_DIGEST_DOMAIN,
    )
    return ATIFShadowDiagnostics.model_validate({**body, "diagnostics_digest": digest})


@final
@dataclass(frozen=True, slots=True, init=False, repr=False)
class ShadowTrace:
    """One immutable bounded Shadow wire trace and its mapping provenance."""

    schema_version: Literal["shadow-trace/v1"]
    run_id: UUID
    binding: ShadowTraceBinding
    diagnostics: ShadowTraceDiagnostics
    mapped_record_digest: str
    _records: tuple[Mapping[str, object], ...] = field(repr=False)
    _record_bytes: tuple[bytes, ...] = field(repr=False, compare=False)
    _adapter_descriptor_bytes: bytes = field(repr=False, compare=False)
    _adapter_configuration_bytes: bytes = field(repr=False, compare=False)
    _binding_bytes: bytes = field(repr=False, compare=False)
    _diagnostics_bytes: bytes = field(repr=False, compare=False)
    _run_id_bytes: bytes = field(repr=False, compare=False)
    _factory_token: object = field(repr=False, compare=False)

    def __init__(self) -> None:
        raise TypeError("ShadowTrace must be created by a validated factory")

    def __init_subclass__(cls, **_kwargs: object) -> None:
        raise TypeError("ShadowTrace cannot be subclassed")

    @staticmethod
    def adapter_profile_digest(adapter_descriptor: dict[str, object]) -> str:
        """Return the exact bounded profile identity used by trace factories."""

        failed = False
        try:
            descriptor_bytes = _copy_descriptor(adapter_descriptor)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            failed = True
            descriptor_bytes = b""
        if failed:
            raise ShadowTraceInputError("invalid_step")
        return length_prefixed_sha256(
            descriptor_bytes,
            domain=_PROFILE_DIGEST_DOMAIN,
        )

    @classmethod
    def from_records(
        cls,
        records: Iterable[dict[str, object]],
        *,
        run_id: UUID,
        adapter_profile_id: str,
        adapter_descriptor: dict[str, object],
        source_bytes: bytes | None = None,
        source_format: str = "shadow-records",
        source_schema_version: str = "shadow-input/v1",
        timestamp_mode: TimestampMode = "record_declared",
        capture_scope: CaptureScope = "unknown",
        task_scope_digest: str | None = None,
        lineage_scope_digest: str | None = None,
        capture_manifest_digest: str | None = None,
    ) -> ShadowTrace:
        """Snapshot and validate direct Shadow wire records without side effects."""

        if type(run_id) is not UUID or run_id.version != 4:
            raise ShadowTraceInputError("invalid_step")
        if (
            type(adapter_profile_id) is not str
            or _PROFILE_ID_PATTERN.fullmatch(adapter_profile_id) is None
        ):
            raise ShadowTraceInputError("invalid_step")
        if adapter_profile_id in _BUILT_IN_ATIF_PROFILE_IDS:
            raise ShadowTraceInputError("profile_mismatch")
        if (
            type(source_format) is not str
            or _SOURCE_FORMAT_PATTERN.fullmatch(source_format) is None
        ):
            raise ShadowTraceInputError("invalid_step")
        if source_format == "atif":
            raise ShadowTraceInputError("profile_mismatch")
        if (
            type(source_schema_version) is not str
            or _SOURCE_SCHEMA_PATTERN.fullmatch(source_schema_version) is None
        ):
            raise ShadowTraceInputError("unsupported_schema")
        if type(timestamp_mode) is not str or timestamp_mode not in _TIMESTAMP_MODES:
            raise ShadowTraceInputError("invalid_timestamp")
        if type(capture_scope) is not str or capture_scope not in _CAPTURE_SCOPES:
            raise ShadowTraceInputError("invalid_step")
        for digest in (task_scope_digest, lineage_scope_digest, capture_manifest_digest):
            if digest is not None and not _exact_digest(digest):
                raise ShadowTraceInputError("invalid_step")
        if source_bytes is not None and (
            type(source_bytes) is not bytes or not 1 <= len(source_bytes) <= MAX_SHADOW_TRACE_BYTES
        ):
            raise ShadowTraceInputError("input_limit_exceeded")
        if source_bytes is None and source_format != "shadow-records":
            raise ShadowTraceInputError("profile_mismatch")
        if source_bytes is None and source_schema_version != "shadow-input/v1":
            raise ShadowTraceInputError("unsupported_schema")

        trace_failure: ShadowTraceInputError | None = None
        unexpected_failure = False
        try:
            descriptor_bytes = _copy_descriptor(adapter_descriptor)
            copied_run_id = UUID(int=run_id.int)
            frozen_records, record_bytes, counts, repeated = _canonical_records(
                records,
                run_id=copied_run_id,
                capture_scope=capture_scope,
                timestamp_mode=timestamp_mode,
            )
            canonical_record_source = b"[" + b",".join(record_bytes) + b"]"
            if len(canonical_record_source) > MAX_SHADOW_TRACE_BYTES:
                raise ShadowTraceInputError("input_limit_exceeded")
            profile_digest = length_prefixed_sha256(
                descriptor_bytes,
                domain=_PROFILE_DIGEST_DOMAIN,
            )
            configuration_bytes = canonical_json(
                {
                    "schema_version": "shadow-adapter-configuration/v1",
                    "adapter_profile_id": adapter_profile_id,
                    "adapter_profile_digest": profile_digest,
                    "source_format": source_format,
                    "source_schema_version": source_schema_version,
                    "timestamp_mode": timestamp_mode,
                    "capture_scope": capture_scope,
                }
            )
            configuration_digest = length_prefixed_sha256(
                configuration_bytes,
                domain=_CONFIGURATION_DIGEST_DOMAIN,
            )
            exact_source = canonical_record_source if source_bytes is None else bytes(source_bytes)
            binding = _build_binding(
                source_format=source_format,
                source_schema_version=source_schema_version,
                source_digest_kind=(
                    "canonical_records" if source_bytes is None else "original_bytes"
                ),
                source_bytes=exact_source,
                adapter_profile_id=adapter_profile_id,
                adapter_profile_digest=profile_digest,
                adapter_configuration_digest=configuration_digest,
                timestamp_mode=timestamp_mode,
                capture_scope=capture_scope,
                task_scope_digest=task_scope_digest,
                lineage_scope_digest=lineage_scope_digest,
                capture_manifest_digest=capture_manifest_digest,
            )
            diagnostics = _build_diagnostics(
                source_record_count=len(frozen_records),
                input_kind_counts=counts,
                repeated_source_identifier_count=repeated,
            )
            trace = _new_shadow_trace(
                run_id=copied_run_id,
                binding=binding,
                diagnostics=diagnostics,
                records=frozen_records,
                record_bytes=record_bytes,
                adapter_descriptor_bytes=descriptor_bytes,
                adapter_configuration_bytes=configuration_bytes,
            )
        except ShadowTraceInputError as error:
            trace_failure = error
            trace = None
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            unexpected_failure = True
            trace = None
        if trace_failure is not None:
            raise trace_failure
        if unexpected_failure or trace is None:
            raise ShadowTraceInputError("invalid_step")
        if cls is not ShadowTrace:
            raise ShadowTraceInputError("invalid_step")
        return trace

    @property
    def records(self) -> tuple[Mapping[str, object], ...]:
        return self._records

    def _is_exact(self) -> bool:
        if (
            type(self) is not ShadowTrace
            or getattr(self, "_factory_token", None) is not _TRACE_FACTORY_TOKEN
        ):
            return False
        try:
            if type(self.schema_version) is not str or self.schema_version != "shadow-trace/v1":
                return False
            if type(self.run_id) is not UUID or self.run_id.version != 4:
                return False
            binding = self.binding
            diagnostics = self.diagnostics
            records = self._records
            record_bytes = self._record_bytes
            descriptor_bytes = self._adapter_descriptor_bytes
            configuration_bytes = self._adapter_configuration_bytes
            binding_bytes = self._binding_bytes
            diagnostics_bytes = self._diagnostics_bytes
            run_id_bytes = self._run_id_bytes
            if type(binding) is not ShadowTraceBinding:
                return False
            if type(diagnostics) not in (ShadowRecordDiagnostics, ATIFShadowDiagnostics):
                return False
            if (type(diagnostics) is ATIFShadowDiagnostics) != (binding.source_format == "atif"):
                return False
            if type(records) is not tuple or type(record_bytes) is not tuple:
                return False
            if (
                type(descriptor_bytes) is not bytes
                or type(configuration_bytes) is not bytes
                or type(binding_bytes) is not bytes
                or type(diagnostics_bytes) is not bytes
                or type(run_id_bytes) is not bytes
            ):
                return False
            if not hmac.compare_digest(self.run_id.bytes, run_id_bytes):
                return False
            ShadowTraceBinding.model_validate(binding)
            if type(diagnostics) is ShadowRecordDiagnostics:
                ShadowRecordDiagnostics.model_validate(diagnostics)
            else:
                ATIFShadowDiagnostics.model_validate(diagnostics)
            if canonical_json(binding) != binding_bytes:
                return False
            if canonical_json(diagnostics) != diagnostics_bytes:
                return False
            profile_digest = length_prefixed_sha256(
                descriptor_bytes,
                domain=_PROFILE_DIGEST_DOMAIN,
            )
            configuration_digest = length_prefixed_sha256(
                configuration_bytes,
                domain=_CONFIGURATION_DIGEST_DOMAIN,
            )
            if not hmac.compare_digest(binding.adapter_profile_digest, profile_digest):
                return False
            if not hmac.compare_digest(
                binding.adapter_configuration_digest,
                configuration_digest,
            ):
                return False
            if len(records) != len(record_bytes):
                return False
            for record, encoded in zip(records, record_bytes, strict=True):
                if (
                    type(record) is not MappingProxyType
                    or not _frozen_json_is_exact(record)
                    or type(encoded) is not bytes
                ):
                    return False
                if canonical_json(record) != encoded:
                    return False
            (
                _validated_records,
                validated_record_bytes,
                validated_kind_counts,
                validated_repeated_count,
            ) = _canonical_records(
                tuple(json.loads(encoded) for encoded in record_bytes),
                run_id=self.run_id,
                capture_scope=binding.capture_scope,
                timestamp_mode=binding.timestamp_mode,
            )
            if validated_record_bytes != record_bytes:
                return False
            mapped_digest = length_prefixed_sha256(
                *record_bytes,
                domain=_MAPPED_RECORD_DIGEST_DOMAIN,
            )
            if not _exact_digest(self.mapped_record_digest) or not hmac.compare_digest(
                self.mapped_record_digest,
                mapped_digest,
            ):
                return False
            if diagnostics.mapped_shadow_record_count != len(records):
                return False
            if type(diagnostics) is ShadowRecordDiagnostics and (
                diagnostics.source_record_count != len(records)
                or diagnostics.input_kind_counts != validated_kind_counts
                or diagnostics.repeated_source_identifier_count != validated_repeated_count
            ):
                return False
            if type(diagnostics) is ATIFShadowDiagnostics:
                from saliencegate.shadow.atif import (
                    _matches_sealed_configuration_contract,
                    _matches_sealed_trace_contract,
                )

                if not _matches_sealed_trace_contract(
                    profile_id=binding.adapter_profile_id,
                    descriptor_bytes=descriptor_bytes,
                    profile_digest=binding.adapter_profile_digest,
                    manifest_digest=diagnostics.profile_audit_manifest_digest,
                    outcome_authority=diagnostics.outcome_evidence_authority,
                ):
                    return False
                if not _matches_sealed_configuration_contract(
                    profile_id=binding.adapter_profile_id,
                    profile_digest=binding.adapter_profile_digest,
                    configuration_bytes=configuration_bytes,
                    configuration_digest=binding.adapter_configuration_digest,
                    source_schema_version=binding.source_schema_version,
                    timestamp_mode=binding.timestamp_mode,
                    capture_scope=binding.capture_scope,
                    records=records,
                    diagnostics=diagnostics,
                ):
                    return False
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            return False
        return True

    def _descriptor_preimage(self) -> bytes:
        return bytes(self._adapter_descriptor_bytes)

    def _configuration_preimage(self) -> bytes:
        return bytes(self._adapter_configuration_bytes)

    def _wire_record_bytes(self) -> tuple[bytes, ...]:
        return tuple(bytes(item) for item in self._record_bytes)

    def _copy_exact(self) -> ShadowTrace:
        """Return a detached, revalidated copy for trusted analyzer admission."""

        if not self._is_exact():
            raise ShadowTraceInputError("invalid_step")
        failed = False
        try:
            binding = ShadowTraceBinding.model_validate_json(self._binding_bytes)
            if type(self.diagnostics) is ShadowRecordDiagnostics:
                diagnostics: ShadowTraceDiagnostics = ShadowRecordDiagnostics.model_validate_json(
                    self._diagnostics_bytes
                )
            else:
                diagnostics = ATIFShadowDiagnostics.model_validate_json(self._diagnostics_bytes)
            records = cast(
                tuple[Mapping[str, object], ...],
                tuple(_copy_frozen_json(record) for record in self._records),
            )
            copied = _new_shadow_trace(
                run_id=UUID(bytes=self._run_id_bytes),
                binding=binding,
                diagnostics=diagnostics,
                records=records,
                record_bytes=self._wire_record_bytes(),
                adapter_descriptor_bytes=self._descriptor_preimage(),
                adapter_configuration_bytes=self._configuration_preimage(),
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            failed = True
            copied = None
        if failed or copied is None or not copied._is_exact():
            raise ShadowTraceInputError("invalid_step")
        return copied

    def __repr__(self) -> str:
        return "ShadowTrace(<validated>)"


def _new_shadow_trace(
    *,
    run_id: UUID,
    binding: ShadowTraceBinding,
    diagnostics: ShadowTraceDiagnostics,
    records: tuple[Mapping[str, object], ...],
    record_bytes: tuple[bytes, ...],
    adapter_descriptor_bytes: bytes,
    adapter_configuration_bytes: bytes,
) -> ShadowTrace:
    """Seal already preflighted records for trusted in-package factories."""

    if type(run_id) is not UUID or run_id.version != 4:
        raise ValueError("trace run identity is invalid")
    if type(binding) is not ShadowTraceBinding or type(diagnostics) not in (
        ShadowRecordDiagnostics,
        ATIFShadowDiagnostics,
    ):
        raise ValueError("trace models are invalid")
    if type(records) is not tuple or type(record_bytes) is not tuple:
        raise ValueError("trace records are invalid")
    if (
        type(adapter_descriptor_bytes) is not bytes
        or type(adapter_configuration_bytes) is not bytes
    ):
        raise ValueError("trace preimages are invalid")
    if len(records) != len(record_bytes) or not 1 <= len(records) <= MAX_SHADOW_TRACE_ROWS:
        raise ValueError("trace record count is invalid")
    if any(type(item) is not bytes for item in record_bytes):
        raise ValueError("trace record bytes are invalid")
    aggregate_record_bytes = 2 + sum(len(item) for item in record_bytes) + len(record_bytes) - 1
    if aggregate_record_bytes > MAX_SHADOW_TRACE_BYTES:
        raise ValueError("trace record bytes are too large")
    if not 1 <= len(adapter_descriptor_bytes) <= MAX_ADAPTER_DESCRIPTOR_BYTES:
        raise ValueError("trace descriptor preimage is invalid")
    if not 1 <= len(adapter_configuration_bytes) <= MAX_SHADOW_TRACE_BYTES:
        raise ValueError("trace configuration preimage is invalid")
    if any(
        type(record) is not MappingProxyType or not _frozen_json_is_exact(record)
        for record in records
    ):
        raise ValueError("trace record snapshot is invalid")
    if diagnostics.mapped_shadow_record_count != len(records):
        raise ValueError("trace diagnostics do not match records")
    if type(diagnostics) is ShadowRecordDiagnostics and diagnostics.source_record_count != len(
        records
    ):
        raise ValueError("direct diagnostics do not match records")
    if (type(diagnostics) is ATIFShadowDiagnostics) != (binding.source_format == "atif"):
        raise ValueError("trace diagnostic discriminant does not match source format")
    ShadowTraceBinding.model_validate(binding)
    if type(diagnostics) is ShadowRecordDiagnostics:
        ShadowRecordDiagnostics.model_validate(diagnostics)
    else:
        ATIFShadowDiagnostics.model_validate(diagnostics)
    expected_profile_digest = length_prefixed_sha256(
        adapter_descriptor_bytes,
        domain=_PROFILE_DIGEST_DOMAIN,
    )
    expected_configuration_digest = length_prefixed_sha256(
        adapter_configuration_bytes,
        domain=_CONFIGURATION_DIGEST_DOMAIN,
    )
    if not hmac.compare_digest(binding.adapter_profile_digest, expected_profile_digest):
        raise ValueError("trace profile digest does not match")
    if not hmac.compare_digest(
        binding.adapter_configuration_digest,
        expected_configuration_digest,
    ):
        raise ValueError("trace configuration digest does not match")
    canonical_records = tuple(canonical_json(record) for record in records)
    if canonical_records != record_bytes:
        raise ValueError("trace record bytes do not match")
    mapped_record_digest = length_prefixed_sha256(
        *record_bytes,
        domain=_MAPPED_RECORD_DIGEST_DOMAIN,
    )
    trace = object.__new__(ShadowTrace)
    object.__setattr__(trace, "schema_version", "shadow-trace/v1")
    object.__setattr__(trace, "run_id", UUID(int=run_id.int))
    object.__setattr__(trace, "binding", binding)
    object.__setattr__(trace, "diagnostics", diagnostics)
    object.__setattr__(trace, "mapped_record_digest", mapped_record_digest)
    object.__setattr__(trace, "_records", records)
    object.__setattr__(trace, "_record_bytes", tuple(bytes(item) for item in record_bytes))
    object.__setattr__(trace, "_adapter_descriptor_bytes", bytes(adapter_descriptor_bytes))
    object.__setattr__(trace, "_adapter_configuration_bytes", bytes(adapter_configuration_bytes))
    object.__setattr__(trace, "_binding_bytes", canonical_json(binding))
    object.__setattr__(trace, "_diagnostics_bytes", canonical_json(diagnostics))
    object.__setattr__(trace, "_run_id_bytes", run_id.bytes)
    object.__setattr__(trace, "_factory_token", _TRACE_FACTORY_TOKEN)
    return trace


__all__ = [
    "MAX_ADAPTER_DESCRIPTOR_BYTES",
    "MAX_SHADOW_TRACE_BYTES",
    "MAX_SHADOW_TRACE_ROWS",
    "ShadowTrace",
    "ShadowTraceBinding",
    "ShadowTraceDiagnostics",
]
