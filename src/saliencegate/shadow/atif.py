"""Bounded, sealed ATIF profiles for provider-free Shadow trace import."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from importlib import resources
from types import MappingProxyType
from typing import Final, Literal, cast, final
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from saliencegate.domain import canonical_json, length_prefixed_sha256
from saliencegate.shadow.errors import (
    ShadowConfigurationError,
    ShadowInvariantError,
    ShadowTraceInputError,
)
from saliencegate.shadow.trace import (
    MAX_SHADOW_TRACE_BYTES,
    MAX_SHADOW_TRACE_ROWS,
    ATIFShadowDiagnostics,
    ResultDisposition,
    ShadowTrace,
    TimestampMode,
    ToolCallDisposition,
    _build_atif_diagnostics,
    _build_binding,
    _canonical_records,
    _copy_descriptor,
    _freeze_json,
    _new_shadow_trace,
)
from saliencegate.signals.fingerprints import ShellActionEvidence

_MAX_JSON_DEPTH = 32
_MAX_JSON_NODES = 1_000_000
_MAX_OBJECT_MEMBERS = 4_096
_MAX_ARRAY_ITEMS = 100_000
_MAX_STRING_BYTES = 2 * 1_024 * 1_024
_MAX_STEPS = 10_000
_MAX_STEP_TOOL_CALLS = 1_024
_MAX_STEP_RESULTS = 1_024
_MAX_TOTAL_TOOL_CALLS = 10_000
_MAX_TOTAL_RESULTS = 10_000
_MAX_SESSION_ID_BYTES = 16 * 1_024
_MAX_COMMAND_BYTES = 128 * 1_024
_MAX_DIRECTORY_BYTES = 16 * 1_024
_MAX_CONTEXT_TEXT_BYTES = 16 * 1_024
_MAX_PREFIX_RULE_ITEMS = 256
_MAX_TERMINUS_DURATION_LEXEME_CHARS = 128
_MAX_TERMINUS_DURATION_SECONDS = Decimal("60")
_MAX_CODEX_INTEGER_ARGUMENT = (1 << 31) - 1
_MIN_EXIT_STATUS = -(1 << 31)
_MAX_EXIT_STATUS = (1 << 31) - 1

_PROFILE_DIGEST_DOMAIN = "saliencegate:shadow:adapter-profile:v1"
_CONFIGURATION_DIGEST_DOMAIN = "saliencegate:shadow:adapter-configuration:v1"
_MANIFEST_DIGEST_DOMAIN = "saliencegate:shadow:atif-profile-compatibility:v1"
_TERMINUS_CONTEXT_DOMAIN = "saliencegate:shadow:atif:terminus-execution-context:v1"
_CODEX_CONTEXT_DOMAIN = "saliencegate:shadow:atif:codex-execution-context:v1"
_TERMINUS_EXECUTION_SEMANTICS: Final[tuple[tuple[str, object], ...]] = (
    ("terminal_transport", "tmux_send_keys"),
    ("execution_delimiter", "line_feed"),
)
_CODEX_EXECUTION_SEMANTIC_KEYS: Final[tuple[str, ...]] = (
    "shell",
    "login",
    "tty",
    "sandbox_permissions",
)

_EXPECTED_MANIFEST_SHA256 = "981bad10e0d7fdb5391656f7df153fec59d24324ccab21dceb17344b88bd0e8d"
_EXPECTED_MANIFEST_DIGEST = "b12cdabc7a25644efb5da81aec3f8036f280e5d638e6b7aa07bb6593893967f2"
_MANIFEST_RESOURCE = "atif_profile_compatibility.json"
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_UTC_TIMESTAMP_PATTERN = re.compile(
    r"^(?P<whole>[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2})"
    r"(?:\.(?P<fraction>[0-9]{1,6}))?(?:Z|\+00:00)$"
)
_ATIF_ACTION_ID_PATTERN = re.compile(r"^atif-s(?P<step>[0-9]{8})-c(?P<call>[0-9]{4})-action$")

_TOOL_DISPOSITIONS: tuple[ToolCallDisposition, ...] = (
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


class ATIFProfile(StrEnum):
    """Explicit built-in ATIF mapping profiles."""

    HARBOR_TERMINUS_2_V1 = "harbor-terminus-2/v1"
    HARBOR_CODEX_V1 = "harbor-codex/v1"


_EXPECTED_PROFILE_DIGESTS: Final[MappingProxyType[ATIFProfile, str]] = MappingProxyType(
    {
        ATIFProfile.HARBOR_TERMINUS_2_V1: (
            "a590e2232ec7957b31234c7ab6c9392e371285cad0e28a4c751d0c78833e70df"
        ),
        ATIFProfile.HARBOR_CODEX_V1: (
            "64f8404359b3630b48780e53f2855d8e692bcfb140e577e2af707c84832150f6"
        ),
    }
)


class ShadowEnvironmentBinding(BaseModel):
    """Caller-attested execution environment used by ATIF action mappings."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    schema_version: Literal["shadow-environment-binding/v1"] = "shadow-environment-binding/v1"
    default_working_directory: str
    environment_digest: str

    def __init_subclass__(cls, **_kwargs: object) -> None:
        raise TypeError("ShadowEnvironmentBinding cannot be subclassed")

    @field_validator(
        "schema_version",
        "default_working_directory",
        "environment_digest",
        mode="before",
    )
    @classmethod
    def require_exact_text(cls, value: object) -> object:
        if type(value) is not str:
            raise ValueError("environment binding text is invalid")
        return value

    @model_validator(mode="after")
    def validate_environment(self) -> ShadowEnvironmentBinding:
        if _DIGEST_PATTERN.fullmatch(self.environment_digest) is None:
            raise ValueError("environment digest is invalid")
        ShellActionEvidence(
            schema_version="1.0",
            kind="shell",
            command="environment-binding-validation",
            working_directory=self.default_working_directory,
            environment_digest=self.environment_digest,
        )
        return self

    def __repr__(self) -> str:
        return "ShadowEnvironmentBinding(<redacted>)"

    def __str__(self) -> str:
        return self.__repr__()


@dataclass(frozen=True, slots=True)
class _JSONNumber:
    lexeme: str
    integer: bool


@dataclass(frozen=True, slots=True)
class _ATIFProfileContract:
    profile: ATIFProfile
    accepted_schema_versions: tuple[str, ...]
    required_agent_name: str
    selected_function_name: str
    outcome_evidence_authority: Literal["none", "producer_claimed_structured"]
    descriptor: MappingProxyType[str, object]
    descriptor_bytes: bytes
    profile_digest: str


@dataclass(frozen=True, slots=True)
class _ActionPlan:
    step_ordinal: int
    call_ordinal: int
    source_event_id: str
    command: str
    working_directory: str
    environment_digest: str
    execution_semantics: tuple[tuple[str, object], ...]
    source_timestamp: str | None
    outcome_exit_statuses: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _CallState:
    call_ordinal: int
    tool_call_id: str | None
    disposition: ToolCallDisposition
    action: _ActionPlan | None


@dataclass(frozen=True, slots=True)
class _MappingPlan:
    actions: tuple[_ActionPlan, ...]
    timestamp_mode: TimestampMode
    total_step_count: int
    ignored_message_step_count: int
    continued_trajectory_ref_present: bool
    embedded_subagent_trajectory_count: int
    tool_counts: tuple[tuple[ToolCallDisposition, int], ...]
    result_counts: tuple[tuple[ResultDisposition, int], ...]


class _JSONSyntaxError(ValueError):
    pass


class _JSONLimitError(ValueError):
    pass


class _JSONDuplicateKeyError(ValueError):
    pass


class _JSONPreflight:
    """Validate JSON grammar and allocation bounds before materialization."""

    __slots__ = ("_index", "_nodes", "_text")

    def __init__(self, text: str) -> None:
        self._text = text
        self._index = 0
        self._nodes = 0

    def run(self) -> None:
        self._skip_whitespace()
        self._parse_value(depth=0)
        self._skip_whitespace()
        if self._index != len(self._text):
            raise _JSONSyntaxError

    def _skip_whitespace(self) -> None:
        text = self._text
        index = self._index
        while index < len(text) and text[index] in " \t\r\n":
            index += 1
        self._index = index

    def _add_node(self) -> None:
        self._nodes += 1
        if self._nodes > _MAX_JSON_NODES:
            raise _JSONLimitError

    def _parse_value(self, *, depth: int) -> None:
        self._add_node()
        if self._index >= len(self._text):
            raise _JSONSyntaxError
        token = self._text[self._index]
        if token == "{":
            self._parse_object(depth=depth + 1)
        elif token == "[":
            self._parse_array(depth=depth + 1)
        elif token == '"':
            self._parse_string()
        elif token == "t":
            self._consume_literal("true")
        elif token == "f":
            self._consume_literal("false")
        elif token == "n":
            self._consume_literal("null")
        elif token == "-" or token.isdigit():
            self._parse_number()
        else:
            raise _JSONSyntaxError

    def _parse_object(self, *, depth: int) -> None:
        if depth > _MAX_JSON_DEPTH:
            raise _JSONLimitError
        self._index += 1
        self._skip_whitespace()
        if self._peek("}"):
            self._index += 1
            return
        members = 0
        while True:
            members += 1
            if members > _MAX_OBJECT_MEMBERS:
                raise _JSONLimitError
            if not self._peek('"'):
                raise _JSONSyntaxError
            self._parse_string()
            self._skip_whitespace()
            if not self._peek(":"):
                raise _JSONSyntaxError
            self._index += 1
            self._skip_whitespace()
            self._parse_value(depth=depth)
            self._skip_whitespace()
            if self._peek("}"):
                self._index += 1
                return
            if not self._peek(","):
                raise _JSONSyntaxError
            self._index += 1
            self._skip_whitespace()

    def _parse_array(self, *, depth: int) -> None:
        if depth > _MAX_JSON_DEPTH:
            raise _JSONLimitError
        self._index += 1
        self._skip_whitespace()
        if self._peek("]"):
            self._index += 1
            return
        items = 0
        while True:
            items += 1
            if items > _MAX_ARRAY_ITEMS:
                raise _JSONLimitError
            self._parse_value(depth=depth)
            self._skip_whitespace()
            if self._peek("]"):
                self._index += 1
                return
            if not self._peek(","):
                raise _JSONSyntaxError
            self._index += 1
            self._skip_whitespace()

    def _parse_string(self) -> None:
        self._index += 1
        decoded_bytes = 0
        text = self._text
        while self._index < len(text):
            character = text[self._index]
            self._index += 1
            if character == '"':
                return
            if ord(character) < 0x20:
                raise _JSONSyntaxError
            if character != "\\":
                decoded_bytes += len(character.encode("utf-8", errors="strict"))
            else:
                if self._index >= len(text):
                    raise _JSONSyntaxError
                escape = text[self._index]
                self._index += 1
                if escape in '"\\/bfnrt':
                    decoded_bytes += 1
                elif escape == "u":
                    codepoint = self._unicode_escape()
                    if 0xD800 <= codepoint <= 0xDBFF:
                        if (
                            self._index + 6 > len(text)
                            or text[self._index : self._index + 2] != "\\u"
                        ):
                            raise _JSONSyntaxError
                        self._index += 2
                        low = self._unicode_escape()
                        if not 0xDC00 <= low <= 0xDFFF:
                            raise _JSONSyntaxError
                        decoded_bytes += 4
                    elif 0xDC00 <= codepoint <= 0xDFFF:
                        raise _JSONSyntaxError
                    else:
                        decoded_bytes += len(chr(codepoint).encode("utf-8"))
                else:
                    raise _JSONSyntaxError
            if decoded_bytes > _MAX_STRING_BYTES:
                raise _JSONLimitError
        raise _JSONSyntaxError

    def _unicode_escape(self) -> int:
        end = self._index + 4
        if end > len(self._text):
            raise _JSONSyntaxError
        token = self._text[self._index : end]
        if any(character not in "0123456789abcdefABCDEF" for character in token):
            raise _JSONSyntaxError
        self._index = end
        return int(token, 16)

    def _parse_number(self) -> None:
        text = self._text
        if self._peek("-"):
            self._index += 1
        if self._index >= len(text):
            raise _JSONSyntaxError
        if self._peek("0"):
            self._index += 1
            if self._index < len(text) and text[self._index].isdigit():
                raise _JSONSyntaxError
        elif "1" <= text[self._index] <= "9":
            while self._index < len(text) and text[self._index].isdigit():
                self._index += 1
        else:
            raise _JSONSyntaxError
        if self._peek("."):
            self._index += 1
            start = self._index
            while self._index < len(text) and text[self._index].isdigit():
                self._index += 1
            if self._index == start:
                raise _JSONSyntaxError
        if self._index < len(text) and text[self._index] in "eE":
            self._index += 1
            if self._index < len(text) and text[self._index] in "+-":
                self._index += 1
            start = self._index
            while self._index < len(text) and text[self._index].isdigit():
                self._index += 1
            if self._index == start:
                raise _JSONSyntaxError

    def _consume_literal(self, value: str) -> None:
        if not self._text.startswith(value, self._index):
            raise _JSONSyntaxError
        self._index += len(value)

    def _peek(self, value: str) -> bool:
        return self._index < len(self._text) and self._text[self._index] == value


def _duplicate_safe_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    if len(pairs) > _MAX_OBJECT_MEMBERS:
        raise _JSONLimitError
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _JSONDuplicateKeyError
        result[key] = value
    return result


def _integer_token(value: str) -> _JSONNumber:
    return _JSONNumber(value, True)


def _float_token(value: str) -> _JSONNumber:
    return _JSONNumber(value, False)


def _reject_constant(_value: str) -> object:
    raise _JSONSyntaxError


def _parse_source(source: bytes) -> dict[str, object]:
    if type(source) is not bytes:
        raise ShadowTraceInputError("invalid_json")
    if not 1 <= len(source) <= MAX_SHADOW_TRACE_BYTES:
        raise ShadowTraceInputError("input_limit_exceeded")
    if source.startswith(b"\xef\xbb\xbf"):
        raise ShadowTraceInputError("invalid_json")
    try:
        text = source.decode("utf-8", errors="strict")
        _JSONPreflight(text).run()
        parsed = json.loads(
            text,
            object_pairs_hook=_duplicate_safe_object,
            parse_int=_integer_token,
            parse_float=_float_token,
            parse_constant=_reject_constant,
        )
    except _JSONLimitError as error:
        raise ShadowTraceInputError("input_limit_exceeded") from error
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        _JSONSyntaxError,
        _JSONDuplicateKeyError,
    ) as error:
        raise ShadowTraceInputError("invalid_json") from error
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise ShadowTraceInputError("invalid_json") from error
    if type(parsed) is not dict:
        raise ShadowTraceInputError("invalid_json")
    return cast(dict[str, object], parsed)


def _exact_text(value: object, *, max_bytes: int, allow_empty: bool = False) -> str:
    if type(value) is not str:
        raise ValueError
    encoded = value.encode("utf-8", errors="strict")
    if (not encoded and not allow_empty) or len(encoded) > max_bytes or "\x00" in value:
        raise ValueError
    return value


def _exact_integer(value: object) -> int | None:
    if type(value) is not _JSONNumber or not value.integer:
        return None
    assert isinstance(value, _JSONNumber)
    if len(value.lexeme) > 20:
        return None
    try:
        return int(value.lexeme)
    except ValueError:
        return None


def _required_step_integer(value: object, *, ordinal: int) -> int:
    parsed = _exact_integer(value)
    if parsed != ordinal or parsed < 1:
        raise ShadowTraceInputError("invalid_step", step_ordinal=ordinal)
    return parsed


def _validate_duration(value: object, *, step: int, call: int) -> None:
    if type(value) is not _JSONNumber:
        raise ShadowTraceInputError("invalid_tool_call", step_ordinal=step, call_ordinal=call)
    assert isinstance(value, _JSONNumber)
    if len(value.lexeme) > _MAX_TERMINUS_DURATION_LEXEME_CHARS:
        raise ShadowTraceInputError("invalid_tool_call", step_ordinal=step, call_ordinal=call)
    try:
        duration = Decimal(value.lexeme)
    except InvalidOperation as error:
        raise ShadowTraceInputError(
            "invalid_tool_call", step_ordinal=step, call_ordinal=call
        ) from error
    if not duration.is_finite() or duration < 0 or duration > _MAX_TERMINUS_DURATION_SECONDS:
        raise ShadowTraceInputError("invalid_tool_call", step_ordinal=step, call_ordinal=call)


def _normalize_timestamp(value: object, *, step: int) -> str:
    if type(value) is not str:
        raise ShadowTraceInputError("invalid_timestamp", step_ordinal=step)
    match = _UTC_TIMESTAMP_PATTERN.fullmatch(value)
    if match is None:
        raise ShadowTraceInputError("invalid_timestamp", step_ordinal=step)
    fraction = match.group("fraction")
    normalized = match.group("whole")
    if fraction is not None:
        normalized = f"{normalized}.{fraction.ljust(6, '0')}"
    normalized = f"{normalized}Z"
    try:
        parsed = datetime.fromisoformat(f"{normalized[:-1]}+00:00").astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise ShadowTraceInputError("invalid_timestamp", step_ordinal=step) from error
    canonical = parsed.isoformat(
        timespec="microseconds" if parsed.microsecond else "seconds"
    ).replace("+00:00", "Z")
    explicit_zero_fraction = (
        parsed.microsecond == 0
        and normalized.endswith(".000000Z")
        and f"{normalized[:-8]}Z" == canonical
    )
    if canonical != normalized and not explicit_zero_fraction:
        raise ShadowTraceInputError("invalid_timestamp", step_ordinal=step)
    return normalized


def _timestamp_value(value: str) -> datetime:
    return datetime.fromisoformat(f"{value[:-1]}+00:00").astimezone(UTC)


def _context_digest(*, domain: str, body: dict[str, object]) -> str:
    return length_prefixed_sha256(canonical_json(body), domain=domain)


def _terminus_environment_digest(environment_digest: str) -> str:
    return _context_digest(
        domain=_TERMINUS_CONTEXT_DOMAIN,
        body={
            "schema_version": "terminus-execution-context/v1",
            "caller_environment_digest": environment_digest,
            "terminal_transport": "tmux_send_keys",
            "execution_delimiter": "line_feed",
        },
    )


def _codex_environment_digest(environment_digest: str, arguments: dict[str, object]) -> str:
    semantic: dict[str, object] = {
        "schema_version": "codex-execution-context/v1",
        "caller_environment_digest": environment_digest,
    }
    for key in _CODEX_EXECUTION_SEMANTIC_KEYS:
        if key in arguments:
            semantic[key] = arguments[key]
    return _context_digest(domain=_CODEX_CONTEXT_DOMAIN, body=semantic)


def _codex_execution_semantics(
    arguments: dict[str, object],
) -> tuple[tuple[str, object], ...]:
    return tuple(
        (key, arguments[key]) for key in _CODEX_EXECUTION_SEMANTIC_KEYS if key in arguments
    )


def _profile_descriptor(
    *,
    profile: ATIFProfile,
    schemas: tuple[str, ...],
    agent_name: str,
    selected_function: str,
    outcome_authority: Literal["none", "producer_claimed_structured"],
) -> dict[str, object]:
    common: dict[str, object] = {
        "schema_version": "atif-shadow-adapter-profile/v1",
        "profile_id": profile.value,
        "accepted_source_schema_versions": list(schemas),
        "required_agent_name": agent_name,
        "source_admission": {
            "source_type": "exact_bytes",
            "source_bytes_min": 1,
            "strict_utf8": True,
            "byte_order_mark": "error",
            "root_type": "exact_object",
            "schema_version_type": "exact_string",
            "session_id": {
                "admission": "absent_null_or_exact_string",
                "utf8_bytes_max": _MAX_SESSION_ID_BYTES,
            },
            "agent": {
                "type": "exact_object",
                "name_type": "exact_string",
            },
            "steps": {
                "type": "exact_array",
                "items_min": 1,
                "step_type": "exact_object",
                "step_id": "exact_positive_integer_equal_to_one_based_ordinal",
                "tool_calls": "absent_null_or_exact_array",
                "observation": "absent_null_or_exact_object",
                "observation_results": "absent_null_or_exact_array",
                "is_copied_context": "absent_null_or_exact_boolean",
                "tool_call_items": "exact_objects",
                "observation_result_items": "exact_objects",
            },
        },
        "selection": {
            "scope": "root_agent_steps_only",
            "candidate_source": "exact_agent",
            "selected_function_names": [selected_function],
            "selected_tool_call_id": "nonempty_exact_string_unique_within_step",
            "unsupported_tool_call_id": "absent_null_or_exact_string_including_empty",
            "nonempty_tool_call_id": "unique_within_step_for_all_non_copied_functions",
            "non_copied_function_name": "required_exact_string",
            "capture_scope": "selected_events",
            "copied_context": "count_and_ignore",
            "copied_container_elements": "exact_objects",
            "continued_trajectory_ref": "non_null_presence_only_no_traversal",
            "subagent_trajectories": "immediate_count_only_no_traversal",
        },
        "result_linking": {
            "keyed": "exact_same_step_unique_tool_call_id",
            "unkeyed": "exactly_one_total_call_and_one_total_result",
            "zero_calls": "ignored_no_parent",
            "otherwise": "ignored_ambiguous_parent",
            "orphan_nonempty_id": "error",
            "null_parent": "absent",
            "empty_or_non_string_parent": "error",
            "linked_unmapped_call": "ignored_unsupported_parent",
            "copied_context_precedes_linking": True,
            "raw_content": "always_discarded",
            "mapped_structured_outcomes_per_action_max": 1,
        },
        "identifiers": {
            "start": "atif-run-start",
            "action": "atif-s{step:08d}-c{call:04d}-action",
            "result": "atif-s{step:08d}-c{call:04d}-result",
            "end": "atif-run-end",
            "raw_identifiers_persisted": False,
            "coordinate_bounds": {
                "step": [1, _MAX_STEPS],
                "call": [1, _MAX_STEP_TOOL_CALLS],
            },
            "mapped_action_coordinate_order": "strictly_increasing_with_gaps_allowed",
        },
        "timestamps": {
            "source_mode": "every_mapped_action_step_has_strict_utc_timestamp",
            "logical_mode": "no_mapped_action_step_has_timestamp",
            "partial": "error",
            "accepted_source_pattern": (
                "YYYY-MM-DDTHH:MM:SS_with_optional_1_to_6_fractional_digits_"
                "and_exact_Z_or_plus_00_00_suffix"
            ),
            "utc_offset_normalization": "exact_plus_00_00_to_Z",
            "source_fraction_normalization": "right_pad_to_six_digits",
            "selected_source_order": "nondecreasing",
            "source_start": "first_mapped_action_timestamp",
            "source_end": "last_mapped_action_or_result_timestamp",
            "logical_epoch": "2000-01-01T00:00:00.000000Z",
            "logical_increment_microseconds": 1,
            "logical_ticks": "every_mapped_record_including_markers",
            "unselected_timestamps": "ignored",
        },
        "parser": {
            "source_bytes_max": MAX_SHADOW_TRACE_BYTES,
            "json_depth_max": _MAX_JSON_DEPTH,
            "json_nodes_max": _MAX_JSON_NODES,
            "object_members_max": _MAX_OBJECT_MEMBERS,
            "unconsumed_array_items_max": _MAX_ARRAY_ITEMS,
            "string_utf8_bytes_max": _MAX_STRING_BYTES,
            "steps_max": _MAX_STEPS,
            "tool_calls_per_step_max": _MAX_STEP_TOOL_CALLS,
            "results_per_step_max": _MAX_STEP_RESULTS,
            "total_tool_calls_max": _MAX_TOTAL_TOOL_CALLS,
            "total_results_max": _MAX_TOTAL_RESULTS,
            "mapped_shadow_rows_max": MAX_SHADOW_TRACE_ROWS,
            "mapped_canonical_bytes_max": MAX_SHADOW_TRACE_BYTES,
            "command_utf8_bytes_max": _MAX_COMMAND_BYTES,
            "working_directory_utf8_bytes_max": _MAX_DIRECTORY_BYTES,
            "session_id_utf8_bytes_max": _MAX_SESSION_ID_BYTES,
            "context_text_utf8_bytes_max": _MAX_CONTEXT_TEXT_BYTES,
            "prefix_rule_items_max": _MAX_PREFIX_RULE_ITEMS,
            "codex_nonnegative_integer_argument_max": _MAX_CODEX_INTEGER_ARGUMENT,
            "unconsumed_json_decimal_lexeme": "grammar_valid_preserved_without_binary64_range",
            "duplicate_keys": "error",
            "non_finite_numbers": "error",
            "unicode_normalization": "none",
            "nul_policy": {
                "rejected_fields": [
                    "session_id",
                    "mapped_command",
                    "working_directory",
                    "codex_shell",
                    "codex_sandbox_permissions",
                    "codex_justification",
                    "codex_prefix_rule_item",
                ],
                "identifier_and_ignored_text": "preserved_only_for_linking_or_discarded",
            },
        },
        "adapter_configuration": {
            "schema_version": "atif-shadow-adapter-configuration/v1",
            "exact_fields": [
                "schema_version",
                "adapter_profile_id",
                "adapter_profile_digest",
                "compatibility_evidence_manifest_digest",
                "source_format",
                "source_schema_version",
                "timestamp_mode",
                "capture_scope",
                "selection_scope",
                "environment",
                "mapped_action_contexts",
            ],
            "selection_scope": "root_segment_selected_events",
            "environment_schema_version": "shadow-environment-binding/v1",
            "digest_domain": _CONFIGURATION_DIGEST_DOMAIN,
            "validation": "canonical_exact_fields_and_profile_constants",
            "mapped_action_contexts": {
                "order": "mapped_action_record_order",
                "fields": [
                    "source_event_id",
                    "working_directory",
                    "execution_semantics",
                    "environment_digest",
                ],
                "commands_persisted": False,
                "raw_source_identifiers_persisted": False,
                "validation": (
                    "cross_link_working_directory_execution_semantics_environment_digest_"
                    "and_mapped_action_record"
                ),
            },
        },
        "diagnostics": {
            "tool_call_disposition_order": list(_TOOL_DISPOSITIONS),
            "result_disposition_order": list(_RESULT_DISPOSITIONS),
            "mapped_record_equation": ("2_plus_mapped_action_plus_mapped_structured_outcome"),
            "record_cross_links": (
                "mapped_action_and_mapped_structured_outcome_counts_equal_record_kinds"
            ),
            "coordinate_diagnostic_cross_links": (
                "step_and_call_bounds_total_steps_total_calls_and_ignored_message_inequality"
            ),
            "ignored_message_step_count": "root_steps_with_zero_tool_calls",
            "root_segment_only": True,
            "complete_execution_session_coverage": False,
        },
        "ignored_data_classes": [
            "messages",
            "reasoning",
            "model_names",
            "metrics",
            "final_metrics",
            "final_answers",
            "artifacts",
            "raw_result_content",
        ],
        "producer_authentication": "none",
        "outcome_evidence_authority": outcome_authority,
        "compatibility_evidence_manifest_digest": _EXPECTED_MANIFEST_DIGEST,
    }
    if profile is ATIFProfile.HARBOR_TERMINUS_2_V1:
        common["mapping"] = {
            "compatibility_claim": "pinned_harbor_converter_and_public_golden_shapes_only",
            "paper_provenance_role": (
                "research_inspiration_not_direct_fixture_or_runtime_compatibility"
            ),
            "arguments": {
                "keystrokes": "required_exact_string",
                "duration": f"optional_finite_number_0_to_{_MAX_TERMINUS_DURATION_SECONDS}",
                "duration_decimal_lexeme_chars_max": (_MAX_TERMINUS_DURATION_LEXEME_CHARS),
                "duration_null": "error",
            },
            "submission": {
                "transport": "tmux_send_keys",
                "delimiter": "one_terminal_line_feed",
                "empty": "ignored_non_command_wait",
                "unterminated": "ignored_unsubmitted_keystrokes",
                "delimiter_only_or_space_tab": "ignored_unresolved_terminal_submission",
            },
            "working_directory": "caller_attested_default",
            "environment_context": [
                "caller_environment_digest",
                "terminal_transport",
                "execution_delimiter",
            ],
            "environment_context_domain": _TERMINUS_CONTEXT_DOMAIN,
            "environment_context_schema_version": "terminus-execution-context/v1",
            "structured_outcome_paths": [],
            "linked_result": "ignored_evidence_absent",
            "impossible_tool_dispositions": ["ignored_continuation"],
            "impossible_result_dispositions": ["mapped_structured_outcome"],
        }
    else:
        common["mapping"] = {
            "compatibility_claim": (
                "pinned_harbor_converter_field_shape_only_no_codex_cli_version_guarantee"
            ),
            "arguments": {
                "allowed_keys": [
                    "cmd",
                    "workdir",
                    "shell",
                    "login",
                    "tty",
                    "sandbox_permissions",
                    "yield_time_ms",
                    "max_output_tokens",
                    "justification",
                    "prefix_rule",
                ],
                "cmd": "required_nonempty_exact_string",
                "workdir": "optional_nonempty_exact_string",
                "unknown_key": "error",
                "shell": f"optional_nonempty_exact_string_max_{_MAX_CONTEXT_TEXT_BYTES}_bytes",
                "login": "optional_exact_boolean",
                "tty": "optional_exact_boolean",
                "sandbox_permissions": (
                    f"optional_nonempty_exact_string_max_{_MAX_CONTEXT_TEXT_BYTES}_bytes"
                ),
                "yield_time_ms": f"optional_exact_integer_0_to_{_MAX_CODEX_INTEGER_ARGUMENT}",
                "max_output_tokens": (f"optional_exact_integer_0_to_{_MAX_CODEX_INTEGER_ARGUMENT}"),
                "justification": f"optional_exact_string_max_{_MAX_STRING_BYTES}_bytes",
                "prefix_rule": {
                    "type": "optional_exact_array",
                    "items_max": _MAX_PREFIX_RULE_ITEMS,
                    "item": (f"nonempty_exact_string_max_{_MAX_CONTEXT_TEXT_BYTES}_bytes"),
                },
            },
            "working_directory": "workdir_or_caller_attested_default",
            "environment_context": [
                "caller_environment_digest",
                "shell",
                "login",
                "tty",
                "sandbox_permissions",
            ],
            "environment_context_domain": _CODEX_CONTEXT_DOMAIN,
            "environment_context_schema_version": "codex-execution-context/v1",
            "continuation_function": "write_stdin",
            "structured_outcome_paths": [
                "step.extra.tool_metadata.exit_code",
                "step.extra.tool_call_details[tool_call_id].metadata.exit_code",
            ],
            "single_tool_metadata_path_condition": "exactly_one_total_call_in_step",
            "structured_outcome_type": "exact_signed_int32_non_boolean",
            "structured_outcome_range": [_MIN_EXIT_STATUS, _MAX_EXIT_STATUS],
            "malformed_single_path": "ignored_evidence_absent",
            "multiple_recognized_paths": "require_same_admissible_exact_int_or_error",
            "structured_outcome_status": "exit_code_zero_succeeded_nonzero_failed",
            "tool_call_status": "ignored",
            "impossible_tool_dispositions": [
                "ignored_non_command_wait",
                "ignored_unsubmitted_keystrokes",
                "ignored_unresolved_terminal_submission",
            ],
        }
    return common


def _make_contract(
    profile: ATIFProfile,
    schemas: tuple[str, ...],
    agent_name: str,
    selected_function: str,
    authority: Literal["none", "producer_claimed_structured"],
) -> _ATIFProfileContract:
    descriptor = _profile_descriptor(
        profile=profile,
        schemas=schemas,
        agent_name=agent_name,
        selected_function=selected_function,
        outcome_authority=authority,
    )
    descriptor_bytes = _copy_descriptor(descriptor)
    digest = length_prefixed_sha256(descriptor_bytes, domain=_PROFILE_DIGEST_DOMAIN)
    if not hmac.compare_digest(digest, _EXPECTED_PROFILE_DIGESTS[profile]):
        raise RuntimeError("ATIF profile descriptor seal does not match")
    frozen = _freeze_json(descriptor)
    if type(frozen) is not MappingProxyType:
        raise RuntimeError("ATIF profile descriptor is invalid")
    return _ATIFProfileContract(
        profile=profile,
        accepted_schema_versions=schemas,
        required_agent_name=agent_name,
        selected_function_name=selected_function,
        outcome_evidence_authority=authority,
        descriptor=cast(MappingProxyType[str, object], frozen),
        descriptor_bytes=descriptor_bytes,
        profile_digest=digest,
    )


_PROFILE_CONTRACTS: Final[MappingProxyType[ATIFProfile, _ATIFProfileContract]] = MappingProxyType(
    {
        ATIFProfile.HARBOR_TERMINUS_2_V1: _make_contract(
            ATIFProfile.HARBOR_TERMINUS_2_V1,
            ("ATIF-v1.6", "ATIF-v1.7"),
            "terminus-2",
            "bash_command",
            "none",
        ),
        ATIFProfile.HARBOR_CODEX_V1: _make_contract(
            ATIFProfile.HARBOR_CODEX_V1,
            ("ATIF-v1.7",),
            "codex",
            "exec_command",
            "producer_claimed_structured",
        ),
    }
)


def _load_manifest_bytes() -> bytes:
    failed = False
    try:
        payload = resources.files("saliencegate.shadow").joinpath(_MANIFEST_RESOURCE).read_bytes()
        decoded = json.loads(payload)
        canonical = canonical_json(decoded)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        failed = True
        payload = b""
        canonical = b""
    if failed:
        raise ShadowInvariantError
    if (
        payload != canonical
        or not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), _EXPECTED_MANIFEST_SHA256)
        or not hmac.compare_digest(
            length_prefixed_sha256(payload, domain=_MANIFEST_DIGEST_DOMAIN),
            _EXPECTED_MANIFEST_DIGEST,
        )
    ):
        raise ShadowInvariantError
    return payload


def _sealed_report_contract(
    profile_id: str,
) -> tuple[str, str, Literal["none", "producer_claimed_structured"]] | None:
    if type(profile_id) is not str:
        return None
    _load_manifest_bytes()
    for contract in _PROFILE_CONTRACTS.values():
        if contract.profile.value == profile_id:
            return (
                contract.profile_digest,
                _EXPECTED_MANIFEST_DIGEST,
                contract.outcome_evidence_authority,
            )
    return None


def _matches_sealed_report_claims(
    *,
    profile_id: str,
    source_schema_version: str,
    timestamp_mode: str,
    capture_scope: str,
    diagnostics: ATIFShadowDiagnostics,
) -> bool:
    contract = next(
        (item for item in _PROFILE_CONTRACTS.values() if item.profile.value == profile_id),
        None,
    )
    if (
        contract is None
        or type(diagnostics) is not ATIFShadowDiagnostics
        or type(source_schema_version) is not str
        or source_schema_version not in contract.accepted_schema_versions
        or timestamp_mode not in ("logical_order", "source_utc")
        or capture_scope != "selected_events"
    ):
        return False
    tool_counts = dict(diagnostics.tool_call_disposition_counts)
    result_counts = dict(diagnostics.result_disposition_counts)
    if result_counts["mapped_structured_outcome"] > tool_counts["mapped_action"]:
        return False
    if contract.profile is ATIFProfile.HARBOR_TERMINUS_2_V1:
        return (
            tool_counts["ignored_continuation"] == 0
            and result_counts["mapped_structured_outcome"] == 0
        )
    return (
        tool_counts["ignored_non_command_wait"] == 0
        and tool_counts["ignored_unsubmitted_keystrokes"] == 0
        and tool_counts["ignored_unresolved_terminal_submission"] == 0
    )


def _matches_sealed_trace_contract(
    *,
    profile_id: str,
    descriptor_bytes: bytes,
    profile_digest: str,
    manifest_digest: str,
    outcome_authority: str,
) -> bool:
    contract_tuple = _sealed_report_contract(profile_id)
    if contract_tuple is None:
        return False
    contract = next(
        item for item in _PROFILE_CONTRACTS.values() if item.profile.value == profile_id
    )
    expected_digest, expected_manifest, expected_authority = contract_tuple
    return (
        type(descriptor_bytes) is bytes
        and hmac.compare_digest(descriptor_bytes, contract.descriptor_bytes)
        and hmac.compare_digest(profile_digest, expected_digest)
        and hmac.compare_digest(manifest_digest, expected_manifest)
        and outcome_authority == expected_authority
    )


def _configuration_object(configuration_bytes: bytes) -> dict[str, object] | None:
    if (
        type(configuration_bytes) is not bytes
        or not 1 <= len(configuration_bytes) <= MAX_SHADOW_TRACE_BYTES
    ):
        return None
    try:
        text = configuration_bytes.decode("utf-8", errors="strict")
        _JSONPreflight(text).run()
        parsed = json.loads(
            text,
            object_pairs_hook=_duplicate_safe_object,
            parse_int=_integer_token,
            parse_float=_float_token,
            parse_constant=_reject_constant,
        )
        if type(parsed) is not dict or canonical_json(parsed) != configuration_bytes:
            return None
        return cast(dict[str, object], parsed)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return None


def _context_matches_action(
    *,
    contract: _ATIFProfileContract,
    environment: ShadowEnvironmentBinding,
    context: object,
    action_record: object,
) -> bool:
    if type(context) is not dict or type(action_record) is not MappingProxyType:
        return False
    context = cast(dict[str, object], context)
    action = cast(MappingProxyType[str, object], action_record)
    if set(context) != {
        "source_event_id",
        "working_directory",
        "execution_semantics",
        "environment_digest",
    }:
        return False
    if set(action) != {
        "schema_version",
        "kind",
        "source_event_id",
        "occurred_at",
        "command",
        "working_directory",
        "environment_digest",
    }:
        return False
    source_event_id = context["source_event_id"]
    working_directory = context["working_directory"]
    environment_digest = context["environment_digest"]
    semantics = context["execution_semantics"]
    if (
        type(source_event_id) is not str
        or _ATIF_ACTION_ID_PATTERN.fullmatch(source_event_id) is None
        or action["kind"] != "action"
        or action["source_event_id"] != source_event_id
        or action["working_directory"] != working_directory
        or action["environment_digest"] != environment_digest
        or type(working_directory) is not str
        or type(environment_digest) is not str
        or _DIGEST_PATTERN.fullmatch(environment_digest) is None
        or type(semantics) is not dict
    ):
        return False
    semantics = cast(dict[str, object], semantics)
    if contract.profile is ATIFProfile.HARBOR_TERMINUS_2_V1:
        if semantics != dict(_TERMINUS_EXECUTION_SEMANTICS):
            return False
        if working_directory != environment.default_working_directory:
            return False
        expected_digest = _terminus_environment_digest(environment.environment_digest)
    else:
        if not set(semantics).issubset(_CODEX_EXECUTION_SEMANTIC_KEYS):
            return False
        try:
            for key in ("shell", "sandbox_permissions"):
                if key in semantics:
                    _exact_text(semantics[key], max_bytes=_MAX_CONTEXT_TEXT_BYTES)
            for key in ("login", "tty"):
                if key in semantics and type(semantics[key]) is not bool:
                    return False
        except (UnicodeEncodeError, ValueError):
            return False
        expected_digest = _codex_environment_digest(
            environment.environment_digest,
            semantics,
        )
    return hmac.compare_digest(environment_digest, expected_digest)


def _records_match_atif_topology(
    *,
    contract: _ATIFProfileContract,
    environment: ShadowEnvironmentBinding,
    contexts: list[object],
    records: tuple[object, ...],
    timestamp_mode: str,
    diagnostics: ATIFShadowDiagnostics,
) -> bool:
    if len(records) < 3:
        return False
    start = records[0]
    end = records[-1]
    if type(start) is not MappingProxyType or type(end) is not MappingProxyType:
        return False
    if set(start) != {"schema_version", "kind", "source_event_id", "occurred_at"} or set(end) != {
        "schema_version",
        "kind",
        "source_event_id",
        "occurred_at",
    }:
        return False
    if (
        start["kind"] != "run_start"
        or start["source_event_id"] != "atif-run-start"
        or end["kind"] != "run_end"
        or end["source_event_id"] != "atif-run-end"
    ):
        return False

    action_records: list[object] = []
    current_action: MappingProxyType[str, object] | None = None
    current_result_seen = False
    for record in records[1:-1]:
        if type(record) is not MappingProxyType:
            return False
        kind = record.get("kind")
        if kind == "action":
            action_records.append(record)
            current_action = cast(MappingProxyType[str, object], record)
            current_result_seen = False
            continue
        if kind != "tool_result" or current_action is None or current_result_seen:
            return False
        if set(record) != {
            "schema_version",
            "kind",
            "source_event_id",
            "occurred_at",
            "action_source_event_id",
            "status",
            "exit_status",
        }:
            return False
        exit_status = record["exit_status"]
        action_source_event_id = current_action["source_event_id"]
        if (
            contract.profile is not ATIFProfile.HARBOR_CODEX_V1
            or type(exit_status) is not int
            or type(exit_status) is bool
            or not _MIN_EXIT_STATUS <= exit_status <= _MAX_EXIT_STATUS
            or record["action_source_event_id"] != action_source_event_id
            or record["source_event_id"] != f"{str(action_source_event_id)[:-6]}result"
            or (
                timestamp_mode == "source_utc"
                and record["occurred_at"] != current_action["occurred_at"]
            )
            or record["status"] != ("succeeded" if exit_status == 0 else "failed")
        ):
            return False
        current_result_seen = True

    if len(contexts) != len(action_records) or not action_records:
        return False
    previous_coordinate = (0, 0)
    mapped_step_ordinals: set[int] = set()
    for action in action_records:
        source_event_id = cast(str, cast(MappingProxyType[str, object], action)["source_event_id"])
        match = _ATIF_ACTION_ID_PATTERN.fullmatch(source_event_id)
        if match is None:
            return False
        coordinate = (int(match.group("step")), int(match.group("call")))
        if (
            not 1 <= coordinate[0] <= _MAX_STEPS
            or not 1 <= coordinate[1] <= _MAX_STEP_TOOL_CALLS
            or coordinate <= previous_coordinate
            or coordinate[0] > diagnostics.total_step_count
            or coordinate[1] > diagnostics.total_tool_call_count
        ):
            return False
        previous_coordinate = coordinate
        mapped_step_ordinals.add(coordinate[0])
    if (
        len(mapped_step_ordinals) + diagnostics.ignored_message_step_count
        > diagnostics.total_step_count
    ):
        return False
    if not all(
        _context_matches_action(
            contract=contract,
            environment=environment,
            context=context,
            action_record=action,
        )
        for context, action in zip(contexts, action_records, strict=True)
    ):
        return False

    frozen_records = cast(tuple[MappingProxyType[str, object], ...], records)
    timestamp_values = tuple(record["occurred_at"] for record in frozen_records)
    if any(type(value) is not str for value in timestamp_values):
        return False
    timestamps = cast(tuple[str, ...], timestamp_values)
    if timestamp_mode == "logical_order":
        return timestamps == tuple(_logical_timestamp(index) for index in range(len(records)))
    if timestamp_mode != "source_utc":
        return False
    try:
        if any(_normalize_timestamp(value, step=1) != value for value in timestamps):
            return False
    except ShadowTraceInputError:
        return False
    action_timestamps = tuple(
        cast(str, cast(MappingProxyType[str, object], action)["occurred_at"])
        for action in action_records
    )
    if any(
        _timestamp_value(action_timestamps[index]) < _timestamp_value(action_timestamps[index - 1])
        for index in range(1, len(action_timestamps))
    ):
        return False
    return timestamps[0] == action_timestamps[0] and timestamps[-1] == timestamps[-2]


def _matches_sealed_configuration_contract(
    *,
    profile_id: str,
    profile_digest: str,
    configuration_bytes: bytes,
    configuration_digest: str,
    source_schema_version: str,
    timestamp_mode: str,
    capture_scope: str,
    records: tuple[object, ...],
    diagnostics: ATIFShadowDiagnostics,
) -> bool:
    parsed = _configuration_object(configuration_bytes)
    contract = next(
        (item for item in _PROFILE_CONTRACTS.values() if item.profile.value == profile_id),
        None,
    )
    if (
        parsed is None
        or contract is None
        or type(records) is not tuple
        or type(diagnostics) is not ATIFShadowDiagnostics
    ):
        return False
    if set(parsed) != {
        "schema_version",
        "adapter_profile_id",
        "adapter_profile_digest",
        "compatibility_evidence_manifest_digest",
        "source_format",
        "source_schema_version",
        "timestamp_mode",
        "capture_scope",
        "selection_scope",
        "environment",
        "mapped_action_contexts",
    }:
        return False
    if (
        parsed["schema_version"] != "atif-shadow-adapter-configuration/v1"
        or parsed["adapter_profile_id"] != profile_id
        or parsed["adapter_profile_digest"] != profile_digest
        or parsed["adapter_profile_digest"] != contract.profile_digest
        or parsed["compatibility_evidence_manifest_digest"] != _EXPECTED_MANIFEST_DIGEST
        or parsed["source_format"] != "atif"
        or parsed["source_schema_version"] != source_schema_version
        or source_schema_version not in contract.accepted_schema_versions
        or parsed["timestamp_mode"] != timestamp_mode
        or parsed["capture_scope"] != capture_scope
        or capture_scope != "selected_events"
        or parsed["selection_scope"] != "root_segment_selected_events"
        or type(parsed["environment"]) is not dict
        or type(parsed["mapped_action_contexts"]) is not list
        or not hmac.compare_digest(
            length_prefixed_sha256(
                configuration_bytes,
                domain=_CONFIGURATION_DIGEST_DOMAIN,
            ),
            configuration_digest,
        )
    ):
        return False
    try:
        environment = ShadowEnvironmentBinding.model_validate(parsed["environment"])
    except Exception:
        return False
    if not _records_match_atif_topology(
        contract=contract,
        environment=environment,
        contexts=cast(list[object], parsed["mapped_action_contexts"]),
        records=records,
        timestamp_mode=timestamp_mode,
        diagnostics=diagnostics,
    ):
        return False
    frozen_records = cast(tuple[MappingProxyType[str, object], ...], records)
    action_count = sum(record.get("kind") == "action" for record in frozen_records)
    result_count = sum(record.get("kind") == "tool_result" for record in frozen_records)
    return (
        _matches_sealed_report_claims(
            profile_id=profile_id,
            source_schema_version=source_schema_version,
            timestamp_mode=timestamp_mode,
            capture_scope=capture_scope,
            diagnostics=diagnostics,
        )
        and dict(diagnostics.tool_call_disposition_counts)["mapped_action"] == action_count
        and dict(diagnostics.result_disposition_counts)["mapped_structured_outcome"] == result_count
    )


def _tool_calls(step: dict[str, object], *, ordinal: int) -> list[object]:
    value = step.get("tool_calls")
    if value is None:
        return []
    if type(value) is not list:
        raise ShadowTraceInputError("invalid_step", step_ordinal=ordinal)
    if len(value) > _MAX_STEP_TOOL_CALLS:
        raise ShadowTraceInputError("input_limit_exceeded", step_ordinal=ordinal)
    return value


def _results(step: dict[str, object], *, ordinal: int) -> list[object]:
    observation = step.get("observation")
    if observation is None:
        return []
    if type(observation) is not dict:
        raise ShadowTraceInputError("invalid_step", step_ordinal=ordinal)
    value = observation.get("results")
    if value is None:
        return []
    if type(value) is not list:
        raise ShadowTraceInputError("invalid_step", step_ordinal=ordinal)
    if len(value) > _MAX_STEP_RESULTS:
        raise ShadowTraceInputError("input_limit_exceeded", step_ordinal=ordinal)
    return value


def _classify_terminus_call(
    call: dict[str, object],
    *,
    step_ordinal: int,
    call_ordinal: int,
    tool_call_id: str | None,
    timestamp: str | None,
    environment: ShadowEnvironmentBinding,
) -> _CallState:
    function_name = call.get("function_name")
    if type(function_name) is not str:
        raise ShadowTraceInputError(
            "invalid_tool_call", step_ordinal=step_ordinal, call_ordinal=call_ordinal
        )
    if function_name != "bash_command":
        return _CallState(call_ordinal, tool_call_id, "ignored_unsupported_function", None)
    arguments = call.get("arguments")
    if type(arguments) is not dict:
        raise ShadowTraceInputError(
            "invalid_tool_call", step_ordinal=step_ordinal, call_ordinal=call_ordinal
        )
    if tool_call_id is None:
        raise ShadowTraceInputError(
            "invalid_tool_call", step_ordinal=step_ordinal, call_ordinal=call_ordinal
        )
    keystrokes = arguments.get("keystrokes")
    if type(keystrokes) is not str:
        raise ShadowTraceInputError(
            "invalid_tool_call", step_ordinal=step_ordinal, call_ordinal=call_ordinal
        )
    if "duration" in arguments:
        _validate_duration(arguments["duration"], step=step_ordinal, call=call_ordinal)
    if keystrokes == "":
        return _CallState(call_ordinal, tool_call_id, "ignored_non_command_wait", None)
    if not keystrokes.endswith("\n"):
        return _CallState(call_ordinal, tool_call_id, "ignored_unsubmitted_keystrokes", None)
    command = keystrokes[:-1]
    if command.strip(" \t") == "":
        return _CallState(
            call_ordinal,
            tool_call_id,
            "ignored_unresolved_terminal_submission",
            None,
        )
    try:
        command = _exact_text(command, max_bytes=_MAX_COMMAND_BYTES)
    except (UnicodeEncodeError, ValueError) as error:
        raise ShadowTraceInputError(
            "invalid_tool_call", step_ordinal=step_ordinal, call_ordinal=call_ordinal
        ) from error
    source_event_id = f"atif-s{step_ordinal:08d}-c{call_ordinal:04d}-action"
    action = _ActionPlan(
        step_ordinal=step_ordinal,
        call_ordinal=call_ordinal,
        source_event_id=source_event_id,
        command=command,
        working_directory=environment.default_working_directory,
        environment_digest=_terminus_environment_digest(environment.environment_digest),
        execution_semantics=_TERMINUS_EXECUTION_SEMANTICS,
        source_timestamp=timestamp,
        outcome_exit_statuses=(),
    )
    return _CallState(call_ordinal, tool_call_id, "mapped_action", action)


_CODEX_ARGUMENT_KEYS = frozenset(
    {
        "cmd",
        "workdir",
        "shell",
        "login",
        "tty",
        "sandbox_permissions",
        "yield_time_ms",
        "max_output_tokens",
        "justification",
        "prefix_rule",
    }
)


def _validate_codex_optional_arguments(
    arguments: dict[str, object], *, step: int, call: int
) -> None:
    try:
        for key in ("shell", "sandbox_permissions"):
            if key in arguments:
                _exact_text(arguments[key], max_bytes=_MAX_CONTEXT_TEXT_BYTES)
        for key in ("login", "tty"):
            if key in arguments and type(arguments[key]) is not bool:
                raise ValueError
        for key in ("yield_time_ms", "max_output_tokens"):
            if key in arguments:
                value = _exact_integer(arguments[key])
                if value is None or not 0 <= value <= _MAX_CODEX_INTEGER_ARGUMENT:
                    raise ValueError
        if "justification" in arguments:
            _exact_text(
                arguments["justification"],
                max_bytes=_MAX_STRING_BYTES,
                allow_empty=True,
            )
        if "prefix_rule" in arguments:
            prefix_rule = arguments["prefix_rule"]
            if type(prefix_rule) is not list or len(prefix_rule) > _MAX_PREFIX_RULE_ITEMS:
                raise ValueError
            for item in prefix_rule:
                _exact_text(item, max_bytes=_MAX_CONTEXT_TEXT_BYTES)
    except (UnicodeEncodeError, ValueError) as error:
        raise ShadowTraceInputError(
            "invalid_tool_call", step_ordinal=step, call_ordinal=call
        ) from error


def _classify_codex_call(
    call: dict[str, object],
    *,
    step_ordinal: int,
    call_ordinal: int,
    tool_call_id: str | None,
    timestamp: str | None,
    environment: ShadowEnvironmentBinding,
) -> _CallState:
    function_name = call.get("function_name")
    if type(function_name) is not str:
        raise ShadowTraceInputError(
            "invalid_tool_call", step_ordinal=step_ordinal, call_ordinal=call_ordinal
        )
    if function_name == "write_stdin":
        return _CallState(call_ordinal, tool_call_id, "ignored_continuation", None)
    if function_name != "exec_command":
        return _CallState(call_ordinal, tool_call_id, "ignored_unsupported_function", None)
    arguments = call.get("arguments")
    if type(arguments) is not dict or not set(arguments).issubset(_CODEX_ARGUMENT_KEYS):
        raise ShadowTraceInputError(
            "invalid_tool_call", step_ordinal=step_ordinal, call_ordinal=call_ordinal
        )
    if tool_call_id is None:
        raise ShadowTraceInputError(
            "invalid_tool_call", step_ordinal=step_ordinal, call_ordinal=call_ordinal
        )
    try:
        command = _exact_text(arguments.get("cmd"), max_bytes=_MAX_COMMAND_BYTES)
        working_directory = (
            _exact_text(arguments["workdir"], max_bytes=_MAX_DIRECTORY_BYTES)
            if "workdir" in arguments
            else environment.default_working_directory
        )
    except (UnicodeEncodeError, ValueError) as error:
        raise ShadowTraceInputError(
            "invalid_tool_call", step_ordinal=step_ordinal, call_ordinal=call_ordinal
        ) from error
    _validate_codex_optional_arguments(arguments, step=step_ordinal, call=call_ordinal)
    source_event_id = f"atif-s{step_ordinal:08d}-c{call_ordinal:04d}-action"
    action = _ActionPlan(
        step_ordinal=step_ordinal,
        call_ordinal=call_ordinal,
        source_event_id=source_event_id,
        command=command,
        working_directory=working_directory,
        environment_digest=_codex_environment_digest(environment.environment_digest, arguments),
        execution_semantics=_codex_execution_semantics(arguments),
        source_timestamp=timestamp,
        outcome_exit_statuses=(),
    )
    return _CallState(call_ordinal, tool_call_id, "mapped_action", action)


def _codex_exit_status(
    step: dict[str, object],
    *,
    tool_call_id: str,
    total_calls: int,
    step_ordinal: int,
    call_ordinal: int,
) -> int | None:
    extra = step.get("extra")
    if type(extra) is not dict:
        return None
    candidates: list[object] = []
    if total_calls == 1 and "tool_metadata" in extra:
        tool_metadata = extra["tool_metadata"]
        if type(tool_metadata) is dict and "exit_code" in tool_metadata:
            candidates.append(tool_metadata["exit_code"])
    if "tool_call_details" in extra:
        tool_call_details = extra["tool_call_details"]
        if type(tool_call_details) is dict and tool_call_id in tool_call_details:
            details = tool_call_details[tool_call_id]
            if type(details) is dict and "metadata" in details:
                metadata = details["metadata"]
                if type(metadata) is dict and "exit_code" in metadata:
                    candidates.append(metadata["exit_code"])
    if not candidates:
        return None
    converted = tuple(_exact_integer(item) for item in candidates)
    admissible = tuple(
        item if item is not None and _MIN_EXIT_STATUS <= item <= _MAX_EXIT_STATUS else None
        for item in converted
    )
    if len(candidates) > 1 and (None in admissible or len(set(admissible)) != 1):
        raise ShadowTraceInputError(
            "invalid_outcome_metadata",
            step_ordinal=step_ordinal,
            call_ordinal=call_ordinal,
        )
    return admissible[0]


def _plan_mapping(
    root: dict[str, object],
    *,
    contract: _ATIFProfileContract,
    environment: ShadowEnvironmentBinding,
) -> tuple[_MappingPlan, str]:
    schema = root.get("schema_version")
    if type(schema) is not str or schema not in contract.accepted_schema_versions:
        raise ShadowTraceInputError("unsupported_schema")
    session_id = root.get("session_id")
    if session_id is not None:
        try:
            _exact_text(session_id, max_bytes=_MAX_SESSION_ID_BYTES, allow_empty=True)
        except (UnicodeEncodeError, ValueError) as error:
            raise ShadowTraceInputError("invalid_step") from error
    agent = root.get("agent")
    if type(agent) is not dict or type(agent.get("name")) is not str:
        raise ShadowTraceInputError("invalid_step")
    if agent["name"] != contract.required_agent_name:
        raise ShadowTraceInputError("profile_mismatch")
    steps = root.get("steps")
    if type(steps) is not list or not steps:
        raise ShadowTraceInputError("invalid_step")
    if len(steps) > _MAX_STEPS:
        raise ShadowTraceInputError("input_limit_exceeded")

    subagents = root.get("subagent_trajectories")
    if subagents is None:
        embedded_subagent_count = 0
    elif type(subagents) is list:
        embedded_subagent_count = len(subagents)
    else:
        raise ShadowTraceInputError("invalid_step")
    continued_present = root.get("continued_trajectory_ref") is not None

    tool_counts: dict[ToolCallDisposition, int] = dict.fromkeys(_TOOL_DISPOSITIONS, 0)
    result_counts: dict[ResultDisposition, int] = dict.fromkeys(_RESULT_DISPOSITIONS, 0)
    actions: list[_ActionPlan] = []
    total_calls = 0
    total_results = 0
    ignored_message_steps = 0

    for step_ordinal, step_value in enumerate(steps, start=1):
        if type(step_value) is not dict:
            raise ShadowTraceInputError("invalid_step", step_ordinal=step_ordinal)
        step = step_value
        _required_step_integer(step.get("step_id"), ordinal=step_ordinal)
        calls = _tool_calls(step, ordinal=step_ordinal)
        results = _results(step, ordinal=step_ordinal)
        total_calls += len(calls)
        total_results += len(results)
        if total_calls > _MAX_TOTAL_TOOL_CALLS or total_results > _MAX_TOTAL_RESULTS:
            raise ShadowTraceInputError("input_limit_exceeded", step_ordinal=step_ordinal)
        if not calls:
            ignored_message_steps += 1

        copied_value = step.get("is_copied_context")
        copied = copied_value is True
        if (
            "is_copied_context" in step
            and copied_value is not None
            and type(copied_value) is not bool
        ):
            raise ShadowTraceInputError("invalid_step", step_ordinal=step_ordinal)
        if copied:
            for call_ordinal, call_value in enumerate(calls, start=1):
                if type(call_value) is not dict:
                    raise ShadowTraceInputError(
                        "invalid_tool_call",
                        step_ordinal=step_ordinal,
                        call_ordinal=call_ordinal,
                    )
            for result_ordinal, result_value in enumerate(results, start=1):
                if type(result_value) is not dict:
                    raise ShadowTraceInputError(
                        "invalid_step",
                        step_ordinal=step_ordinal,
                        result_ordinal=result_ordinal,
                    )
            tool_counts["ignored_copied_context"] += len(calls)
            result_counts["ignored_copied_context"] += len(results)
            continue

        source = step.get("source")
        source_is_agent = source == "agent" and type(source) is str
        timestamp_present = "timestamp" in step and step.get("timestamp") is not None
        raw_timestamp = step.get("timestamp") if timestamp_present else None

        call_states: list[_CallState] = []
        identifiers: dict[str, _CallState] = {}
        for call_ordinal, call_value in enumerate(calls, start=1):
            if type(call_value) is not dict:
                raise ShadowTraceInputError(
                    "invalid_tool_call",
                    step_ordinal=step_ordinal,
                    call_ordinal=call_ordinal,
                )
            raw_id = call_value.get("tool_call_id")
            if raw_id is None:
                tool_call_id = None
            elif type(raw_id) is str:
                tool_call_id = raw_id or None
            else:
                raise ShadowTraceInputError(
                    "invalid_tool_call",
                    step_ordinal=step_ordinal,
                    call_ordinal=call_ordinal,
                )
            if tool_call_id is not None and tool_call_id in identifiers:
                raise ShadowTraceInputError(
                    "duplicate_tool_call_id",
                    step_ordinal=step_ordinal,
                    call_ordinal=call_ordinal,
                )
            if not source_is_agent:
                function_name = call_value.get("function_name")
                if type(function_name) is not str:
                    raise ShadowTraceInputError(
                        "invalid_tool_call",
                        step_ordinal=step_ordinal,
                        call_ordinal=call_ordinal,
                    )
                state = _CallState(
                    call_ordinal,
                    tool_call_id,
                    "ignored_unsupported_function",
                    None,
                )
            elif contract.profile is ATIFProfile.HARBOR_TERMINUS_2_V1:
                state = _classify_terminus_call(
                    call_value,
                    step_ordinal=step_ordinal,
                    call_ordinal=call_ordinal,
                    tool_call_id=tool_call_id,
                    timestamp=cast(str | None, raw_timestamp),
                    environment=environment,
                )
            else:
                state = _classify_codex_call(
                    call_value,
                    step_ordinal=step_ordinal,
                    call_ordinal=call_ordinal,
                    tool_call_id=tool_call_id,
                    timestamp=cast(str | None, raw_timestamp),
                    environment=environment,
                )
            call_states.append(state)
            tool_counts[state.disposition] += 1
            if tool_call_id is not None:
                identifiers[tool_call_id] = state

        mapped_by_call: dict[int, list[int]] = {}
        for result_ordinal, result_value in enumerate(results, start=1):
            if type(result_value) is not dict:
                raise ShadowTraceInputError(
                    "invalid_step",
                    step_ordinal=step_ordinal,
                    result_ordinal=result_ordinal,
                )
            raw_parent = result_value.get("source_call_id")
            if raw_parent is None:
                if not calls:
                    result_counts["ignored_no_parent"] += 1
                    continue
                if len(calls) == 1 and len(results) == 1:
                    parent = call_states[0]
                else:
                    result_counts["ignored_ambiguous_parent"] += 1
                    continue
            elif type(raw_parent) is str and raw_parent:
                resolved_parent = identifiers.get(raw_parent)
                if resolved_parent is None:
                    raise ShadowTraceInputError(
                        "orphan_result",
                        step_ordinal=step_ordinal,
                        result_ordinal=result_ordinal,
                    )
                parent = resolved_parent
            else:
                raise ShadowTraceInputError(
                    "invalid_step",
                    step_ordinal=step_ordinal,
                    result_ordinal=result_ordinal,
                )
            if parent.action is None:
                result_counts["ignored_unsupported_parent"] += 1
                continue
            if contract.profile is ATIFProfile.HARBOR_TERMINUS_2_V1:
                result_counts["ignored_evidence_absent"] += 1
                continue
            assert parent.tool_call_id is not None
            exit_status = _codex_exit_status(
                step,
                tool_call_id=parent.tool_call_id,
                total_calls=len(calls),
                step_ordinal=step_ordinal,
                call_ordinal=parent.call_ordinal,
            )
            if exit_status is None:
                result_counts["ignored_evidence_absent"] += 1
            else:
                result_counts["mapped_structured_outcome"] += 1
                mapped_by_call.setdefault(parent.call_ordinal, []).append(exit_status)

        for state in call_states:
            if state.action is None:
                continue
            outcomes = tuple(mapped_by_call.get(state.call_ordinal, ()))
            if len(outcomes) > 1:
                raise ShadowTraceInputError(
                    "invalid_outcome_metadata",
                    step_ordinal=step_ordinal,
                    call_ordinal=state.call_ordinal,
                )
            action = _ActionPlan(
                step_ordinal=state.action.step_ordinal,
                call_ordinal=state.action.call_ordinal,
                source_event_id=state.action.source_event_id,
                command=state.action.command,
                working_directory=state.action.working_directory,
                environment_digest=state.action.environment_digest,
                execution_semantics=state.action.execution_semantics,
                source_timestamp=state.action.source_timestamp,
                outcome_exit_statuses=outcomes,
            )
            actions.append(action)

    if not actions:
        raise ShadowTraceInputError("no_supported_action")
    timestamp_presence = tuple(action.source_timestamp is not None for action in actions)
    if all(timestamp_presence):
        timestamp_mode: TimestampMode = "source_utc"
        previous: datetime | None = None
        normalized_actions: list[_ActionPlan] = []
        for action in actions:
            normalized = _normalize_timestamp(action.source_timestamp, step=action.step_ordinal)
            parsed = _timestamp_value(normalized)
            if previous is not None and parsed < previous:
                raise ShadowTraceInputError("invalid_timestamp", step_ordinal=action.step_ordinal)
            previous = parsed
            normalized_actions.append(
                _ActionPlan(
                    step_ordinal=action.step_ordinal,
                    call_ordinal=action.call_ordinal,
                    source_event_id=action.source_event_id,
                    command=action.command,
                    working_directory=action.working_directory,
                    environment_digest=action.environment_digest,
                    execution_semantics=action.execution_semantics,
                    source_timestamp=normalized,
                    outcome_exit_statuses=action.outcome_exit_statuses,
                )
            )
        actions = normalized_actions
    elif any(timestamp_presence):
        raise ShadowTraceInputError("partial_timestamps")
    else:
        timestamp_mode = "logical_order"

    plan = _MappingPlan(
        actions=tuple(actions),
        timestamp_mode=timestamp_mode,
        total_step_count=len(steps),
        ignored_message_step_count=ignored_message_steps,
        continued_trajectory_ref_present=continued_present,
        embedded_subagent_trajectory_count=embedded_subagent_count,
        tool_counts=tuple((key, tool_counts[key]) for key in _TOOL_DISPOSITIONS),
        result_counts=tuple((key, result_counts[key]) for key in _RESULT_DISPOSITIONS),
    )
    return plan, schema


def _logical_timestamp(index: int) -> str:
    value = datetime(2000, 1, 1, tzinfo=UTC) + timedelta(microseconds=index)
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _emit_records(plan: _MappingPlan) -> list[dict[str, object]]:
    mapped_count = (
        2 + len(plan.actions) + sum(len(action.outcome_exit_statuses) for action in plan.actions)
    )
    if mapped_count > MAX_SHADOW_TRACE_ROWS:
        raise ShadowTraceInputError("input_limit_exceeded")
    logical_index = 0
    first_source = plan.actions[0].source_timestamp
    start_timestamp = first_source if plan.timestamp_mode == "source_utc" else _logical_timestamp(0)
    assert start_timestamp is not None
    records: list[dict[str, object]] = [
        {
            "schema_version": "shadow-input/v1",
            "kind": "run_start",
            "source_event_id": "atif-run-start",
            "occurred_at": start_timestamp,
        }
    ]
    last_timestamp = start_timestamp
    for action in plan.actions:
        if plan.timestamp_mode == "logical_order":
            logical_index += 1
            action_timestamp = _logical_timestamp(logical_index)
        else:
            assert action.source_timestamp is not None
            action_timestamp = action.source_timestamp
        records.append(
            {
                "schema_version": "shadow-input/v1",
                "kind": "action",
                "source_event_id": action.source_event_id,
                "occurred_at": action_timestamp,
                "command": action.command,
                "working_directory": action.working_directory,
                "environment_digest": action.environment_digest,
            }
        )
        last_timestamp = action_timestamp
        for exit_status in action.outcome_exit_statuses:
            if plan.timestamp_mode == "logical_order":
                logical_index += 1
                result_timestamp = _logical_timestamp(logical_index)
            else:
                result_timestamp = action_timestamp
            records.append(
                {
                    "schema_version": "shadow-input/v1",
                    "kind": "tool_result",
                    "source_event_id": (
                        f"atif-s{action.step_ordinal:08d}-c{action.call_ordinal:04d}-result"
                    ),
                    "occurred_at": result_timestamp,
                    "action_source_event_id": action.source_event_id,
                    "status": "succeeded" if exit_status == 0 else "failed",
                    "exit_status": exit_status,
                }
            )
            last_timestamp = result_timestamp
    if plan.timestamp_mode == "logical_order":
        logical_index += 1
        end_timestamp = _logical_timestamp(logical_index)
    else:
        end_timestamp = last_timestamp
    records.append(
        {
            "schema_version": "shadow-input/v1",
            "kind": "run_end",
            "source_event_id": "atif-run-end",
            "occurred_at": end_timestamp,
        }
    )
    return records


def _build_trace(
    *,
    source: bytes,
    run_id: UUID,
    environment: ShadowEnvironmentBinding,
    contract: _ATIFProfileContract,
    plan: _MappingPlan,
    source_schema: str,
    task_scope_digest: str | None,
    lineage_scope_digest: str | None,
    capture_manifest_digest: str | None,
) -> ShadowTrace:
    records = _emit_records(plan)
    frozen_records, record_bytes, _counts, _repeated = _canonical_records(
        records,
        run_id=run_id,
        capture_scope="selected_events",
        timestamp_mode=plan.timestamp_mode,
    )
    configuration_bytes = canonical_json(
        {
            "schema_version": "atif-shadow-adapter-configuration/v1",
            "adapter_profile_id": contract.profile.value,
            "adapter_profile_digest": contract.profile_digest,
            "compatibility_evidence_manifest_digest": _EXPECTED_MANIFEST_DIGEST,
            "source_format": "atif",
            "source_schema_version": source_schema,
            "timestamp_mode": plan.timestamp_mode,
            "capture_scope": "selected_events",
            "selection_scope": "root_segment_selected_events",
            "environment": environment.model_dump(mode="json", warnings=False),
            "mapped_action_contexts": [
                {
                    "source_event_id": action.source_event_id,
                    "working_directory": action.working_directory,
                    "execution_semantics": dict(action.execution_semantics),
                    "environment_digest": action.environment_digest,
                }
                for action in plan.actions
            ],
        }
    )
    configuration_digest = length_prefixed_sha256(
        configuration_bytes,
        domain=_CONFIGURATION_DIGEST_DOMAIN,
    )
    binding = _build_binding(
        source_format="atif",
        source_schema_version=source_schema,
        source_digest_kind="original_bytes",
        source_bytes=source,
        adapter_profile_id=contract.profile.value,
        adapter_profile_digest=contract.profile_digest,
        adapter_configuration_digest=configuration_digest,
        timestamp_mode=plan.timestamp_mode,
        capture_scope="selected_events",
        task_scope_digest=task_scope_digest,
        lineage_scope_digest=lineage_scope_digest,
        capture_manifest_digest=capture_manifest_digest,
    )
    diagnostics: ATIFShadowDiagnostics = _build_atif_diagnostics(
        continued_trajectory_ref_present=plan.continued_trajectory_ref_present,
        embedded_subagent_trajectory_count=plan.embedded_subagent_trajectory_count,
        outcome_evidence_authority=contract.outcome_evidence_authority,
        profile_audit_manifest_digest=_EXPECTED_MANIFEST_DIGEST,
        total_step_count=plan.total_step_count,
        ignored_message_step_count=plan.ignored_message_step_count,
        total_tool_call_count=sum(value for _, value in plan.tool_counts),
        tool_call_disposition_counts=plan.tool_counts,
        total_observation_result_count=sum(value for _, value in plan.result_counts),
        result_disposition_counts=plan.result_counts,
        mapped_shadow_record_count=len(records),
    )
    trace = _new_shadow_trace(
        run_id=run_id,
        binding=binding,
        diagnostics=diagnostics,
        records=frozen_records,
        record_bytes=record_bytes,
        adapter_descriptor_bytes=contract.descriptor_bytes,
        adapter_configuration_bytes=configuration_bytes,
    )
    if not trace._is_exact():
        raise ShadowInvariantError
    return trace


@final
class ATIFShadowAdapter:
    """A sealed, side-effect-free adapter for one explicit built-in ATIF profile."""

    __slots__ = ("_environment", "_profile")

    _environment: ShadowEnvironmentBinding
    _profile: ATIFProfile

    def __init__(
        self,
        *,
        profile: ATIFProfile,
        environment: ShadowEnvironmentBinding,
    ) -> None:
        if type(profile) is not ATIFProfile or type(environment) is not ShadowEnvironmentBinding:
            raise TypeError("ATIF adapter configuration is invalid")
        invariant_failure = False
        configuration_failure = False
        try:
            copied_environment = ShadowEnvironmentBinding.model_validate_json(
                canonical_json(environment)
            )
            _load_manifest_bytes()
        except (KeyboardInterrupt, SystemExit):
            raise
        except ShadowInvariantError:
            invariant_failure = True
            copied_environment = None
        except Exception:
            configuration_failure = True
            copied_environment = None
        if invariant_failure:
            raise ShadowInvariantError
        if configuration_failure or copied_environment is None:
            raise ShadowConfigurationError
        object.__setattr__(self, "_profile", profile)
        object.__setattr__(self, "_environment", copied_environment)

    def __init_subclass__(cls, **_kwargs: object) -> None:
        raise TypeError("ATIFShadowAdapter cannot be subclassed")

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("ATIFShadowAdapter is immutable")

    @property
    def profile_id(self) -> str:
        return str(self._profile.value)

    @property
    def profile_digest(self) -> str:
        return _PROFILE_CONTRACTS[self._profile].profile_digest

    def adapt_bytes(
        self,
        source: bytes,
        *,
        run_id: UUID,
        task_scope_digest: str | None = None,
        lineage_scope_digest: str | None = None,
        capture_manifest_digest: str | None = None,
    ) -> ShadowTrace:
        if type(source) is not bytes:
            raise ShadowTraceInputError("invalid_json")
        if type(run_id) is not UUID or run_id.version != 4:
            raise ShadowTraceInputError("invalid_step")
        for digest in (task_scope_digest, lineage_scope_digest, capture_manifest_digest):
            if digest is not None and (
                type(digest) is not str or _DIGEST_PATTERN.fullmatch(digest) is None
            ):
                raise ShadowTraceInputError("invalid_step")
        contract = _PROFILE_CONTRACTS[self._profile]
        trace_error: tuple[str, int | None, int | None, int | None] | None = None
        invariant_failure = False
        unexpected_failure = False
        trace: ShadowTrace | None = None
        try:
            _load_manifest_bytes()
            root = _parse_source(bytes(source))
            plan, source_schema = _plan_mapping(
                root,
                contract=contract,
                environment=self._environment,
            )
            trace = _build_trace(
                source=bytes(source),
                run_id=UUID(int=run_id.int),
                environment=self._environment,
                contract=contract,
                plan=plan,
                source_schema=source_schema,
                task_scope_digest=task_scope_digest,
                lineage_scope_digest=lineage_scope_digest,
                capture_manifest_digest=capture_manifest_digest,
            )
        except ShadowTraceInputError as error:
            trace_error = (
                error.reason_code,
                error.step_ordinal,
                error.call_ordinal,
                error.result_ordinal,
            )
        except ShadowInvariantError:
            invariant_failure = True
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            unexpected_failure = True
        if trace_error is not None:
            reason, step, call, result = trace_error
            raise ShadowTraceInputError(
                reason,
                step_ordinal=step,
                call_ordinal=call,
                result_ordinal=result,
            )
        if invariant_failure:
            raise ShadowInvariantError
        if unexpected_failure or trace is None:
            raise ShadowTraceInputError("invalid_step")
        return trace

    def __repr__(self) -> str:
        return "ATIFShadowAdapter(<configured>)"


__all__ = [
    "ATIFProfile",
    "ATIFShadowAdapter",
    "ShadowEnvironmentBinding",
]
