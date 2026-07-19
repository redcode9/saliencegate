from __future__ import annotations

import re
from dataclasses import InitVar, dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Literal, Never, Self, SupportsIndex, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from saliencegate.domain import EventType, TraceEvent, length_prefixed_sha256
from saliencegate.security import REDACTED, REDACTED_PRIVATE_KEY
from saliencegate.signals.base import AbstentionReason

_FINGERPRINT_VERSION: Literal["structured-fingerprint-v1"] = "structured-fingerprint-v1"

_MAX_ACTION_PAYLOAD_BYTES = 512 * 1_024
_MAX_ACTION_PAYLOAD_NODES = 1_024
_MAX_ARGV_ITEMS = 256
_MAX_ARG_BYTES = 16 * 1_024
_MAX_ARGV_BYTES = 256 * 1_024
_MAX_COMMAND_BYTES = 128 * 1_024
_MAX_DIRECTORY_BYTES = 16 * 1_024
_MAX_SHORT_TEXT_BYTES = 16 * 1_024
_MAX_SIGNATURE_BYTES = 128 * 1_024
_MAX_TOOL_PAYLOAD_BYTES = 512 * 1_024
_MAX_TOOL_PAYLOAD_NODES = 128
_MAX_TEST_FAILURES = 10_000
_MAX_TEST_REPORT_BYTES = 2 * 1_024 * 1_024
_MAX_TEST_REPORT_NODES = 60_000
_MAX_TEST_ID_BYTES = 32 * 1_024
_MAX_PAYLOAD_DEPTH = 8
_MAX_EXIT_STATUS = (1 << 31) - 1
_MIN_EXIT_STATUS = -(1 << 31)

_PYTEST_COMMUTATIVE_FLAGS = frozenset({"--quiet", "--verbose", "-q", "-v"})
_UNSAFE_RAW_SHELL_CHARACTERS = frozenset("$`*?[]{}~#!%^\r\n\x00();<>|&'\"\\=")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class FingerprintUnavailableError(ValueError):
    """A sanitized reason why deterministic equivalence cannot be established."""

    __slots__ = ("reason",)

    def __init__(self, reason: AbstentionReason) -> None:
        self.reason = reason
        super().__init__("structured fingerprint input is unavailable")


class ToolOutcomeStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class TestReportStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class _EvidenceModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
        validate_default=True,
    )


ActionCommandText = Annotated[str, StringConstraints(min_length=1, max_length=128 * 1_024)]
ArgumentText = Annotated[str, StringConstraints(min_length=1, max_length=16 * 1_024)]
DirectoryText = Annotated[str, StringConstraints(min_length=1, max_length=16 * 1_024)]
ShortText = Annotated[str, StringConstraints(min_length=1, max_length=16 * 1_024)]
SignatureText = Annotated[str, StringConstraints(min_length=1, max_length=128 * 1_024)]
TestIdText = Annotated[str, StringConstraints(min_length=1, max_length=32 * 1_024)]


def _contract_text(value: str, *, max_bytes: int) -> str:
    if type(value) is not str:
        raise ValueError("text subclasses are not accepted")
    encoded: bytes | None = None
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        encoded = None
    if encoded is None:
        raise ValueError("text must be UTF-8 encodable")
    if not encoded or len(encoded) > max_bytes or "\x00" in value:
        raise ValueError("text violates its local bound")
    return value


def _optional_contract_text(value: str | None, *, max_bytes: int) -> str | None:
    return None if value is None else _contract_text(value, max_bytes=max_bytes)


class ShellActionEvidence(_EvidenceModel):
    """Versioned adapter-owned envelope for an intercepted shell action."""

    schema_version: Literal["1.0"]
    kind: Literal["shell"]
    command: Annotated[ActionCommandText | None, Field(repr=False)] = None
    argv: Annotated[
        tuple[ArgumentText, ...] | None,
        Field(min_length=1, max_length=_MAX_ARGV_ITEMS, repr=False),
    ] = None
    working_directory: Annotated[DirectoryText, Field(repr=False)]
    environment_digest: str = Field(pattern=r"^[0-9a-f]{64}$", repr=False)

    @field_validator("command")
    @classmethod
    def bounded_command(cls, value: str | None) -> str | None:
        return _optional_contract_text(value, max_bytes=_MAX_COMMAND_BYTES)

    @field_validator("argv")
    @classmethod
    def bounded_argv(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is None:
            return None
        checked = tuple(_contract_text(item, max_bytes=_MAX_ARG_BYTES) for item in value)
        if sum(len(item.encode("utf-8")) for item in checked) > _MAX_ARGV_BYTES:
            raise ValueError("argv violates its aggregate byte bound")
        return checked

    @field_validator("working_directory")
    @classmethod
    def bounded_directory(cls, value: str) -> str:
        return _contract_text(value, max_bytes=_MAX_DIRECTORY_BYTES)

    @field_validator("environment_digest", mode="before")
    @classmethod
    def exact_environment_digest(cls, value: object) -> object:
        if type(value) is not str or _DIGEST.fullmatch(value) is None:
            raise ValueError("environment digest is invalid")
        return value

    @model_validator(mode="after")
    def has_one_source_form(self) -> Self:
        if (self.command is None) == (self.argv is None):
            raise ValueError("shell action requires exactly one of command or argv")
        return self


class ToolOutcomeEvidence(_EvidenceModel):
    """Versioned adapter-owned envelope for a structured tool completion."""

    schema_version: Literal["1.0"]
    status: Literal["succeeded", "failed"] | None = None
    exit_status: Annotated[
        int | None,
        Field(ge=_MIN_EXIT_STATUS, le=_MAX_EXIT_STATUS),
    ] = None
    exception_type: Annotated[ShortText | None, Field(repr=False)] = None
    error_code: Annotated[ShortText | None, Field(repr=False)] = None
    failure_signature: Annotated[SignatureText | None, Field(repr=False)] = None

    @field_validator("exception_type", "error_code")
    @classmethod
    def bounded_optional_short_text(cls, value: str | None) -> str | None:
        return _optional_contract_text(value, max_bytes=_MAX_SHORT_TEXT_BYTES)

    @field_validator("failure_signature")
    @classmethod
    def bounded_optional_signature(cls, value: str | None) -> str | None:
        return _optional_contract_text(value, max_bytes=_MAX_SIGNATURE_BYTES)

    @model_validator(mode="after")
    def evidence_is_consistent(self) -> Self:
        auxiliary_failure = any(
            item is not None
            for item in (self.exception_type, self.error_code, self.failure_signature)
        )
        nonzero_exit = self.exit_status not in (None, 0)
        if self.exit_status == 0 and self.status != "failed" and auxiliary_failure:
            raise ValueError("zero exit status contradicts failure evidence")
        if self.status == "succeeded" and (nonzero_exit or auxiliary_failure):
            raise ValueError("successful tool outcome carries failure evidence")
        if self.status is None and self.exit_status is None and not auxiliary_failure:
            raise ValueError("tool outcome carries no classifiable evidence")
        return self


class TestFailureEvidence(_EvidenceModel):
    """Versioned failure item nested in a :class:`TestReportEvidence`."""

    schema_version: Literal["1.0"]
    test_id: Annotated[TestIdText, Field(repr=False)]
    failure_type: Annotated[ShortText | None, Field(repr=False)] = None
    signature: Annotated[SignatureText | None, Field(repr=False)] = None

    @field_validator("test_id")
    @classmethod
    def bounded_test_id(cls, value: str) -> str:
        return _contract_text(value, max_bytes=_MAX_TEST_ID_BYTES)

    @field_validator("failure_type")
    @classmethod
    def bounded_optional_failure_type(cls, value: str | None) -> str | None:
        return _optional_contract_text(value, max_bytes=_MAX_SHORT_TEXT_BYTES)

    @field_validator("signature")
    @classmethod
    def bounded_optional_signature(cls, value: str | None) -> str | None:
        return _optional_contract_text(value, max_bytes=_MAX_SIGNATURE_BYTES)


class TestReportEvidence(_EvidenceModel):
    """Versioned adapter-owned envelope for a structured test report."""

    schema_version: Literal["1.0"]
    framework: ShortText
    status: Literal["passed", "failed"]
    failures: Annotated[
        tuple[TestFailureEvidence, ...],
        Field(max_length=_MAX_TEST_FAILURES, repr=False),
    ]

    @field_validator("failures", mode="before")
    @classmethod
    def bounded_failures_input(cls, value: object) -> object:
        if type(value) is tuple and len(value) > _MAX_TEST_FAILURES:
            raise ValueError("test report violates its aggregate byte bound")
        if type(value) is tuple and all(type(item) is TestFailureEvidence for item in value):
            size_bound = 2 + len(value)
            for item in value:
                assert isinstance(item, TestFailureEvidence)
                size_bound += 6 * len(item.test_id) + 128
                if item.failure_type is not None:
                    size_bound += 6 * len(item.failure_type) + 2
                if item.signature is not None:
                    size_bound += 6 * len(item.signature) + 2
            if size_bound > _MAX_TEST_REPORT_BYTES:
                raise ValueError("test report violates its aggregate byte bound")
        elif not _payload_is_bounded(
            value,
            max_bytes=_MAX_TEST_REPORT_BYTES,
            max_nodes=_MAX_TEST_REPORT_NODES,
        ):
            raise ValueError("test report violates its aggregate byte bound")
        return value

    @field_validator("framework")
    @classmethod
    def bounded_framework(cls, value: str) -> str:
        return _contract_text(value, max_bytes=_MAX_SHORT_TEXT_BYTES)

    @model_validator(mode="after")
    def failures_match_status(self) -> Self:
        if (self.status == "failed") != bool(self.failures):
            raise ValueError("test report status and failures disagree")
        size_bound = 6 * len(self.framework) + 64
        for failure in self.failures:
            size_bound += 6 * len(failure.test_id) + 128
            if failure.failure_type is not None:
                size_bound += 6 * len(failure.failure_type) + 2
            if failure.signature is not None:
                size_bound += 6 * len(failure.signature) + 2
        if size_bound > _MAX_TEST_REPORT_BYTES:
            raise ValueError("test report violates its aggregate byte bound")
        return self


def _equivalence_text(value: str) -> str:
    if REDACTED in value or REDACTED_PRIVATE_KEY in value:
        raise FingerprintUnavailableError(AbstentionReason.REDACTED_EQUIVALENCE_INPUT)
    return value


def _validate_derived_text(
    value: str,
    *,
    max_bytes: int,
    allow_empty: bool = False,
) -> None:
    if type(value) is not str or (not value and not allow_empty) or len(value) > max_bytes:
        raise ValueError("derived text is invalid")
    encoded: bytes | None = None
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        encoded = None
    if encoded is None:
        raise ValueError("derived text is invalid")
    if (
        len(encoded) > max_bytes
        or "\x00" in value
        or REDACTED in value
        or REDACTED_PRIVATE_KEY in value
    ):
        raise ValueError("derived text is invalid")


@dataclass(frozen=True, slots=True, repr=False)
class ActionFingerprint:
    """Transient, non-serializable action equivalence value."""

    execution_mode: Literal["argv", "shell"]
    tokens: InitVar[tuple[str, ...]]
    working_directory: InitVar[str]
    environment_digest: InitVar[str]
    _digest: str = field(init=False, repr=False)
    algorithm_version: Literal["structured-fingerprint-v1"] = field(
        default=_FINGERPRINT_VERSION,
        init=False,
        repr=False,
    )

    def __post_init__(
        self,
        tokens: tuple[str, ...],
        working_directory: str,
        environment_digest: str,
    ) -> None:
        if (
            type(self.execution_mode) is not str
            or self.execution_mode not in ("argv", "shell")
            or not tokens
        ):
            raise ValueError("action fingerprint is invalid")
        if type(tokens) is not tuple or len(tokens) > _MAX_ARGV_ITEMS:
            raise ValueError("action fingerprint is invalid")
        for token in tokens:
            _validate_derived_text(token, max_bytes=_MAX_ARG_BYTES)
        if sum(len(token.encode("utf-8")) for token in tokens) > _MAX_ARGV_BYTES:
            raise ValueError("action fingerprint is invalid")
        _validate_derived_text(working_directory, max_bytes=_MAX_DIRECTORY_BYTES)
        if type(environment_digest) is not str or _DIGEST.fullmatch(environment_digest) is None:
            raise ValueError("action fingerprint is invalid")
        object.__setattr__(
            self,
            "_digest",
            length_prefixed_sha256(
                self.execution_mode,
                working_directory,
                environment_digest,
                *tokens,
                domain="saliencegate:signals:action-fingerprint:v1",
            ),
        )

    def __repr__(self) -> str:
        return "ActionFingerprint(<structured>)"

    __str__ = __repr__

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        del protocol
        raise TypeError("action fingerprints are transient")


@dataclass(frozen=True, slots=True, repr=False)
class ToolOutcome:
    """Validated transient view of a tool outcome envelope."""

    status: ToolOutcomeStatus
    exit_status: int | None = None
    exception_type: str | None = None
    error_code: str | None = None
    failure_signature: str | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not ToolOutcomeStatus:
            raise ValueError("tool outcome is invalid")
        if self.exit_status is not None and (
            type(self.exit_status) is not int
            or not _MIN_EXIT_STATUS <= self.exit_status <= _MAX_EXIT_STATUS
        ):
            raise ValueError("tool outcome is invalid")
        for item in (self.exception_type, self.error_code):
            if item is not None:
                _validate_derived_text(item, max_bytes=_MAX_SHORT_TEXT_BYTES)
        if self.failure_signature is not None:
            _validate_derived_text(
                self.failure_signature,
                max_bytes=_MAX_SIGNATURE_BYTES,
            )
        failure_detail = any(
            item is not None
            for item in (self.exception_type, self.error_code, self.failure_signature)
        )
        if self.status is ToolOutcomeStatus.SUCCEEDED and (
            self.exit_status not in (None, 0) or failure_detail
        ):
            raise ValueError("tool outcome is invalid")

    def __repr__(self) -> str:
        return "ToolOutcome(<structured>)"


@dataclass(frozen=True, slots=True, repr=False)
class NormalizedTestFailure:
    """Validated transient view of one test failure."""

    test_id: str
    failure_type: str | None = None
    signature: str | None = None

    def __post_init__(self) -> None:
        if normalize_test_id(self.test_id) != self.test_id:
            raise ValueError("normalized test failure is invalid")
        if self.failure_type is not None:
            _validate_derived_text(self.failure_type, max_bytes=_MAX_SHORT_TEXT_BYTES)
        if self.signature is not None:
            _validate_derived_text(self.signature, max_bytes=_MAX_SIGNATURE_BYTES)

    def __repr__(self) -> str:
        return "NormalizedTestFailure(<structured>)"


@dataclass(frozen=True, slots=True, repr=False)
class TestReport:
    """Validated transient view of a complete test report envelope."""

    framework: str
    status: TestReportStatus
    failures: tuple[NormalizedTestFailure, ...] = ()

    def __post_init__(self) -> None:
        _validate_derived_text(self.framework, max_bytes=_MAX_SHORT_TEXT_BYTES)
        if type(self.status) is not TestReportStatus or type(self.failures) is not tuple:
            raise ValueError("test report is invalid")
        if len(self.failures) > _MAX_TEST_FAILURES:
            raise ValueError("test report is invalid")
        if any(type(item) is not NormalizedTestFailure for item in self.failures):
            raise ValueError("test report is invalid")
        size_bound = 6 * len(self.framework) + 64
        for failure in self.failures:
            size_bound += 6 * len(failure.test_id) + 128
            if failure.failure_type is not None:
                size_bound += 6 * len(failure.failure_type) + 2
            if failure.signature is not None:
                size_bound += 6 * len(failure.signature) + 2
        if size_bound > _MAX_TEST_REPORT_BYTES:
            raise ValueError("test report is invalid")
        if (self.status is TestReportStatus.FAILED) != bool(self.failures):
            raise ValueError("test report is invalid")
        if len(set(self.failures)) != len(self.failures) or self.failures != tuple(
            sorted(self.failures, key=_test_failure_sort_key)
        ):
            raise ValueError("test report is invalid")

    def __repr__(self) -> str:
        return "TestReport(<structured>)"


@dataclass(frozen=True, slots=True, repr=False)
class FailureFingerprint:
    """Transient, non-serializable failure equivalence value."""

    category: Literal["test", "tool"]
    components: InitVar[tuple[str, ...]]
    _digest: str = field(init=False, repr=False)
    algorithm_version: Literal["structured-fingerprint-v1"] = field(
        default=_FINGERPRINT_VERSION,
        init=False,
        repr=False,
    )

    def __post_init__(self, components: tuple[str, ...]) -> None:
        if (
            type(self.category) is not str
            or self.category not in ("test", "tool")
            or not components
        ):
            raise ValueError("failure fingerprint is invalid")
        if type(components) is not tuple or len(components) > 2 + 3 * _MAX_TEST_FAILURES:
            raise ValueError("failure fingerprint is invalid")
        for component in components:
            _validate_derived_text(
                component,
                max_bytes=_MAX_SIGNATURE_BYTES,
                allow_empty=True,
            )
        if sum(6 * len(component) + 2 for component in components) > _MAX_TEST_REPORT_BYTES:
            raise ValueError("failure fingerprint is invalid")
        object.__setattr__(
            self,
            "_digest",
            length_prefixed_sha256(
                self.category,
                *components,
                domain="saliencegate:signals:failure-fingerprint:v1",
            ),
        )

    def __repr__(self) -> str:
        return "FailureFingerprint(<structured>)"

    __str__ = __repr__

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        del protocol
        raise TypeError("failure fingerprints are transient")


EvidenceModelT = TypeVar("EvidenceModelT", bound=_EvidenceModel)


def _payload_is_bounded(value: object, *, max_bytes: int, max_nodes: int) -> bool:
    remaining = max_bytes
    nodes = 0
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > max_nodes or depth > _MAX_PAYLOAD_DEPTH:
            return False
        if type(item) in (dict, MappingProxyType):
            assert isinstance(item, (dict, MappingProxyType))
            if nodes + len(item) > max_nodes:
                return False
            remaining -= 2 + len(item) * 2
            for key, nested in item.items():
                if type(key) is not str:
                    return False
                remaining -= 6 * len(key) + 3
                stack.append((nested, depth + 1))
        elif type(item) is tuple:
            if nodes + len(item) > max_nodes:
                return False
            remaining -= 2 + len(item)
            stack.extend((nested, depth + 1) for nested in item)
        elif type(item) is str:
            remaining -= 6 * len(item) + 2
        elif item is None or type(item) is bool:
            remaining -= 5
        elif type(item) is int:
            remaining -= max(2, item.bit_length() // 3 + 2)
        elif type(item) is float:
            remaining -= 32
        else:
            return False
        if remaining < 0:
            return False
    return True


def _thaw_json(value: object) -> object:
    if type(value) in (dict, MappingProxyType):
        assert isinstance(value, (dict, MappingProxyType))
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_thaw_json(item) for item in value)
    return value


def _load_payload(
    model: type[EvidenceModelT],
    value: object,
    *,
    max_bytes: int,
    max_nodes: int,
) -> EvidenceModelT:
    if not _payload_is_bounded(value, max_bytes=max_bytes, max_nodes=max_nodes):
        raise FingerprintUnavailableError(AbstentionReason.STRUCTURED_EVIDENCE_INVALID)
    try:
        parsed = model.model_validate(_thaw_json(value))
    except (TypeError, ValueError, ValidationError, RecursionError):
        parsed = None
    if parsed is None:
        raise FingerprintUnavailableError(AbstentionReason.STRUCTURED_EVIDENCE_INVALID)
    return parsed


def _load_action(event: TraceEvent) -> ShellActionEvidence:
    if type(event) is not TraceEvent or event.event_type is not EventType.ACTION_PROPOSAL:
        raise FingerprintUnavailableError(AbstentionReason.EVENT_NOT_APPLICABLE)
    if type(event.payload) is not MappingProxyType:
        raise FingerprintUnavailableError(AbstentionReason.STRUCTURED_EVIDENCE_INVALID)
    if "action" not in event.payload:
        raise FingerprintUnavailableError(AbstentionReason.STRUCTURED_EVIDENCE_MISSING)
    value = event.payload["action"]
    return _load_payload(
        ShellActionEvidence,
        value,
        max_bytes=_MAX_ACTION_PAYLOAD_BYTES,
        max_nodes=_MAX_ACTION_PAYLOAD_NODES,
    )


def _load_tool_outcome(event: TraceEvent) -> ToolOutcomeEvidence:
    if type(event) is not TraceEvent or event.event_type is not EventType.TOOL_COMPLETION:
        raise FingerprintUnavailableError(AbstentionReason.EVENT_NOT_APPLICABLE)
    if type(event.payload) is not MappingProxyType:
        raise FingerprintUnavailableError(AbstentionReason.STRUCTURED_EVIDENCE_INVALID)
    if "test_report" in event.payload:
        raise FingerprintUnavailableError(AbstentionReason.STRUCTURED_EVIDENCE_INVALID)
    if "tool_outcome" not in event.payload:
        raise FingerprintUnavailableError(AbstentionReason.STRUCTURED_EVIDENCE_MISSING)
    value = event.payload["tool_outcome"]
    return _load_payload(
        ToolOutcomeEvidence,
        value,
        max_bytes=_MAX_TOOL_PAYLOAD_BYTES,
        max_nodes=_MAX_TOOL_PAYLOAD_NODES,
    )


def _load_test_report(event: TraceEvent) -> TestReportEvidence:
    if type(event) is not TraceEvent or event.event_type not in (
        EventType.TOOL_COMPLETION,
        EventType.OBSERVATION,
    ):
        raise FingerprintUnavailableError(AbstentionReason.EVENT_NOT_APPLICABLE)
    if type(event.payload) is not MappingProxyType:
        raise FingerprintUnavailableError(AbstentionReason.STRUCTURED_EVIDENCE_INVALID)
    if "tool_outcome" in event.payload:
        raise FingerprintUnavailableError(AbstentionReason.STRUCTURED_EVIDENCE_INVALID)
    if "test_report" not in event.payload:
        raise FingerprintUnavailableError(AbstentionReason.STRUCTURED_EVIDENCE_MISSING)
    value = event.payload["test_report"]
    return _load_payload(
        TestReportEvidence,
        value,
        max_bytes=_MAX_TEST_REPORT_BYTES,
        max_nodes=_MAX_TEST_REPORT_NODES,
    )


def _normalize_pytest_flags(tokens: tuple[str, ...]) -> tuple[str, ...]:
    executable = tokens[0]
    prefix_length = 1 if executable in {"py.test", "pytest"} else 0
    if (
        prefix_length == 0
        and executable in {"python", "python3"}
        and len(tokens) >= 3
        and tokens[1:3] == ("-m", "pytest")
    ):
        prefix_length = 3
    if prefix_length == 0:
        return tokens

    prefix = tokens[:prefix_length]
    arguments = tokens[prefix_length:]
    try:
        terminator = arguments.index("--")
    except ValueError:
        terminator = len(arguments)
    before_terminator = arguments[:terminator]
    after_terminator = arguments[terminator:]
    unknown_option_present = any(
        item.startswith("-") and item not in _PYTEST_COMMUTATIVE_FLAGS for item in before_terminator
    )
    if unknown_option_present:
        return tokens
    safe = sorted(item for item in before_terminator if item in _PYTEST_COMMUTATIVE_FLAGS)
    positional = (item for item in before_terminator if item not in _PYTEST_COMMUTATIVE_FLAGS)
    return (*prefix, *safe, *positional, *after_terminator)


def _shell_tokens(payload: ShellActionEvidence) -> tuple[str, ...]:
    if payload.argv is not None:
        tokens = payload.argv
    else:
        assert payload.command is not None
        if REDACTED in payload.command or REDACTED_PRIVATE_KEY in payload.command:
            raise FingerprintUnavailableError(AbstentionReason.REDACTED_EQUIVALENCE_INPUT)
        if any(character in _UNSAFE_RAW_SHELL_CHARACTERS for character in payload.command) or any(
            character.isspace() and character not in " \t" for character in payload.command
        ):
            raise FingerprintUnavailableError(AbstentionReason.STRUCTURED_EVIDENCE_INVALID)
        tokens = tuple(token for token in re.split(r"[ \t]+", payload.command) if token)
    token_sizes = tuple(len(token.encode("utf-8")) for token in tokens)
    if (
        not tokens
        or len(tokens) > _MAX_ARGV_ITEMS
        or any(
            not token or size > _MAX_ARG_BYTES
            for token, size in zip(tokens, token_sizes, strict=True)
        )
        or sum(token_sizes) > _MAX_ARGV_BYTES
    ):
        raise FingerprintUnavailableError(AbstentionReason.STRUCTURED_EVIDENCE_INVALID)
    normalized = tuple(_equivalence_text(token) for token in tokens)
    return _normalize_pytest_flags(normalized)


def action_fingerprint(event: TraceEvent) -> ActionFingerprint:
    payload = _load_action(event)
    return ActionFingerprint(
        execution_mode="argv" if payload.argv is not None else "shell",
        tokens=_shell_tokens(payload),
        working_directory=_equivalence_text(payload.working_directory),
        environment_digest=payload.environment_digest,
    )


def _tool_status(payload: ToolOutcomeEvidence) -> ToolOutcomeStatus:
    if payload.status is not None:
        return ToolOutcomeStatus(payload.status)
    if payload.exit_status == 0:
        return ToolOutcomeStatus.SUCCEEDED
    return ToolOutcomeStatus.FAILED


def classify_tool_outcome(event: TraceEvent) -> ToolOutcomeStatus:
    """Classify failure without consuming optional equivalence details."""

    return _tool_status(_load_tool_outcome(event))


def parse_tool_outcome(event: TraceEvent) -> ToolOutcome:
    payload = _load_tool_outcome(event)
    return ToolOutcome(
        status=_tool_status(payload),
        exit_status=payload.exit_status,
        exception_type=(
            None if payload.exception_type is None else _equivalence_text(payload.exception_type)
        ),
        error_code=None if payload.error_code is None else _equivalence_text(payload.error_code),
        failure_signature=(
            None
            if payload.failure_signature is None
            else _equivalence_text(payload.failure_signature)
        ),
    )


def normalize_test_id(value: str) -> str:
    try:
        checked = _contract_text(value, max_bytes=_MAX_TEST_ID_BYTES)
    except ValueError:
        checked = None
    if checked is None:
        raise FingerprintUnavailableError(AbstentionReason.STRUCTURED_EVIDENCE_INVALID)
    checked = _equivalence_text(checked)
    if checked == "./":
        raise FingerprintUnavailableError(AbstentionReason.STRUCTURED_EVIDENCE_INVALID)
    remainder = checked[2:] if checked.startswith("./") else ""
    drive_qualified = re.match(r"^[A-Za-z]:[\\/]", remainder) is not None
    normalized = (
        remainder
        if remainder and not drive_qualified and not remainder.startswith(("./", "/", "\\"))
        else checked
    )
    if not normalized:
        raise FingerprintUnavailableError(AbstentionReason.STRUCTURED_EVIDENCE_INVALID)
    path, separator, selector = normalized.partition("::")
    if not path or (separator and not selector):
        raise FingerprintUnavailableError(AbstentionReason.STRUCTURED_EVIDENCE_INVALID)
    if ".." in re.split(r"[\\/]", path):
        raise FingerprintUnavailableError(AbstentionReason.STRUCTURED_EVIDENCE_INVALID)
    return normalized


def _validated_test_ids(payload: TestReportEvidence) -> tuple[str, ...]:
    return tuple(normalize_test_id(failure.test_id) for failure in payload.failures)


def classify_test_report(event: TraceEvent) -> TestReportStatus:
    """Classify a report while ignoring optional details and duplicate failures."""

    payload = _load_test_report(event)
    _equivalence_text(payload.framework)
    _validated_test_ids(payload)
    return TestReportStatus(payload.status)


def _test_failure_sort_key(failure: NormalizedTestFailure) -> tuple[str, str, str]:
    return failure.test_id, failure.failure_type or "", failure.signature or ""


def parse_test_report(event: TraceEvent) -> TestReport:
    payload = _load_test_report(event)
    failures = tuple(
        sorted(
            (
                NormalizedTestFailure(
                    test_id=normalize_test_id(failure.test_id),
                    failure_type=(
                        None
                        if failure.failure_type is None
                        else _equivalence_text(failure.failure_type)
                    ),
                    signature=(
                        None if failure.signature is None else _equivalence_text(failure.signature)
                    ),
                )
                for failure in payload.failures
            ),
            key=_test_failure_sort_key,
        )
    )
    if len(set(failures)) != len(failures):
        raise FingerprintUnavailableError(AbstentionReason.STRUCTURED_EVIDENCE_INVALID)
    return TestReport(
        framework=_equivalence_text(payload.framework),
        status=TestReportStatus(payload.status),
        failures=failures,
    )


def failure_fingerprint(event: TraceEvent) -> FailureFingerprint:
    if type(event) is not TraceEvent:
        raise FingerprintUnavailableError(AbstentionReason.EVENT_NOT_APPLICABLE)
    has_test_report = "test_report" in event.payload
    has_tool_outcome = "tool_outcome" in event.payload
    if has_test_report and has_tool_outcome:
        raise FingerprintUnavailableError(AbstentionReason.STRUCTURED_EVIDENCE_INVALID)
    if has_test_report:
        report = parse_test_report(event)
        if report.status is not TestReportStatus.FAILED:
            raise FingerprintUnavailableError(AbstentionReason.EVENT_NOT_APPLICABLE)
        if any(
            failure.failure_type is None or failure.signature is None for failure in report.failures
        ):
            raise FingerprintUnavailableError(AbstentionReason.STRUCTURED_EVIDENCE_MISSING)
        test_components = [report.framework, str(len(report.failures))]
        for failure in report.failures:
            assert failure.failure_type is not None
            assert failure.signature is not None
            test_components.extend((failure.test_id, failure.failure_type, failure.signature))
        return FailureFingerprint(category="test", components=tuple(test_components))

    outcome = parse_tool_outcome(event)
    if outcome.status is not ToolOutcomeStatus.FAILED:
        raise FingerprintUnavailableError(AbstentionReason.EVENT_NOT_APPLICABLE)
    has_failure_identity = outcome.exit_status not in (None, 0) or any(
        item is not None
        for item in (outcome.exception_type, outcome.error_code, outcome.failure_signature)
    )
    if not has_failure_identity:
        raise FingerprintUnavailableError(AbstentionReason.STRUCTURED_EVIDENCE_MISSING)
    tool_components = (
        "exit:none" if outcome.exit_status is None else f"exit:{outcome.exit_status}",
        outcome.exception_type or "",
        outcome.error_code or "",
        outcome.failure_signature or "",
    )
    return FailureFingerprint(category="tool", components=tool_components)


__all__ = [
    "ActionFingerprint",
    "FailureFingerprint",
    "FingerprintUnavailableError",
    "NormalizedTestFailure",
    "ShellActionEvidence",
    "TestFailureEvidence",
    "TestReport",
    "TestReportEvidence",
    "TestReportStatus",
    "ToolOutcome",
    "ToolOutcomeEvidence",
    "ToolOutcomeStatus",
    "action_fingerprint",
    "classify_test_report",
    "classify_tool_outcome",
    "failure_fingerprint",
    "normalize_test_id",
    "parse_test_report",
    "parse_tool_outcome",
]
