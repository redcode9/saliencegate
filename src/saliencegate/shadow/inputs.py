from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Literal, TypeAlias
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from saliencegate.domain import (
    EventPhase,
    EventType,
    JsonObject,
    NormalizedTraceEventDraft,
    SignalType,
    TrustLabel,
    length_prefixed_sha256,
    trace_event_payload_is_bounded,
)
from saliencegate.domain.records import UUID4, ComponentIdentifier, PositiveInt, UtcDatetime
from saliencegate.shadow.errors import ShadowInputError
from saliencegate.signals.fingerprints import (
    ActionCommandText,
    ArgumentText,
    DirectoryText,
    ShellActionEvidence,
    ShortText,
    SignatureText,
    TestFailureEvidence,
    TestReportEvidence,
    ToolOutcomeEvidence,
)

_EVENT_ID_DOMAIN = "saliencegate:shadow:event-id:v1"
_SOURCE_EVENT_DOMAIN = "saliencegate:shadow:source-event:v1"
_MAX_TEST_FAILURES = 10_000
_MAX_TEST_REPORT_BYTES = 2 * 1_024 * 1_024
_SOURCE_EVENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
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


def _require_exact_source_event_id(value: str) -> str:
    if type(value) is not str or _SOURCE_EVENT_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("source event identifier is invalid")
    return value


SourceEventId = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$"),
    AfterValidator(_require_exact_source_event_id),
]


class _ShadowModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class ShadowInputKind(StrEnum):
    START = "start"
    ACTION = "action"
    TOOL_RESULT = "tool_result"
    TEST_RESULT = "test_result"
    OBSERVATION = "observation"
    CONTROLLER_ERROR = "controller_error"
    FINISH = "finish"


class ShadowObservationSource(StrEnum):
    TASK_INPUT = "task_input"
    TOOL_OUTPUT = "tool_output"
    MODEL_OUTPUT = "model_output"
    EXTERNAL_MEMORY = "external_memory"


class ShadowEventRef(_ShadowModel):
    schema_version: Literal["shadow-event-ref/v1"] = "shadow-event-ref/v1"
    run_id: Annotated[UUID4, Field(repr=False)]
    event_id: Annotated[UUID4, Field(repr=False)]
    sequence: Annotated[PositiveInt, Field(repr=False)]

    @field_validator("schema_version", mode="before")
    @classmethod
    def require_exact_schema_version(cls, value: object) -> object:
        if type(value) is not str:
            raise ValueError("event reference schema is invalid")
        return value


class _ShadowInputBase(_ShadowModel):
    schema_version: Literal["shadow-input/v1"] = "shadow-input/v1"
    source_event_id: Annotated[SourceEventId, Field(repr=False)]
    occurred_at: Annotated[UtcDatetime, Field(repr=False)]

    @field_validator("schema_version", mode="before")
    @classmethod
    def require_exact_schema_version(cls, value: object) -> object:
        if type(value) is not str:
            raise ValueError("shadow input schema is invalid")
        return value

    @field_validator("kind", mode="before", check_fields=False)
    @classmethod
    def require_exact_kind(cls, value: object) -> object:
        if type(value) is not str:
            raise ValueError("shadow input kind is invalid")
        return value

    @field_validator("source_event_id", mode="before")
    @classmethod
    def require_exact_source_event_id(cls, value: object) -> object:
        if type(value) is not str:
            raise ValueError("source event identifier is invalid")
        return value


class ShadowStartInput(_ShadowInputBase):
    kind: Literal["start"] = "start"


class ShadowActionInput(_ShadowInputBase):
    kind: Literal["action"] = "action"
    command: Annotated[ActionCommandText | None, Field(repr=False)] = None
    argv: Annotated[
        tuple[ArgumentText, ...] | None,
        Field(min_length=1, max_length=256, repr=False),
    ] = None
    working_directory: Annotated[DirectoryText, Field(repr=False)]
    environment_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", repr=False)]

    @field_validator("command", "working_directory", "environment_digest", mode="before")
    @classmethod
    def require_exact_action_text(cls, value: object) -> object:
        if value is not None and type(value) is not str:
            raise ValueError("action text is invalid")
        return value

    @field_validator("argv", mode="before")
    @classmethod
    def require_exact_argv(cls, value: object) -> object:
        if value is None:
            return value
        if type(value) is not tuple or any(type(item) is not str for item in value):
            raise ValueError("action arguments are invalid")
        return value

    @model_validator(mode="after")
    def validate_shell_evidence(self) -> ShadowActionInput:
        ShellActionEvidence(
            schema_version="1.0",
            kind="shell",
            command=self.command,
            argv=self.argv,
            working_directory=self.working_directory,
            environment_digest=self.environment_digest,
        )
        return self


class ShadowToolResultInput(_ShadowInputBase):
    kind: Literal["tool_result"] = "tool_result"
    action: Annotated[ShadowEventRef, Field(repr=False)]
    status: Literal["succeeded", "failed"] | None = None
    exit_status: Annotated[int | None, Field(ge=-(1 << 31), le=(1 << 31) - 1)] = None
    exception_type: Annotated[ShortText | None, Field(repr=False)] = None
    error_code: Annotated[ShortText | None, Field(repr=False)] = None
    failure_signature: Annotated[SignatureText | None, Field(repr=False)] = None

    @field_validator("status", mode="before")
    @classmethod
    def require_exact_optional_tool_status(cls, value: object) -> object:
        if value is not None and type(value) is not str:
            raise ValueError("tool evidence status is invalid")
        return value

    @field_validator("exception_type", "error_code", "failure_signature", mode="before")
    @classmethod
    def require_exact_optional_tool_text(cls, value: object) -> object:
        if value is not None and type(value) is not str:
            raise ValueError("tool evidence text is invalid")
        return value

    @model_validator(mode="after")
    def validate_tool_evidence(self) -> ShadowToolResultInput:
        ToolOutcomeEvidence(
            schema_version="1.0",
            status=self.status,
            exit_status=self.exit_status,
            exception_type=self.exception_type,
            error_code=self.error_code,
            failure_signature=self.failure_signature,
        )
        return self


class ShadowTestResultInput(_ShadowInputBase):
    kind: Literal["test_result"] = "test_result"
    action: Annotated[ShadowEventRef, Field(repr=False)]
    framework: Annotated[ShortText, Field(repr=False)]
    status: Literal["passed", "failed"]
    failures: Annotated[
        tuple[TestFailureEvidence, ...],
        Field(max_length=10_000, repr=False),
    ]

    @field_validator("framework", mode="before")
    @classmethod
    def require_exact_framework(cls, value: object) -> object:
        if type(value) is not str:
            raise ValueError("test framework is invalid")
        return value

    @field_validator("status", mode="before")
    @classmethod
    def require_exact_test_status(cls, value: object) -> object:
        if type(value) is not str:
            raise ValueError("test status is invalid")
        return value

    @field_validator("failures", mode="before")
    @classmethod
    def defensively_copy_failures(cls, value: object) -> object:
        if type(value) is not tuple or len(value) > _MAX_TEST_FAILURES:
            raise ValueError("test failure evidence is invalid")

        size_bound = 2 + len(value)
        allowed_fields = frozenset({"schema_version", "test_id", "failure_type", "signature"})
        for item in value:
            if type(item) is TestFailureEvidence:
                if (
                    type(item.schema_version) is not str
                    or item.schema_version != "1.0"
                    or type(item.test_id) is not str
                    or (item.failure_type is not None and type(item.failure_type) is not str)
                    or (item.signature is not None and type(item.signature) is not str)
                ):
                    raise ValueError("test failure evidence is invalid")
                test_id = item.test_id
                failure_type = item.failure_type
                signature = item.signature
            elif type(item) is dict:
                if len(item) > len(allowed_fields) or not set(item).issubset(allowed_fields):
                    raise ValueError("test failure evidence is invalid")
                if (
                    type(item.get("schema_version")) is not str
                    or item.get("schema_version") != "1.0"
                    or type(item.get("test_id")) is not str
                ):
                    raise ValueError("test failure evidence is invalid")
                failure_type = item.get("failure_type")
                signature = item.get("signature")
                if (failure_type is not None and type(failure_type) is not str) or (
                    signature is not None and type(signature) is not str
                ):
                    raise ValueError("test failure evidence is invalid")
                test_id = item["test_id"]
                assert isinstance(test_id, str)
            else:
                raise ValueError("test failure evidence is invalid")

            size_bound += 6 * len(test_id) + 128
            if failure_type is not None:
                size_bound += 6 * len(failure_type) + 2
            if signature is not None:
                size_bound += 6 * len(signature) + 2
            if size_bound > _MAX_TEST_REPORT_BYTES:
                raise ValueError("test failure evidence is invalid")

        copied: list[TestFailureEvidence] = []
        for item in value:
            candidate: object
            if type(item) is TestFailureEvidence:
                candidate = TestFailureEvidence.__pydantic_serializer__.to_python(
                    item,
                    mode="python",
                    warnings=False,
                )
            elif type(item) is dict:
                for field_name in ("schema_version", "test_id"):
                    if type(item.get(field_name)) is not str:
                        raise ValueError("test failure evidence is invalid")
                for field_name in ("failure_type", "signature"):
                    field_value = item.get(field_name)
                    if field_value is not None and type(field_value) is not str:
                        raise ValueError("test failure evidence is invalid")
                candidate = dict(item)
            else:
                raise ValueError("test failure evidence is invalid")
            copied.append(TestFailureEvidence.model_validate(candidate))
        return tuple(copied)

    @model_validator(mode="after")
    def validate_test_evidence(self) -> ShadowTestResultInput:
        TestReportEvidence(
            schema_version="1.0",
            framework=self.framework,
            status=self.status,
            failures=self.failures,
        )
        return self


class ShadowObservationInput(_ShadowInputBase):
    kind: Literal["observation"] = "observation"
    source: ShadowObservationSource
    payload: Annotated[JsonObject, Field(repr=False)]

    @field_validator("payload", mode="before")
    @classmethod
    def validate_observation_payload(cls, value: object) -> object:
        if not isinstance(value, Mapping) or not trace_event_payload_is_bounded(value):
            raise ValueError("observation payload is invalid")
        if any(key in _RESERVED_OBSERVATION_KEYS for key in value):
            raise ValueError("observation payload uses a reserved namespace")
        return value


class ShadowControllerErrorInput(_ShadowInputBase):
    kind: Literal["controller_error"] = "controller_error"
    error_code: Annotated[ComponentIdentifier, Field(repr=False)]

    @field_validator("error_code", mode="before")
    @classmethod
    def require_exact_error_code(cls, value: object) -> object:
        if type(value) is not str:
            raise ValueError("controller error code is invalid")
        return value


class ShadowFinishInput(_ShadowInputBase):
    kind: Literal["finish"] = "finish"


ShadowInputRecord: TypeAlias = (
    ShadowStartInput
    | ShadowActionInput
    | ShadowToolResultInput
    | ShadowTestResultInput
    | ShadowObservationInput
    | ShadowControllerErrorInput
    | ShadowFinishInput
)


class ShadowProjectionSpec(_ShadowModel):
    event_type: EventType
    phase: EventPhase
    trust_label: TrustLabel
    payload_namespace: ComponentIdentifier
    parent: Literal["none", "action"]
    applicable_detectors: tuple[SignalType, ...]


SHADOW_PROJECTION_MATRIX: Mapping[ShadowInputKind, ShadowProjectionSpec] = MappingProxyType(
    {
        ShadowInputKind.START: ShadowProjectionSpec(
            event_type=EventType.RUN_START,
            phase=EventPhase.INITIALIZATION,
            trust_label=TrustLabel.TRUSTED_CONTROLLER,
            payload_namespace="shadow_run",
            parent="none",
            applicable_detectors=(),
        ),
        ShadowInputKind.ACTION: ShadowProjectionSpec(
            event_type=EventType.ACTION_PROPOSAL,
            phase=EventPhase.PRE_ACTION,
            trust_label=TrustLabel.UNTRUSTED_MODEL_OUTPUT,
            payload_namespace="action",
            parent="none",
            applicable_detectors=(SignalType.REPEATED_ACTION,),
        ),
        ShadowInputKind.TOOL_RESULT: ShadowProjectionSpec(
            event_type=EventType.TOOL_COMPLETION,
            phase=EventPhase.POST_ACTION,
            trust_label=TrustLabel.UNTRUSTED_TOOL_OUTPUT,
            payload_namespace="tool_outcome",
            parent="action",
            applicable_detectors=(SignalType.TOOL_ERROR, SignalType.REPEATED_FAILURE),
        ),
        ShadowInputKind.TEST_RESULT: ShadowProjectionSpec(
            event_type=EventType.OBSERVATION,
            phase=EventPhase.POST_ACTION,
            trust_label=TrustLabel.UNTRUSTED_TOOL_OUTPUT,
            payload_namespace="test_report",
            parent="action",
            applicable_detectors=(SignalType.TEST_FAILURE, SignalType.REPEATED_FAILURE),
        ),
        ShadowInputKind.OBSERVATION: ShadowProjectionSpec(
            event_type=EventType.OBSERVATION,
            phase=EventPhase.POST_ACTION,
            trust_label=TrustLabel.UNTRUSTED_TASK_INPUT,
            payload_namespace="observation",
            parent="none",
            applicable_detectors=(),
        ),
        ShadowInputKind.CONTROLLER_ERROR: ShadowProjectionSpec(
            event_type=EventType.CONTROLLER_ERROR,
            phase=EventPhase.INTERNAL,
            trust_label=TrustLabel.TRUSTED_CONTROLLER,
            payload_namespace="controller_error",
            parent="none",
            applicable_detectors=(SignalType.TOOL_ERROR,),
        ),
        ShadowInputKind.FINISH: ShadowProjectionSpec(
            event_type=EventType.RUN_END,
            phase=EventPhase.TERMINAL,
            trust_label=TrustLabel.TRUSTED_CONTROLLER,
            payload_namespace="shadow_run_end",
            parent="none",
            applicable_detectors=(),
        ),
    }
)

_OBSERVATION_TRUST_LABELS: Mapping[ShadowObservationSource, TrustLabel] = MappingProxyType(
    {
        ShadowObservationSource.TASK_INPUT: TrustLabel.UNTRUSTED_TASK_INPUT,
        ShadowObservationSource.TOOL_OUTPUT: TrustLabel.UNTRUSTED_TOOL_OUTPUT,
        ShadowObservationSource.MODEL_OUTPUT: TrustLabel.UNTRUSTED_MODEL_OUTPUT,
        ShadowObservationSource.EXTERNAL_MEMORY: TrustLabel.UNTRUSTED_EXTERNAL_MEMORY,
    }
)


def _identity_inputs_are_valid(run_id: object, source_event_id: object) -> bool:
    return (
        type(run_id) is UUID
        and run_id.version == 4
        and type(source_event_id) is str
        and _SOURCE_EVENT_ID_PATTERN.fullmatch(source_event_id) is not None
    )


def derive_shadow_event_id(run_id: UUID, source_event_id: str) -> UUID:
    """Derive a stable RFC 4122 UUID4 from one run-bound source identifier."""

    if not _identity_inputs_are_valid(run_id, source_event_id):
        raise ShadowInputError()
    digest = length_prefixed_sha256(
        str(run_id),
        source_event_id,
        domain=_EVENT_ID_DOMAIN,
    )
    raw = bytearray(bytes.fromhex(digest)[:16])
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return UUID(bytes=bytes(raw))


def derive_shadow_source_event_digest(run_id: UUID, source_event_id: str) -> str:
    """Bind an opaque source identifier to one run without exposing it."""

    if not _identity_inputs_are_valid(run_id, source_event_id):
        raise ShadowInputError()
    return length_prefixed_sha256(
        str(run_id),
        source_event_id,
        domain=_SOURCE_EVENT_DOMAIN,
    )


def _validated_input(value: object) -> ShadowInputRecord:
    model_type = type(value)
    if model_type not in (
        ShadowStartInput,
        ShadowActionInput,
        ShadowToolResultInput,
        ShadowTestResultInput,
        ShadowObservationInput,
        ShadowControllerErrorInput,
        ShadowFinishInput,
    ):
        raise ValueError("unsupported shadow input")
    return model_type.model_validate(value)


def _copy_marker_payload(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or not trace_event_payload_is_bounded(value):
        raise ValueError("shadow marker payload is invalid")
    return dict(value)


def _project_validated_input(
    value: ShadowInputRecord,
    *,
    run_id: UUID,
    source_adapter: str,
    start_payload: Mapping[str, object] | None,
    finish_payload: Mapping[str, object] | None,
) -> NormalizedTraceEventDraft:
    if type(source_adapter) is not str:
        raise ValueError("source adapter is invalid")
    kind = ShadowInputKind(value.kind)
    spec = SHADOW_PROJECTION_MATRIX[kind]
    payload: dict[str, object]
    parent_ids: tuple[UUID, ...] = ()
    trust_label = spec.trust_label

    if type(value) is ShadowStartInput:
        if start_payload is None or finish_payload is not None:
            raise ValueError("start marker payload is invalid")
        payload = {spec.payload_namespace: _copy_marker_payload(start_payload)}
    elif type(value) is ShadowFinishInput:
        if finish_payload is None or start_payload is not None:
            raise ValueError("finish marker payload is invalid")
        payload = {spec.payload_namespace: _copy_marker_payload(finish_payload)}
    else:
        if start_payload is not None or finish_payload is not None:
            raise ValueError("marker payload is misplaced")
        if type(value) is ShadowActionInput:
            action_evidence = ShellActionEvidence(
                schema_version="1.0",
                kind="shell",
                command=value.command,
                argv=value.argv,
                working_directory=value.working_directory,
                environment_digest=value.environment_digest,
            )
            payload = {spec.payload_namespace: action_evidence.model_dump(mode="json")}
        elif type(value) is ShadowToolResultInput:
            if value.action.run_id != run_id:
                raise ValueError("tool result parent belongs to another run")
            tool_evidence = ToolOutcomeEvidence(
                schema_version="1.0",
                status=value.status,
                exit_status=value.exit_status,
                exception_type=value.exception_type,
                error_code=value.error_code,
                failure_signature=value.failure_signature,
            )
            payload = {spec.payload_namespace: tool_evidence.model_dump(mode="json")}
            parent_ids = (value.action.event_id,)
        elif type(value) is ShadowTestResultInput:
            if value.action.run_id != run_id:
                raise ValueError("test result parent belongs to another run")
            test_evidence = TestReportEvidence(
                schema_version="1.0",
                framework=value.framework,
                status=value.status,
                failures=value.failures,
            )
            payload = {spec.payload_namespace: test_evidence.model_dump(mode="json")}
            parent_ids = (value.action.event_id,)
        elif type(value) is ShadowObservationInput:
            payload = {
                spec.payload_namespace: dict(value.payload),
            }
            trust_label = _OBSERVATION_TRUST_LABELS[value.source]
        elif type(value) is ShadowControllerErrorInput:
            payload = {
                spec.payload_namespace: {
                    "schema_version": "controller_error/v1",
                    "error_code": value.error_code,
                }
            }
        else:  # pragma: no cover - exhaustive exact-type validation above
            raise ValueError("unsupported shadow input")

    return NormalizedTraceEventDraft(
        run_id=run_id,
        source_event_id=value.source_event_id,
        timestamp=value.occurred_at,
        event_type=spec.event_type,
        phase=spec.phase,
        payload=payload,
        parent_ids=parent_ids,
        source_adapter=source_adapter,
        trust_label=trust_label,
    )


def project_shadow_input(
    value: object,
    *,
    run_id: UUID,
    source_adapter: str,
    start_payload: Mapping[str, object] | None = None,
    finish_payload: Mapping[str, object] | None = None,
) -> NormalizedTraceEventDraft:
    """Project one strict public input into its frozen normalized trace shape."""

    projected: NormalizedTraceEventDraft | None = None
    try:
        validated = _validated_input(value)
        projected = _project_validated_input(
            validated,
            run_id=run_id,
            source_adapter=source_adapter,
            start_payload=start_payload,
            finish_payload=finish_payload,
        )
    except Exception:
        pass
    if projected is None:
        raise ShadowInputError()
    return projected


__all__ = [
    "SHADOW_PROJECTION_MATRIX",
    "ShadowActionInput",
    "ShadowControllerErrorInput",
    "ShadowEventRef",
    "ShadowFinishInput",
    "ShadowInputKind",
    "ShadowInputRecord",
    "ShadowObservationInput",
    "ShadowObservationSource",
    "ShadowProjectionSpec",
    "ShadowStartInput",
    "ShadowTestResultInput",
    "ShadowToolResultInput",
    "derive_shadow_event_id",
    "derive_shadow_source_event_digest",
    "project_shadow_input",
]
