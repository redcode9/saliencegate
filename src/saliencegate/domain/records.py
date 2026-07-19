from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Annotated, Literal, Self, TypeAlias
from uuid import UUID

from pydantic import UUID4 as PYDANTIC_UUID4
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    StringConstraints,
    field_validator,
    model_validator,
)

from saliencegate.domain.enums import (
    ClaimKind,
    ConstraintStatus,
    CycleState,
    DeduplicationGuarantee,
    DeliveryOutcome,
    DeliveryState,
    DeliveryTarget,
    EventPhase,
    EventType,
    EvidenceSource,
    ExpirationAction,
    InterventionAction,
    MemoryKind,
    OutcomeEvidenceMode,
    PayloadDigestAlgorithm,
    ReasonCode,
    RepeatedErrorStatus,
    SignalType,
    TrustLabel,
    UtilityLabel,
    ValidityState,
)
from saliencegate.domain.ids import cycle_id as derive_cycle_id

CURRENT_SCHEMA_VERSION: Literal["1.0"] = "1.0"
SUPPORTED_SCHEMA_VERSIONS = (CURRENT_SCHEMA_VERSION,)


def _require_exact_uuid(value: UUID) -> UUID:
    if type(value) is not UUID:
        raise ValueError("UUID subclasses are not accepted")
    return UUID(int=value.int)


UUID4 = Annotated[PYDANTIC_UUID4, AfterValidator(_require_exact_uuid)]


def _require_exact_string(value: str) -> str:
    if type(value) is not str:
        raise ValueError("string subclasses are not accepted")
    return value


NonEmptyString = Annotated[
    str,
    StringConstraints(min_length=1),
    AfterValidator(_require_exact_string),
]
EventMetadataIdentifier = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9_][A-Za-z0-9._:/+\-]*$",
    ),
    AfterValidator(_require_exact_string),
]
ComponentIdentifier = EventMetadataIdentifier
Sha256Digest = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
    AfterValidator(_require_exact_string),
]
JsonPointer = Annotated[
    str,
    StringConstraints(max_length=1024, pattern=r"^(?:/(?:[^~/]|~[01])*)+$"),
    AfterValidator(_require_exact_string),
]
MAX_TRACE_EVENT_PAYLOAD_BYTES = 1024 * 1024
MAX_TRACE_EVENT_PAYLOAD_NODES = 50_000
MAX_TRACE_EVENT_PAYLOAD_DEPTH = 64
MAX_TRACE_EVENT_PARENTS = 64
MAX_MEMORY_CONTENT_BYTES = 64 * 1024
MAX_MEMORY_DELTA_ITEMS = 64
MAX_MEMORY_PROVENANCE_ITEMS = 8
MAX_SIGNAL_EVIDENCE_EVENTS = 64
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(ge=1)]
Signed64Offset = Annotated[int, Field(ge=0, le=(1 << 63) - 1)]
PositiveSigned64Offset = Annotated[int, Field(ge=1, le=(1 << 63) - 1)]


def _require_utc(value: datetime) -> datetime:
    if type(value) is not datetime:
        raise ValueError("datetime subclasses are not accepted")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware UTC")
    if value.utcoffset() != timedelta(0):
        raise ValueError("timestamp must use UTC rather than a non-zero offset")
    return value.astimezone(UTC)


UtcDatetime = Annotated[datetime, AfterValidator(_require_utc)]


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("JSON object keys must be strings")
            frozen[key] = _freeze_json(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, str):
        if type(value) is not str:
            raise ValueError("JSON string subclasses are not accepted")
        return value
    if value is None or type(value) in (bool, int):
        return value
    if isinstance(value, float):
        if type(value) is not float:
            raise ValueError("JSON number subclasses are not accepted")
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    raise ValueError(f"unsupported JSON value type: {type(value).__name__}")


def _freeze_json_object(value: Mapping[str, object]) -> Mapping[str, object]:
    frozen = _freeze_json(value)
    if not isinstance(frozen, Mapping):  # pragma: no cover - guarded by the field schema
        raise ValueError("expected a JSON object")
    return frozen


def _json_is_bounded(
    value: object,
    *,
    max_bytes: int,
    max_nodes: int,
    max_depth: int,
) -> bool:
    """Conservatively bound JSON before recursive freezing or serialization."""

    try:
        remaining = max_bytes
        nodes = 0
        stack = [(value, 0)]
        while stack:
            item, depth = stack.pop()
            nodes += 1
            if nodes > max_nodes or depth > max_depth:
                return False
            if isinstance(item, Mapping):
                declared_length = len(item)
                if declared_length > max_nodes - nodes - len(stack):
                    return False
                remaining -= 2 + declared_length * 2
                observed = 0
                for key, nested in item.items():
                    if observed >= declared_length or type(key) is not str:
                        return False
                    observed += 1
                    remaining -= len(key.encode("utf-8", errors="strict")) + 3
                    stack.append((nested, depth + 1))
                if observed != declared_length:
                    return False
            elif type(item) in (list, tuple):
                assert isinstance(item, (list, tuple))
                if len(item) > max_nodes - nodes - len(stack):
                    return False
                remaining -= 2 + len(item)
                stack.extend((nested, depth + 1) for nested in item)
            elif type(item) is str:
                remaining -= len(item.encode("utf-8", errors="strict")) + 2
            elif item is None or type(item) is bool:
                remaining -= 5
            elif type(item) is int:
                remaining -= max(2, item.bit_length() // 3 + 2)
            elif type(item) is float:
                if not math.isfinite(item):
                    return False
                remaining -= 32
            else:
                return False
            if remaining < 0:
                return False
        return True
    except Exception:
        return False


def _bound_trace_payload(value: object) -> object:
    if not trace_event_payload_is_bounded(value):
        raise ValueError(
            "trace event payload exceeds its structural bound, contains unsupported JSON, "
            "or has non-finite numbers"
        )
    return value


def trace_event_payload_is_bounded(value: object) -> bool:
    return _json_is_bounded(
        value,
        max_bytes=MAX_TRACE_EVENT_PAYLOAD_BYTES,
        max_nodes=MAX_TRACE_EVENT_PAYLOAD_NODES,
        max_depth=MAX_TRACE_EVENT_PAYLOAD_DEPTH,
    )


def _bound_memory_content(value: object) -> object:
    if type(value) is not str:
        return value
    try:
        if len(value.encode("utf-8", errors="strict")) <= MAX_MEMORY_CONTENT_BYTES:
            return value
    except UnicodeError:
        pass
    raise ValueError("memory content exceeds its UTF-8 byte bound")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _thaw_json_object(value: Mapping[str, object]) -> dict[str, object]:
    thawed = _thaw_json(value)
    if not isinstance(thawed, dict):  # pragma: no cover - guarded by the field schema
        raise TypeError("expected a JSON object")
    return thawed


JsonObject: TypeAlias = Annotated[
    Mapping[str, object],
    AfterValidator(_freeze_json_object),
    PlainSerializer(_thaw_json_object, return_type=dict[str, object]),
]


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
        validate_default=True,
    )


class VersionedRecord(FrozenModel):
    schema_version: Literal["1.0"] = CURRENT_SCHEMA_VERSION


class TextSpan(FrozenModel):
    start_byte: Signed64Offset
    end_byte: PositiveSigned64Offset

    @model_validator(mode="after")
    def end_follows_start(self) -> Self:
        if self.end_byte <= self.start_byte:
            raise ValueError("span end must be greater than start")
        return self


class EvidenceReference(FrozenModel):
    source: EvidenceSource
    source_id: UUID4
    revision: PositiveInt | None = None
    field_path: JsonPointer
    span: TextSpan | None = None

    @model_validator(mode="after")
    def identify_exact_evidence(self) -> Self:
        if self.source is EvidenceSource.MEMORY and self.revision is None:
            raise ValueError("memory evidence must identify a revision")
        if self.source is EvidenceSource.EVENT and self.revision is not None:
            raise ValueError("event evidence cannot carry a memory revision")
        return self


def evidence_reference_is_bounded(value: object) -> bool:
    try:
        return (
            type(value) is EvidenceReference
            and type(value.source) is EvidenceSource
            and type(value.source_id) is UUID
            and value.source_id.version == 4
            and (
                value.revision is None
                or (type(value.revision) is int and 1 <= value.revision <= (1 << 63) - 1)
            )
            and type(value.field_path) is str
            and len(value.field_path) <= 1_024
            and len(value.field_path.encode("utf-8", errors="strict")) <= 4_096
            and (value.span is None or type(value.span) is TextSpan)
        )
    except Exception:
        return False


class PayloadDigest(FrozenModel):
    algorithm: PayloadDigestAlgorithm
    value: Sha256Digest


class BudgetAmounts(FrozenModel):
    model_calls: NonNegativeInt = 0
    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0
    canonical_token_equivalents: NonNegativeInt = 0
    latency_us: NonNegativeInt = 0
    interventions: NonNegativeInt = 0
    schema_repairs: NonNegativeInt = 0


class BudgetLimits(BudgetAmounts):
    max_call_latency_us: NonNegativeInt = 0


class BudgetSnapshot(FrozenModel):
    limits: BudgetLimits
    reserved: BudgetAmounts
    consumed: BudgetAmounts

    @model_validator(mode="after")
    def stay_within_limits(self) -> Self:
        fields = (
            "model_calls",
            "input_tokens",
            "output_tokens",
            "canonical_token_equivalents",
            "latency_us",
            "interventions",
            "schema_repairs",
        )
        for field_name in fields:
            limit = getattr(self.limits, field_name)
            allocated = getattr(self.reserved, field_name) + getattr(self.consumed, field_name)
            if allocated > limit:
                raise ValueError(f"reserved and consumed {field_name} exceed the limit")
        return self


class _TraceEventDraftBase(VersionedRecord):
    run_id: UUID4
    source_event_id: EventMetadataIdentifier
    timestamp: UtcDatetime
    event_type: EventType
    phase: EventPhase
    payload: JsonObject = Field(repr=False)
    parent_ids: Annotated[tuple[UUID4, ...], Field(max_length=MAX_TRACE_EVENT_PARENTS)] = ()
    source_adapter: EventMetadataIdentifier
    trust_label: TrustLabel

    @field_validator("payload", mode="before")
    @classmethod
    def bound_payload(cls, value: object) -> object:
        return _bound_trace_payload(value)

    @field_validator("parent_ids")
    @classmethod
    def canonicalize_parent_set(cls, value: tuple[UUID, ...]) -> tuple[UUID, ...]:
        return tuple(sorted(value, key=str))

    @model_validator(mode="after")
    def parent_ids_are_unique(self) -> Self:
        if len(set(self.parent_ids)) != len(self.parent_ids):
            raise ValueError("parent event IDs must be unique")
        return self


class NormalizedTraceEventDraft(_TraceEventDraftBase):
    record_type: Literal["normalized_trace_event_draft"] = "normalized_trace_event_draft"


def normalized_trace_event_draft_is_bounded(value: object) -> bool:
    try:
        return (
            type(value) is NormalizedTraceEventDraft
            and type(value.run_id) is UUID
            and value.run_id.version == 4
            and type(value.source_event_id) is str
            and 0 < len(value.source_event_id) <= 256
            and type(value.timestamp) is datetime
            and type(value.event_type) is EventType
            and type(value.phase) is EventPhase
            and type(value.parent_ids) is tuple
            and len(value.parent_ids) <= MAX_TRACE_EVENT_PARENTS
            and all(
                type(parent_id) is UUID and parent_id.version == 4 for parent_id in value.parent_ids
            )
            and type(value.source_adapter) is str
            and 0 < len(value.source_adapter) <= 256
            and type(value.trust_label) is TrustLabel
            and trace_event_payload_is_bounded(value.payload)
        )
    except Exception:
        return False


class RedactedTraceEventDraft(_TraceEventDraftBase):
    record_type: Literal["redacted_trace_event_draft"] = "redacted_trace_event_draft"
    payload_digest: PayloadDigest


class TraceEvent(VersionedRecord):
    record_type: Literal["trace_event"] = "trace_event"
    event_id: UUID4
    run_id: UUID4
    sequence: PositiveInt
    source_event_id: EventMetadataIdentifier
    timestamp: UtcDatetime
    event_type: EventType
    phase: EventPhase
    payload: JsonObject
    payload_digest: PayloadDigest
    parent_ids: Annotated[tuple[UUID4, ...], Field(max_length=MAX_TRACE_EVENT_PARENTS)] = ()
    source_adapter: EventMetadataIdentifier
    trust_label: TrustLabel

    @field_validator("payload", mode="before")
    @classmethod
    def bound_payload(cls, value: object) -> object:
        return _bound_trace_payload(value)

    @field_validator("parent_ids")
    @classmethod
    def canonicalize_parent_set(cls, value: tuple[UUID, ...]) -> tuple[UUID, ...]:
        return tuple(sorted(value, key=str))

    @model_validator(mode="after")
    def parent_ids_are_unique(self) -> Self:
        if len(set(self.parent_ids)) != len(self.parent_ids):
            raise ValueError("parent event IDs must be unique")
        return self


class Signal(VersionedRecord):
    record_type: Literal["signal"] = "signal"
    signal_id: UUID4
    run_id: UUID4
    created_at: UtcDatetime
    signal_type: SignalType
    strength: UnitInterval
    evidence_event_ids: Annotated[
        tuple[UUID4, ...], Field(min_length=1, max_length=MAX_SIGNAL_EVIDENCE_EVENTS)
    ]
    detector_version: ComponentIdentifier
    reason_code: ReasonCode

    @model_validator(mode="after")
    def evidence_event_ids_are_unique(self) -> Self:
        if len(set(self.evidence_event_ids)) != len(self.evidence_event_ids):
            raise ValueError("signal evidence event IDs must be unique")
        return self

    @model_validator(mode="after")
    def reason_code_matches_signal_type(self) -> Self:
        if self.reason_code is not ReasonCode(self.signal_type.value):
            raise ValueError("signal reason_code must match signal_type")
        return self


class MemoryRecord(VersionedRecord):
    record_type: Literal["memory_record"] = "memory_record"
    memory_id: UUID4
    run_id: UUID4
    kind: MemoryKind
    content: NonEmptyString
    provenance: Annotated[tuple[EvidenceReference, ...], Field(min_length=1)]
    confidence: UnitInterval
    validity: ValidityState
    revision: PositiveInt
    created_at: UtcDatetime
    updated_at: UtcDatetime
    access_count: NonNegativeInt = 0
    last_accessed_at: UtcDatetime | None = None
    expires_at: UtcDatetime | None = None
    invalidated_at: UtcDatetime | None = None
    trust_label: TrustLabel

    @model_validator(mode="after")
    def timestamps_follow_creation(self) -> Self:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        if self.last_accessed_at is not None and self.last_accessed_at < self.created_at:
            raise ValueError("last_accessed_at cannot precede created_at")
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ValueError("expires_at must follow created_at")
        if self.validity is ValidityState.INVALIDATED and self.invalidated_at is None:
            raise ValueError("invalidated memory requires invalidated_at")
        if self.validity is not ValidityState.INVALIDATED and self.invalidated_at is not None:
            raise ValueError("only invalidated memory can carry invalidated_at")
        if self.invalidated_at is not None and self.invalidated_at < self.created_at:
            raise ValueError("invalidated_at cannot precede created_at")
        return self


_FORCED_INVOKE_REASON_CODES: frozenset[ReasonCode] = frozenset(
    {
        ReasonCode.POLICY_ALWAYS,
        ReasonCode.SCRIPTED_INVOKE,
        ReasonCode.BOOTSTRAP,
        ReasonCode.WATCHDOG,
        ReasonCode.HARD_SIGNAL,
        ReasonCode.RISK_THRESHOLD_MET,
    }
)
_FORCED_SILENCE_REASON_CODES: frozenset[ReasonCode] = frozenset(
    {
        ReasonCode.POLICY_NEVER,
        ReasonCode.SCRIPTED_SILENCE,
        ReasonCode.SCRIPT_EXHAUSTED,
        ReasonCode.BUDGET_EXHAUSTED,
        ReasonCode.COOLDOWN_ACTIVE,
        ReasonCode.RISK_BELOW_THRESHOLD,
    }
)
_RISK_THRESHOLD_REASON_CODES: frozenset[ReasonCode] = frozenset(
    {
        ReasonCode.RISK_THRESHOLD_MET,
        ReasonCode.RISK_BELOW_THRESHOLD,
    }
)
_SIGNAL_REASON_CODES: frozenset[ReasonCode] = frozenset(
    ReasonCode(signal_type.value) for signal_type in SignalType
)
_INVOCATION_REASON_CODES = (
    _FORCED_INVOKE_REASON_CODES | _FORCED_SILENCE_REASON_CODES | _SIGNAL_REASON_CODES
)


class InvocationDecision(VersionedRecord):
    record_type: Literal["invocation_decision"] = "invocation_decision"
    decision_id: UUID4
    run_id: UUID4
    event_sequence: PositiveInt
    invoke: bool
    risk_score: UnitInterval | None
    reason_codes: Annotated[
        tuple[ReasonCode, ...],
        Field(min_length=1, max_length=len(_INVOCATION_REASON_CODES)),
    ]
    policy_version: ComponentIdentifier
    configuration_digest: Sha256Digest
    budget_snapshot: BudgetSnapshot
    cooldown_active: bool
    created_at: UtcDatetime

    @model_validator(mode="after")
    def reasons_match_decision(self) -> Self:
        reasons = set(self.reason_codes)
        if len(reasons) != len(self.reason_codes):
            raise ValueError("invocation reason_codes must be unique")

        unsupported = reasons - _INVOCATION_REASON_CODES
        if unsupported:
            raise ValueError("reason_codes contain a non-invocation reason")
        if not self.invoke and reasons & _FORCED_INVOKE_REASON_CODES:
            raise ValueError("forced-invocation reason requires invoke=True")
        if self.invoke and reasons & _FORCED_SILENCE_REASON_CODES:
            raise ValueError("forced-silence reason requires invoke=False")
        if self.risk_score is None and reasons & _RISK_THRESHOLD_REASON_CODES:
            raise ValueError("risk-threshold reason requires risk_score")
        if ReasonCode.COOLDOWN_ACTIVE in reasons and not self.cooldown_active:
            raise ValueError("cooldown_active reason requires cooldown_active=True")
        return self


class MemoryCreate(FrozenModel):
    handle: Annotated[NonEmptyString, Field(max_length=256)]
    kind: MemoryKind
    content: NonEmptyString
    provenance: Annotated[
        tuple[EvidenceReference, ...],
        Field(min_length=1, max_length=MAX_MEMORY_PROVENANCE_ITEMS),
    ]
    confidence: UnitInterval
    trust_label: TrustLabel
    expires_at: UtcDatetime | None = None

    @field_validator("content", mode="before")
    @classmethod
    def bound_content(cls, value: object) -> object:
        return _bound_memory_content(value)


class ExpirationPatch(FrozenModel):
    action: ExpirationAction = ExpirationAction.KEEP
    value: UtcDatetime | None = None

    @model_validator(mode="after")
    def match_action_and_value(self) -> Self:
        if self.action is ExpirationAction.SET and self.value is None:
            raise ValueError("setting expiration requires a timestamp")
        if self.action is not ExpirationAction.SET and self.value is not None:
            raise ValueError("only an expiration set operation can carry a timestamp")
        return self


class MemoryUpdate(FrozenModel):
    memory_id: UUID4
    expected_revision: PositiveInt
    content: NonEmptyString | None = None
    provenance: (
        Annotated[
            tuple[EvidenceReference, ...],
            Field(min_length=1, max_length=MAX_MEMORY_PROVENANCE_ITEMS),
        ]
        | None
    ) = None
    confidence: UnitInterval | None = None
    expiration: ExpirationPatch = ExpirationPatch()

    @field_validator("content", mode="before")
    @classmethod
    def bound_content(cls, value: object) -> object:
        return _bound_memory_content(value)

    @model_validator(mode="after")
    def include_a_change(self) -> Self:
        if (
            all(value is None for value in (self.content, self.provenance, self.confidence))
            and self.expiration.action is ExpirationAction.KEEP
        ):
            raise ValueError("memory update must include at least one changed field")
        return self


class MemoryInvalidation(FrozenModel):
    memory_id: UUID4
    expected_revision: PositiveInt
    reason_code: ReasonCode


class PrivateStatusReplacement(FrozenModel):
    expected_memory_id: UUID4 | None = None
    expected_revision: PositiveInt | None = None
    replacement: MemoryCreate

    @model_validator(mode="after")
    def validate_expected_revision_and_kind(self) -> Self:
        if (self.expected_memory_id is None) != (self.expected_revision is None):
            raise ValueError("expected private-status ID and revision must be supplied together")
        if self.replacement.kind is not MemoryKind.PRIVATE_STATUS:
            raise ValueError("private-status replacement must have kind private_status")
        return self


class MemoryDelta(VersionedRecord):
    record_type: Literal["memory_delta"] = "memory_delta"
    delta_id: UUID4
    run_id: UUID4
    creates: Annotated[tuple[MemoryCreate, ...], Field(max_length=MAX_MEMORY_DELTA_ITEMS)] = ()
    updates: Annotated[tuple[MemoryUpdate, ...], Field(max_length=MAX_MEMORY_DELTA_ITEMS)] = ()
    invalidations: Annotated[
        tuple[MemoryInvalidation, ...], Field(max_length=MAX_MEMORY_DELTA_ITEMS)
    ] = ()
    private_status_replacement: PrivateStatusReplacement | None = None
    created_at: UtcDatetime

    @model_validator(mode="after")
    def keep_private_status_in_its_replacement_channel(self) -> Self:
        if any(item.kind is MemoryKind.PRIVATE_STATUS for item in self.creates):
            raise ValueError("private status must use private_status_replacement")
        handles = [item.handle for item in self.creates]
        if self.private_status_replacement is not None:
            handles.append(self.private_status_replacement.replacement.handle)
        if len(set(handles)) != len(handles):
            raise ValueError("memory create handles must be unique within a delta")

        update_ids = [item.memory_id for item in self.updates]
        invalidation_ids = [item.memory_id for item in self.invalidations]
        if len(set(update_ids)) != len(update_ids):
            raise ValueError("a delta cannot update the same memory more than once")
        if len(set(invalidation_ids)) != len(invalidation_ids):
            raise ValueError("a delta cannot invalidate the same memory more than once")
        if set(update_ids).intersection(invalidation_ids):
            raise ValueError("a delta cannot both update and invalidate the same memory")
        return self


def memory_delta_is_bounded(value: object) -> bool:
    """Check model-controlled delta cardinality and text before reserialization."""

    try:
        if type(value) is not MemoryDelta:
            return False
        if (
            value.schema_version != CURRENT_SCHEMA_VERSION
            or value.record_type != "memory_delta"
            or type(value.delta_id) is not UUID
            or value.delta_id.version != 4
            or type(value.run_id) is not UUID
            or value.run_id.version != 4
            or type(value.created_at) is not datetime
        ):
            return False
        collections = (value.creates, value.updates, value.invalidations)
        if any(
            type(items) is not tuple or len(items) > MAX_MEMORY_DELTA_ITEMS for items in collections
        ):
            return False

        def valid_create(item: object) -> bool:
            return (
                type(item) is MemoryCreate
                and type(item.handle) is str
                and 0 < len(item.handle) <= 256
                and type(item.content) is str
                and len(item.content.encode("utf-8", errors="strict")) <= MAX_MEMORY_CONTENT_BYTES
                and type(item.provenance) is tuple
                and 1 <= len(item.provenance) <= MAX_MEMORY_PROVENANCE_ITEMS
                and all(evidence_reference_is_bounded(reference) for reference in item.provenance)
                and type(item.kind) is MemoryKind
                and type(item.confidence) is float
                and 0.0 <= item.confidence <= 1.0
                and type(item.trust_label) is TrustLabel
                and (item.expires_at is None or type(item.expires_at) is datetime)
            )

        if any(not valid_create(item) for item in value.creates):
            return False
        if any(
            type(item) is not MemoryUpdate
            or type(item.memory_id) is not UUID
            or item.memory_id.version != 4
            or type(item.expected_revision) is not int
            or not 1 <= item.expected_revision <= (1 << 63) - 1
            or (
                item.content is not None
                and (
                    type(item.content) is not str
                    or len(item.content.encode("utf-8", errors="strict")) > MAX_MEMORY_CONTENT_BYTES
                )
            )
            or (
                item.provenance is not None
                and (
                    type(item.provenance) is not tuple
                    or not 1 <= len(item.provenance) <= MAX_MEMORY_PROVENANCE_ITEMS
                    or any(
                        not evidence_reference_is_bounded(reference)
                        for reference in item.provenance
                    )
                )
            )
            or (
                item.confidence is not None
                and (type(item.confidence) is not float or not 0.0 <= item.confidence <= 1.0)
            )
            or type(item.expiration) is not ExpirationPatch
            or type(item.expiration.action) is not ExpirationAction
            or (item.expiration.value is not None and type(item.expiration.value) is not datetime)
            for item in value.updates
        ):
            return False
        if any(
            type(item) is not MemoryInvalidation
            or type(item.memory_id) is not UUID
            or item.memory_id.version != 4
            or type(item.expected_revision) is not int
            or not 1 <= item.expected_revision <= (1 << 63) - 1
            or type(item.reason_code) is not ReasonCode
            for item in value.invalidations
        ):
            return False
        replacement = value.private_status_replacement
        return replacement is None or (
            type(replacement) is PrivateStatusReplacement
            and (replacement.expected_memory_id is None) is (replacement.expected_revision is None)
            and (
                replacement.expected_memory_id is None
                or (
                    type(replacement.expected_memory_id) is UUID
                    and replacement.expected_memory_id.version == 4
                )
            )
            and (
                replacement.expected_revision is None
                or (
                    type(replacement.expected_revision) is int
                    and 1 <= replacement.expected_revision <= (1 << 63) - 1
                )
            )
            and valid_create(replacement.replacement)
            and replacement.replacement.kind is MemoryKind.PRIVATE_STATUS
        )
    except Exception:
        return False


class InterventionClaim(FrozenModel):
    kind: ClaimKind
    fields: JsonObject = Field(repr=False)
    evidence: Annotated[tuple[EvidenceReference, ...], Field(min_length=1, max_length=8)]

    @model_validator(mode="after")
    def reject_duplicate_evidence(self) -> Self:
        if len(set(self.evidence)) != len(self.evidence):
            raise ValueError("claim evidence cannot contain duplicates")
        return self


def _cited_ids(
    claims: tuple[InterventionClaim, ...],
    source: EvidenceSource,
) -> tuple[UUID, ...]:
    ordered: dict[UUID, None] = {}
    for claim in claims:
        for evidence in claim.evidence:
            if evidence.source is source:
                ordered[evidence.source_id] = None
    return tuple(ordered)


_SILENCE_REASON_CODES = frozenset(
    {
        ReasonCode.SILENCE_SELECTED,
        ReasonCode.NO_GROUNDED_CLAIMS,
        ReasonCode.SCHEMA_INVALID,
        ReasonCode.CLAIM_OVER_LIMIT,
        ReasonCode.CITATION_MISSING,
        ReasonCode.CITATION_CROSS_RUN,
        ReasonCode.CITATION_EXPIRED,
        ReasonCode.CITATION_INVALIDATED,
        ReasonCode.INVALID_PROVENANCE,
        ReasonCode.UNGROUNDED,
        ReasonCode.DUPLICATE_REMINDER,
        ReasonCode.COOLDOWN_BLOCKED,
        ReasonCode.UNSUPPORTED_DELIVERY_TARGET,
    }
)


class InterventionDecision(VersionedRecord):
    record_type: Literal["intervention_decision"] = "intervention_decision"
    intervention_id: UUID4
    run_id: UUID4
    cycle_id: Sha256Digest
    grounding_version: ComponentIdentifier
    grounding_configuration: JsonObject = Field(repr=False)
    grounding_configuration_digest: Sha256Digest
    grounding_receipt: JsonObject = Field(repr=False)
    action: InterventionAction
    delivery_target: DeliveryTarget | None = None
    claims: Annotated[tuple[InterventionClaim, ...], Field(max_length=2)] = ()
    rendered_text: NonEmptyString | None = Field(default=None, repr=False)
    cited_memory_ids: tuple[UUID4, ...] = ()
    cited_event_ids: tuple[UUID4, ...] = ()
    confidence: UnitInterval
    reason_code: ReasonCode
    ttl_steps: NonNegativeInt = 0
    created_at: UtcDatetime

    @model_validator(mode="after")
    def match_action_to_delivery(self) -> Self:
        if self.cited_memory_ids != _cited_ids(self.claims, EvidenceSource.MEMORY):
            raise ValueError("cited_memory_ids must match structured claim evidence")
        if self.cited_event_ids != _cited_ids(self.claims, EvidenceSource.EVENT):
            raise ValueError("cited_event_ids must match structured claim evidence")
        if self.action is InterventionAction.SILENCE:
            if self.delivery_target is not None or self.rendered_text is not None:
                raise ValueError("silence cannot carry a delivery target or rendered text")
            if self.claims or self.cited_memory_ids or self.cited_event_ids:
                raise ValueError("silence cannot carry claims or citation indexes")
            if self.reason_code not in _SILENCE_REASON_CODES:
                raise ValueError("silence must use an allowlisted grounding reason")
            if self.ttl_steps != 0:
                raise ValueError("silence must have a zero-step time-to-live")
            return self

        if self.reason_code is not ReasonCode.GROUNDED_REMINDER:
            raise ValueError("a reminder must use the grounded_reminder reason")
        if self.delivery_target is None:
            raise ValueError("a reminder requires a delivery target")
        if not self.claims:
            raise ValueError("a reminder requires at least one structured claim")
        if self.rendered_text is None:
            raise ValueError("a reminder requires deterministically rendered text")
        if self.ttl_steps != 1:
            raise ValueError("a reminder requires a one-step time-to-live")
        return self


class InterventionOutcome(VersionedRecord):
    record_type: Literal["intervention_outcome"] = "intervention_outcome"
    outcome_id: UUID4
    run_id: UUID4
    intervention_id: UUID4
    next_action_fingerprint: Sha256Digest | None = None
    repeated_error_status: RepeatedErrorStatus
    constraint_status: ConstraintStatus
    evidence_mode: OutcomeEvidenceMode
    utility: UtilityLabel | None = None
    action_changed: bool | None = None
    task_reward: FiniteFloat | None = None
    task_passed: bool | None = None
    steps: NonNegativeInt = 0
    tool_calls: NonNegativeInt = 0
    memory_calls: NonNegativeInt = 0
    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0
    canonical_token_equivalents: NonNegativeInt = 0
    latency_us: NonNegativeInt = 0
    created_at: UtcDatetime

    @model_validator(mode="after")
    def keep_policy_replay_noncausal(self) -> Self:
        if self.evidence_mode is OutcomeEvidenceMode.POLICY_REPLAY and self.utility is not None:
            raise ValueError("policy replay cannot assign a causal utility label")
        return self


class MemoryIdAssignment(FrozenModel):
    handle: NonEmptyString
    memory_id: UUID4


class CycleRecord(VersionedRecord):
    record_type: Literal["cycle_record"] = "cycle_record"
    cycle_id: Sha256Digest
    run_id: UUID4
    revision: PositiveInt
    invocation_decision_id: UUID4
    policy_version: ComponentIdentifier
    configuration_digest: Sha256Digest
    grounding_version: ComponentIdentifier
    grounding_configuration: JsonObject = Field(repr=False)
    grounding_configuration_digest: Sha256Digest
    requested_delivery_target: DeliveryTarget | None
    first_event_sequence: PositiveInt
    last_event_sequence: PositiveInt
    state: CycleState
    budget_reservation: BudgetAmounts | None = None
    budget_settlement: BudgetAmounts | None = None
    batch_digest: Sha256Digest | None = None
    model_call_digests: tuple[Sha256Digest, ...] = ()
    model_call_latencies_us: tuple[NonNegativeInt, ...] = ()
    validated_delta: MemoryDelta | None = None
    memory_id_assignments: tuple[MemoryIdAssignment, ...] = ()
    intervention: InterventionDecision | None = None
    selector_provenance: JsonObject | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        repr=False,
    )
    failure_reason: ReasonCode | None = None
    created_at: UtcDatetime
    updated_at: UtcDatetime

    @model_validator(mode="after")
    def validate_range_and_embedded_run(self) -> Self:
        if self.last_event_sequence < self.first_event_sequence:
            raise ValueError("last_event_sequence cannot precede first_event_sequence")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        expected_cycle_id = derive_cycle_id(
            self.run_id,
            self.first_event_sequence,
            self.last_event_sequence,
            self.policy_version,
            self.configuration_digest,
            self.grounding_version,
            self.grounding_configuration_digest,
            self.requested_delivery_target,
        )
        if self.cycle_id != expected_cycle_id:
            raise ValueError("cycle_id does not match the cycle identity fields")
        if self.validated_delta is not None and self.validated_delta.run_id != self.run_id:
            raise ValueError("validated delta belongs to a different run")
        if self.intervention is not None and self.intervention.run_id != self.run_id:
            raise ValueError("intervention belongs to a different run")
        if self.intervention is not None and self.intervention.cycle_id != self.cycle_id:
            raise ValueError("intervention belongs to a different cycle")
        if self.intervention is not None and (
            self.intervention.grounding_version != self.grounding_version
            or self.intervention.grounding_configuration != self.grounding_configuration
            or self.intervention.grounding_configuration_digest
            != self.grounding_configuration_digest
        ):
            raise ValueError("intervention grounding pin does not match the cycle")
        if (
            self.intervention is not None
            and self.intervention.action is InterventionAction.REMIND
            and self.intervention.delivery_target is not self.requested_delivery_target
        ):
            raise ValueError("intervention delivery target does not match the cycle request")
        if len(self.model_call_digests) != len(self.model_call_latencies_us):
            raise ValueError("model-call digests and latencies must have equal length")
        if self.budget_reservation is not None and self.budget_reservation.model_calls < 1:
            raise ValueError("cycle reservation requires a model call")
        if self.budget_settlement is not None and self.budget_reservation is not None:
            budget_fields = (
                "model_calls",
                "input_tokens",
                "output_tokens",
                "canonical_token_equivalents",
                "latency_us",
                "interventions",
                "schema_repairs",
            )
            if any(
                getattr(self.budget_settlement, field_name)
                > getattr(self.budget_reservation, field_name)
                for field_name in budget_fields
            ):
                raise ValueError("cycle settlement cannot exceed its reservation")
            if sum(self.model_call_latencies_us) > self.budget_settlement.latency_us:
                raise ValueError("model-call latency exceeds settled cycle latency")

        assignment_handles = tuple(item.handle for item in self.memory_id_assignments)
        assignment_ids = tuple(item.memory_id for item in self.memory_id_assignments)
        if len(set(assignment_handles)) != len(assignment_handles):
            raise ValueError("memory ID assignment handles must be unique")
        if len(set(assignment_ids)) != len(assignment_ids):
            raise ValueError("assigned memory IDs must be unique")
        expected_handles: tuple[str, ...] = ()
        if self.validated_delta is not None:
            expected_handles = tuple(item.handle for item in self.validated_delta.creates)
            replacement = self.validated_delta.private_status_replacement
            if replacement is not None:
                expected_handles += (replacement.replacement.handle,)
        if assignment_handles != expected_handles:
            raise ValueError("memory ID assignments must exactly match created handles")

        committed_outputs = (
            self.budget_settlement,
            self.validated_delta,
            self.intervention,
        )
        if self.state is not CycleState.COMMITTED and self.selector_provenance is not None:
            raise ValueError("only a committed cycle can bind selector provenance")
        if self.state is CycleState.PENDING:
            if self.batch_digest is not None:
                raise ValueError("pending cycle cannot carry a batch digest")
            if (
                self.budget_reservation is not None
                or any(item is not None for item in committed_outputs)
                or self.memory_id_assignments
            ):
                raise ValueError("pending cycle cannot carry reservations or committed outputs")
            if self.failure_reason is not None:
                raise ValueError("pending cycle cannot carry a failure reason")
            if self.model_call_digests or self.model_call_latencies_us:
                raise ValueError("pending cycle cannot carry model-call receipts")
        elif self.state in (CycleState.RESERVED, CycleState.RUNNING):
            if self.state is CycleState.RUNNING and self.batch_digest is None:
                raise ValueError("running cycle requires a batch digest")
            if self.budget_reservation is None:
                raise ValueError("reserved or running cycle requires a budget reservation")
            if any(item is not None for item in committed_outputs) or self.memory_id_assignments:
                raise ValueError("uncommitted cycle cannot carry committed outputs")
            if self.failure_reason is not None:
                raise ValueError("uncommitted cycle cannot carry a failure reason")
            if self.model_call_digests or self.model_call_latencies_us:
                raise ValueError("uncommitted cycle cannot carry model-call receipts")
        elif self.state is CycleState.COMMITTED:
            if self.batch_digest is None:
                raise ValueError("committed cycle requires a batch digest")
            if self.budget_reservation is None or any(item is None for item in committed_outputs):
                raise ValueError("committed cycle requires reservation, settlement, and verdicts")
            if self.failure_reason is not None:
                raise ValueError("committed cycle cannot carry a failure reason")
            if self.budget_settlement is None or self.intervention is None:  # pragma: no cover
                raise ValueError("committed cycle requires settlement and intervention")
            if self.budget_settlement.model_calls < 1:
                raise ValueError("committed cycle must settle a model call")
            if len(self.model_call_digests) != self.budget_settlement.model_calls:
                raise ValueError("committed model-call digests must match settled calls")
            expected_interventions = int(self.intervention.action is InterventionAction.REMIND)
            if self.budget_settlement.interventions != expected_interventions:
                raise ValueError("committed intervention usage must match its verdict")
        elif self.state is CycleState.FAILED:
            if self.failure_reason is None:
                raise ValueError("failed cycle requires a failure reason")
            if self.validated_delta is not None or self.intervention is not None:
                raise ValueError("failed cycle cannot carry committed mutations or intervention")
            if self.budget_reservation is None:
                if self.budget_settlement is not None or self.batch_digest is not None:
                    raise ValueError("failure before reservation cannot carry settlement or batch")
                if self.model_call_digests or self.model_call_latencies_us:
                    raise ValueError("failure before reservation cannot carry model-call receipts")
            elif self.budget_settlement is None:
                raise ValueError("failed reserved cycle requires a budget settlement")
            if self.failure_reason is ReasonCode.FAILED_UNKNOWN_COST:
                if self.batch_digest is None or self.budget_reservation is None:
                    raise ValueError("failed_unknown_cost requires a running cycle")
                if self.budget_settlement != self.budget_reservation:
                    raise ValueError("failed_unknown_cost must consume the full reservation")
                if len(self.model_call_digests) > self.budget_reservation.model_calls:
                    raise ValueError("known model-call receipts exceed the reservation")
            elif self.budget_settlement is not None:
                if self.budget_settlement.interventions != 0:
                    raise ValueError("failed cycle cannot consume an intervention")
                if self.batch_digest is None:
                    if (
                        self.model_call_digests
                        or self.model_call_latencies_us
                        or self.budget_settlement.model_calls != 0
                        or self.budget_settlement.input_tokens != 0
                        or self.budget_settlement.output_tokens != 0
                        or self.budget_settlement.canonical_token_equivalents != 0
                        or self.budget_settlement.schema_repairs != 0
                    ):
                        raise ValueError("failure before running cannot consume model usage")
                elif self.budget_settlement.model_calls < 1:
                    raise ValueError("running failure must settle a model call")
                elif len(self.model_call_digests) != self.budget_settlement.model_calls:
                    raise ValueError("failed model-call digests must match settled calls")
        return self


class DeliveryRecord(VersionedRecord):
    record_type: Literal["delivery_record"] = "delivery_record"
    delivery_id: UUID4
    run_id: UUID4
    revision: PositiveInt
    cycle_id: Sha256Digest
    intervention_id: UUID4
    rendered_text_digest: Sha256Digest
    target_request_id: EventMetadataIdentifier
    target: DeliveryTarget
    state: DeliveryState
    attempt_count: NonNegativeInt
    adapter_id: ComponentIdentifier
    adapter_deduplicates: bool
    adapter_deduplication_guarantee: DeduplicationGuarantee
    adapter_supports_pre_action: bool
    adapter_contract_version: ComponentIdentifier
    adapter_capabilities_digest: Sha256Digest
    claim_id: UUID4 | None = None
    attempt_id: UUID4 | None = None
    receipt: JsonObject | None = Field(default=None, repr=False)
    outcome: DeliveryOutcome | None = None
    reason_code: ReasonCode | None = None
    created_at: UtcDatetime
    updated_at: UtcDatetime

    @model_validator(mode="after")
    def update_follows_creation(self) -> Self:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        if not self.adapter_deduplicates and self.attempt_count > 1:
            raise ValueError("a non-deduplicating adapter can be attempted at most once")
        is_durable = (
            self.adapter_deduplication_guarantee is DeduplicationGuarantee.DURABLE_DELIVERY_ID
        )
        if self.adapter_deduplicates is not is_durable:
            raise ValueError("adapter deduplication declaration is inconsistent")
        if (
            self.target is DeliveryTarget.PRE_ACTION_REPLAN
            and self.attempt_count > 0
            and not self.adapter_supports_pre_action
        ):
            raise ValueError("pre-action delivery attempt requires interception capability")
        empty_result = self.receipt is None and self.outcome is None and self.reason_code is None
        if self.state is DeliveryState.PENDING:
            if (
                self.attempt_count != 0
                or self.claim_id is not None
                or self.attempt_id is not None
                or not empty_result
            ):
                raise ValueError("pending delivery cannot have ownership, attempts, or outcome")
        elif self.state is DeliveryState.CLAIMED:
            if self.claim_id is None or self.attempt_id is not None or not empty_result:
                raise ValueError("claimed delivery requires only a claim owner")
            if self.attempt_count > 0 and not self.adapter_deduplicates:
                raise ValueError("only a deduplicating adapter can reclaim an attempted delivery")
        elif self.state is DeliveryState.ATTEMPTING:
            if (
                self.attempt_count < 1
                or self.claim_id is None
                or self.attempt_id is None
                or not empty_result
            ):
                raise ValueError(
                    "attempting delivery requires claimed attempt ownership without an outcome"
                )
        elif self.state is DeliveryState.DELIVERED:
            provider_receipt_id: object | None = None
            if self.receipt is not None and set(self.receipt) == {"provider_receipt_id"}:
                provider_receipt_id = self.receipt.get("provider_receipt_id")
            if (
                self.attempt_count < 1
                or self.claim_id is None
                or self.attempt_id is None
                or type(provider_receipt_id) is not str
                or not 1 <= len(provider_receipt_id) <= 256
                or not provider_receipt_id[0].isalnum()
                or any(
                    character
                    not in ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:/+-")
                    for character in provider_receipt_id
                )
                or self.outcome is not DeliveryOutcome.DELIVERED
                or self.reason_code is not ReasonCode.DELIVERY_SUCCEEDED
            ):
                raise ValueError("delivered state requires a bounded successful receipt")
        elif self.state is DeliveryState.UNKNOWN:
            if (
                self.attempt_count < 1
                or self.claim_id is None
                or self.attempt_id is None
                or self.receipt is not None
                or self.outcome is not DeliveryOutcome.UNKNOWN
                or self.reason_code is not ReasonCode.DELIVERY_UNKNOWN
            ):
                raise ValueError("unknown state requires an attempt and unknown outcome")
        elif self.state is DeliveryState.FAILED:
            if (
                self.attempt_count < 1
                or self.claim_id is None
                or self.attempt_id is None
                or self.receipt is not None
                or self.outcome is not DeliveryOutcome.FAILED
                or self.reason_code is not ReasonCode.DELIVERY_FAILED
            ):
                raise ValueError("failed state requires an attempt and failed outcome")
        elif self.state is DeliveryState.REJECTED:
            allowed_reasons = {
                ReasonCode.UNSUPPORTED_DELIVERY_TARGET,
                ReasonCode.UNSUPPORTED_DELIVERY_CHANNEL,
                ReasonCode.UNSAFE_ROLE_MAPPING,
                ReasonCode.TARGET_UNAVAILABLE,
            }
            if (
                self.attempt_id is not None
                or self.receipt is not None
                or self.outcome is not DeliveryOutcome.REFUSED
                or self.reason_code not in allowed_reasons
                or (self.claim_id is None and self.attempt_count != 0)
                or (self.attempt_count > 0 and not self.adapter_deduplicates)
            ):
                raise ValueError("rejected state requires a safe refusal without a new attempt")
        return self


RuntimeRecord: TypeAlias = (
    TraceEvent
    | Signal
    | MemoryRecord
    | InvocationDecision
    | MemoryDelta
    | InterventionDecision
    | InterventionOutcome
    | CycleRecord
    | DeliveryRecord
)

LedgerRecord: TypeAlias = (
    TraceEvent | Signal | InvocationDecision | CycleRecord | InterventionOutcome | DeliveryRecord
)
