from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Annotated, Literal, Protocol, Self
from uuid import UUID

from pydantic import UUID4 as PYDANTIC_UUID4
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from saliencegate.domain import (
    MAX_MEMORY_DELTA_ITEMS,
    BudgetAmounts,
    BudgetSnapshot,
    CycleRecord,
    CycleState,
    DeduplicationGuarantee,
    DeliveryOutcome,
    DeliveryRecord,
    DeliveryState,
    DeliveryTarget,
    InterventionAction,
    InterventionDecision,
    InterventionOutcome,
    InvocationDecision,
    JsonObject,
    LedgerRecord,
    MemoryDelta,
    MemoryIdAssignment,
    MemoryKind,
    MemoryRecord,
    NormalizedTraceEventDraft,
    PayloadDigest,
    ReasonCode,
    Signal,
    TraceEvent,
    TrustLabel,
    ValidityState,
    canonical_json,
    length_prefixed_sha256,
)

DirectLedgerRecord = Signal | InvocationDecision | InterventionOutcome
LedgerEntryRecord = Annotated[LedgerRecord, Field(discriminator="record_type")]

MAX_CONDITIONAL_BATCH_OPERATIONS = 5_000
MAX_CONDITIONAL_BATCH_EVENTS = 1_000
MAX_CONDITIONAL_BATCH_SIGNALS = 4_000
MAX_CONDITIONAL_BATCH_REQUEST_BYTES = 128 * 1024 * 1024
MAX_CONDITIONAL_BATCH_RECEIPT_BYTES = 256 * 1024 * 1024


def _require_exact_uuid(value: UUID) -> UUID:
    if type(value) is not UUID:
        raise ValueError("UUID subclasses are not accepted")
    return UUID(int=value.int)


UUID4 = Annotated[PYDANTIC_UUID4, AfterValidator(_require_exact_uuid)]
CycleId = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Sha256Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ComponentIdentifier = Annotated[
    str,
    Field(
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9_][A-Za-z0-9._:/+\-]*$",
    ),
]
PositiveInt = Annotated[int, Field(ge=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
_PREVIEW_COMMAND_DIGEST_DOMAIN = "saliencegate:repository:memory-delta-preview-command:v1"


def _require_utc(value: datetime) -> datetime:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("timestamp must be an exact UTC datetime")
    return value.astimezone(UTC)


UtcDatetime = Annotated[datetime, AfterValidator(_require_utc)]


class RepositoryError(RuntimeError):
    """Base class for repository-boundary failures without sensitive values."""


class InvalidAppendTypeError(RepositoryError, TypeError):
    def __init__(self, received_type: type[object]) -> None:
        super().__init__(
            f"append accepts exactly NormalizedTraceEventDraft; received {received_type.__name__}"
        )


class InvalidDraftError(RepositoryError):
    def __init__(self) -> None:
        super().__init__("normalized event draft failed repository validation")


class UnsafeEventMetadataError(RepositoryError):
    def __init__(self, fields: tuple[str, ...]) -> None:
        self.fields = fields
        super().__init__(f"event metadata requires redaction: {', '.join(fields)}")


class InvalidRecordTypeError(RepositoryError, TypeError):
    def __init__(self, operation: str, received_type: type[object]) -> None:
        super().__init__(f"{operation} does not accept {received_type.__name__}")


class InvalidRecordError(RepositoryError):
    def __init__(self, operation: str) -> None:
        super().__init__(f"{operation} record failed repository validation")


class UnsafeRecordContentError(RepositoryError):
    def __init__(self, operation: str) -> None:
        super().__init__(f"{operation} record contains content that requires redaction")


class InvalidQueryError(RepositoryError):
    def __init__(self) -> None:
        super().__init__("memory query failed repository validation")


class InvalidRunIdError(RepositoryError, TypeError):
    def __init__(self) -> None:
        super().__init__("run ID must be an exact UUID4 value")


class InvalidCycleStateError(RepositoryError):
    def __init__(self, operation: str, expected: CycleState) -> None:
        super().__init__(f"{operation} requires a {expected.value} cycle record")


class CycleConflictError(RepositoryError):
    def __init__(self) -> None:
        super().__init__("cycle revision conflicts with the authoritative ledger")


class CycleRevisionConflictError(RepositoryError):
    def __init__(self, expected: int, actual: int | None) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__("cycle revision does not match the authoritative state")


class DeliveryNotFoundError(RepositoryError):
    def __init__(self) -> None:
        super().__init__("delivery was not found in the authoritative ledger")


class DeliveryRevisionConflictError(RepositoryError):
    def __init__(self, expected: int, actual: int | None) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__("delivery revision does not match the authoritative state")


class InvalidDeliveryStateError(RepositoryError):
    def __init__(self, operation: str) -> None:
        super().__init__(f"{operation} is not valid for the authoritative delivery state")


class DeliveryOwnershipError(RepositoryError):
    def __init__(self) -> None:
        super().__init__("delivery ownership token does not match the authoritative attempt")


class InvalidRecoveryTimeError(RepositoryError):
    def __init__(self) -> None:
        super().__init__("cycle recovery time must be an exact UTC datetime")


class RunNotFoundError(RepositoryError):
    def __init__(self, run_id: UUID) -> None:
        self.run_id = run_id
        super().__init__(f"run not found: {run_id}")


class RecordCollisionError(RepositoryError):
    def __init__(self, record_type: str, record_id: UUID) -> None:
        self.record_type = record_type
        self.record_id = record_id
        super().__init__(f"{record_type} ID collision: {record_id}")


class RevisionConflictError(RepositoryError):
    def __init__(self, memory_id: UUID, expected: int, actual: int | None) -> None:
        self.memory_id = memory_id
        self.expected = expected
        self.actual = actual
        super().__init__(f"memory revision conflict: {memory_id}")


class CrossRunReferenceError(RepositoryError):
    def __init__(self, reference_type: str) -> None:
        super().__init__(f"{reference_type} references a missing or cross-run record")


class DigestVerificationError(RepositoryError):
    def __init__(self, scope: str) -> None:
        super().__init__(f"{scope} integrity verification failed")


class LedgerHeadConflictError(RepositoryError):
    def __init__(self) -> None:
        super().__init__("ledger head does not match the conditional write precondition")


class ProjectionInvariantError(RepositoryError):
    pass


class PreviewConflictError(RepositoryError):
    def __init__(self) -> None:
        super().__init__("memory delta preview anchor is stale")


class RebuildError(RepositoryError):
    def __init__(self) -> None:
        super().__init__("ledger rebuild failed; the previous projection remains active")


class AppendDisposition(StrEnum):
    APPENDED = "appended"
    DUPLICATE = "duplicate"
    COLLISION = "collision"


class RepositoryModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
        validate_default=True,
    )


class LedgerEntry(RepositoryModel):
    run_id: UUID4
    position: Annotated[int, Field(ge=1)]
    record_key: Annotated[str, Field(min_length=1, max_length=200)]
    record_tag: PayloadDigest
    previous_chain_tag: PayloadDigest | None = None
    chain_tag: PayloadDigest
    record: LedgerEntryRecord

    @model_validator(mode="after")
    def record_belongs_to_entry_run(self) -> Self:
        if self.record.run_id != self.run_id:
            raise ValueError("ledger record belongs to a different run")
        if (self.position == 1) != (self.previous_chain_tag is None):
            raise ValueError("only the first ledger entry can omit its previous chain tag")
        algorithms = {self.record_tag.algorithm, self.chain_tag.algorithm}
        if self.previous_chain_tag is not None:
            algorithms.add(self.previous_chain_tag.algorithm)
        if len(algorithms) != 1:
            raise ValueError("ledger entry integrity algorithms must match")
        return self


class LedgerHead(RepositoryModel):
    run_id: UUID4
    entry_count: Annotated[int, Field(ge=1)]
    chain_tag: PayloadDigest
    projection_tag: PayloadDigest
    head_tag: PayloadDigest

    @model_validator(mode="after")
    def algorithms_match(self) -> Self:
        if (
            len(
                {
                    self.chain_tag.algorithm,
                    self.projection_tag.algorithm,
                    self.head_tag.algorithm,
                }
            )
            != 1
        ):
            raise ValueError("ledger head integrity algorithms must match")
        return self


class AppendReceipt(RepositoryModel):
    disposition: AppendDisposition
    event: TraceEvent
    ledger_position: Annotated[int, Field(ge=1)]
    ingestion_cursor: Annotated[int, Field(ge=1)]
    collision_event: TraceEvent | None = None

    @model_validator(mode="after")
    def collision_event_matches_disposition(self) -> Self:
        if (self.disposition is AppendDisposition.COLLISION) != (self.collision_event is not None):
            raise ValueError("only collision receipts can carry a collision event")
        if self.collision_event is not None and self.collision_event.run_id != self.event.run_id:
            raise ValueError("collision event belongs to a different run")
        if self.ingestion_cursor < self.event.sequence:
            raise ValueError("receipt cursor cannot precede its event")
        if (
            self.collision_event is not None
            and self.ingestion_cursor != self.collision_event.sequence
        ):
            raise ValueError("collision receipt cursor must identify its audit event")
        return self


class LedgerReceipt(RepositoryModel):
    appended: bool
    record_id: UUID4
    record_tag: PayloadDigest
    ledger_position: Annotated[int, Field(ge=1)]
    chain_tag: PayloadDigest


def _validated_repository_model(
    value: object,
    expected_type: type[BaseModel],
    *,
    json_mode: bool = False,
) -> BaseModel:
    """Return a detached, recursively revalidated exact model instance."""

    if type(value) is dict:
        try:
            if json_mode:
                return expected_type.model_validate_json(canonical_json(value))
            validated = expected_type.model_validate(value)
            return expected_type.model_validate_json(validated.model_dump_json(warnings=False))
        except Exception:
            raise ValueError(f"{expected_type.__name__} failed validation") from None
    if json_mode:
        raise ValueError(f"JSON value for {expected_type.__name__} must be an object")
    if type(value) is not expected_type:
        raise ValueError(f"value must be exactly {expected_type.__name__}")
    try:
        encoded = expected_type.model_dump_json(value, warnings=False)
        return expected_type.model_validate_json(encoded)
    except Exception:
        raise ValueError(f"{expected_type.__name__} failed validation") from None


class ConditionalEventAppend(RepositoryModel):
    model_config = ConfigDict(revalidate_instances="always")

    operation: Literal["append_event"] = "append_event"
    event: NormalizedTraceEventDraft = Field(repr=False)
    event_id: UUID4

    @field_validator("event", mode="before")
    @classmethod
    def event_is_detached_and_valid(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> NormalizedTraceEventDraft:
        return _validated_repository_model(  # type: ignore[return-value]
            value,
            NormalizedTraceEventDraft,
            json_mode=info.mode == "json",
        )


class ConditionalSignalAppend(RepositoryModel):
    model_config = ConfigDict(revalidate_instances="always")

    operation: Literal["record_signal"] = "record_signal"
    signal: Signal = Field(repr=False)

    @field_validator("signal", mode="before")
    @classmethod
    def signal_is_detached_and_valid(cls, value: object, info: ValidationInfo) -> Signal:
        return _validated_repository_model(  # type: ignore[return-value]
            value,
            Signal,
            json_mode=info.mode == "json",
        )


ConditionalAppendOperation = Annotated[
    ConditionalEventAppend | ConditionalSignalAppend,
    Field(discriminator="operation"),
]


class ConditionalBatchReceipt(RepositoryModel):
    """Ordered batch result; generic record receipts pair with the input operations by index."""

    model_config = ConfigDict(revalidate_instances="always")

    initial_head: LedgerHead | None
    receipts: Annotated[
        tuple[AppendReceipt | LedgerReceipt, ...],
        Field(min_length=1, max_length=MAX_CONDITIONAL_BATCH_OPERATIONS, repr=False),
    ]
    final_head: LedgerHead

    @field_validator("initial_head", "final_head", mode="before")
    @classmethod
    def heads_are_detached_and_valid(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> LedgerHead | None:
        if value is None:
            return None
        return _validated_repository_model(  # type: ignore[return-value]
            value,
            LedgerHead,
            json_mode=info.mode == "json",
        )

    @field_validator("receipts", mode="before")
    @classmethod
    def receipts_are_detached_and_valid(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> tuple[AppendReceipt | LedgerReceipt, ...]:
        json_mode = info.mode == "json"
        expected_container_type = list if json_mode else tuple
        if type(value) is not expected_container_type:
            raise ValueError(
                "receipts must be a JSON array" if json_mode else "receipts must be an exact tuple"
            )
        detached: list[AppendReceipt | LedgerReceipt] = []
        for receipt in value:
            if type(receipt) is dict:
                is_event = "disposition" in receipt
                is_record = "appended" in receipt
                if is_event == is_record:
                    raise ValueError("receipt has an unsupported shape")
                expected_type = AppendReceipt if is_event else LedgerReceipt
                checked = _validated_repository_model(
                    receipt,
                    expected_type,
                    json_mode=json_mode,
                )
            elif type(receipt) is AppendReceipt:
                checked = _validated_repository_model(receipt, AppendReceipt)
            elif type(receipt) is LedgerReceipt:
                checked = _validated_repository_model(receipt, LedgerReceipt)
            else:
                raise ValueError("receipt has an unsupported type")
            detached.append(checked)  # type: ignore[arg-type]
        return tuple(detached)

    @model_validator(mode="after")
    def batch_receipt_is_consistent(self) -> Self:
        initial_count = 0
        if self.initial_head is not None:
            if self.initial_head.run_id != self.final_head.run_id:
                raise ValueError("batch heads belong to different runs")
            if self.initial_head.head_tag.algorithm is not self.final_head.head_tag.algorithm:
                raise ValueError("batch head algorithms do not match")
            initial_count = self.initial_head.entry_count

        appended = 0
        current_count = initial_count
        algorithm = self.final_head.head_tag.algorithm
        event_receipts = 0
        signal_receipts = 0
        for receipt in self.receipts:
            if isinstance(receipt, AppendReceipt):
                event_receipts += 1
                if receipt.event.run_id != self.final_head.run_id:
                    raise ValueError("event receipt belongs to a different run")
                if receipt.disposition is AppendDisposition.COLLISION:
                    raise ValueError("conditional batches cannot return collisions")
                if receipt.event.payload_digest.algorithm is not algorithm:
                    raise ValueError("event receipt integrity algorithm does not match the head")
                receipt_appended = receipt.disposition is AppendDisposition.APPENDED
            else:
                signal_receipts += 1
                if (
                    receipt.record_tag.algorithm is not algorithm
                    or receipt.chain_tag.algorithm is not algorithm
                ):
                    raise ValueError("record receipt integrity algorithm does not match the head")
                receipt_appended = receipt.appended
            if receipt_appended:
                current_count += 1
                appended += 1
                if receipt.ledger_position != current_count:
                    raise ValueError("appended receipt positions must be contiguous and ordered")
            elif receipt.ledger_position > current_count:
                raise ValueError("duplicate receipt refers to a future ledger position")
        if (
            event_receipts > MAX_CONDITIONAL_BATCH_EVENTS
            or signal_receipts > MAX_CONDITIONAL_BATCH_SIGNALS
        ):
            raise ValueError("conditional batch receipt exceeds its operation-kind limit")
        if self.initial_head is None and (
            not isinstance(self.receipts[0], AppendReceipt)
            or self.receipts[0].disposition is not AppendDisposition.APPENDED
        ):
            raise ValueError("a genesis batch must begin with an appended event receipt")
        if self.final_head.entry_count != initial_count + appended or (
            self.final_head.entry_count != current_count
        ):
            raise ValueError("final head does not match appended receipt count")
        if appended == 0 and self.final_head != self.initial_head:
            raise ValueError("a no-op batch must preserve the exact ledger head")
        if len(canonical_json(self)) > MAX_CONDITIONAL_BATCH_RECEIPT_BYTES:
            raise ValueError("conditional batch receipt exceeds its canonical byte limit")
        return self


class GroundingPin(RepositoryModel):
    """Resolved grounding inputs fixed before a memory-model call."""

    grounding_version: Annotated[
        str,
        Field(
            min_length=1,
            max_length=256,
            pattern=r"^[A-Za-z0-9_][A-Za-z0-9._:/+\-]*$",
        ),
    ]
    grounding_configuration: JsonObject = Field(repr=False)
    grounding_configuration_digest: Sha256Digest
    requested_delivery_target: DeliveryTarget | None


class BeginCycle(RepositoryModel):
    run_id: UUID4
    invocation_decision_id: UUID4
    grounding_version: Annotated[
        str,
        Field(
            min_length=1,
            max_length=256,
            pattern=r"^[A-Za-z0-9_][A-Za-z0-9._:/+\-]*$",
        ),
    ]
    grounding_configuration: JsonObject = Field(repr=False)
    grounding_configuration_digest: Sha256Digest
    requested_delivery_target: DeliveryTarget | None
    created_at: UtcDatetime


class ReserveCycle(RepositoryModel):
    run_id: UUID4
    cycle_id: CycleId
    expected_revision: PositiveInt = 1
    reservation: BudgetAmounts
    updated_at: UtcDatetime


class StartCycle(RepositoryModel):
    run_id: UUID4
    cycle_id: CycleId
    expected_revision: PositiveInt = 2
    batch_digest: Sha256Digest
    updated_at: UtcDatetime


class EnqueueDelivery(RepositoryModel):
    """Adapter binding fixed atomically with an accepted reminder."""

    target_request_id: ComponentIdentifier
    adapter_id: ComponentIdentifier
    adapter_deduplicates: bool
    adapter_deduplication_guarantee: DeduplicationGuarantee
    adapter_supports_pre_action: bool
    adapter_contract_version: ComponentIdentifier
    adapter_capabilities_digest: Sha256Digest

    @model_validator(mode="after")
    def deduplication_declaration_is_consistent(self) -> Self:
        is_durable = (
            self.adapter_deduplication_guarantee is DeduplicationGuarantee.DURABLE_DELIVERY_ID
        )
        if self.adapter_deduplicates is not is_durable:
            raise ValueError("deduplication flag and guarantee disagree")
        return self


class CommitCycle(RepositoryModel):
    run_id: UUID4
    cycle_id: CycleId
    expected_revision: PositiveInt = 3
    settlement: BudgetAmounts
    model_call_digests: tuple[Sha256Digest, ...] = ()
    model_call_latencies_us: tuple[NonNegativeInt, ...] = ()
    validated_delta: MemoryDelta
    memory_id_assignments: tuple[MemoryIdAssignment, ...] = ()
    intervention: InterventionDecision
    selector_provenance: JsonObject | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        repr=False,
    )
    delivery: EnqueueDelivery | None = None
    updated_at: UtcDatetime

    @model_validator(mode="after")
    def reconcile_model_call_receipts(self) -> Self:
        if (
            len(self.model_call_digests) != self.settlement.model_calls
            or len(self.model_call_latencies_us) != self.settlement.model_calls
        ):
            raise ValueError("model-call receipts must match settled calls")
        if sum(self.model_call_latencies_us) > self.settlement.latency_us:
            raise ValueError("model-call latency exceeds settled cycle latency")
        expects_delivery = self.intervention.action is InterventionAction.REMIND
        if expects_delivery != (self.delivery is not None):
            raise ValueError("only a reminder cycle must enqueue one delivery")
        return self


def _preview_command_digest(values: Mapping[str, object]) -> str:
    return length_prefixed_sha256(
        canonical_json(
            {
                "schema_version": values["schema_version"],
                "run_id": str(values["run_id"]),
                "expected_ledger_position": values["expected_ledger_position"],
                "expected_ingestion_cursor": values["expected_ingestion_cursor"],
                "expected_memory_cursor": values["expected_memory_cursor"],
                "expected_projection_digest": values["expected_projection_digest"],
                "last_event_sequence": values["last_event_sequence"],
                "delta": values["delta"],
                "memory_id_assignments": values["memory_id_assignments"],
            }
        ),
        domain=_PREVIEW_COMMAND_DIGEST_DOMAIN,
    )


class PreviewMemoryDelta(RepositoryModel):
    """Read-only delta projection pinned to one authoritative repository state."""

    schema_version: Literal["memory-delta-preview-command/v1"]
    run_id: UUID4
    expected_ledger_position: NonNegativeInt
    expected_ingestion_cursor: NonNegativeInt
    expected_memory_cursor: NonNegativeInt
    expected_projection_digest: PayloadDigest
    last_event_sequence: PositiveInt
    delta: MemoryDelta
    memory_id_assignments: Annotated[
        tuple[MemoryIdAssignment, ...],
        Field(max_length=MAX_MEMORY_DELTA_ITEMS + 1),
    ] = ()
    command_digest: Sha256Digest = Field(default_factory=_preview_command_digest)

    @model_validator(mode="after")
    def delta_and_anchor_match_the_run(self) -> Self:
        if self.delta.run_id != self.run_id:
            raise ValueError("memory preview delta belongs to a different run")
        if self.expected_memory_cursor > self.expected_ingestion_cursor:
            raise ValueError("memory preview cursor anchor is inconsistent")
        if self.last_event_sequence > self.expected_ingestion_cursor:
            raise ValueError("memory preview event range exceeds its anchor")
        values = self.model_dump(mode="json", exclude={"command_digest"})
        if self.command_digest != _preview_command_digest(values):
            raise ValueError("memory preview command digest does not match")
        return self


class FailCycle(RepositoryModel):
    run_id: UUID4
    cycle_id: CycleId
    expected_revision: PositiveInt
    reason: ReasonCode
    settlement: BudgetAmounts | None = None
    model_call_digests: tuple[Sha256Digest, ...] = ()
    model_call_latencies_us: tuple[NonNegativeInt, ...] = ()
    updated_at: UtcDatetime

    @model_validator(mode="after")
    def reconcile_known_model_call_receipts(self) -> Self:
        if len(self.model_call_digests) != len(self.model_call_latencies_us):
            raise ValueError("model-call receipts must have equal length")
        if self.settlement is None:
            if self.model_call_digests:
                raise ValueError("model-call receipts require a settlement")
            return self
        if sum(self.model_call_latencies_us) > self.settlement.latency_us:
            raise ValueError("model-call latency exceeds settled cycle latency")
        if self.reason is ReasonCode.FAILED_UNKNOWN_COST:
            if len(self.model_call_digests) > self.settlement.model_calls:
                raise ValueError("known model-call receipts exceed settled calls")
        elif len(self.model_call_digests) != self.settlement.model_calls:
            raise ValueError("model-call receipts must match settled calls")
        return self


class ClaimDelivery(RepositoryModel):
    run_id: UUID4
    delivery_id: UUID4
    expected_revision: PositiveInt
    claim_id: UUID4
    updated_at: UtcDatetime


class BeginDeliveryAttempt(RepositoryModel):
    run_id: UUID4
    delivery_id: UUID4
    expected_revision: PositiveInt
    claim_id: UUID4
    attempt_id: UUID4
    updated_at: UtcDatetime


class CompleteDelivery(RepositoryModel):
    run_id: UUID4
    delivery_id: UUID4
    expected_revision: PositiveInt
    claim_id: UUID4
    attempt_id: UUID4
    outcome: DeliveryOutcome
    provider_receipt_id: ComponentIdentifier | None = Field(default=None, repr=False)
    updated_at: UtcDatetime

    @model_validator(mode="after")
    def receipt_matches_outcome(self) -> Self:
        if self.outcome not in (DeliveryOutcome.DELIVERED, DeliveryOutcome.FAILED):
            raise ValueError("completion outcome must be delivered or failed")
        if (self.outcome is DeliveryOutcome.DELIVERED) != (self.provider_receipt_id is not None):
            raise ValueError("only delivered completion requires a provider receipt ID")
        return self


class MarkDeliveryUnknown(RepositoryModel):
    run_id: UUID4
    delivery_id: UUID4
    expected_revision: PositiveInt
    claim_id: UUID4
    attempt_id: UUID4
    updated_at: UtcDatetime


class RejectDelivery(RepositoryModel):
    run_id: UUID4
    delivery_id: UUID4
    expected_revision: PositiveInt
    claim_id: UUID4 | None = None
    reason_code: ReasonCode
    updated_at: UtcDatetime

    @model_validator(mode="after")
    def reason_is_delivery_refusal(self) -> Self:
        if self.reason_code not in {
            ReasonCode.UNSUPPORTED_DELIVERY_TARGET,
            ReasonCode.UNSUPPORTED_DELIVERY_CHANNEL,
            ReasonCode.UNSAFE_ROLE_MAPPING,
            ReasonCode.TARGET_UNAVAILABLE,
        }:
            raise ValueError("rejection requires a delivery refusal reason")
        return self


class DeliveryAttemptEnvelope(RepositoryModel):
    """Repository-derived reminder input for exactly one owned adapter attempt."""

    delivery_id: UUID4
    run_id: UUID4
    cycle_id: CycleId
    intervention_id: UUID4
    rendered_text_digest: Sha256Digest
    claim_id: UUID4
    attempt_id: UUID4
    attempt_number: PositiveInt
    target_request_id: ComponentIdentifier
    target: DeliveryTarget
    adapter_id: ComponentIdentifier
    adapter_deduplicates: bool
    adapter_deduplication_guarantee: DeduplicationGuarantee
    adapter_supports_pre_action: bool
    adapter_contract_version: ComponentIdentifier
    adapter_capabilities_digest: Sha256Digest
    rendered_text: Annotated[str, Field(min_length=1, max_length=4_096)] = Field(repr=False)
    ttl_steps: Literal[1] = 1


class DeliveryTransitionReceipt(RepositoryModel):
    appended: bool
    delivery: DeliveryRecord
    record_tag: PayloadDigest
    ledger_position: Annotated[int, Field(ge=1)]
    chain_tag: PayloadDigest

    @model_validator(mode="after")
    def receipt_matches_delivery_and_integrity_algorithm(self) -> Self:
        if self.delivery.run_id.version != 4:  # pragma: no cover - field invariant
            raise ValueError("delivery receipt has an invalid run")
        if self.record_tag.algorithm is not self.chain_tag.algorithm:
            raise ValueError("delivery receipt integrity algorithms differ")
        return self


class DeliveryAttemptReceipt(DeliveryTransitionReceipt):
    envelope: DeliveryAttemptEnvelope | None = Field(default=None, repr=False)

    @model_validator(mode="after")
    def new_attempt_is_the_only_envelope_source(self) -> Self:
        if self.appended != (self.envelope is not None):
            raise ValueError("only a newly persisted attempt can expose its envelope")
        if self.envelope is not None and (
            self.delivery.state is not DeliveryState.ATTEMPTING
            or self.envelope.delivery_id != self.delivery.delivery_id
            or self.envelope.run_id != self.delivery.run_id
            or self.envelope.cycle_id != self.delivery.cycle_id
            or self.envelope.intervention_id != self.delivery.intervention_id
            or self.envelope.rendered_text_digest != self.delivery.rendered_text_digest
            or self.envelope.claim_id != self.delivery.claim_id
            or self.envelope.attempt_id != self.delivery.attempt_id
            or self.envelope.attempt_number != self.delivery.attempt_count
            or self.envelope.target_request_id != self.delivery.target_request_id
            or self.envelope.target is not self.delivery.target
            or self.envelope.adapter_id != self.delivery.adapter_id
            or self.envelope.adapter_deduplicates is not self.delivery.adapter_deduplicates
            or self.envelope.adapter_deduplication_guarantee
            is not self.delivery.adapter_deduplication_guarantee
            or self.envelope.adapter_supports_pre_action
            is not self.delivery.adapter_supports_pre_action
            or self.envelope.adapter_contract_version != self.delivery.adapter_contract_version
            or self.envelope.adapter_capabilities_digest
            != self.delivery.adapter_capabilities_digest
        ):
            raise ValueError("attempt envelope does not match its delivery revision")
        return self


class DeliveryRecoveryReceipt(RepositoryModel):
    run_id: UUID4
    marked_unknown: tuple[DeliveryTransitionReceipt, ...]
    resumable_pending: tuple[DeliveryRecord, ...]
    resumable_claimed: tuple[DeliveryRecord, ...]
    retryable_unknown: tuple[DeliveryRecord, ...]
    non_retryable_unknown: tuple[DeliveryRecord, ...]

    @model_validator(mode="after")
    def categories_match_authoritative_states(self) -> Self:
        categories = (
            (self.resumable_pending, DeliveryState.PENDING),
            (self.resumable_claimed, DeliveryState.CLAIMED),
            (self.retryable_unknown, DeliveryState.UNKNOWN),
            (self.non_retryable_unknown, DeliveryState.UNKNOWN),
        )
        classified_ids: list[UUID] = []
        for records, expected_state in categories:
            for record in records:
                if record.run_id != self.run_id or record.state is not expected_state:
                    raise ValueError("delivery recovery category has the wrong run or state")
                classified_ids.append(record.delivery_id)
        if len(set(classified_ids)) != len(classified_ids):
            raise ValueError("delivery recovery categories must be disjoint")
        if any(not record.adapter_deduplicates for record in self.retryable_unknown):
            raise ValueError("retryable unknown delivery must support durable deduplication")
        if any(record.adapter_deduplicates for record in self.non_retryable_unknown):
            raise ValueError("non-retryable unknown delivery cannot support deduplication")
        if any(
            receipt.delivery.run_id != self.run_id
            or receipt.delivery.state is not DeliveryState.UNKNOWN
            or not receipt.appended
            for receipt in self.marked_unknown
        ):
            raise ValueError("recovered attempts must be newly marked unknown in this run")
        return self


class CycleReceipt(RepositoryModel):
    appended: bool
    cycle: CycleRecord
    record_tag: PayloadDigest
    ledger_position: Annotated[int, Field(ge=1)]
    chain_tag: PayloadDigest
    budget_snapshot: BudgetSnapshot
    delivery: DeliveryRecord | None = None

    @model_validator(mode="after")
    def delivery_matches_committed_cycle(self) -> Self:
        intervention = self.cycle.intervention
        expects_delivery = (
            self.cycle.state is CycleState.COMMITTED
            and intervention is not None
            and intervention.action is InterventionAction.REMIND
        )
        if expects_delivery != (self.delivery is not None):
            raise ValueError("only a committed reminder receipt must carry its delivery")
        if self.delivery is None:
            return self
        if (
            self.cycle.state is not CycleState.COMMITTED
            or intervention is None
            or intervention.action is not InterventionAction.REMIND
            or self.delivery.state is not DeliveryState.PENDING
            or self.delivery.revision != 1
            or self.delivery.run_id != self.cycle.run_id
            or self.delivery.cycle_id != self.cycle.cycle_id
            or self.delivery.intervention_id != intervention.intervention_id
            or self.delivery.target is not intervention.delivery_target
            or self.delivery.created_at != self.cycle.updated_at
        ):
            raise ValueError("cycle receipt delivery does not match its committed reminder")
        return self


class CycleRecoveryReceipt(RepositoryModel):
    run_id: UUID4
    resumable_pending: tuple[CycleRecord, ...]
    resumable_reserved: tuple[CycleRecord, ...]
    failed_unknown_cost: tuple[CycleReceipt, ...]


class MemoryQuery(RepositoryModel):
    run_id: UUID4
    text: Annotated[str, Field(min_length=1, max_length=1_000)] | None = None
    kinds: tuple[MemoryKind, ...] = ()
    validity: tuple[ValidityState, ...] = (ValidityState.ACTIVE,)
    trust_labels: tuple[TrustLabel, ...] = ()
    limit: Annotated[int, Field(ge=1, le=100)] = 10


class MemoryHit(RepositoryModel):
    memory: MemoryRecord
    score: Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
    matched_terms: tuple[str, ...] = ()
    ranker_version: Literal["lexical-v1"] = "lexical-v1"


class ProjectionDigests(RepositoryModel):
    events: PayloadDigest
    memory: PayloadDigest
    signals: PayloadDigest
    decisions: PayloadDigest
    cycles: PayloadDigest
    interventions: PayloadDigest
    budgets: PayloadDigest
    cursors: PayloadDigest
    deliveries: PayloadDigest
    outcomes: PayloadDigest
    overall: PayloadDigest

    @model_validator(mode="after")
    def algorithms_match(self) -> Self:
        algorithms = {getattr(self, field_name).algorithm for field_name in type(self).model_fields}
        if len(algorithms) != 1:
            raise ValueError("projection digest algorithms must match")
        return self


class MemorySnapshot(RepositoryModel):
    run_id: UUID4
    ledger_position: Annotated[int, Field(ge=0)]
    ingestion_cursor: Annotated[int, Field(ge=0)]
    memory_cursor: Annotated[int, Field(ge=0)]
    records: tuple[MemoryRecord, ...]
    projection_digest: PayloadDigest


class MemoryDeltaPreview(RepositoryModel):
    """Detached post-delta memory view carrying the immutable source anchor."""

    schema_version: Literal["memory-delta-preview/v1"]
    run_id: UUID4
    command_digest: Sha256Digest
    source_ledger_position: NonNegativeInt
    source_ingestion_cursor: NonNegativeInt
    source_memory_cursor: NonNegativeInt
    source_projection_digest: PayloadDigest
    records: tuple[MemoryRecord, ...]
    current_private_status_id: UUID4 | None = None
    preview_projection_digest: PayloadDigest

    @model_validator(mode="after")
    def records_belong_to_the_preview_run(self) -> Self:
        if any(record.run_id != self.run_id for record in self.records):
            raise ValueError("memory preview records belong to a different run")
        memory_ids = tuple(record.memory_id for record in self.records)
        if len(set(memory_ids)) != len(memory_ids):
            raise ValueError("memory preview records must have unique IDs")
        if self.source_memory_cursor > self.source_ingestion_cursor:
            raise ValueError("memory preview cursor anchor is inconsistent")
        if self.source_projection_digest.algorithm is not self.preview_projection_digest.algorithm:
            raise ValueError("memory preview projection digest algorithms must match")
        if (
            self.current_private_status_id is not None
            and self.current_private_status_id not in memory_ids
        ):
            raise ValueError("memory preview private status is unavailable")
        if self.current_private_status_id is not None:
            current = next(
                record
                for record in self.records
                if record.memory_id == self.current_private_status_id
            )
            if (
                current.kind is not MemoryKind.PRIVATE_STATUS
                or current.validity is not ValidityState.ACTIVE
            ):
                raise ValueError("memory preview private status is not active")
        return self


class RebuildReceipt(RepositoryModel):
    run_id: UUID4
    entries_replayed: Annotated[int, Field(ge=0)]
    before: ProjectionDigests | None
    after: ProjectionDigests
    equivalent: bool


class RunRepository(Protocol):
    async def append(
        self,
        event: NormalizedTraceEventDraft,
        *,
        event_id: UUID | None = None,
    ) -> AppendReceipt: ...

    async def append_event_if_head(
        self,
        event: NormalizedTraceEventDraft,
        *,
        event_id: UUID,
        expected_head: LedgerHead | None,
    ) -> AppendReceipt: ...

    async def record_signal(self, signal: Signal) -> LedgerReceipt: ...

    async def record_signal_if_head(
        self,
        signal: Signal,
        *,
        expected_head: LedgerHead,
    ) -> LedgerReceipt: ...

    async def append_records_if_head(
        self,
        operations: tuple[ConditionalAppendOperation, ...],
        *,
        expected_head: LedgerHead | None,
    ) -> ConditionalBatchReceipt: ...

    async def record_invocation_decision(
        self,
        decision: InvocationDecision,
    ) -> LedgerReceipt: ...

    async def record_outcome(self, outcome: InterventionOutcome) -> LedgerReceipt: ...

    async def begin_cycle(self, command: BeginCycle) -> CycleReceipt: ...

    async def reserve_cycle(self, command: ReserveCycle) -> CycleReceipt: ...

    async def mark_cycle_running(self, command: StartCycle) -> CycleReceipt: ...

    async def commit_cycle(self, command: CommitCycle) -> CycleReceipt: ...

    async def fail_cycle(self, command: FailCycle) -> CycleReceipt: ...

    async def preview_memory_delta(
        self,
        command: PreviewMemoryDelta,
    ) -> MemoryDeltaPreview: ...

    async def budget_snapshot(self, run_id: UUID) -> BudgetSnapshot: ...

    async def recover_cycles(
        self,
        run_id: UUID,
        *,
        recovered_at: datetime,
    ) -> CycleRecoveryReceipt: ...

    async def delivery(self, run_id: UUID, delivery_id: UUID) -> DeliveryRecord: ...

    async def claim_delivery(
        self,
        command: ClaimDelivery,
    ) -> DeliveryTransitionReceipt: ...

    async def begin_delivery_attempt(
        self,
        command: BeginDeliveryAttempt,
    ) -> DeliveryAttemptReceipt: ...

    async def complete_delivery(
        self,
        command: CompleteDelivery,
    ) -> DeliveryTransitionReceipt: ...

    async def mark_delivery_unknown(
        self,
        command: MarkDeliveryUnknown,
    ) -> DeliveryTransitionReceipt: ...

    async def reject_delivery(
        self,
        command: RejectDelivery,
    ) -> DeliveryTransitionReceipt: ...

    async def recover_deliveries(
        self,
        run_id: UUID,
        *,
        recovered_at: datetime,
    ) -> DeliveryRecoveryReceipt: ...

    async def ledger(self, run_id: UUID) -> tuple[LedgerEntry, ...]: ...

    async def ledger_head(self, run_id: UUID) -> LedgerHead: ...

    async def search(self, query: MemoryQuery) -> tuple[MemoryHit, ...]: ...

    async def snapshot(self, run_id: UUID) -> MemorySnapshot: ...

    async def rebuild(self, run_id: UUID) -> RebuildReceipt: ...


class ManagedRunRepository(RunRepository, Protocol):
    """A repository whose caller owns an explicit resource lifecycle."""

    def close(self) -> None: ...

    async def aclose(self) -> None: ...
