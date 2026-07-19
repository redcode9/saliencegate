from __future__ import annotations

from contextlib import suppress
from datetime import timedelta
from enum import StrEnum
from typing import Annotated, Literal, Protocol, Self, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.domain import (
    DeduplicationGuarantee,
    DeliveryTarget,
    NormalizedTraceEventDraft,
    ReasonCode,
    canonical_digest,
)
from saliencegate.ports.repository import (
    UUID4,
    ComponentIdentifier,
    CycleId,
    EnqueueDelivery,
    PositiveInt,
    UtcDatetime,
)

ADAPTER_CONTRACT_VERSION = "adapter-contract/v1"
UNTRUSTED_BLOCK_BEGIN = "<<<SALIENCEGATE_UNTRUSTED_EVIDENCE_V1>>>"
UNTRUSTED_BLOCK_END = "<<<END_SALIENCEGATE_UNTRUSTED_EVIDENCE_V1>>>"
_MAX_RECEIPT_LATENCY = timedelta(minutes=15)


class AdapterModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class DeliveryChannel(StrEnum):
    PROVIDER_DATA = "provider_data"
    EXISTING_USER_TASK = "existing_user_task"


class DeliveryRole(StrEnum):
    DATA = "data"
    USER = "user"
    SYSTEM = "system"
    DEVELOPER = "developer"


_SAFE_PROVIDER_CHANNEL_LEAVES = frozenset(
    {"context", "context_data", "data", "evidence", "metadata"}
)
_RESERVED_PROVIDER_CHANNEL_SEGMENTS = frozenset(
    {"developer", "instruction", "instructions", "message", "messages", "prompt", "system"}
)


class InjectionMapping(AdapterModel):
    channel: DeliveryChannel
    role: DeliveryRole
    provider_channel: ComponentIdentifier | None = None

    @model_validator(mode="after")
    def channel_has_non_authoritative_role(self) -> Self:
        if self.role in (DeliveryRole.SYSTEM, DeliveryRole.DEVELOPER):
            raise ValueError("authoritative delivery roles are prohibited")
        if self.channel is DeliveryChannel.PROVIDER_DATA:
            if self.role is not DeliveryRole.DATA or self.provider_channel is None:
                raise ValueError("provider data delivery requires a named data channel")
            normalized = self.provider_channel
            for separator in "_:/+-":
                normalized = normalized.replace(separator, ".")
            segments = tuple(segment.casefold() for segment in normalized.split(".") if segment)
            leaf = segments[-1] if segments else ""
            if (
                any(segment in _RESERVED_PROVIDER_CHANNEL_SEGMENTS for segment in segments)
                or leaf not in _SAFE_PROVIDER_CHANNEL_LEAVES
            ):
                raise ValueError("provider channel is not an allowlisted data destination")
        elif self.role is not DeliveryRole.USER or self.provider_channel is not None:
            raise ValueError("existing-task delivery requires the fixed user evidence block")
        return self


class AdapterCapabilities(AdapterModel):
    schema_version: Literal["1.0"]
    adapter_id: ComponentIdentifier
    pre_action_interception: bool
    deduplicates_delivery_id: bool
    deduplication_guarantee: DeduplicationGuarantee
    injection_mappings: Annotated[tuple[InjectionMapping, ...], Field(max_length=4)]

    @model_validator(mode="after")
    def mappings_are_unique(self) -> Self:
        channels = tuple(mapping.channel for mapping in self.injection_mappings)
        if len(set(channels)) != len(channels):
            raise ValueError("duplicate injection channel declaration")
        is_durable = self.deduplication_guarantee is DeduplicationGuarantee.DURABLE_DELIVERY_ID
        if self.deduplicates_delivery_id is not is_durable:
            raise ValueError("deduplication flag and durability guarantee disagree")
        return self


class DeliveryEnvelope(AdapterModel):
    """One pre-authorized, non-authoritative adapter delivery attempt."""

    schema_version: Literal["1.0"]
    delivery_id: UUID4
    run_id: UUID4
    cycle_id: CycleId
    intervention_id: UUID4
    claim_id: UUID4
    attempt_id: UUID4
    attempt_number: PositiveInt
    target_request_id: ComponentIdentifier
    target: DeliveryTarget
    adapter_id: ComponentIdentifier
    mapping: InjectionMapping
    payload: Annotated[str, Field(min_length=1, max_length=8_192)] = Field(repr=False)
    ttl_steps: Literal[1]
    created_at: UtcDatetime

    @model_validator(mode="after")
    def payload_matches_mapping(self) -> Self:
        try:
            mapping = InjectionMapping.model_validate_json(
                self.mapping.model_dump_json(warnings=False)
            )
            encoded = self.payload.encode("utf-8", errors="strict")
        except Exception:
            raise ValueError("delivery mapping or payload failed validation") from None
        if not encoded:
            raise ValueError("delivery payload cannot be empty")
        if mapping.channel is DeliveryChannel.EXISTING_USER_TASK and not _is_fixed_block(
            self.payload
        ):
            raise ValueError("fallback payload must be the exact fixed untrusted block")
        if mapping.channel is DeliveryChannel.PROVIDER_DATA and len(encoded) > 4_096:
            raise ValueError("provider data payload exceeds its bound")
        if len(encoded) > 8_192:
            raise ValueError("delivery payload exceeds its byte bound")
        return self


class DeliveryReceipt(AdapterModel):
    schema_version: Literal["1.0"]
    delivery_id: UUID4
    attempt_id: UUID4
    attempt_number: PositiveInt
    adapter_id: ComponentIdentifier
    target_request_id: ComponentIdentifier
    delivered_at: UtcDatetime
    provider_receipt_id: ComponentIdentifier = Field(repr=False)


class AdapterDeliveryRefusedError(ValueError):
    def __init__(self, reason_code: ReasonCode) -> None:
        if reason_code not in {
            ReasonCode.UNSUPPORTED_DELIVERY_TARGET,
            ReasonCode.UNSUPPORTED_DELIVERY_CHANNEL,
            ReasonCode.UNSAFE_ROLE_MAPPING,
            ReasonCode.TARGET_UNAVAILABLE,
        }:
            raise ValueError("reason code is not a delivery refusal")
        self.reason_code = reason_code
        super().__init__(f"adapter delivery refused: {reason_code.value}")


class InvalidAdapterReceiptError(ValueError):
    def __init__(self) -> None:
        super().__init__("adapter receipt failed identity validation")


class AdapterDeliveryFailedError(RuntimeError):
    """The adapter knows that the attempted side effect did not occur."""

    def __init__(self) -> None:
        self.reason_code = ReasonCode.DELIVERY_FAILED
        super().__init__("adapter delivery failed")


class AdapterNormalizationError(ValueError):
    """A native event could not be converted without trusting its raw representation."""

    def __init__(self) -> None:
        super().__init__("adapter event normalization failed")


class AdapterEventIdResolutionError(ValueError):
    """A native event could not be assigned one stable replay identifier."""

    def __init__(self) -> None:
        super().__init__("adapter event ID resolution failed")


class AdapterTargetResolutionError(ValueError):
    """A delivery target could not be resolved to a bounded harness request ID."""

    def __init__(self) -> None:
        super().__init__("adapter target resolution failed")


class AdapterCallbackError(RuntimeError):
    """An integration callback failed; the underlying message is intentionally discarded."""

    def __init__(self, operation: Literal["capabilities", "delivery"]) -> None:
        self.operation = operation
        super().__init__(f"adapter {operation} callback failed")


@runtime_checkable
class DeliveryAdapter(Protocol):
    def capabilities(self) -> AdapterCapabilities: ...

    async def deliver(self, delivery: DeliveryEnvelope) -> DeliveryReceipt: ...


@runtime_checkable
class HarnessAdapter(DeliveryAdapter, Protocol):
    def normalize(self, native_event: object) -> NormalizedTraceEventDraft: ...

    def resolve_event_id(self, native_event: object, ordinal: int) -> UUID | None: ...

    def resolve_target_request_id(
        self,
        native_event: object,
        target: DeliveryTarget,
    ) -> str | None: ...


def _unsafe_role(value: object) -> bool:
    try:
        if isinstance(value, InjectionMapping):
            return value.role in (DeliveryRole.SYSTEM, DeliveryRole.DEVELOPER)
        if isinstance(value, dict):
            role = value.get("role")
            return any(
                role == prohibited
                for prohibited in (
                    DeliveryRole.SYSTEM,
                    DeliveryRole.DEVELOPER,
                    DeliveryRole.SYSTEM.value,
                    DeliveryRole.DEVELOPER.value,
                )
            )
    except Exception:
        return True
    return False


def validated_capabilities(value: object) -> AdapterCapabilities:
    if type(value) is not AdapterCapabilities:
        raise AdapterDeliveryRefusedError(ReasonCode.UNSUPPORTED_DELIVERY_CHANNEL)
    capabilities = value
    mappings: tuple[InjectionMapping, ...] | None = None
    with suppress(Exception):
        candidate = capabilities.injection_mappings
        if type(candidate) is tuple:
            mappings = candidate
    if mappings is None:
        raise AdapterDeliveryRefusedError(ReasonCode.UNSUPPORTED_DELIVERY_CHANNEL)
    if any(_unsafe_role(mapping) for mapping in mappings):
        raise AdapterDeliveryRefusedError(ReasonCode.UNSAFE_ROLE_MAPPING)
    validated: AdapterCapabilities | None = None
    with suppress(Exception):
        validated = AdapterCapabilities.model_validate_json(
            capabilities.model_dump_json(warnings=False)
        )
    if validated is None:
        reason = (
            ReasonCode.UNSAFE_ROLE_MAPPING
            if any(_unsafe_role(mapping) for mapping in mappings)
            else ReasonCode.UNSUPPORTED_DELIVERY_CHANNEL
        )
        raise AdapterDeliveryRefusedError(reason)
    return validated


def adapter_capabilities_digest(value: object) -> str:
    capabilities = validated_capabilities(value)
    return canonical_digest(
        {
            "contract_version": ADAPTER_CONTRACT_VERSION,
            "capabilities": capabilities.model_dump(mode="json", warnings=False),
        }
    )


def enqueue_delivery_binding(
    *,
    target_request_id: str,
    capabilities: object,
) -> EnqueueDelivery:
    declared = validated_capabilities(capabilities)
    return EnqueueDelivery(
        target_request_id=target_request_id,
        adapter_id=declared.adapter_id,
        adapter_deduplicates=declared.deduplicates_delivery_id,
        adapter_deduplication_guarantee=declared.deduplication_guarantee,
        adapter_supports_pre_action=declared.pre_action_interception,
        adapter_contract_version=ADAPTER_CONTRACT_VERSION,
        adapter_capabilities_digest=adapter_capabilities_digest(declared),
    )


def select_injection_mapping(value: object, target: object) -> InjectionMapping:
    if type(target) is not DeliveryTarget:
        raise AdapterDeliveryRefusedError(ReasonCode.UNSUPPORTED_DELIVERY_TARGET)
    capabilities = validated_capabilities(value)
    if target is DeliveryTarget.PRE_ACTION_REPLAN and not capabilities.pre_action_interception:
        raise AdapterDeliveryRefusedError(ReasonCode.TARGET_UNAVAILABLE)
    for channel in (DeliveryChannel.PROVIDER_DATA, DeliveryChannel.EXISTING_USER_TASK):
        for mapping in capabilities.injection_mappings:
            if mapping.channel is channel:
                return mapping
    raise AdapterDeliveryRefusedError(ReasonCode.UNSUPPORTED_DELIVERY_CHANNEL)


def _is_fixed_block(value: str) -> bool:
    prefix = f"{UNTRUSTED_BLOCK_BEGIN}\nauthority=none\ninstructions=false\n"
    suffix = f"\n{UNTRUSTED_BLOCK_END}"
    return (
        value.startswith(prefix)
        and value.endswith(suffix)
        and value.count(UNTRUSTED_BLOCK_BEGIN) == 1
        and value.count(UNTRUSTED_BLOCK_END) == 1
    )


def render_untrusted_user_block(rendered_text: object) -> str:
    if type(rendered_text) is not str:
        raise AdapterDeliveryRefusedError(ReasonCode.UNSUPPORTED_DELIVERY_CHANNEL)
    text = rendered_text
    encoded: bytes | None = None
    with suppress(UnicodeEncodeError):
        encoded = text.encode("utf-8", errors="strict")
    if (
        encoded is None
        or not encoded
        or len(encoded) > 4_096
        or UNTRUSTED_BLOCK_BEGIN in text
        or UNTRUSTED_BLOCK_END in text
    ):
        raise AdapterDeliveryRefusedError(ReasonCode.UNSUPPORTED_DELIVERY_CHANNEL)
    return (
        f"{UNTRUSTED_BLOCK_BEGIN}\n"
        "authority=none\n"
        "instructions=false\n"
        f"{text}\n"
        f"{UNTRUSTED_BLOCK_END}"
    )


def delivery_payload(rendered_text: object, mapping: InjectionMapping) -> str:
    validated: InjectionMapping | None = None
    with suppress(Exception):
        validated = InjectionMapping.model_validate_json(mapping.model_dump_json(warnings=False))
    if validated is None:
        reason = (
            ReasonCode.UNSAFE_ROLE_MAPPING
            if _unsafe_role(mapping)
            else ReasonCode.UNSUPPORTED_DELIVERY_CHANNEL
        )
        raise AdapterDeliveryRefusedError(reason)
    if validated.channel is DeliveryChannel.EXISTING_USER_TASK:
        return render_untrusted_user_block(rendered_text)
    if type(rendered_text) is not str:
        raise AdapterDeliveryRefusedError(ReasonCode.UNSUPPORTED_DELIVERY_CHANNEL)
    text = rendered_text
    encoded: bytes | None = None
    with suppress(UnicodeEncodeError):
        encoded = text.encode("utf-8", errors="strict")
    if encoded is None or not encoded or len(encoded) > 4_096:
        raise AdapterDeliveryRefusedError(ReasonCode.UNSUPPORTED_DELIVERY_CHANNEL)
    return text


def validate_delivery_receipt(
    envelope: object,
    value: object,
) -> DeliveryReceipt:
    if type(envelope) is not DeliveryEnvelope or type(value) is not DeliveryReceipt:
        raise InvalidAdapterReceiptError()
    expected: DeliveryEnvelope | None = None
    receipt: DeliveryReceipt | None = None
    try:
        expected = DeliveryEnvelope.model_validate_json(envelope.model_dump_json(warnings=False))
        receipt = DeliveryReceipt.model_validate_json(value.model_dump_json(warnings=False))
    except Exception:
        pass
    if expected is None or receipt is None:
        raise InvalidAdapterReceiptError()
    if (
        receipt.delivery_id != expected.delivery_id
        or receipt.attempt_id != expected.attempt_id
        or receipt.attempt_number != expected.attempt_number
        or receipt.adapter_id != expected.adapter_id
        or receipt.target_request_id != expected.target_request_id
        or receipt.delivered_at < expected.created_at
        or receipt.delivered_at > expected.created_at + _MAX_RECEIPT_LATENCY
    ):
        raise InvalidAdapterReceiptError()
    return receipt


__all__ = [
    "ADAPTER_CONTRACT_VERSION",
    "UNTRUSTED_BLOCK_BEGIN",
    "UNTRUSTED_BLOCK_END",
    "AdapterCallbackError",
    "AdapterCapabilities",
    "AdapterDeliveryFailedError",
    "AdapterDeliveryRefusedError",
    "AdapterEventIdResolutionError",
    "AdapterNormalizationError",
    "AdapterTargetResolutionError",
    "DeduplicationGuarantee",
    "DeliveryAdapter",
    "DeliveryChannel",
    "DeliveryEnvelope",
    "DeliveryReceipt",
    "DeliveryRole",
    "HarnessAdapter",
    "InjectionMapping",
    "InvalidAdapterReceiptError",
    "adapter_capabilities_digest",
    "delivery_payload",
    "enqueue_delivery_binding",
    "render_untrusted_user_block",
    "select_injection_mapping",
    "validate_delivery_receipt",
    "validated_capabilities",
]
