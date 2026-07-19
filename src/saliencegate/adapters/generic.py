from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal
from uuid import UUID

from pydantic import TypeAdapter

from saliencegate.domain import (
    DeliveryTarget,
    NormalizedTraceEventDraft,
    ReasonCode,
    canonical_json,
    normalized_trace_event_draft_is_bounded,
    validate_normalized_trace_event_draft,
)
from saliencegate.ports.adapters import (
    AdapterCallbackError,
    AdapterCapabilities,
    AdapterDeliveryFailedError,
    AdapterDeliveryRefusedError,
    AdapterEventIdResolutionError,
    AdapterNormalizationError,
    AdapterTargetResolutionError,
    DeliveryEnvelope,
    DeliveryReceipt,
    InvalidAdapterReceiptError,
    validate_delivery_receipt,
    validated_capabilities,
)
from saliencegate.ports.repository import UUID4, ComponentIdentifier

NormalizeCallback = Callable[[object], object]
CapabilitiesCallback = Callable[[], object]
TargetRequestIdCallback = Callable[[object, DeliveryTarget], object]
EventIdCallback = Callable[[object, int], object]
DeliveryCallback = Callable[[DeliveryEnvelope], Awaitable[object]]

_TARGET_REQUEST_ID = TypeAdapter(ComponentIdentifier)
_EVENT_ID = TypeAdapter(UUID4)


def _copy_draft(value: object) -> NormalizedTraceEventDraft:
    validated: NormalizedTraceEventDraft | None = None
    try:
        validated = validate_normalized_trace_event_draft(value)
    except Exception:
        try:
            validated = NormalizedTraceEventDraft.model_validate_json(canonical_json(value))
        except Exception:
            raise AdapterNormalizationError() from None
    try:
        if not normalized_trace_event_draft_is_bounded(validated):
            raise ValueError
        return NormalizedTraceEventDraft.model_validate_json(
            validated.model_dump_json(warnings=False)
        )
    except Exception:
        raise AdapterNormalizationError() from None


def _copy_envelope(value: object) -> DeliveryEnvelope:
    if type(value) is not DeliveryEnvelope:
        raise InvalidAdapterReceiptError()
    try:
        return DeliveryEnvelope.model_validate_json(value.model_dump_json(warnings=False))
    except Exception:
        raise InvalidAdapterReceiptError() from None


def _fresh_refusal(
    error: AdapterDeliveryRefusedError,
    operation: Literal["capabilities", "delivery"],
) -> AdapterDeliveryRefusedError:
    try:
        reason = error.reason_code
        if type(reason) is not ReasonCode:
            raise TypeError
        return AdapterDeliveryRefusedError(reason)
    except Exception:
        raise AdapterCallbackError(operation) from None


class GenericHarnessAdapter:
    """Framework-neutral callbacks with validation and sanitized failure boundaries."""

    __slots__ = (
        "_capabilities_callback",
        "_delivery_callback",
        "_event_id_callback",
        "_normalize_callback",
        "_target_request_id_callback",
    )

    _normalize_callback: NormalizeCallback
    _capabilities_callback: CapabilitiesCallback
    _target_request_id_callback: TargetRequestIdCallback
    _delivery_callback: DeliveryCallback
    _event_id_callback: EventIdCallback | None

    def __init__(
        self,
        *,
        normalize_callback: NormalizeCallback,
        capabilities_callback: CapabilitiesCallback,
        target_request_id_callback: TargetRequestIdCallback,
        delivery_callback: DeliveryCallback,
        event_id_callback: EventIdCallback | None = None,
    ) -> None:
        callbacks = (
            normalize_callback,
            capabilities_callback,
            target_request_id_callback,
            delivery_callback,
        )
        if not all(callable(callback) for callback in callbacks):
            raise TypeError("adapter callbacks must be callable")
        if event_id_callback is not None and not callable(event_id_callback):
            raise TypeError("adapter callbacks must be callable")
        self._normalize_callback = normalize_callback
        self._capabilities_callback = capabilities_callback
        self._target_request_id_callback = target_request_id_callback
        self._delivery_callback = delivery_callback
        self._event_id_callback = event_id_callback

    def normalize(self, native_event: object) -> NormalizedTraceEventDraft:
        try:
            candidate = self._normalize_callback(native_event)
        except Exception:
            raise AdapterNormalizationError() from None
        return _copy_draft(candidate)

    def resolve_target_request_id(
        self,
        native_event: object,
        target: DeliveryTarget,
    ) -> str | None:
        if type(target) is not DeliveryTarget:
            raise AdapterTargetResolutionError()
        try:
            candidate = self._target_request_id_callback(native_event, target)
            if candidate is None:
                return None
            return _TARGET_REQUEST_ID.validate_python(candidate, strict=True)
        except Exception:
            raise AdapterTargetResolutionError() from None

    def resolve_event_id(self, native_event: object, ordinal: int) -> UUID | None:
        if type(ordinal) is not int or ordinal < 1:
            raise AdapterEventIdResolutionError()
        callback = self._event_id_callback
        if callback is None:
            return None
        try:
            candidate = callback(native_event, ordinal)
            if candidate is None:
                return None
            return _EVENT_ID.validate_python(candidate, strict=True)
        except Exception:
            raise AdapterEventIdResolutionError() from None

    def capabilities(self) -> AdapterCapabilities:
        try:
            validated = validated_capabilities(self._capabilities_callback())
            return AdapterCapabilities.model_validate_json(
                validated.model_dump_json(warnings=False)
            )
        except AdapterDeliveryRefusedError as error:
            raise _fresh_refusal(error, "capabilities") from None
        except Exception:
            raise AdapterCallbackError("capabilities") from None

    async def deliver(self, delivery: DeliveryEnvelope) -> DeliveryReceipt:
        envelope = _copy_envelope(delivery)
        try:
            candidate = await self._delivery_callback(envelope)
        except AdapterDeliveryFailedError:
            raise AdapterDeliveryFailedError() from None
        except AdapterDeliveryRefusedError as error:
            raise _fresh_refusal(error, "delivery") from None
        except InvalidAdapterReceiptError:
            raise InvalidAdapterReceiptError() from None
        except Exception:
            raise AdapterCallbackError("delivery") from None
        try:
            return validate_delivery_receipt(envelope, candidate)
        except Exception:
            raise InvalidAdapterReceiptError() from None


__all__ = ["GenericHarnessAdapter"]
