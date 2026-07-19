from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID

import pytest

from saliencegate.adapters.generic import GenericHarnessAdapter
from saliencegate.domain import (
    MAX_TRACE_EVENT_PAYLOAD_BYTES,
    DeduplicationGuarantee,
    DeliveryTarget,
    EventPhase,
    EventType,
    NormalizedTraceEventDraft,
    ReasonCode,
    TrustLabel,
)
from saliencegate.ports.adapters import (
    AdapterCallbackError,
    AdapterCapabilities,
    AdapterDeliveryFailedError,
    AdapterDeliveryRefusedError,
    AdapterEventIdResolutionError,
    AdapterNormalizationError,
    AdapterTargetResolutionError,
    DeliveryChannel,
    DeliveryEnvelope,
    DeliveryReceipt,
    DeliveryRole,
    HarnessAdapter,
    InjectionMapping,
    InvalidAdapterReceiptError,
)

RUN_ID = UUID("00000000-0000-4000-8000-000000007001")
DELIVERY_ID = UUID("00000000-0000-4000-8000-000000007002")
INTERVENTION_ID = UUID("00000000-0000-4000-8000-000000007003")
CLAIM_ID = UUID("00000000-0000-4000-8000-000000007004")
ATTEMPT_ID = UUID("00000000-0000-4000-8000-000000007005")
NOW = datetime(2026, 7, 11, 14, 0, tzinfo=UTC)


def _draft(**changes: object) -> NormalizedTraceEventDraft:
    values: dict[str, object] = {
        "run_id": RUN_ID,
        "source_event_id": "native-event-1",
        "timestamp": NOW,
        "event_type": EventType.OBSERVATION,
        "phase": EventPhase.POST_ACTION,
        "payload": {"message": "tests failed"},
        "parent_ids": (),
        "source_adapter": "generic.fixture/1",
        "trust_label": TrustLabel.SYNTHETIC_FIXTURE,
    }
    values.update(changes)
    return NormalizedTraceEventDraft.model_validate(values)


def _capabilities() -> AdapterCapabilities:
    return AdapterCapabilities(
        schema_version="1.0",
        adapter_id="generic.fixture/1",
        pre_action_interception=True,
        deduplicates_delivery_id=True,
        deduplication_guarantee=DeduplicationGuarantee.DURABLE_DELIVERY_ID,
        injection_mappings=(
            InjectionMapping(
                channel=DeliveryChannel.PROVIDER_DATA,
                role=DeliveryRole.DATA,
                provider_channel="responses.context_data",
            ),
        ),
    )


def _envelope(**changes: object) -> DeliveryEnvelope:
    values: dict[str, object] = {
        "schema_version": "1.0",
        "delivery_id": DELIVERY_ID,
        "run_id": RUN_ID,
        "cycle_id": "a" * 64,
        "intervention_id": INTERVENTION_ID,
        "claim_id": CLAIM_ID,
        "attempt_id": ATTEMPT_ID,
        "attempt_number": 1,
        "target_request_id": "request-1",
        "target": DeliveryTarget.NEXT_MODEL_CALL,
        "adapter_id": "generic.fixture/1",
        "mapping": _capabilities().injection_mappings[0],
        "payload": "grounded reminder",
        "ttl_steps": 1,
        "created_at": NOW,
    }
    values.update(changes)
    return DeliveryEnvelope.model_validate(values)


def _receipt(**changes: object) -> DeliveryReceipt:
    values: dict[str, object] = {
        "schema_version": "1.0",
        "delivery_id": DELIVERY_ID,
        "attempt_id": ATTEMPT_ID,
        "attempt_number": 1,
        "adapter_id": "generic.fixture/1",
        "target_request_id": "request-1",
        "delivered_at": NOW + timedelta(milliseconds=1),
        "provider_receipt_id": "receipt-1",
    }
    values.update(changes)
    return DeliveryReceipt.model_validate(values)


def _adapter(
    *,
    normalize: object | None = None,
    capabilities: object | None = None,
    resolve_target_request_id: object | None = None,
    resolve_event_id: object | None = None,
    deliver: object | None = None,
) -> GenericHarnessAdapter:
    async def deliver_default(_delivery: DeliveryEnvelope) -> DeliveryReceipt:
        return _receipt()

    return GenericHarnessAdapter(
        normalize_callback=cast(object, normalize) if normalize is not None else lambda _: _draft(),
        capabilities_callback=(
            cast(object, capabilities) if capabilities is not None else _capabilities
        ),
        target_request_id_callback=(
            cast(object, resolve_target_request_id)
            if resolve_target_request_id is not None
            else lambda _event, target: (
                "request-1" if target is DeliveryTarget.NEXT_MODEL_CALL else None
            )
        ),
        delivery_callback=cast(object, deliver) if deliver is not None else deliver_default,
        event_id_callback=(
            cast(object, resolve_event_id) if resolve_event_id is not None else None
        ),
    )


@pytest.mark.asyncio
async def test_generic_adapter_normalizes_resolves_capabilities_and_delivers() -> None:
    native = {"native": "event"}
    seen: list[object] = []

    def normalize_callback(value: object) -> NormalizedTraceEventDraft:
        seen.append(value)
        return _draft()

    async def delivery_callback(value: DeliveryEnvelope) -> DeliveryReceipt:
        seen.append(value)
        return _receipt()

    adapter = _adapter(normalize=normalize_callback, deliver=delivery_callback)

    normalized = adapter.normalize(native)
    capabilities = adapter.capabilities()
    target = adapter.resolve_target_request_id(native, DeliveryTarget.NEXT_MODEL_CALL)
    receipt = await adapter.deliver(_envelope())

    assert isinstance(adapter, HarnessAdapter)
    assert seen == [native, _envelope()]
    assert normalized == _draft()
    assert normalized is not _draft()
    assert capabilities == _capabilities()
    assert capabilities is not _capabilities()
    assert target == "request-1"
    assert receipt == _receipt()
    assert receipt is not _receipt()


def test_generic_adapter_exposes_a_validated_event_id_mapping() -> None:
    event_id = UUID("00000000-0000-4000-8000-000000007006")
    native = {"native": "event"}
    adapter = _adapter(resolve_event_id=lambda value, ordinal: event_id)

    assert adapter.resolve_event_id(native, 1) == event_id
    assert _adapter().resolve_event_id(native, 1) is None
    with pytest.raises(AdapterEventIdResolutionError):
        adapter.resolve_event_id(native, 0)
    with pytest.raises(AdapterEventIdResolutionError):
        _adapter(resolve_event_id=lambda _value, _ordinal: "not-a-uuid").resolve_event_id(
            native,
            1,
        )


def test_generic_adapter_revalidates_a_mapping_and_returns_a_deep_copy() -> None:
    mutable = _draft().model_dump(mode="json")
    adapter = _adapter(normalize=lambda _: mutable)

    normalized = adapter.normalize(object())
    cast(dict[str, object], mutable["payload"])["message"] = "changed later"

    assert normalized.payload["message"] == "tests failed"


@pytest.mark.parametrize("returned", (None, object(), {"record_type": "wrong"}))
def test_normalization_rejects_invalid_callback_results_without_input(returned: object) -> None:
    secret = "sk-raw-normalization-secret"
    adapter = _adapter(normalize=lambda _: returned)

    with pytest.raises(AdapterNormalizationError) as raised:
        adapter.normalize({"payload": secret})

    assert secret not in str(raised.value)
    assert raised.value.__cause__ is None


def test_normalization_sanitizes_arbitrary_callback_exceptions() -> None:
    secret = "raw-callback-secret"

    def fail(_: object) -> object:
        raise RuntimeError(secret)

    with pytest.raises(AdapterNormalizationError) as raised:
        _adapter(normalize=fail).normalize(object())

    assert secret not in str(raised.value)
    assert raised.value.__cause__ is None


def test_normalization_revalidates_forged_draft_instances() -> None:
    forged = _draft().model_copy(update={"source_event_id": "invalid id"})

    with pytest.raises(AdapterNormalizationError):
        _adapter(normalize=lambda _: forged).normalize(object())


@pytest.mark.parametrize("returned", (None, 3, "bad id", "invalid$id"))
def test_target_resolution_rejects_invalid_ids_without_exposing_values(returned: object) -> None:
    adapter = _adapter(resolve_target_request_id=lambda _event, _target: returned)

    if returned is None:
        assert adapter.resolve_target_request_id(object(), DeliveryTarget.PRE_ACTION_REPLAN) is None
    else:
        with pytest.raises(AdapterTargetResolutionError) as raised:
            adapter.resolve_target_request_id(object(), DeliveryTarget.NEXT_MODEL_CALL)
        assert str(returned) not in str(raised.value)
        assert raised.value.__cause__ is None


def test_target_resolution_validates_target_before_callback() -> None:
    called = False

    def resolve(_event: object, _target: DeliveryTarget) -> str:
        nonlocal called
        called = True
        return "request-1"

    adapter = _adapter(resolve_target_request_id=resolve)
    with pytest.raises(AdapterTargetResolutionError):
        adapter.resolve_target_request_id(object(), cast(DeliveryTarget, "next_model_call"))
    assert called is False


def test_target_resolution_sanitizes_callback_exceptions() -> None:
    secret = "target callback leaked me"

    def fail(_event: object, _target: DeliveryTarget) -> str:
        raise RuntimeError(secret)

    with pytest.raises(AdapterTargetResolutionError) as raised:
        _adapter(resolve_target_request_id=fail).resolve_target_request_id(
            object(), DeliveryTarget.NEXT_MODEL_CALL
        )
    assert secret not in str(raised.value)
    assert raised.value.__cause__ is None


def test_capabilities_are_revalidated_and_callback_errors_are_sanitized() -> None:
    secret = "capability callback leaked me"

    def fail() -> AdapterCapabilities:
        raise RuntimeError(secret)

    with pytest.raises(AdapterCallbackError) as raised:
        _adapter(capabilities=fail).capabilities()
    assert raised.value.operation == "capabilities"
    assert secret not in str(raised.value)
    assert raised.value.__cause__ is None

    forged = _capabilities().model_copy(
        update={"injection_mappings": ({"role": "system", "channel": "provider_data"},)}
    )
    with pytest.raises(AdapterDeliveryRefusedError) as refusal:
        _adapter(capabilities=lambda: forged).capabilities()
    assert refusal.value.reason_code is ReasonCode.UNSAFE_ROLE_MAPPING


@pytest.mark.asyncio
async def test_delivery_preserves_known_failures_and_sanitizes_unknown_exceptions() -> None:
    async def known(_delivery: DeliveryEnvelope) -> DeliveryReceipt:
        raise AdapterDeliveryFailedError()

    with pytest.raises(AdapterDeliveryFailedError):
        await _adapter(deliver=known).deliver(_envelope())

    secret = "provider leaked a raw response"

    async def unknown(_delivery: DeliveryEnvelope) -> DeliveryReceipt:
        raise RuntimeError(secret)

    with pytest.raises(AdapterCallbackError) as raised:
        await _adapter(deliver=unknown).deliver(_envelope())
    assert raised.value.operation == "delivery"
    assert secret not in str(raised.value)
    assert raised.value.__cause__ is None


@pytest.mark.asyncio
async def test_known_callback_errors_are_rebuilt_without_hostile_arguments() -> None:
    secret = "known-callback-error-secret"
    failure = AdapterDeliveryFailedError()
    failure.args = (secret,)
    refusal = AdapterDeliveryRefusedError(ReasonCode.TARGET_UNAVAILABLE)
    refusal.args = (secret,)
    invalid_receipt = InvalidAdapterReceiptError()
    invalid_receipt.args = (secret,)

    async def fail(_delivery: DeliveryEnvelope) -> DeliveryReceipt:
        raise failure

    async def refuse(_delivery: DeliveryEnvelope) -> DeliveryReceipt:
        raise refusal

    async def invalid(_delivery: DeliveryEnvelope) -> DeliveryReceipt:
        raise invalid_receipt

    with pytest.raises(AdapterDeliveryFailedError) as failed:
        await _adapter(deliver=fail).deliver(_envelope())
    assert failed.value is not failure
    assert str(failed.value) == "adapter delivery failed"

    with pytest.raises(AdapterDeliveryRefusedError) as refused:
        await _adapter(deliver=refuse).deliver(_envelope())
    assert refused.value is not refusal
    assert refused.value.reason_code is ReasonCode.TARGET_UNAVAILABLE
    assert secret not in str(refused.value)

    with pytest.raises(InvalidAdapterReceiptError) as invalid_result:
        await _adapter(deliver=invalid).deliver(_envelope())
    assert invalid_result.value is not invalid_receipt
    assert secret not in str(invalid_result.value)


def test_capability_callback_refusals_are_rebuilt_and_invalid_reasons_are_sanitized() -> None:
    secret = "capability-refusal-secret"
    refusal = AdapterDeliveryRefusedError(ReasonCode.UNSAFE_ROLE_MAPPING)
    refusal.args = (secret,)

    def refuse() -> AdapterCapabilities:
        raise refusal

    with pytest.raises(AdapterDeliveryRefusedError) as raised:
        _adapter(capabilities=refuse).capabilities()
    assert raised.value is not refusal
    assert raised.value.reason_code is ReasonCode.UNSAFE_ROLE_MAPPING
    assert secret not in str(raised.value)

    refusal.reason_code = cast(ReasonCode, secret)
    with pytest.raises(AdapterCallbackError) as invalid:
        _adapter(capabilities=refuse).capabilities()
    assert invalid.value.operation == "capabilities"
    assert secret not in str(invalid.value)


@pytest.mark.asyncio
async def test_delivery_revalidates_input_and_receipt_identity() -> None:
    seen: list[DeliveryEnvelope] = []

    async def delivery_callback(value: DeliveryEnvelope) -> DeliveryReceipt:
        seen.append(value)
        return _receipt(attempt_number=2)

    envelope = _envelope()
    with pytest.raises(InvalidAdapterReceiptError):
        await _adapter(deliver=delivery_callback).deliver(envelope)
    assert seen == [envelope]
    assert seen[0] is not envelope

    with pytest.raises(InvalidAdapterReceiptError):
        await _adapter().deliver(cast(DeliveryEnvelope, object()))

    forged = _envelope().model_copy(update={"attempt_number": 0})
    with pytest.raises(InvalidAdapterReceiptError):
        await _adapter().deliver(forged)


def test_constructor_rejects_non_callbacks_without_echoing_them() -> None:
    with pytest.raises(TypeError, match="adapter callbacks must be callable"):
        GenericHarnessAdapter(
            normalize_callback=cast(object, "raw secret"),
            capabilities_callback=_capabilities,
            target_request_id_callback=lambda _event, _target: None,
            delivery_callback=cast(object, None),
        )


def test_normalization_rejects_a_forged_oversized_payload() -> None:
    forged = _draft().model_copy(
        update={"payload": {"text": "x" * (MAX_TRACE_EVENT_PAYLOAD_BYTES + 1)}}
    )

    with pytest.raises(AdapterNormalizationError):
        _adapter(normalize=lambda _value: forged).normalize(object())
