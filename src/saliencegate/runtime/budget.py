from __future__ import annotations

from pydantic import ValidationError

from saliencegate.domain import BudgetAmounts, BudgetSnapshot

BUDGET_FIELDS = (
    "model_calls",
    "input_tokens",
    "output_tokens",
    "canonical_token_equivalents",
    "latency_us",
    "interventions",
    "schema_repairs",
)


class BudgetError(ValueError):
    """Base class for budget failures that never embeds caller-provided values."""


class BudgetInputError(BudgetError):
    def __init__(self, input_type: str) -> None:
        super().__init__(f"{input_type} failed budget validation")


class BudgetReservationDeniedError(BudgetError):
    def __init__(self) -> None:
        super().__init__("budget reservation exceeds the available balance")


class BudgetSettlementError(BudgetError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"budget settlement rejected: {reason}")


def _validate_amounts(amounts: object) -> BudgetAmounts:
    if type(amounts) is not BudgetAmounts:
        raise BudgetInputError("amounts")
    try:
        return BudgetAmounts.model_validate_json(amounts.model_dump_json(warnings=False))
    except (AttributeError, ValidationError):
        raise BudgetInputError("amounts") from None


def _validate_snapshot(snapshot: object) -> BudgetSnapshot:
    if type(snapshot) is not BudgetSnapshot:
        raise BudgetInputError("snapshot")
    try:
        return BudgetSnapshot.model_validate_json(snapshot.model_dump_json(warnings=False))
    except (AttributeError, ValidationError):
        raise BudgetInputError("snapshot") from None


def _validate_call_latencies(value: object) -> tuple[int, ...]:
    if type(value) is not tuple or any(type(item) is not int or item < 0 for item in value):
        raise BudgetInputError("model-call latencies")
    return value


def _reconcile(
    snapshot: BudgetSnapshot,
    reservation: BudgetAmounts,
    actual: BudgetAmounts,
) -> BudgetSnapshot:
    if any(
        getattr(reservation, field_name) > getattr(snapshot.reserved, field_name)
        for field_name in BUDGET_FIELDS
    ):
        raise BudgetSettlementError("reservation is not held")
    if any(
        getattr(actual, field_name) > getattr(reservation, field_name)
        for field_name in BUDGET_FIELDS
    ):
        raise BudgetSettlementError("actual usage exceeds the reservation")
    reserved = BudgetAmounts(
        **{
            field_name: getattr(snapshot.reserved, field_name) - getattr(reservation, field_name)
            for field_name in BUDGET_FIELDS
        }
    )
    consumed = BudgetAmounts(
        **{
            field_name: getattr(snapshot.consumed, field_name) + getattr(actual, field_name)
            for field_name in BUDGET_FIELDS
        }
    )
    return BudgetSnapshot(
        limits=snapshot.limits,
        reserved=reserved,
        consumed=consumed,
    )


class BudgetGovernor:
    """Pure, integer-only reservation and settlement arithmetic."""

    __slots__ = ()

    def available(self, snapshot: BudgetSnapshot) -> BudgetAmounts:
        validated = _validate_snapshot(snapshot)
        return BudgetAmounts(
            **{
                field_name: getattr(validated.limits, field_name)
                - getattr(validated.reserved, field_name)
                - getattr(validated.consumed, field_name)
                for field_name in BUDGET_FIELDS
            }
        )

    def reserve(
        self,
        snapshot: BudgetSnapshot,
        requested: BudgetAmounts,
    ) -> BudgetSnapshot:
        validated = _validate_snapshot(snapshot)
        request = _validate_amounts(requested)
        available = self.available(validated)
        if any(
            getattr(request, field_name) > getattr(available, field_name)
            for field_name in BUDGET_FIELDS
        ):
            raise BudgetReservationDeniedError()
        reserved = BudgetAmounts(
            **{
                field_name: getattr(validated.reserved, field_name) + getattr(request, field_name)
                for field_name in BUDGET_FIELDS
            }
        )
        return BudgetSnapshot(
            limits=validated.limits,
            reserved=reserved,
            consumed=validated.consumed,
        )

    def settle(
        self,
        snapshot: BudgetSnapshot,
        reservation: BudgetAmounts,
        actual: BudgetAmounts,
        *,
        model_call_latencies_us: tuple[int, ...],
    ) -> BudgetSnapshot:
        validated = _validate_snapshot(snapshot)
        held = _validate_amounts(reservation)
        consumed_now = _validate_amounts(actual)
        call_latencies = _validate_call_latencies(model_call_latencies_us)
        if len(call_latencies) != consumed_now.model_calls:
            raise BudgetSettlementError("model-call latency count does not match actual calls")
        if sum(call_latencies) > consumed_now.latency_us:
            raise BudgetSettlementError("model-call latency exceeds actual cycle latency")
        if any(latency_us > validated.limits.max_call_latency_us for latency_us in call_latencies):
            raise BudgetSettlementError("a model call exceeds the per-call latency limit")
        return _reconcile(validated, held, consumed_now)

    def consume_unknown(
        self,
        snapshot: BudgetSnapshot,
        reservation: BudgetAmounts,
    ) -> BudgetSnapshot:
        validated = _validate_snapshot(snapshot)
        held = _validate_amounts(reservation)
        return _reconcile(validated, held, held)
