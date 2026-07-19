from __future__ import annotations

from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from saliencegate.domain import BudgetAmounts, BudgetLimits, BudgetSnapshot
from saliencegate.runtime import (
    BudgetGovernor,
    BudgetInputError,
    BudgetReservationDeniedError,
    BudgetSettlementError,
)


def snapshot(*, limit: int = 100) -> BudgetSnapshot:
    return BudgetSnapshot(
        limits=BudgetLimits(
            model_calls=limit,
            input_tokens=limit,
            output_tokens=limit,
            canonical_token_equivalents=limit,
            latency_us=limit,
            max_call_latency_us=limit,
            interventions=limit,
            schema_repairs=limit,
        ),
        reserved=BudgetAmounts(),
        consumed=BudgetAmounts(),
    )


def test_reservation_and_settlement_release_unused_balance() -> None:
    governor = BudgetGovernor()
    requested = BudgetAmounts(
        model_calls=1,
        input_tokens=40,
        output_tokens=20,
        canonical_token_equivalents=60,
        latency_us=50,
    )
    reserved = governor.reserve(snapshot(), requested)
    actual = BudgetAmounts(
        model_calls=1,
        input_tokens=30,
        output_tokens=10,
        canonical_token_equivalents=40,
        latency_us=25,
    )
    settled = governor.settle(
        reserved,
        requested,
        actual,
        model_call_latencies_us=(25,),
    )

    assert settled.reserved == BudgetAmounts()
    assert settled.consumed == actual
    assert governor.available(settled).input_tokens == 70
    assert governor.available(settled).output_tokens == 90


@pytest.mark.parametrize(
    "field_name",
    (
        "model_calls",
        "input_tokens",
        "output_tokens",
        "canonical_token_equivalents",
        "latency_us",
        "interventions",
        "schema_repairs",
    ),
)
def test_each_budget_dimension_has_a_hard_reservation_ceiling(field_name: str) -> None:
    governor = BudgetGovernor()
    request = BudgetAmounts(**{field_name: 2})

    with pytest.raises(BudgetReservationDeniedError):
        governor.reserve(snapshot(limit=1), request)


def test_settlement_requires_a_held_reservation_and_bounded_actuals() -> None:
    governor = BudgetGovernor()
    initial = snapshot(limit=10)
    held = BudgetAmounts(model_calls=1, input_tokens=5)
    reserved = governor.reserve(initial, held)

    with pytest.raises(BudgetSettlementError, match="not held"):
        governor.settle(
            initial,
            held,
            BudgetAmounts(),
            model_call_latencies_us=(),
        )
    with pytest.raises(BudgetSettlementError, match="exceeds"):
        governor.settle(
            reserved,
            held,
            BudgetAmounts(model_calls=1, input_tokens=6),
            model_call_latencies_us=(0,),
        )


def test_per_call_latency_uses_individual_authenticated_measurements() -> None:
    governor = BudgetGovernor()
    initial = BudgetSnapshot(
        limits=BudgetLimits(
            model_calls=10,
            latency_us=100,
            max_call_latency_us=10,
        ),
        reserved=BudgetAmounts(),
        consumed=BudgetAmounts(),
    )

    overhead_reservation = governor.reserve(
        initial,
        BudgetAmounts(model_calls=1, latency_us=11),
    )
    assert overhead_reservation.reserved.latency_us == 11

    held = BudgetAmounts(model_calls=10, latency_us=100)
    reserved = governor.reserve(initial, held)
    with pytest.raises(BudgetSettlementError, match="per-call latency"):
        governor.settle(
            reserved,
            held,
            BudgetAmounts(model_calls=1, latency_us=100),
            model_call_latencies_us=(100,),
        )

    settled = governor.settle(
        reserved,
        held,
        BudgetAmounts(model_calls=1, latency_us=10),
        model_call_latencies_us=(10,),
    )
    assert settled.consumed.latency_us == 10

    two_calls = BudgetAmounts(model_calls=2, latency_us=20)
    two_reserved = governor.reserve(initial, two_calls)
    with pytest.raises(BudgetSettlementError, match="per-call latency"):
        governor.settle(
            two_reserved,
            two_calls,
            two_calls,
            model_call_latencies_us=(20, 0),
        )


def test_settlement_requires_complete_call_latency_receipts() -> None:
    governor = BudgetGovernor()
    held = BudgetAmounts(model_calls=1, latency_us=10)
    reserved = governor.reserve(snapshot(), held)

    with pytest.raises(BudgetSettlementError, match="count"):
        governor.settle(
            reserved,
            held,
            held,
            model_call_latencies_us=(),
        )
    with pytest.raises(BudgetSettlementError, match="exceeds actual"):
        governor.settle(
            reserved,
            held,
            held,
            model_call_latencies_us=(11,),
        )
    with pytest.raises(BudgetInputError, match="model-call latencies"):
        governor.settle(
            reserved,
            held,
            held,
            model_call_latencies_us=cast(tuple[int, ...], [10]),
        )


def test_unknown_cost_consumes_the_full_reservation() -> None:
    governor = BudgetGovernor()
    held = BudgetAmounts(model_calls=1, input_tokens=8, latency_us=10)
    reserved = governor.reserve(snapshot(limit=10), held)

    settled = governor.consume_unknown(reserved, held)

    assert settled.reserved == BudgetAmounts()
    assert settled.consumed == held


def test_unchecked_invalid_models_fail_without_echoing_values() -> None:
    secret = "fixture-secret-must-not-echo"
    invalid = BudgetAmounts().model_copy(update={"model_calls": secret})

    with pytest.raises(BudgetInputError) as error:
        BudgetGovernor().reserve(snapshot(), invalid)

    assert secret not in str(error.value)


def test_budget_inputs_require_exact_revalidatable_models() -> None:
    governor = BudgetGovernor()

    with pytest.raises(BudgetInputError, match="amounts"):
        governor.reserve(snapshot(), cast(BudgetAmounts, object()))
    with pytest.raises(BudgetInputError, match="snapshot"):
        governor.available(cast(BudgetSnapshot, object()))

    invalid_snapshot = snapshot().model_copy(update={"reserved": "unchecked"})
    with pytest.raises(BudgetInputError, match="snapshot"):
        governor.available(invalid_snapshot)


@given(
    limit=st.integers(min_value=0, max_value=10_000),
    requested=st.integers(min_value=0, max_value=10_000),
)
def test_reservation_property_never_crosses_the_limit(limit: int, requested: int) -> None:
    governor = BudgetGovernor()
    request = BudgetAmounts(input_tokens=requested)

    if requested > limit:
        with pytest.raises(BudgetReservationDeniedError):
            governor.reserve(snapshot(limit=limit), request)
    else:
        reserved = governor.reserve(snapshot(limit=limit), request)
        assert reserved.reserved.input_tokens == requested
        assert reserved.reserved.input_tokens <= reserved.limits.input_tokens
