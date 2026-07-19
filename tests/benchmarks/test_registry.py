from __future__ import annotations

import pytest

from saliencegate.benchmarks.registry import (
    BenchmarkNotFoundError,
    available_benchmarks,
    get_benchmark,
)


def test_registry_exposes_only_the_public_smoke_suite() -> None:
    definitions = available_benchmarks()

    assert tuple(item.suite_id for item in definitions) == ("state-decay-smoke",)
    definition = definitions[0]
    assert definition.suite_version == "v1"
    assert definition.title == "StateDecayBench smoke"
    assert definition.diagnostic is True
    assert definition.synthetic is True
    assert definition.balanced is True
    assert definition.external_claims_supported is False
    assert definition.scenario_count == 32
    assert get_benchmark("state-decay-smoke") == definition


def test_registry_rejects_unknown_or_ill_typed_names_without_echoing_them() -> None:
    for name in ("fixture-secret-suite", "", object()):
        with pytest.raises(BenchmarkNotFoundError) as error:
            get_benchmark(name)  # type: ignore[arg-type]
        assert "fixture-secret" not in str(error.value)
        assert error.value.__cause__ is None
