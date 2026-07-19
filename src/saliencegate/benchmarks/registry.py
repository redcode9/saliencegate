from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, PositiveInt


class BenchmarkNotFoundError(LookupError):
    """A value-free unknown benchmark identifier error."""

    def __init__(self) -> None:
        super().__init__("benchmark is not registered")


class BenchmarkDefinition(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    schema_version: Literal["benchmark-definition/v1"] = "benchmark-definition/v1"
    suite_id: Literal["state-decay-smoke"]
    suite_version: Literal["v1"]
    title: Literal["StateDecayBench smoke"]
    diagnostic: Literal[True]
    synthetic: Literal[True]
    balanced: Literal[True]
    external_claims_supported: Literal[False]
    scenario_count: PositiveInt


_DEFINITIONS = (
    BenchmarkDefinition(
        suite_id="state-decay-smoke",
        suite_version="v1",
        title="StateDecayBench smoke",
        diagnostic=True,
        synthetic=True,
        balanced=True,
        external_claims_supported=False,
        scenario_count=32,
    ),
)


def available_benchmarks() -> tuple[BenchmarkDefinition, ...]:
    return tuple(
        BenchmarkDefinition.model_validate_json(definition.model_dump_json(warnings=False))
        for definition in _DEFINITIONS
    )


def get_benchmark(name: str) -> BenchmarkDefinition:
    if type(name) is not str:
        raise BenchmarkNotFoundError() from None
    for definition in _DEFINITIONS:
        if name == definition.suite_id:
            return BenchmarkDefinition.model_validate_json(
                definition.model_dump_json(warnings=False)
            )
    raise BenchmarkNotFoundError() from None


__all__ = [
    "BenchmarkDefinition",
    "BenchmarkNotFoundError",
    "available_benchmarks",
    "get_benchmark",
]
