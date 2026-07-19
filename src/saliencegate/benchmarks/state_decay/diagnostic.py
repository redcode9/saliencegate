from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Never

from saliencegate.benchmarks.state_decay.generator import (
    generate_smoke_scenarios,
    validate_smoke_coverage,
)
from saliencegate.benchmarks.state_decay.oracle import (
    OracleResult,
    evaluate_scenario,
)
from saliencegate.benchmarks.state_decay.schema import StateDecayScenario
from saliencegate.domain import canonical_json


class StateDecayDiagnosticError(ValueError):
    """A value-free failure at the in-memory smoke-diagnostic boundary."""

    def __init__(self) -> None:
        super().__init__("state decay diagnostic failed")


@dataclass(frozen=True, slots=True)
class StateDecayDiagnosticResult:
    """One canonically revalidated smoke suite and its ordered oracle results."""

    scenarios: tuple[StateDecayScenario, ...]
    oracle_results: tuple[OracleResult, ...]


def _fail() -> Never:
    raise StateDecayDiagnosticError() from None


def evaluate_state_decay_scenarios(
    scenarios: Sequence[StateDecayScenario],
) -> StateDecayDiagnosticResult:
    """Revalidate and evaluate one complete frozen StateDecayBench smoke suite."""

    diagnostic: StateDecayDiagnosticResult | None = None
    try:
        supplied = tuple(scenarios)
        validate_smoke_coverage(supplied)
        rebuilt_scenarios = tuple(
            StateDecayScenario.model_validate_json(canonical_json(scenario))
            for scenario in supplied
        )
        if rebuilt_scenarios != supplied:
            raise ValueError

        evaluated = tuple(evaluate_scenario(scenario) for scenario in rebuilt_scenarios)
        rebuilt_results = tuple(
            OracleResult.model_validate_json(canonical_json(result)) for result in evaluated
        )
        scenario_ids = tuple(scenario.scenario_id for scenario in rebuilt_scenarios)
        result_ids = tuple(result.scenario_id for result in rebuilt_results)
        if (
            rebuilt_results != evaluated
            or len(rebuilt_scenarios) != 32
            or len(rebuilt_results) != len(rebuilt_scenarios)
            or result_ids != scenario_ids
            or any(result.matched is not True for result in rebuilt_results)
        ):
            raise ValueError
        diagnostic = StateDecayDiagnosticResult(
            scenarios=rebuilt_scenarios,
            oracle_results=rebuilt_results,
        )
    except Exception:
        pass
    if diagnostic is None:
        _fail()
    return diagnostic


def run_state_decay_diagnostic() -> StateDecayDiagnosticResult:
    """Generate and evaluate the frozen smoke suite without external access."""

    scenarios: tuple[StateDecayScenario, ...] | None = None
    with suppress(Exception):
        scenarios = generate_smoke_scenarios()
    if scenarios is None:
        _fail()
    return evaluate_state_decay_scenarios(scenarios)


__all__ = [
    "StateDecayDiagnosticError",
    "StateDecayDiagnosticResult",
    "evaluate_state_decay_scenarios",
    "run_state_decay_diagnostic",
]
