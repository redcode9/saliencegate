from __future__ import annotations

import builtins
import os
import socket
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import cast

import pytest

import saliencegate.benchmarks.state_decay.diagnostic as diagnostic_module
from saliencegate.benchmarks.state_decay.diagnostic import (
    StateDecayDiagnosticError,
    StateDecayDiagnosticResult,
    evaluate_state_decay_scenarios,
    run_state_decay_diagnostic,
)
from saliencegate.benchmarks.state_decay.generator import generate_smoke_scenarios
from saliencegate.benchmarks.state_decay.oracle import OracleEvaluationError, OracleResult
from saliencegate.benchmarks.state_decay.schema import (
    InterventionLabel,
    ScenarioFamily,
    StateDecayScenario,
)


class _SinglePassScenarios(Sequence[StateDecayScenario]):
    def __init__(self, values: tuple[StateDecayScenario, ...]) -> None:
        self._values = values
        self.iterations = 0

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> StateDecayScenario:
        return self._values[index]

    def __iter__(self) -> Iterator[StateDecayScenario]:
        self.iterations += 1
        if self.iterations != 1:
            raise AssertionError("scenario input was materialized more than once")
        return iter(self._values)


def _assert_value_free(error: StateDecayDiagnosticError) -> None:
    assert str(error) == "state decay diagnostic failed"
    assert error.__cause__ is None
    assert error.__context__ is None


def test_evaluator_returns_one_frozen_canonically_revalidated_result() -> None:
    original = generate_smoke_scenarios()
    source = _SinglePassScenarios(original)

    result = evaluate_state_decay_scenarios(source)

    assert isinstance(result, StateDecayDiagnosticResult)
    assert source.iterations == 1
    assert result.scenarios == original
    assert result.scenarios is not original
    assert all(
        rebuilt is not supplied
        for rebuilt, supplied in zip(result.scenarios, original, strict=True)
    )
    assert len(result.scenarios) == len(result.oracle_results) == 32
    assert Counter(scenario.family for scenario in result.scenarios) == {
        family: 4 for family in ScenarioFamily
    }
    assert Counter(scenario.label for scenario in result.scenarios) == {
        InterventionLabel.INTERVENE: 16,
        InterventionLabel.SILENCE: 16,
    }
    assert all(type(item) is OracleResult and item.matched for item in result.oracle_results)
    assert tuple(item.scenario_id for item in result.oracle_results) == tuple(
        scenario.scenario_id for scenario in result.scenarios
    )
    with pytest.raises(FrozenInstanceError):
        result.scenarios = ()  # type: ignore[misc]


def test_frozen_generator_diagnostic_is_repeatable() -> None:
    first = run_state_decay_diagnostic()
    second = run_state_decay_diagnostic()

    assert first == second
    assert first.scenarios is not second.scenarios
    assert first.oracle_results is not second.oracle_results


def test_generator_failure_is_mapped_to_the_value_free_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        diagnostic_module,
        "generate_smoke_scenarios",
        lambda: (_ for _ in ()).throw(ValueError("generator-secret")),
    )

    with pytest.raises(StateDecayDiagnosticError) as captured:
        run_state_decay_diagnostic()

    _assert_value_free(captured.value)
    assert "generator-secret" not in repr(captured.value)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_evaluator_rejects_incomplete_extra_and_duplicate_scenarios(mutation: str) -> None:
    scenarios = generate_smoke_scenarios()
    if mutation == "missing":
        invalid = scenarios[:-1]
    elif mutation == "extra":
        invalid = (*scenarios, scenarios[0])
    else:
        invalid = (scenarios[0], scenarios[0], *scenarios[2:])

    with pytest.raises(StateDecayDiagnosticError) as captured:
        evaluate_state_decay_scenarios(invalid)

    _assert_value_free(captured.value)


def test_evaluator_rejects_coverage_failure_without_leaking_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_coverage(values: tuple[StateDecayScenario, ...]) -> None:
        del values
        raise ValueError("scenario-secret")

    monkeypatch.setattr(diagnostic_module, "validate_smoke_coverage", fail_coverage)

    with pytest.raises(StateDecayDiagnosticError) as captured:
        evaluate_state_decay_scenarios(generate_smoke_scenarios())

    _assert_value_free(captured.value)
    assert "scenario-secret" not in repr(captured.value)


def test_evaluator_does_not_trust_changed_scenario_revalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenarios = generate_smoke_scenarios()
    monkeypatch.setattr(diagnostic_module, "validate_smoke_coverage", lambda values: None)
    monkeypatch.setattr(
        diagnostic_module.StateDecayScenario,
        "model_validate_json",
        lambda value: scenarios[0],
    )

    with pytest.raises(StateDecayDiagnosticError) as captured:
        evaluate_state_decay_scenarios(scenarios)

    _assert_value_free(captured.value)


@pytest.mark.parametrize(
    "mutation",
    ["oracle_error", "non_match", "wrong_identity", "reordered_identity"],
)
def test_evaluator_rejects_invalid_oracle_results(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    scenarios = generate_smoke_scenarios()
    real_evaluate = diagnostic_module.evaluate_scenario
    if mutation == "oracle_error":

        def evaluate(scenario: StateDecayScenario) -> OracleResult:
            del scenario
            raise OracleEvaluationError()

    elif mutation == "non_match":

        def evaluate(scenario: StateDecayScenario) -> OracleResult:
            return real_evaluate(scenario).model_copy(update={"matched": False})

    elif mutation == "wrong_identity":

        def evaluate(scenario: StateDecayScenario) -> OracleResult:
            return real_evaluate(scenario).model_copy(update={"scenario_id": "a" * 64})

    else:
        reordered = iter(tuple(real_evaluate(item) for item in reversed(scenarios)))

        def evaluate(scenario: StateDecayScenario) -> OracleResult:
            del scenario
            return next(reordered)

    monkeypatch.setattr(diagnostic_module, "evaluate_scenario", evaluate)

    with pytest.raises(StateDecayDiagnosticError) as captured:
        evaluate_state_decay_scenarios(scenarios)

    _assert_value_free(captured.value)


def test_evaluator_does_not_trust_changed_oracle_revalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenarios = generate_smoke_scenarios()
    changed = diagnostic_module.evaluate_scenario(scenarios[0])
    monkeypatch.setattr(
        diagnostic_module.OracleResult,
        "model_validate_json",
        lambda value: changed,
    )

    with pytest.raises(StateDecayDiagnosticError) as captured:
        evaluate_state_decay_scenarios(scenarios)

    _assert_value_free(captured.value)


def test_diagnostic_is_in_memory_and_does_not_load_optional_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("external access attempted")

    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "httpx" or name.startswith("saliencegate.models"):
            raise AssertionError("optional runtime import attempted")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    monkeypatch.setattr(os, "getenv", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    before = frozenset(sys.modules)

    result = run_state_decay_diagnostic()

    assert len(result.scenarios) == 32
    loaded = frozenset(sys.modules) - before
    assert not any(name == "httpx" or name.startswith("saliencegate.models") for name in loaded)
    assert not hasattr(diagnostic_module, "os")
    assert not hasattr(diagnostic_module, "subprocess")
    assert not hasattr(diagnostic_module, "socket")


def test_evaluator_rejects_non_sequence_content_value_free() -> None:
    with pytest.raises(StateDecayDiagnosticError) as captured:
        evaluate_state_decay_scenarios(cast("Sequence[StateDecayScenario]", [object()]))

    _assert_value_free(captured.value)
