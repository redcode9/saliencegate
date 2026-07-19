from __future__ import annotations

from inspect import signature
from itertools import product

import pytest
from pydantic import ValidationError

from saliencegate.benchmarks.state_decay_v2.public_contract import (
    PublicCausalExposure,
    PublicCausalFactor,
    PublicCausalFactorValue,
    PublicTerminalState,
    PublicTransition,
    PublicTransitionGraph,
    PublicTransitionState,
    derive_causal_outcome,
    execute_public_transition_graph,
    transition_graph_digest,
)
from saliencegate.benchmarks.state_decay_v2.schema import ScenarioOutcome
from saliencegate.domain import canonical_json

_FACTOR_IDS = ("guidance-relevant", "baseline-can-recover")
_FACTOR_VECTORS = tuple(product((False, True), repeat=2))

_PAIRED_TERMINALS = {
    (False, False): (
        PublicTerminalState.GOAL_REACHED,
        PublicTerminalState.GOAL_NOT_REACHED,
    ),
    (False, True): (
        PublicTerminalState.GOAL_NOT_REACHED,
        PublicTerminalState.GOAL_REACHED,
    ),
    (True, False): (
        PublicTerminalState.GOAL_REACHED,
        PublicTerminalState.GOAL_REACHED,
    ),
    (True, True): (
        PublicTerminalState.GOAL_NOT_REACHED,
        PublicTerminalState.GOAL_NOT_REACHED,
    ),
}

_PAIRED_OUTCOMES = {
    (False, False): ScenarioOutcome.HELPFUL,
    (False, True): ScenarioOutcome.HARMFUL,
    (True, False): ScenarioOutcome.REDUNDANT,
    (True, True): ScenarioOutcome.UNRESOLVED,
}


def _factor_values(vector: tuple[bool, bool]) -> tuple[PublicCausalFactorValue, ...]:
    return tuple(
        PublicCausalFactorValue(factor_id=factor_id, value=value)
        for factor_id, value in zip(_FACTOR_IDS, vector, strict=True)
    )


def _transition(
    *,
    vector: tuple[bool, bool],
    exposure: PublicCausalExposure,
    terminal: PublicTerminalState,
) -> PublicTransition:
    vector_key = "".join("1" if value else "0" for value in vector)
    exposure_key = "guidance" if exposure is PublicCausalExposure.GUIDANCE_APPLIED else "baseline"
    goal_reached = terminal is PublicTerminalState.GOAL_REACHED
    return PublicTransition(
        source_state_id="initial",
        target_state_id=("goal-reached" if goal_reached else "goal-not-reached"),
        exposure=exposure,
        factor_values=_factor_values(vector),
        action_fingerprint_id=f"action-{exposure_key}-{vector_key}",
        failure_fingerprint_id=(None if goal_reached else f"failure-{vector_key}"),
        trigger=f"Execute the {exposure.value} path for factor vector {vector_key}.",
    )


def _graph() -> PublicTransitionGraph:
    transitions = tuple(
        _transition(vector=vector, exposure=exposure, terminal=terminal)
        for vector in _FACTOR_VECTORS
        for exposure, terminal in zip(
            (
                PublicCausalExposure.GUIDANCE_APPLIED,
                PublicCausalExposure.BASELINE_CONTINUED,
            ),
            _PAIRED_TERMINALS[vector],
            strict=True,
        )
    )
    values: dict[str, object] = {
        "schema_version": "state-decay-v2-public-transition-graph/v1",
        "initial_state_id": "initial",
        "factors": (
            PublicCausalFactor(
                factor_id=_FACTOR_IDS[0],
                true_description="The guidance changes the decisive action path.",
                false_description="The guidance leaves the decisive action path available.",
            ),
            PublicCausalFactor(
                factor_id=_FACTOR_IDS[1],
                true_description="The uninterrupted baseline can recover the goal.",
                false_description="The uninterrupted baseline cannot recover the goal.",
            ),
        ),
        "states": (
            PublicTransitionState(
                state_id="initial",
                description="The action path has not terminated.",
                terminal=None,
            ),
            PublicTransitionState(
                state_id="goal-reached",
                description="The task goal is reached.",
                terminal=PublicTerminalState.GOAL_REACHED,
            ),
            PublicTransitionState(
                state_id="goal-not-reached",
                description="The task goal is not reached.",
                terminal=PublicTerminalState.GOAL_NOT_REACHED,
            ),
        ),
        "transitions": transitions,
    }
    values["transition_graph_digest"] = transition_graph_digest(values)
    return PublicTransitionGraph.model_validate(values)


def _graph_payload() -> dict[str, object]:
    return _graph().model_dump(mode="python")


def _redigest(payload: dict[str, object]) -> None:
    payload["transition_graph_digest"] = transition_graph_digest(payload)


def test_public_causal_execution_contract_is_closed_and_outcome_free() -> None:
    assert tuple(PublicCausalExposure) == (
        PublicCausalExposure.GUIDANCE_APPLIED,
        PublicCausalExposure.BASELINE_CONTINUED,
    )
    assert tuple(PublicTerminalState) == (
        PublicTerminalState.GOAL_REACHED,
        PublicTerminalState.GOAL_NOT_REACHED,
    )
    assert tuple(PublicCausalFactor.model_fields) == (
        "factor_id",
        "true_description",
        "false_description",
    )
    assert tuple(PublicCausalFactorValue.model_fields) == ("factor_id", "value")
    assert tuple(PublicTransitionState.model_fields) == (
        "state_id",
        "description",
        "terminal",
    )
    assert tuple(PublicTransition.model_fields) == (
        "source_state_id",
        "target_state_id",
        "exposure",
        "factor_values",
        "action_fingerprint_id",
        "failure_fingerprint_id",
        "trigger",
    )
    assert tuple(PublicTransitionGraph.model_fields) == (
        "schema_version",
        "initial_state_id",
        "factors",
        "states",
        "transitions",
        "transition_graph_digest",
    )

    graph = _graph()
    assert PublicTransitionGraph.model_validate_json(canonical_json(graph)) == graph
    public_bytes = canonical_json(graph).decode("utf-8")
    assert all(f'"{outcome.value}"' not in public_bytes for outcome in ScenarioOutcome)
    assert "allocation" not in public_bytes
    assert "delta" not in public_bytes


def test_executor_and_deriver_accept_only_outcome_free_inputs() -> None:
    assert tuple(signature(execute_public_transition_graph).parameters) == (
        "graph",
        "exposure",
        "factor_values",
    )
    assert tuple(signature(derive_causal_outcome).parameters) == (
        "graph",
        "factor_values",
    )


def test_executor_and_deriver_recover_all_four_paired_outcomes_without_index_order() -> None:
    graph = _graph()
    actual_outcomes: dict[tuple[bool, bool], ScenarioOutcome] = {}

    for vector in reversed(_FACTOR_VECTORS):
        factor_values = _factor_values(vector)
        guidance = execute_public_transition_graph(
            graph=graph,
            exposure=PublicCausalExposure.GUIDANCE_APPLIED,
            factor_values=factor_values,
        )
        baseline = execute_public_transition_graph(
            graph=graph,
            exposure=PublicCausalExposure.BASELINE_CONTINUED,
            factor_values=factor_values,
        )

        assert (guidance.terminal, baseline.terminal) == _PAIRED_TERMINALS[vector]
        assert len(guidance.action_fingerprint_ids) == 1
        assert len(baseline.action_fingerprint_ids) == 1
        assert len(guidance.failure_fingerprint_ids) == 1
        assert len(baseline.failure_fingerprint_ids) == 1
        assert len(guidance.visited_state_ids) == 2
        assert len(baseline.visited_state_ids) == 2
        assert guidance.repeated_action_count == baseline.repeated_action_count == 0
        assert guidance.failure_loop_count == baseline.failure_loop_count == 0
        actual_outcomes[vector] = derive_causal_outcome(
            graph=graph,
            factor_values=factor_values,
        )

    assert actual_outcomes == _PAIRED_OUTCOMES
    assert set(actual_outcomes.values()) == set(ScenarioOutcome)


def test_transition_graph_rejects_duplicate_factor_definitions() -> None:
    payload = _graph_payload()
    payload["factors"] = (payload["factors"][0], payload["factors"][0])  # type: ignore[index]
    _redigest(payload)

    with pytest.raises(ValidationError, match="factor"):
        PublicTransitionGraph.model_validate(payload)


def test_transition_graph_rejects_incomplete_and_unknown_factor_guards() -> None:
    payload = _graph_payload()
    transitions = payload["transitions"]
    transitions[0]["factor_values"] = transitions[0]["factor_values"][:1]  # type: ignore[index]
    _redigest(payload)
    with pytest.raises(ValidationError):
        PublicTransitionGraph.model_validate(payload)

    payload = _graph_payload()
    transitions = payload["transitions"]
    transitions[0]["factor_values"][1]["factor_id"] = "undeclared-factor"  # type: ignore[index]
    _redigest(payload)
    with pytest.raises(ValidationError, match="factor"):
        PublicTransitionGraph.model_validate(payload)


def test_transition_graph_rejects_missing_and_ambiguous_branches() -> None:
    payload = _graph_payload()
    payload["transitions"] = payload["transitions"][1:]  # type: ignore[index]
    _redigest(payload)
    with pytest.raises(ValidationError, match=r"transition|branch|guard"):
        PublicTransitionGraph.model_validate(payload)

    payload = _graph_payload()
    transitions = payload["transitions"]
    transitions[2]["factor_values"] = transitions[0]["factor_values"]  # type: ignore[index]
    _redigest(payload)
    with pytest.raises(ValidationError, match=r"ambiguous|transition|guard"):
        PublicTransitionGraph.model_validate(payload)


def test_transition_graph_rejects_unresolved_states_and_terminal_outgoing_edges() -> None:
    payload = _graph_payload()
    payload["transitions"][0]["target_state_id"] = "missing-state"  # type: ignore[index]
    _redigest(payload)
    with pytest.raises(ValidationError, match=r"resolve|endpoint|state"):
        PublicTransitionGraph.model_validate(payload)

    payload = _graph_payload()
    outgoing = payload["transitions"][0].copy()  # type: ignore[index]
    outgoing["source_state_id"] = "goal-reached"
    outgoing["target_state_id"] = "goal-not-reached"
    outgoing["trigger"] = "A terminal state attempts another transition."
    payload["transitions"] = (*payload["transitions"], outgoing)  # type: ignore[arg-type]
    _redigest(payload)
    with pytest.raises(ValidationError, match="terminal"):
        PublicTransitionGraph.model_validate(payload)


def test_transition_graph_rejects_nontermination() -> None:
    payload = _graph_payload()
    payload["transitions"][0]["target_state_id"] = "initial"  # type: ignore[index]
    _redigest(payload)

    with pytest.raises(ValidationError, match=r"terminat|cycle|step"):
        PublicTransitionGraph.model_validate(payload)


def test_transition_graph_rejects_digest_tampering() -> None:
    payload = _graph_payload()
    payload["transition_graph_digest"] = "0" * 64

    with pytest.raises(ValidationError, match="transition graph digest"):
        PublicTransitionGraph.model_validate(payload)


@pytest.mark.parametrize(
    ("section", "field"),
    (
        ("states", "description"),
        ("transitions", "trigger"),
        ("transitions", "action_fingerprint_id"),
    ),
)
def test_transition_graph_rejects_outcome_label_shortcuts(section: str, field: str) -> None:
    payload = _graph_payload()
    payload[section][0][field] = "helpful-path"  # type: ignore[index]
    _redigest(payload)

    with pytest.raises(ValidationError, match="outcome label"):
        PublicTransitionGraph.model_validate(payload)
