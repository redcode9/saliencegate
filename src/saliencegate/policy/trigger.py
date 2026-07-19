from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from saliencegate.domain import (
    BudgetAmounts,
    BudgetLimits,
    BudgetSnapshot,
    InvocationDecision,
    ReasonCode,
    Signal,
    SignalType,
)
from saliencegate.policy.config import (
    MAX_SIGNED_64,
    AlwaysInvokeConfig,
    BudgetedPolicyConfig,
    NeverInvokeConfig,
    PolicyConfigurationError,
    ResolvedPolicyConfiguration,
    RunState,
    ScriptedPolicyConfig,
    _frozen_json_is_bounded,
    _seal_policy_configuration,
    resolve_policy_configuration,
)
from saliencegate.runtime import BudgetGovernor, BudgetReservationDeniedError

MAX_POLICY_SIGNALS = 64
_POLICY_VERSION = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._:/+\-]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BUDGET_AMOUNT_FIELDS = (
    "model_calls",
    "input_tokens",
    "output_tokens",
    "canonical_token_equivalents",
    "latency_us",
    "interventions",
    "schema_repairs",
)
_BUDGET_LIMIT_FIELDS = (*_BUDGET_AMOUNT_FIELDS, "max_call_latency_us")


class PolicyError(ValueError):
    """Base class for policy failures that never embeds caller-owned values."""


class PolicyInputError(PolicyError):
    def __init__(self) -> None:
        super().__init__("trigger policy input failed validation")


class PolicyContractError(PolicyError):
    def __init__(self) -> None:
        super().__init__("trigger policy plugin violated its contract")


class TriggerPolicy(Protocol):
    """Stable synchronous trigger contract; implementations must be replay-pure."""

    def decide(
        self,
        signals: list[Signal],
        state: RunState,
        budget: BudgetSnapshot,
    ) -> InvocationDecision: ...


class ConfiguredTriggerPolicy(TriggerPolicy, Protocol):
    """A trigger with replay metadata that can be wrapped by ``BudgetedPolicy``."""

    @property
    def policy_version(self) -> str: ...

    @property
    def configuration_digest(self) -> str: ...

    @property
    def resolved_configuration(self) -> ResolvedPolicyConfiguration: ...


def _exact_bounded_int_fields(value: object, fields: tuple[str, ...]) -> bool:
    return all(
        type(getattr(value, field_name, None)) is int
        and 0 <= getattr(value, field_name) <= MAX_SIGNED_64
        for field_name in fields
    )


def _budget_is_preflight_safe(value: object) -> bool:
    try:
        return (
            type(value) is BudgetSnapshot
            and type(value.limits) is BudgetLimits
            and type(value.reserved) is BudgetAmounts
            and type(value.consumed) is BudgetAmounts
            and _exact_bounded_int_fields(value.limits, _BUDGET_LIMIT_FIELDS)
            and _exact_bounded_int_fields(value.reserved, _BUDGET_AMOUNT_FIELDS)
            and _exact_bounded_int_fields(value.consumed, _BUDGET_AMOUNT_FIELDS)
        )
    except Exception:
        return False


def _state_is_preflight_safe(value: object) -> bool:
    try:
        return (
            type(value) is RunState
            and value.schema_version == "1.0"
            and type(value.schema_version) is str
            and type(value.decision_id) is UUID
            and value.decision_id.version == 4
            and type(value.run_id) is UUID
            and value.run_id.version == 4
            and type(value.current_event_id) is UUID
            and value.current_event_id.version == 4
            and type(value.event_sequence) is int
            and 1 <= value.event_sequence <= MAX_SIGNED_64
            and type(value.decision_ordinal) is int
            and 1 <= value.decision_ordinal <= MAX_SIGNED_64
            and type(value.created_at) is datetime
        )
    except Exception:
        return False


def _signal_is_preflight_safe(value: object) -> bool:
    try:
        return (
            type(value) is Signal
            and value.schema_version == "1.0"
            and value.record_type == "signal"
            and type(value.signal_id) is UUID
            and value.signal_id.version == 4
            and type(value.run_id) is UUID
            and value.run_id.version == 4
            and type(value.created_at) is datetime
            and type(value.signal_type) is SignalType
            and type(value.strength) is float
            and math.isfinite(value.strength)
            and type(value.evidence_event_ids) is tuple
            and len(value.evidence_event_ids) <= 10_000
            and all(
                type(event_id) is UUID and event_id.version == 4
                for event_id in value.evidence_event_ids
            )
            and type(value.detector_version) is str
            and len(value.detector_version) <= 256
            and type(value.reason_code) is ReasonCode
        )
    except Exception:
        return False


def _validated_model(value: object, model_type: type[object]) -> object | None:
    validated: object | None = None
    try:
        candidate = model_type.model_validate(  # type: ignore[attr-defined]
            value.model_dump(mode="python", warnings=False)  # type: ignore[attr-defined]
        )
        if candidate == value:
            validated = candidate
    except Exception:
        validated = None
    return validated


@dataclass(frozen=True, slots=True)
class _PolicyInputs:
    signals: tuple[Signal, ...]
    state: RunState
    budget: BudgetSnapshot


def _validated_inputs(
    signals: object,
    state: object,
    budget: object,
) -> _PolicyInputs:
    if (
        type(signals) is not list
        or len(signals) > MAX_POLICY_SIGNALS
        or not _state_is_preflight_safe(state)
        or not _budget_is_preflight_safe(budget)
        or any(not _signal_is_preflight_safe(signal) for signal in signals)
    ):
        raise PolicyInputError()

    validated_state = _validated_model(state, RunState)
    validated_budget = _validated_model(budget, BudgetSnapshot)
    validated_signals = tuple(_validated_model(signal, Signal) for signal in signals)
    if (
        type(validated_state) is not RunState
        or type(validated_budget) is not BudgetSnapshot
        or any(type(signal) is not Signal for signal in validated_signals)
    ):
        raise PolicyInputError()
    typed_signals = tuple(signal for signal in validated_signals if type(signal) is Signal)
    if (
        len({signal.signal_id for signal in typed_signals}) != len(typed_signals)
        or any(signal.run_id != validated_state.run_id for signal in typed_signals)
        or any(
            validated_state.current_event_id not in signal.evidence_event_ids
            for signal in typed_signals
        )
        or any(signal.created_at > validated_state.created_at for signal in typed_signals)
    ):
        raise PolicyInputError()
    return _PolicyInputs(
        signals=typed_signals,
        state=validated_state,
        budget=validated_budget,
    )


def _decision(
    inputs: _PolicyInputs,
    *,
    invoke: bool,
    reason: ReasonCode,
    policy_version: str,
    configuration_digest: str,
    risk_score: float | None = None,
    cooldown_active: bool = False,
) -> InvocationDecision:
    return InvocationDecision(
        decision_id=inputs.state.decision_id,
        run_id=inputs.state.run_id,
        event_sequence=inputs.state.event_sequence,
        invoke=invoke,
        risk_score=risk_score,
        reason_codes=(reason,),
        policy_version=policy_version,
        configuration_digest=configuration_digest,
        budget_snapshot=inputs.budget,
        cooldown_active=cooldown_active,
        created_at=inputs.state.created_at,
    )


class _ReferencePolicy:
    __slots__ = ("_configuration_digest_value", "_resolved_json")

    _configuration_digest_value: str
    _resolved_json: str

    @property
    def configuration_digest(self) -> str:
        return self._configuration_digest_value

    @property
    def resolved_configuration(self) -> ResolvedPolicyConfiguration:
        return ResolvedPolicyConfiguration.model_validate_json(self._resolved_json)


def _install_resolved(
    policy: _ReferencePolicy,
    resolved: ResolvedPolicyConfiguration,
) -> None:
    object.__setattr__(policy, "_configuration_digest_value", resolved.configuration_digest)
    object.__setattr__(
        policy,
        "_resolved_json",
        resolved.model_dump_json(warnings=False),
    )


@dataclass(frozen=True, slots=True, init=False, repr=False)
class NeverInvoke(_ReferencePolicy):
    def __init__(self, configuration: NeverInvokeConfig) -> None:
        if type(configuration) is not NeverInvokeConfig:
            raise PolicyConfigurationError()
        _install_resolved(
            self,
            resolve_policy_configuration(self.policy_version, configuration),
        )

    @property
    def policy_version(self) -> str:
        return "never-invoke/v1"

    def decide(
        self,
        signals: list[Signal],
        state: RunState,
        budget: BudgetSnapshot,
    ) -> InvocationDecision:
        inputs = _validated_inputs(signals, state, budget)
        return _decision(
            inputs,
            invoke=False,
            reason=ReasonCode.POLICY_NEVER,
            policy_version=self.policy_version,
            configuration_digest=self.configuration_digest,
        )


@dataclass(frozen=True, slots=True, init=False, repr=False)
class AlwaysInvoke(_ReferencePolicy):
    def __init__(self, configuration: AlwaysInvokeConfig) -> None:
        if type(configuration) is not AlwaysInvokeConfig:
            raise PolicyConfigurationError()
        _install_resolved(
            self,
            resolve_policy_configuration(self.policy_version, configuration),
        )

    @property
    def policy_version(self) -> str:
        return "always-invoke/v1"

    def decide(
        self,
        signals: list[Signal],
        state: RunState,
        budget: BudgetSnapshot,
    ) -> InvocationDecision:
        inputs = _validated_inputs(signals, state, budget)
        return _decision(
            inputs,
            invoke=True,
            reason=ReasonCode.POLICY_ALWAYS,
            policy_version=self.policy_version,
            configuration_digest=self.configuration_digest,
        )


@dataclass(frozen=True, slots=True, init=False, repr=False)
class ScriptedPolicy(_ReferencePolicy):
    _decisions: tuple[bool, ...]

    def __init__(self, configuration: ScriptedPolicyConfig) -> None:
        if type(configuration) is not ScriptedPolicyConfig:
            raise PolicyConfigurationError()
        resolved = resolve_policy_configuration(self.policy_version, configuration)
        decisions = resolved.configuration["decisions"]
        assert type(decisions) is tuple
        object.__setattr__(self, "_decisions", tuple(decisions))
        _install_resolved(self, resolved)

    @property
    def policy_version(self) -> str:
        return "scripted/v1"

    def decide(
        self,
        signals: list[Signal],
        state: RunState,
        budget: BudgetSnapshot,
    ) -> InvocationDecision:
        inputs = _validated_inputs(signals, state, budget)
        index = inputs.state.decision_ordinal - 1
        if index >= len(self._decisions):
            invoke = False
            reason = ReasonCode.SCRIPT_EXHAUSTED
        else:
            invoke = self._decisions[index]
            reason = ReasonCode.SCRIPTED_INVOKE if invoke else ReasonCode.SCRIPTED_SILENCE
        return _decision(
            inputs,
            invoke=invoke,
            reason=reason,
            policy_version=self.policy_version,
            configuration_digest=self.configuration_digest,
        )


def _resolved_is_preflight_safe(value: object) -> bool:
    try:
        return (
            type(value) is ResolvedPolicyConfiguration
            and value.schema_version == "1.0"
            and type(value.policy_version) is str
            and _POLICY_VERSION.fullmatch(value.policy_version) is not None
            and type(value.configuration_digest) is str
            and _SHA256.fullmatch(value.configuration_digest) is not None
            and _frozen_json_is_bounded(value.configuration)
        )
    except Exception:
        return False


def _invocation_decision_is_preflight_safe(value: object) -> bool:
    if type(value) is not InvocationDecision:
        return False
    try:
        risk_score = value.risk_score
        return (
            value.schema_version == "1.0"
            and value.record_type == "invocation_decision"
            and type(value.decision_id) is UUID
            and value.decision_id.version == 4
            and type(value.run_id) is UUID
            and value.run_id.version == 4
            and type(value.event_sequence) is int
            and 1 <= value.event_sequence <= MAX_SIGNED_64
            and type(value.invoke) is bool
            and (
                risk_score is None
                or (
                    type(risk_score) is float
                    and math.isfinite(risk_score)
                    and 0.0 <= risk_score <= 1.0
                )
            )
            and type(value.reason_codes) is tuple
            and 1 <= len(value.reason_codes) <= len(ReasonCode)
            and all(type(reason) is ReasonCode for reason in value.reason_codes)
            and type(value.policy_version) is str
            and _POLICY_VERSION.fullmatch(value.policy_version) is not None
            and type(value.configuration_digest) is str
            and _SHA256.fullmatch(value.configuration_digest) is not None
            and _budget_is_preflight_safe(value.budget_snapshot)
            and type(value.cooldown_active) is bool
            and type(value.created_at) is datetime
        )
    except Exception:
        return False


def _snapshot_inner_policy(
    inner: object,
) -> (
    tuple[
        Callable[[list[Signal], RunState, BudgetSnapshot], InvocationDecision],
        str,
        ResolvedPolicyConfiguration,
    ]
    | None
):
    snapshot: (
        tuple[
            Callable[[list[Signal], RunState, BudgetSnapshot], InvocationDecision],
            str,
            ResolvedPolicyConfiguration,
        ]
        | None
    ) = None
    try:
        decide = getattr(inner, "decide", None)
        policy_version = getattr(inner, "policy_version", None)
        configuration_digest = getattr(inner, "configuration_digest", None)
        resolved = getattr(inner, "resolved_configuration", None)
        if (
            callable(decide)
            and type(policy_version) is str
            and _POLICY_VERSION.fullmatch(policy_version) is not None
            and type(configuration_digest) is str
            and _resolved_is_preflight_safe(resolved)
        ):
            assert type(resolved) is ResolvedPolicyConfiguration
            validated = ResolvedPolicyConfiguration.model_validate(
                resolved.model_dump(mode="python", warnings=False)
            )
            if (
                validated == resolved
                and validated.policy_version == policy_version
                and validated.configuration_digest == configuration_digest
            ):
                snapshot = decide, policy_version, validated
    except Exception:
        snapshot = None
    return snapshot


def _inner_decision_matches(
    decision: object,
    inputs: _PolicyInputs,
    *,
    policy_version: str,
    configuration_digest: str,
) -> InvocationDecision | None:
    if not _invocation_decision_is_preflight_safe(decision):
        return None
    assert type(decision) is InvocationDecision
    validated: InvocationDecision | None = None
    try:
        candidate = InvocationDecision.model_validate(
            decision.model_dump(mode="python", warnings=False)
        )
        if candidate == decision:
            validated = candidate
    except Exception:
        validated = None
    if validated is None or (
        validated.decision_id != inputs.state.decision_id
        or validated.run_id != inputs.state.run_id
        or validated.event_sequence != inputs.state.event_sequence
        or validated.policy_version != policy_version
        or validated.configuration_digest != configuration_digest
        or validated.budget_snapshot != inputs.budget
        or validated.created_at != inputs.state.created_at
    ):
        return None
    return validated


def _detached_plugin_inputs(
    inputs: _PolicyInputs,
) -> tuple[list[Signal], RunState, BudgetSnapshot]:
    signals = [
        Signal.model_validate_json(signal.model_dump_json(warnings=False))
        for signal in inputs.signals
    ]
    state = RunState.model_validate_json(inputs.state.model_dump_json(warnings=False))
    budget = BudgetSnapshot.model_validate_json(inputs.budget.model_dump_json(warnings=False))
    return signals, state, budget


@dataclass(frozen=True, slots=True, init=False, repr=False)
class BudgetedPolicy(_ReferencePolicy):
    _inner_decide: Callable[[list[Signal], RunState, BudgetSnapshot], InvocationDecision]
    _inner_policy_version: str
    _inner_configuration_digest: str
    _reservation_values: tuple[int, ...]

    def __init__(
        self,
        inner: ConfiguredTriggerPolicy,
        configuration: BudgetedPolicyConfig,
    ) -> None:
        if type(configuration) is not BudgetedPolicyConfig:
            raise PolicyConfigurationError()
        base = resolve_policy_configuration(self.policy_version, configuration)
        inner_snapshot = _snapshot_inner_policy(inner)
        if inner_snapshot is None:
            raise PolicyContractError()
        inner_decide, inner_version, inner_resolved = inner_snapshot
        if (
            isinstance(inner, BudgetedPolicy)
            or inner_version == self.policy_version
            or inner_resolved.configuration.get("policy_kind") == "budgeted"
        ):
            raise PolicyContractError()
        base_payload = base.model_dump(mode="json", warnings=False)["configuration"]
        assert isinstance(base_payload, dict)
        composite = {
            **base_payload,
            "inner_policy_version": inner_version,
            "inner_configuration_digest": inner_resolved.configuration_digest,
            "inner_configuration": inner_resolved.model_dump(mode="json", warnings=False)[
                "configuration"
            ],
        }
        resolved = _seal_policy_configuration(self.policy_version, composite)
        object.__setattr__(self, "_inner_decide", inner_decide)
        object.__setattr__(self, "_inner_policy_version", inner_version)
        object.__setattr__(
            self,
            "_inner_configuration_digest",
            inner_resolved.configuration_digest,
        )
        reservation = base.configuration["reservation"]
        assert isinstance(reservation, Mapping)
        reservation_values = tuple(reservation[field_name] for field_name in _BUDGET_AMOUNT_FIELDS)
        assert all(type(value) is int for value in reservation_values)
        object.__setattr__(self, "_reservation_values", reservation_values)
        _install_resolved(self, resolved)

    @property
    def policy_version(self) -> str:
        return "budgeted/v1"

    @property
    def reservation(self) -> BudgetAmounts:
        return BudgetAmounts(
            **dict(zip(_BUDGET_AMOUNT_FIELDS, self._reservation_values, strict=True))
        )

    def decide(
        self,
        signals: list[Signal],
        state: RunState,
        budget: BudgetSnapshot,
    ) -> InvocationDecision:
        inputs = _validated_inputs(signals, state, budget)
        plugin_signals, plugin_state, plugin_budget = _detached_plugin_inputs(inputs)
        inner_result: object | None = None
        try:
            inner_result = self._inner_decide(plugin_signals, plugin_state, plugin_budget)
        except Exception:
            inner_result = None
        inner = _inner_decision_matches(
            inner_result,
            inputs,
            policy_version=self._inner_policy_version,
            configuration_digest=self._inner_configuration_digest,
        )
        if inner is None:
            raise PolicyContractError()

        denied = False
        if inner.invoke:
            try:
                BudgetGovernor().reserve(inputs.budget, self.reservation)
            except BudgetReservationDeniedError:
                denied = True
        return InvocationDecision(
            decision_id=inputs.state.decision_id,
            run_id=inputs.state.run_id,
            event_sequence=inputs.state.event_sequence,
            invoke=inner.invoke and not denied,
            risk_score=inner.risk_score,
            reason_codes=(ReasonCode.BUDGET_EXHAUSTED,) if denied else inner.reason_codes,
            policy_version=self.policy_version,
            configuration_digest=self.configuration_digest,
            budget_snapshot=inputs.budget,
            cooldown_active=inner.cooldown_active,
            created_at=inputs.state.created_at,
        )


__all__ = [
    "MAX_POLICY_SIGNALS",
    "AlwaysInvoke",
    "BudgetedPolicy",
    "ConfiguredTriggerPolicy",
    "NeverInvoke",
    "PolicyContractError",
    "PolicyError",
    "PolicyInputError",
    "ScriptedPolicy",
    "TriggerPolicy",
]
