from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any, cast
from uuid import UUID

import pytest

from saliencegate.domain import (
    BudgetAmounts,
    BudgetLimits,
    BudgetSnapshot,
    InvocationDecision,
    ReasonCode,
    Signal,
    SignalType,
    canonical_json,
    load_record,
)
from saliencegate.policy import (
    MAX_POLICY_SIGNALS,
    AlwaysInvoke,
    AlwaysInvokeConfig,
    BudgetedPolicy,
    BudgetedPolicyConfig,
    ConfiguredTriggerPolicy,
    NeverInvoke,
    NeverInvokeConfig,
    PolicyConfigurationError,
    PolicyContractError,
    PolicyInputError,
    ResolvedPolicyConfiguration,
    RunState,
    ScriptedPolicy,
    ScriptedPolicyConfig,
    TriggerPolicy,
)

RUN_ID = UUID("00000000-0000-4000-8000-000000003001")
OTHER_RUN_ID = UUID("00000000-0000-4000-8000-000000003002")
EVENT_ID = UUID("00000000-0000-4000-8000-000000003003")
DECISION_ID = UUID("00000000-0000-4000-8000-000000003004")
SIGNAL_ID = UUID("00000000-0000-4000-8000-000000003005")
NOW = datetime(2026, 7, 11, 21, 0, tzinfo=UTC)
SCHEMA_VERSION = "1.0"


def state(
    *,
    ordinal: int = 1,
    event_sequence: int = 1,
    run_id: UUID = RUN_ID,
    current_event_id: UUID = EVENT_ID,
    decision_id: UUID = DECISION_ID,
) -> RunState:
    return RunState(
        schema_version=SCHEMA_VERSION,
        decision_id=decision_id,
        run_id=run_id,
        current_event_id=current_event_id,
        event_sequence=event_sequence,
        decision_ordinal=ordinal,
        created_at=NOW,
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


def signal(
    *,
    signal_id: UUID = SIGNAL_ID,
    run_id: UUID = RUN_ID,
    evidence_event_id: UUID = EVENT_ID,
    created_at: datetime = NOW,
    signal_type: SignalType = SignalType.TOOL_ERROR,
) -> Signal:
    return Signal(
        signal_id=signal_id,
        run_id=run_id,
        created_at=created_at,
        signal_type=signal_type,
        strength=1.0,
        evidence_event_ids=(evidence_event_id,),
        detector_version=f"{signal_type.value}/v1",
        reason_code=ReasonCode(signal_type.value),
    )


def always_policy() -> AlwaysInvoke:
    return AlwaysInvoke(
        AlwaysInvokeConfig(
            schema_version=SCHEMA_VERSION,
            policy_kind="always_invoke",
        )
    )


def never_policy() -> NeverInvoke:
    return NeverInvoke(
        NeverInvokeConfig(
            schema_version=SCHEMA_VERSION,
            policy_kind="never_invoke",
        )
    )


def scripted_policy(*decisions: bool) -> ScriptedPolicy:
    return ScriptedPolicy(
        ScriptedPolicyConfig(
            schema_version=SCHEMA_VERSION,
            policy_kind="scripted",
            decisions=decisions,
            on_exhaustion="silence",
        )
    )


def budget_config(reservation: BudgetAmounts) -> BudgetedPolicyConfig:
    return BudgetedPolicyConfig(
        schema_version=SCHEMA_VERSION,
        policy_kind="budgeted",
        reservation=reservation,
        on_denial="silence",
    )


def assert_reference_decision(
    decision: InvocationDecision,
    *,
    invoke: bool,
    reason: ReasonCode,
    policy_version: str,
    configuration_digest: str,
    budget: BudgetSnapshot,
) -> None:
    assert decision.decision_id == DECISION_ID
    assert decision.run_id == RUN_ID
    assert decision.event_sequence == 1
    assert decision.invoke is invoke
    assert decision.risk_score is None
    assert decision.reason_codes == (reason,)
    assert decision.policy_version == policy_version
    assert decision.configuration_digest == configuration_digest
    assert decision.budget_snapshot == budget
    assert not decision.cooldown_active
    assert decision.created_at == NOW


class DecideOnlyPolicy:
    def decide(
        self,
        signals: list[Signal],
        selected_state: RunState,
        budget: BudgetSnapshot,
    ) -> InvocationDecision:
        return always_policy().decide(signals, selected_state, budget)


def test_stable_trigger_policy_protocol_matches_reference_implementations() -> None:
    policies: tuple[TriggerPolicy, ...] = (
        always_policy(),
        never_policy(),
        scripted_policy(True),
    )

    assert tuple(policy.policy_version for policy in policies) == (
        "always-invoke/v1",
        "never-invoke/v1",
        "scripted/v1",
    )
    assert all(not hasattr(policy, "__dict__") for policy in policies)

    decide_only: TriggerPolicy = DecideOnlyPolicy()
    assert decide_only.decide([], state(), snapshot()).invoke
    with pytest.raises(PolicyContractError):
        BudgetedPolicy(
            cast(ConfiguredTriggerPolicy, decide_only),
            budget_config(BudgetAmounts(model_calls=1)),
        )


@pytest.mark.parametrize(
    "factory",
    (
        lambda: AlwaysInvoke(
            cast(
                AlwaysInvokeConfig,
                NeverInvokeConfig(schema_version=SCHEMA_VERSION, policy_kind="never_invoke"),
            )
        ),
        lambda: AlwaysInvoke(AlwaysInvokeConfig.model_construct()),
        lambda: NeverInvoke(
            cast(
                NeverInvokeConfig,
                AlwaysInvokeConfig(schema_version=SCHEMA_VERSION, policy_kind="always_invoke"),
            )
        ),
        lambda: NeverInvoke(NeverInvokeConfig.model_construct()),
        lambda: ScriptedPolicy(cast(ScriptedPolicyConfig, always_policy().resolved_configuration)),
        lambda: ScriptedPolicy(ScriptedPolicyConfig.model_construct()),
        lambda: BudgetedPolicy(
            always_policy(),
            cast(BudgetedPolicyConfig, always_policy().resolved_configuration),
        ),
        lambda: BudgetedPolicy(always_policy(), BudgetedPolicyConfig.model_construct()),
    ),
)
def test_reference_policy_constructors_reject_wrong_or_incomplete_config_types(
    factory: object,
) -> None:
    with pytest.raises(PolicyConfigurationError) as error:
        factory()  # type: ignore[operator]

    assert error.value.__context__ is None
    assert error.value.__cause__ is None


def test_always_and_never_emit_complete_replay_stable_decisions() -> None:
    budget = snapshot()
    always = always_policy()
    never = never_policy()

    invoking = always.decide([signal()], state(), budget)
    silence = never.decide([], state(), budget)

    assert_reference_decision(
        invoking,
        invoke=True,
        reason=ReasonCode.POLICY_ALWAYS,
        policy_version=always.policy_version,
        configuration_digest=always.configuration_digest,
        budget=budget,
    )
    assert_reference_decision(
        silence,
        invoke=False,
        reason=ReasonCode.POLICY_NEVER,
        policy_version=never.policy_version,
        configuration_digest=never.configuration_digest,
        budget=budget,
    )
    assert always.decide([signal()], state(), budget) == invoking
    assert load_record(canonical_json(invoking)) == invoking


def test_script_is_stateless_indexed_and_explicitly_exhausted() -> None:
    policy = scripted_policy(True, False)
    budget = snapshot()

    second_state = state(ordinal=2, event_sequence=99)
    second = policy.decide([], second_state, budget)
    first = policy.decide([], state(ordinal=1), budget)
    exhausted = policy.decide([], state(ordinal=3), budget)

    assert first.invoke
    assert first.reason_codes == (ReasonCode.SCRIPTED_INVOKE,)
    assert not second.invoke
    assert second.event_sequence == 99
    assert second.reason_codes == (ReasonCode.SCRIPTED_SILENCE,)
    assert not exhausted.invoke
    assert exhausted.reason_codes == (ReasonCode.SCRIPT_EXHAUSTED,)
    assert policy.decide([], second_state, budget) == second


def test_empty_script_is_immediately_and_repeatably_exhausted() -> None:
    policy = scripted_policy()

    first = policy.decide([], state(), snapshot())
    second = policy.decide([], state(), snapshot())

    assert first == second
    assert first.reason_codes == (ReasonCode.SCRIPT_EXHAUSTED,)


def test_budget_wrapper_allows_invocation_without_mutating_snapshot() -> None:
    inner = always_policy()
    config = budget_config(
        BudgetAmounts(
            model_calls=1,
            input_tokens=20,
            output_tokens=10,
            canonical_token_equivalents=30,
            latency_us=10,
        )
    )
    policy = BudgetedPolicy(inner, config)
    budget = snapshot()

    decision = policy.decide([], state(), budget)

    assert decision.invoke
    assert decision.reason_codes == (ReasonCode.POLICY_ALWAYS,)
    assert decision.policy_version == "budgeted/v1"
    assert decision.configuration_digest == policy.configuration_digest
    assert decision.configuration_digest != inner.configuration_digest
    assert decision.budget_snapshot == budget
    assert decision.budget_snapshot.reserved == BudgetAmounts()
    assert policy.reservation == config.reservation


def test_budget_wrapper_owns_reservation_and_resolved_configuration_snapshots() -> None:
    config = budget_config(BudgetAmounts(model_calls=1))
    policy = BudgetedPolicy(always_policy(), config)
    original_digest = policy.configuration_digest

    config.reservation.__dict__["model_calls"] = 2
    leaked_reservation = policy.reservation
    leaked_reservation.__dict__["model_calls"] = 3
    leaked_resolved = policy.resolved_configuration
    leaked_resolved.__dict__["configuration_digest"] = "e" * 64

    decision = policy.decide([], state(), snapshot(limit=1))

    assert decision.invoke
    assert policy.reservation.model_calls == 1
    assert policy.configuration_digest == original_digest
    assert policy.resolved_configuration.configuration_digest == original_digest
    assert decision.configuration_digest == original_digest


def test_budget_denial_is_explicit_silence_with_the_preflight_snapshot() -> None:
    policy = BudgetedPolicy(
        always_policy(),
        budget_config(BudgetAmounts(model_calls=1)),
    )
    budget = snapshot(limit=0)

    decision = policy.decide([], state(), budget)

    assert not decision.invoke
    assert decision.reason_codes == (ReasonCode.BUDGET_EXHAUSTED,)
    assert decision.risk_score is None
    assert decision.budget_snapshot == budget
    assert decision.policy_version == policy.policy_version
    assert decision.configuration_digest == policy.configuration_digest


def test_silent_inner_policy_does_not_consult_or_claim_budget_denial() -> None:
    policy = BudgetedPolicy(
        never_policy(),
        budget_config(BudgetAmounts(model_calls=1)),
    )

    decision = policy.decide([], state(), snapshot(limit=0))

    assert not decision.invoke
    assert decision.reason_codes == (ReasonCode.POLICY_NEVER,)
    assert decision.policy_version == policy.policy_version


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
def test_budget_wrapper_denies_each_exhausted_dimension(field_name: str) -> None:
    requested = {"model_calls": 1, field_name: 2}
    if field_name == "model_calls":
        requested = {"model_calls": 2}
    reservation = BudgetAmounts(**requested)
    limits = BudgetLimits(
        model_calls=10,
        input_tokens=10,
        output_tokens=10,
        canonical_token_equivalents=10,
        latency_us=10,
        max_call_latency_us=10,
        interventions=10,
        schema_repairs=10,
    ).model_copy(update={field_name: 1})
    budget = BudgetSnapshot(
        limits=limits,
        reserved=BudgetAmounts(),
        consumed=BudgetAmounts(),
    )
    policy = BudgetedPolicy(always_policy(), budget_config(reservation))

    decision = policy.decide([], state(), budget)

    assert not decision.invoke
    assert decision.reason_codes == (ReasonCode.BUDGET_EXHAUSTED,)


def test_reserved_and_consumed_amounts_participate_in_budget_denial() -> None:
    policy = BudgetedPolicy(
        always_policy(),
        budget_config(BudgetAmounts(model_calls=1, input_tokens=2)),
    )
    budget = BudgetSnapshot(
        limits=snapshot(limit=3).limits,
        reserved=BudgetAmounts(input_tokens=1),
        consumed=BudgetAmounts(input_tokens=1),
    )

    assert policy.decide([], state(), budget).reason_codes == (ReasonCode.BUDGET_EXHAUSTED,)


def test_composed_configuration_contains_inner_config_and_full_reservation() -> None:
    inner = scripted_policy(True, False)
    policy = BudgetedPolicy(
        inner,
        budget_config(BudgetAmounts(model_calls=1, input_tokens=5)),
    )
    configuration = policy.resolved_configuration.configuration

    assert configuration["policy_kind"] == "budgeted"
    assert configuration["reservation"]["model_calls"] == 1  # type: ignore[index]
    assert configuration["reservation"]["schema_repairs"] == 0  # type: ignore[index]
    assert configuration["inner_policy_version"] == inner.policy_version
    assert configuration["inner_configuration_digest"] == inner.configuration_digest
    assert configuration["inner_configuration"] == inner.resolved_configuration.configuration


def test_budget_wrapper_digest_commits_to_inner_policy_and_every_reservation_field() -> None:
    base = BudgetedPolicy(
        always_policy(),
        budget_config(BudgetAmounts(model_calls=1)),
    )
    variants = [
        BudgetedPolicy(
            always_policy(),
            budget_config(
                BudgetAmounts(
                    **{
                        "model_calls": 2 if field_name == "model_calls" else 1,
                        **({field_name: 1} if field_name != "model_calls" else {}),
                    }
                )
            ),
        )
        for field_name in (
            "model_calls",
            "input_tokens",
            "output_tokens",
            "canonical_token_equivalents",
            "latency_us",
            "interventions",
            "schema_repairs",
        )
    ]
    different_inner = BudgetedPolicy(
        scripted_policy(True),
        budget_config(BudgetAmounts(model_calls=1)),
    )

    digests = {
        base.configuration_digest,
        different_inner.configuration_digest,
        *(policy.configuration_digest for policy in variants),
    }
    assert len(digests) == 9
    assert (
        BudgetedPolicy(
            always_policy(),
            budget_config(BudgetAmounts(model_calls=1)),
        ).configuration_digest
        == base.configuration_digest
    )


@pytest.mark.parametrize(
    "signals",
    (
        (),
        [signal()] * (MAX_POLICY_SIGNALS + 1),
        [signal(), signal()],
        [signal(run_id=OTHER_RUN_ID)],
        [signal(evidence_event_id=UUID("00000000-0000-4000-8000-000000003099"))],
        [signal(created_at=NOW + timedelta(seconds=1))],
    ),
)
def test_policy_inputs_require_exact_current_run_signals(signals: object) -> None:
    with pytest.raises(PolicyInputError):
        always_policy().decide(cast(list[Signal], signals), state(), snapshot())


def test_policy_accepts_one_current_signal_of_every_reserved_type() -> None:
    signals = [
        signal(
            signal_id=UUID(f"00000000-0000-4000-8000-{index:012d}"),
            signal_type=signal_type,
        )
        for index, signal_type in enumerate(SignalType, start=1)
    ]

    decision = always_policy().decide(signals, state(), snapshot())

    assert decision.invoke


def test_policy_accepts_distinct_current_signals_of_the_same_type() -> None:
    signals = [
        signal(),
        signal(signal_id=UUID("00000000-0000-4000-8000-000000003006")),
    ]

    assert always_policy().decide(signals, state(), snapshot()).invoke


def test_policy_revalidates_forged_state_budget_and_signal_without_echo() -> None:
    secret = "policy-input-secret"
    forged_state = state().model_copy(update={"event_sequence": secret})
    forged_budget = snapshot().model_copy(update={"reserved": secret})
    forged_signal = signal().model_copy(update={"detector_version": secret * 100})
    mismatched_signal = signal().model_copy(update={"reason_code": ReasonCode.TEST_FAILURE})

    for signals, selected_state, budget in (
        ([], forged_state, snapshot()),
        ([], state(), forged_budget),
        ([forged_signal], state(), snapshot()),
        ([mismatched_signal], state(), snapshot()),
    ):
        with pytest.raises(PolicyInputError) as error:
            always_policy().decide(signals, selected_state, budget)
        assert secret not in str(error.value)
        assert error.value.__context__ is None
        assert error.value.__cause__ is None


@pytest.mark.parametrize(
    ("signals", "selected_state", "budget"),
    (
        ([], RunState.model_construct(), snapshot()),
        ([], state(), BudgetSnapshot.model_construct()),
        ([Signal.model_construct()], state(), snapshot()),
        (
            [],
            state(),
            BudgetSnapshot.model_construct(
                limits=BudgetLimits().model_copy(update={"model_calls": object()}),
                reserved=BudgetAmounts(),
                consumed=BudgetAmounts(),
            ),
        ),
    ),
)
def test_policy_sanitizes_incomplete_constructed_inputs(
    signals: list[Signal],
    selected_state: RunState,
    budget: BudgetSnapshot,
) -> None:
    with pytest.raises(PolicyInputError) as error:
        always_policy().decide(signals, selected_state, budget)

    assert error.value.__context__ is None
    assert error.value.__cause__ is None


class BrokenPolicy:
    def __init__(self) -> None:
        self._resolved = always_policy().resolved_configuration

    @property
    def policy_version(self) -> str:
        return self._resolved.policy_version

    @property
    def configuration_digest(self) -> str:
        return self._resolved.configuration_digest

    @property
    def resolved_configuration(self) -> Any:
        return self._resolved

    def decide(
        self,
        _signals: list[Signal],
        _state: RunState,
        _budget: BudgetSnapshot,
    ) -> InvocationDecision:
        raise RuntimeError("plugin-policy-secret")


class ReturningPolicy(BrokenPolicy):
    def __init__(self, updates: dict[str, object]) -> None:
        super().__init__()
        self._updates = updates

    def decide(
        self,
        signals: list[Signal],
        selected_state: RunState,
        budget: BudgetSnapshot,
    ) -> InvocationDecision:
        return (
            always_policy().decide(signals, selected_state, budget).model_copy(update=self._updates)
        )


class MutatingPolicy(BrokenPolicy):
    def decide(
        self,
        signals: list[Signal],
        selected_state: RunState,
        budget: BudgetSnapshot,
    ) -> InvocationDecision:
        selected_state.__dict__["event_sequence"] = 99
        budget.limits.__dict__["model_calls"] = 100
        return always_policy().decide(signals, selected_state, budget)


class BudgetOnlyMutatingPolicy(BrokenPolicy):
    def decide(
        self,
        signals: list[Signal],
        selected_state: RunState,
        budget: BudgetSnapshot,
    ) -> InvocationDecision:
        budget.limits.__dict__["model_calls"] = 100
        return always_policy().decide(signals, selected_state, budget)


class IncompleteDecisionPolicy(BrokenPolicy):
    def decide(
        self,
        _signals: list[Signal],
        _state: RunState,
        _budget: BudgetSnapshot,
    ) -> InvocationDecision:
        return InvocationDecision.model_construct()


class OversizedResolvedPolicy(BrokenPolicy):
    @property
    def resolved_configuration(self) -> Any:
        return ResolvedPolicyConfiguration.model_construct(
            schema_version=SCHEMA_VERSION,
            policy_version=self.policy_version,
            configuration={"items": [None] * 50_001},
            configuration_digest=self.configuration_digest,
        )


class ExplodingResolvedMapping(Mapping[str, object]):
    def __getitem__(self, _key: str) -> object:
        raise RuntimeError("resolved-mapping-secret")

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("resolved-mapping-secret")

    def __len__(self) -> int:
        raise RuntimeError("resolved-mapping-secret")


class ExplodingResolvedPolicy(BrokenPolicy):
    @property
    def resolved_configuration(self) -> Any:
        return ResolvedPolicyConfiguration.model_construct(
            schema_version=SCHEMA_VERSION,
            policy_version=self.policy_version,
            configuration=MappingProxyType(ExplodingResolvedMapping()),
            configuration_digest=self.configuration_digest,
        )


class RenamedBudgetedPolicy(BudgetedPolicy):
    @property
    def policy_version(self) -> str:
        return "renamed-budget/v1"


class ExplodingMetadataPolicy(BrokenPolicy):
    @property
    def policy_version(self) -> str:
        raise RuntimeError("metadata-policy-secret")


@pytest.mark.parametrize(
    "updates",
    (
        {"decision_id": UUID("00000000-0000-4000-8000-000000003099")},
        {"run_id": OTHER_RUN_ID},
        {"event_sequence": 2},
        {"policy_version": "other/v1"},
        {"configuration_digest": "e" * 64},
        {"budget_snapshot": snapshot(limit=99)},
        {"created_at": NOW + timedelta(seconds=1)},
        {"invoke": "policy-contract-secret"},
    ),
)
def test_budget_wrapper_rejects_inner_decisions_with_mismatched_context(
    updates: dict[str, object],
) -> None:
    policy = BudgetedPolicy(
        cast(ConfiguredTriggerPolicy, ReturningPolicy(updates)),
        budget_config(BudgetAmounts(model_calls=1)),
    )

    with pytest.raises(PolicyContractError) as error:
        policy.decide([], state(), snapshot())

    assert "policy-contract-secret" not in str(error.value)
    assert error.value.__context__ is None
    assert error.value.__cause__ is None


@pytest.mark.parametrize("inner", (object(), ExplodingMetadataPolicy()))
def test_budget_wrapper_rejects_invalid_inner_metadata_without_echo(inner: object) -> None:
    with pytest.raises(PolicyContractError) as error:
        BudgetedPolicy(
            cast(ConfiguredTriggerPolicy, inner),
            budget_config(BudgetAmounts(model_calls=1)),
        )

    assert "metadata-policy-secret" not in str(error.value)
    assert error.value.__context__ is None
    assert error.value.__cause__ is None


def test_budget_wrapper_isolates_authoritative_inputs_from_mutating_plugins() -> None:
    selected_state = state()
    budget = snapshot(limit=1)
    policy = BudgetedPolicy(
        cast(ConfiguredTriggerPolicy, MutatingPolicy()),
        budget_config(BudgetAmounts(model_calls=2)),
    )

    with pytest.raises(PolicyContractError):
        policy.decide([], selected_state, budget)

    assert selected_state.event_sequence == 1
    assert budget.limits.model_calls == 1


def test_mutating_plugin_cannot_bypass_the_authoritative_budget_snapshot() -> None:
    budget = snapshot(limit=1)
    policy = BudgetedPolicy(
        cast(ConfiguredTriggerPolicy, BudgetOnlyMutatingPolicy()),
        budget_config(BudgetAmounts(model_calls=2)),
    )

    with pytest.raises(PolicyContractError):
        policy.decide([], state(), budget)

    assert budget.limits.model_calls == 1


@pytest.mark.parametrize(
    "inner",
    (
        IncompleteDecisionPolicy(),
        OversizedResolvedPolicy(),
        ExplodingResolvedPolicy(),
    ),
)
def test_budget_wrapper_sanitizes_incomplete_or_oversized_plugin_models(
    inner: object,
) -> None:
    policy_config = budget_config(BudgetAmounts(model_calls=1))

    if isinstance(inner, (OversizedResolvedPolicy, ExplodingResolvedPolicy)):
        with pytest.raises(PolicyContractError) as error:
            BudgetedPolicy(cast(ConfiguredTriggerPolicy, inner), policy_config)
        assert "resolved-mapping-secret" not in str(error.value)
        assert error.value.__context__ is None
        assert error.value.__cause__ is None
    else:
        policy = BudgetedPolicy(cast(ConfiguredTriggerPolicy, inner), policy_config)
        with pytest.raises(PolicyContractError):
            policy.decide([], state(), snapshot())


def test_budget_wrapper_rejects_ambiguous_nested_budget_policies() -> None:
    inner = BudgetedPolicy(
        always_policy(),
        budget_config(BudgetAmounts(model_calls=1)),
    )

    with pytest.raises(PolicyContractError):
        BudgetedPolicy(
            inner,
            budget_config(BudgetAmounts(model_calls=1)),
        )

    renamed = RenamedBudgetedPolicy(
        always_policy(),
        budget_config(BudgetAmounts(model_calls=1)),
    )
    with pytest.raises(PolicyContractError):
        BudgetedPolicy(
            renamed,
            budget_config(BudgetAmounts(model_calls=1)),
        )


def test_budget_wrapper_sanitizes_inner_policy_contract_failures() -> None:
    policy = BudgetedPolicy(
        cast(ConfiguredTriggerPolicy, BrokenPolicy()),
        budget_config(BudgetAmounts(model_calls=1)),
    )

    with pytest.raises(PolicyContractError) as error:
        policy.decide([], state(), snapshot())

    assert "plugin-policy-secret" not in str(error.value)
    assert error.value.__context__ is None
    assert error.value.__cause__ is None
