from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import cast

import pytest
from pydantic import ValidationError

import saliencegate.policy.config as config_module
from saliencegate.domain import BudgetAmounts
from saliencegate.policy import (
    AlwaysInvokeConfig,
    BudgetedPolicyConfig,
    NeverInvokeConfig,
    PolicyConfigurationError,
    ResolvedPolicyConfiguration,
    ScriptedPolicyConfig,
    resolve_policy_configuration,
    seal_policy_configuration,
)

SCHEMA_VERSION = "1.0"


class ExplodingMapping(Mapping[str, object]):
    def __getitem__(self, _key: str) -> object:
        raise RuntimeError("sealer-mapping-secret")

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("sealer-mapping-secret")

    def __len__(self) -> int:
        raise RuntimeError("sealer-mapping-secret")


def deeply_nested_configuration() -> dict[str, object]:
    nested: object = None
    for _ in range(66):
        nested = {"next": nested}
    return cast(dict[str, object], nested)


def never_config() -> NeverInvokeConfig:
    return NeverInvokeConfig(
        schema_version=SCHEMA_VERSION,
        policy_kind="never_invoke",
    )


def always_config() -> AlwaysInvokeConfig:
    return AlwaysInvokeConfig(
        schema_version=SCHEMA_VERSION,
        policy_kind="always_invoke",
    )


def script_config(*decisions: bool) -> ScriptedPolicyConfig:
    return ScriptedPolicyConfig(
        schema_version=SCHEMA_VERSION,
        policy_kind="scripted",
        decisions=decisions,
        on_exhaustion="silence",
    )


def budget_config(**amounts: int) -> BudgetedPolicyConfig:
    return BudgetedPolicyConfig(
        schema_version=SCHEMA_VERSION,
        policy_kind="budgeted",
        reservation=BudgetAmounts(model_calls=1, **amounts),
        on_denial="silence",
    )


def test_resolved_configuration_digest_is_stable_complete_and_versioned() -> None:
    first = resolve_policy_configuration("scripted/v1", script_config(True, False))
    second = resolve_policy_configuration("scripted/v1", script_config(True, False))

    assert first == second
    assert first.schema_version == SCHEMA_VERSION
    assert first.policy_version == "scripted/v1"
    assert len(first.configuration_digest) == 64
    assert first.configuration["schema_version"] == SCHEMA_VERSION
    assert first.configuration["policy_kind"] == "scripted"
    assert first.configuration["decisions"] == (True, False)
    assert first.configuration["on_exhaustion"] == "silence"
    assert (
        first.configuration_digest
        == "73526ef2654ace7ccc595c6ab6e283f72347d92c72e9f300158d9c80189b0410"
    )
    assert type(first).model_validate_json(first.model_dump_json()) == first


def test_every_resolved_configuration_change_changes_the_digest() -> None:
    never = resolve_policy_configuration("never-invoke/v1", never_config())
    always = resolve_policy_configuration("always-invoke/v1", always_config())
    one = resolve_policy_configuration("scripted/v1", script_config(True))
    two = resolve_policy_configuration("scripted/v1", script_config(False))
    versioned = resolve_policy_configuration("scripted/v2", script_config(True))

    assert (
        len(
            {
                never.configuration_digest,
                always.configuration_digest,
                one.configuration_digest,
                two.configuration_digest,
                versioned.configuration_digest,
            }
        )
        == 5
    )


def test_budget_configuration_serializes_every_dimension_including_zero() -> None:
    config = budget_config(input_tokens=20, output_tokens=10)
    payload = config.model_dump(mode="json")

    assert payload == {
        "schema_version": SCHEMA_VERSION,
        "policy_kind": "budgeted",
        "reservation": {
            "model_calls": 1,
            "input_tokens": 20,
            "output_tokens": 10,
            "canonical_token_equivalents": 0,
            "latency_us": 0,
            "interventions": 0,
            "schema_repairs": 0,
        },
        "on_denial": "silence",
    }


@pytest.mark.parametrize(
    "config",
    (
        never_config(),
        always_config(),
        script_config(),
        script_config(True, False),
        budget_config(),
    ),
)
def test_policy_configurations_are_strict_frozen_and_round_trip(config: object) -> None:
    restored = type(config).model_validate_json(config.model_dump_json())  # type: ignore[union-attr]
    assert restored == config
    with pytest.raises(ValidationError):
        config.schema_version = "9.9"  # type: ignore[attr-defined,union-attr]


@pytest.mark.parametrize(
    "factory",
    (
        lambda: NeverInvokeConfig(policy_kind="never_invoke"),
        lambda: AlwaysInvokeConfig(schema_version=SCHEMA_VERSION),
        lambda: ScriptedPolicyConfig(
            schema_version=SCHEMA_VERSION,
            policy_kind="scripted",
            decisions=(True,),
        ),
        lambda: BudgetedPolicyConfig(
            schema_version=SCHEMA_VERSION,
            policy_kind="budgeted",
            reservation=BudgetAmounts(model_calls=1),
        ),
    ),
)
def test_experiment_configuration_has_no_hidden_required_fields(factory: object) -> None:
    with pytest.raises(ValidationError):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize("value", (1, 1.0, "1", -1, None))
def test_script_values_are_exact_booleans(value: object) -> None:
    with pytest.raises(ValidationError):
        ScriptedPolicyConfig(
            schema_version=SCHEMA_VERSION,
            policy_kind="scripted",
            decisions=(value,),  # type: ignore[arg-type]
            on_exhaustion="silence",
        )


@pytest.mark.parametrize("value", (True, 1.0, "1", -1, (1 << 63)))
def test_budget_values_are_exact_signed_64_bit_integers(value: object) -> None:
    forged = BudgetAmounts().model_copy(update={"model_calls": value})
    with pytest.raises(ValidationError):
        BudgetedPolicyConfig(
            schema_version=SCHEMA_VERSION,
            policy_kind="budgeted",
            reservation=forged,
            on_denial="silence",
        )


def test_script_length_and_reservation_are_locally_bounded() -> None:
    with pytest.raises(ValidationError):
        script_config(*(True,) * 10_001)
    with pytest.raises(ValidationError):
        BudgetedPolicyConfig(
            schema_version=SCHEMA_VERSION,
            policy_kind="budgeted",
            reservation=BudgetAmounts(),
            on_denial="silence",
        )


def test_resolver_revalidates_exact_models_without_leaking_forged_values() -> None:
    secret = "configuration-secret"
    forged = script_config(True).model_copy(update={"decisions": (secret,)})

    with pytest.raises(PolicyConfigurationError) as error:
        resolve_policy_configuration("scripted/v1", forged)

    assert secret not in str(error.value)
    assert error.value.__context__ is None
    assert error.value.__cause__ is None
    with pytest.raises(PolicyConfigurationError):
        resolve_policy_configuration("scripted/v1", cast(ScriptedPolicyConfig, object()))


@pytest.mark.parametrize(
    ("policy_version", "configuration"),
    (
        ("always-invoke/v1", never_config()),
        ("never-invoke/v1", always_config()),
        ("scripted/v1", always_config()),
        ("budgeted/v1", always_config()),
    ),
)
def test_reserved_policy_versions_require_their_exact_configuration_type(
    policy_version: str,
    configuration: object,
) -> None:
    with pytest.raises(PolicyConfigurationError):
        resolve_policy_configuration(policy_version, configuration)


@pytest.mark.parametrize(
    ("policy_version", "configuration"),
    (
        ("always-invoke/v1", AlwaysInvokeConfig.model_construct()),
        ("never-invoke/v1", NeverInvokeConfig.model_construct()),
        ("scripted/v1", ScriptedPolicyConfig.model_construct()),
        ("budgeted/v1", BudgetedPolicyConfig.model_construct()),
    ),
)
def test_resolver_sanitizes_incomplete_constructed_models(
    policy_version: str,
    configuration: object,
) -> None:
    with pytest.raises(PolicyConfigurationError) as error:
        resolve_policy_configuration(policy_version, configuration)

    assert error.value.__context__ is None
    assert error.value.__cause__ is None


def test_external_policy_can_seal_only_a_non_reserved_secret_free_json_config() -> None:
    first = seal_policy_configuration(
        "example-policy/v1",
        {"schema_version": SCHEMA_VERSION, "threshold_micros": 500_000},
    )
    second = seal_policy_configuration(
        "example-policy/v1",
        {"schema_version": SCHEMA_VERSION, "threshold_micros": 500_000},
    )

    assert first == second
    assert first.configuration["threshold_micros"] == 500_000
    with pytest.raises(PolicyConfigurationError):
        seal_policy_configuration("always-invoke/v1", {"policy_kind": "spoofed"})
    with pytest.raises(PolicyConfigurationError):
        seal_policy_configuration("example-policy/v1", {"unsafe": object()})


@pytest.mark.parametrize(
    "unsafe",
    (
        {"nested": MappingProxyType(ExplodingMapping())},
        {"invalid_utf8": "sealer-secret-\ud800"},
    ),
)
def test_external_policy_sealer_sanitizes_hostile_json_boundaries(unsafe: object) -> None:
    with pytest.raises(PolicyConfigurationError) as error:
        seal_policy_configuration("example-policy/v1", unsafe)

    assert "sealer-secret" not in str(error.value)
    assert error.value.__context__ is None
    assert error.value.__cause__ is None


@pytest.mark.parametrize("policy_version", (True, "", "has space", "v" * 257))
def test_resolver_rejects_invalid_policy_versions_without_echo(policy_version: object) -> None:
    with pytest.raises(PolicyConfigurationError) as error:
        resolve_policy_configuration(cast(str, policy_version), always_config())

    assert error.value.__context__ is None
    assert error.value.__cause__ is None


def test_resolved_configuration_rejects_digest_tampering() -> None:
    resolved = resolve_policy_configuration("always-invoke/v1", always_config())
    payload = resolved.model_dump(mode="python")
    payload["configuration"] = {"finite_measurement": 1.25}

    with pytest.raises(ValidationError, match="digest does not match"):
        ResolvedPolicyConfiguration.model_validate(payload)


@pytest.mark.parametrize(
    "configuration",
    (
        {1: "non-string-key"},
        {"non_finite": float("nan")},
        {"unsupported": object()},
        {"too_large": "x" * 1_000_001},
        {"too_many_nodes": [None] * 50_001},
        deeply_nested_configuration(),
    ),
)
def test_resolved_configuration_rejects_unsafe_or_unbounded_json(
    configuration: object,
) -> None:
    with pytest.raises(ValidationError, match="local bound"):
        ResolvedPolicyConfiguration(
            schema_version=SCHEMA_VERSION,
            policy_version="fixture/v1",
            configuration=cast(dict[str, object], configuration),
            configuration_digest="0" * 64,
        )


def test_script_validator_rejects_container_subclasses() -> None:
    class _Decisions(list[bool]):
        pass

    with pytest.raises(ValueError, match="exact bounded"):
        ScriptedPolicyConfig.exact_bounded_decisions(_Decisions([True]))


@pytest.mark.parametrize(
    ("node_limit", "byte_limit", "value"),
    (
        (1, 1_000, {"a": 1}),
        (50_000, 1, {}),
        (50_000, 5, {"long": None}),
        (50_000, 1, []),
    ),
)
def test_bounded_json_accounts_for_declared_children_keys_and_containers(
    monkeypatch: pytest.MonkeyPatch,
    node_limit: int,
    byte_limit: int,
    value: object,
) -> None:
    monkeypatch.setattr(config_module, "_MAX_CONFIGURATION_NODES", node_limit)
    monkeypatch.setattr(config_module, "_MAX_CONFIGURATION_BYTES", byte_limit)
    assert not config_module._bounded_json(value, allow_mapping_proxy=False)


def test_resolved_configuration_and_sealer_reject_internal_digest_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = resolve_policy_configuration("always-invoke/v1", always_config())
    monkeypatch.setattr(config_module, "_configuration_digest", lambda *_args: None)
    with pytest.raises(ValueError, match="canonical JSON"):
        resolved.digest_matches_configuration()
    with pytest.raises(PolicyConfigurationError):
        config_module._seal_policy_configuration("example/v1", {})


def test_budget_preflight_and_constructor_failure_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert config_module._config_is_preflight_safe(budget_config())
    monkeypatch.setattr(
        config_module,
        "ResolvedPolicyConfiguration",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError()),
    )
    with pytest.raises(PolicyConfigurationError):
        config_module._seal_policy_configuration("example/v1", {})
