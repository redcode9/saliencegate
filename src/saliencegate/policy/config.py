from __future__ import annotations

import math
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from saliencegate.domain import BudgetAmounts, JsonObject, canonical_json, length_prefixed_sha256
from saliencegate.domain.records import UUID4, ComponentIdentifier, Sha256Digest, UtcDatetime

MAX_SIGNED_64 = (1 << 63) - 1
MAX_SCRIPT_DECISIONS = 10_000
_MAX_CONFIGURATION_BYTES = 1_000_000
_MAX_CONFIGURATION_NODES = 50_000
_MAX_CONFIGURATION_DEPTH = 64
_POLICY_VERSION = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._:/+\-]{0,255}$")
_CONFIGURATION_DIGEST_DOMAIN = "saliencegate:policy:configuration:v1"
_BUDGET_FIELDS = (
    "model_calls",
    "input_tokens",
    "output_tokens",
    "canonical_token_equivalents",
    "latency_us",
    "interventions",
    "schema_repairs",
)


class PolicyConfigurationError(ValueError):
    """A configuration boundary failed without exposing caller-owned values."""

    def __init__(self) -> None:
        super().__init__("policy configuration failed validation")


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
        validate_default=True,
    )


class RunState(_FrozenModel):
    """Minimal immutable run state required by every deterministic trigger policy."""

    schema_version: Literal["1.0"]
    decision_id: UUID4
    run_id: UUID4
    current_event_id: UUID4
    event_sequence: Annotated[int, Field(ge=1, le=MAX_SIGNED_64)]
    decision_ordinal: Annotated[int, Field(ge=1, le=MAX_SIGNED_64)]
    created_at: UtcDatetime


class NeverInvokeConfig(_FrozenModel):
    schema_version: Literal["1.0"]
    policy_kind: Literal["never_invoke"]


class AlwaysInvokeConfig(_FrozenModel):
    schema_version: Literal["1.0"]
    policy_kind: Literal["always_invoke"]


class ScriptedPolicyConfig(_FrozenModel):
    schema_version: Literal["1.0"]
    policy_kind: Literal["scripted"]
    decisions: Annotated[tuple[bool, ...], Field(max_length=MAX_SCRIPT_DECISIONS)]
    on_exhaustion: Literal["silence"]

    @field_validator("decisions", mode="before")
    @classmethod
    def exact_bounded_decisions(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)) or type(value) not in (list, tuple):
            raise ValueError("script decisions must be exact bounded booleans")
        if len(value) > MAX_SCRIPT_DECISIONS or any(type(item) is not bool for item in value):
            raise ValueError("script decisions must be exact bounded booleans")
        return tuple(value)


def _budget_amounts_are_bounded(value: object) -> bool:
    return type(value) is BudgetAmounts and all(
        type(getattr(value, field_name, None)) is int
        and 0 <= getattr(value, field_name, -1) <= MAX_SIGNED_64
        for field_name in _BUDGET_FIELDS
    )


def _validated_budget_amounts(value: object) -> BudgetAmounts | None:
    raw: object
    if _budget_amounts_are_bounded(value):
        assert type(value) is BudgetAmounts
        raw = value.model_dump(mode="python", warnings=False)
    elif (
        type(value) is dict
        and set(value) == set(_BUDGET_FIELDS)
        and all(type(item) is int and 0 <= item <= MAX_SIGNED_64 for item in value.values())
    ):
        raw = value
    else:
        return None
    return BudgetAmounts.model_validate(raw)


class BudgetedPolicyConfig(_FrozenModel):
    schema_version: Literal["1.0"]
    policy_kind: Literal["budgeted"]
    reservation: BudgetAmounts
    on_denial: Literal["silence"]

    @field_validator("reservation", mode="before")
    @classmethod
    def exact_bounded_reservation(cls, value: object) -> BudgetAmounts:
        validated = _validated_budget_amounts(value)
        if validated is None:
            raise ValueError("budget reservation is invalid")
        return validated

    @model_validator(mode="after")
    def reservation_can_run_memory(self) -> Self:
        if self.reservation.model_calls < 1:
            raise ValueError("budgeted policy requires at least one model call")
        return self


PolicyConfig: TypeAlias = (
    NeverInvokeConfig | AlwaysInvokeConfig | ScriptedPolicyConfig | BudgetedPolicyConfig
)
_RESERVED_POLICY_CONFIG_TYPES: dict[str, type[PolicyConfig]] = {
    "always-invoke/v1": AlwaysInvokeConfig,
    "budgeted/v1": BudgetedPolicyConfig,
    "never-invoke/v1": NeverInvokeConfig,
    "scripted/v1": ScriptedPolicyConfig,
}


def _bounded_json(value: object, *, allow_mapping_proxy: bool) -> bool:
    try:
        remaining = _MAX_CONFIGURATION_BYTES
        nodes = 0
        stack = [(value, 0)]
        while stack:
            item, depth = stack.pop()
            nodes += 1
            if nodes > _MAX_CONFIGURATION_NODES or depth > _MAX_CONFIGURATION_DEPTH:
                return False
            mapping = type(item) is dict or (allow_mapping_proxy and type(item) is MappingProxyType)
            if mapping:
                assert isinstance(item, Mapping)
                available_children = _MAX_CONFIGURATION_NODES - nodes - len(stack)
                declared_length = len(item)
                if declared_length > available_children:
                    return False
                remaining -= 2 + declared_length * 2
                if remaining < 0:
                    return False
                observed_length = 0
                for key, nested in item.items():
                    if observed_length >= available_children or type(key) is not str:
                        return False
                    observed_length += 1
                    remaining -= 6 * len(key) + 3
                    if remaining < 0:
                        return False
                    stack.append((nested, depth + 1))
                if observed_length != declared_length:
                    return False
            elif type(item) in (list, tuple):
                assert isinstance(item, (list, tuple))
                if len(item) > _MAX_CONFIGURATION_NODES - nodes - len(stack):
                    return False
                remaining -= 2 + len(item)
                if remaining < 0:
                    return False
                stack.extend((nested, depth + 1) for nested in item)
            elif type(item) is str:
                remaining -= 6 * len(item) + 2
            elif item is None or type(item) is bool:
                remaining -= 5
            elif type(item) is int:
                remaining -= max(2, item.bit_length() // 3 + 2)
            elif type(item) is float:
                if not math.isfinite(item):
                    return False
                remaining -= 32
            else:
                return False
            if remaining < 0:
                return False
        return True
    except Exception:
        return False


def _json_is_bounded(value: object) -> bool:
    return _bounded_json(value, allow_mapping_proxy=False)


def _frozen_json_is_bounded(value: object) -> bool:
    return _bounded_json(value, allow_mapping_proxy=True)


def _configuration_digest(policy_version: str, configuration: object) -> str | None:
    digest: str | None = None
    try:
        digest = length_prefixed_sha256(
            policy_version,
            canonical_json(configuration),
            domain=_CONFIGURATION_DIGEST_DOMAIN,
        )
    except Exception:
        digest = None
    return digest


class ResolvedPolicyConfiguration(_FrozenModel):
    """Self-verifying fully resolved policy configuration for replay artifacts."""

    schema_version: Literal["1.0"]
    policy_version: ComponentIdentifier
    configuration: Annotated[JsonObject, Field(repr=False)]
    configuration_digest: Sha256Digest

    @field_validator("configuration", mode="before")
    @classmethod
    def bounded_configuration(cls, value: object) -> object:
        if not _json_is_bounded(value):
            raise ValueError("resolved policy configuration exceeds its local bound")
        return value

    @model_validator(mode="after")
    def digest_matches_configuration(self) -> Self:
        expected = _configuration_digest(self.policy_version, self.configuration)
        if expected is None:
            raise ValueError("resolved policy configuration is not canonical JSON")
        if self.configuration_digest != expected:
            raise ValueError("resolved policy configuration digest does not match")
        return self


def _config_is_preflight_safe(value: object) -> bool:
    try:
        if type(value) in (NeverInvokeConfig, AlwaysInvokeConfig):
            return True
        if type(value) is ScriptedPolicyConfig:
            return (
                type(value.decisions) is tuple
                and len(value.decisions) <= MAX_SCRIPT_DECISIONS
                and all(type(item) is bool for item in value.decisions)
            )
        if type(value) is BudgetedPolicyConfig:
            return _budget_amounts_are_bounded(value.reservation)
        return False
    except Exception:
        return False


def _validated_config(value: object) -> PolicyConfig | None:
    if not _config_is_preflight_safe(value):
        return None
    validated: PolicyConfig | None = None
    try:
        if type(value) is NeverInvokeConfig:
            candidate: PolicyConfig = NeverInvokeConfig.model_validate(
                value.model_dump(mode="python", warnings=False)
            )
        elif type(value) is AlwaysInvokeConfig:
            candidate = AlwaysInvokeConfig.model_validate(
                value.model_dump(mode="python", warnings=False)
            )
        elif type(value) is ScriptedPolicyConfig:
            candidate = ScriptedPolicyConfig.model_validate(
                value.model_dump(mode="python", warnings=False)
            )
        else:
            assert type(value) is BudgetedPolicyConfig
            candidate = BudgetedPolicyConfig.model_validate(
                value.model_dump(mode="python", warnings=False)
            )
        if candidate == value:
            validated = candidate
    except Exception:
        validated = None
    return validated


def resolve_policy_configuration(
    policy_version: str,
    configuration: object,
) -> ResolvedPolicyConfiguration:
    """Validate and seal a typed configuration with a domain-separated digest."""

    if type(policy_version) is not str or _POLICY_VERSION.fullmatch(policy_version) is None:
        raise PolicyConfigurationError()
    expected_type = _RESERVED_POLICY_CONFIG_TYPES.get(policy_version)
    if expected_type is not None and type(configuration) is not expected_type:
        raise PolicyConfigurationError()
    validated = _validated_config(configuration)
    if validated is None:
        raise PolicyConfigurationError()
    payload = validated.model_dump(mode="json", warnings=False)
    return _seal_policy_configuration(policy_version, payload)


def seal_policy_configuration(
    policy_version: str,
    configuration: object,
) -> ResolvedPolicyConfiguration:
    """Seal one fully resolved, secret-free JSON configuration for an external policy."""

    if type(policy_version) is not str or policy_version in _RESERVED_POLICY_CONFIG_TYPES:
        raise PolicyConfigurationError()
    return _seal_policy_configuration(policy_version, configuration)


def _seal_policy_configuration(
    policy_version: str,
    payload: object,
) -> ResolvedPolicyConfiguration:
    if (
        type(policy_version) is not str
        or _POLICY_VERSION.fullmatch(policy_version) is None
        or type(payload) is not dict
        or not _json_is_bounded(payload)
    ):
        raise PolicyConfigurationError()
    assert isinstance(payload, dict)
    digest = _configuration_digest(policy_version, payload)
    if digest is None:
        raise PolicyConfigurationError()
    resolved: ResolvedPolicyConfiguration | None = None
    try:
        resolved = ResolvedPolicyConfiguration(
            schema_version="1.0",
            policy_version=policy_version,
            configuration=payload,
            configuration_digest=digest,
        )
    except Exception:
        resolved = None
    if resolved is None:
        raise PolicyConfigurationError()
    return resolved


__all__ = [
    "MAX_SCRIPT_DECISIONS",
    "MAX_SIGNED_64",
    "AlwaysInvokeConfig",
    "BudgetedPolicyConfig",
    "NeverInvokeConfig",
    "PolicyConfig",
    "PolicyConfigurationError",
    "ResolvedPolicyConfiguration",
    "RunState",
    "ScriptedPolicyConfig",
    "resolve_policy_configuration",
    "seal_policy_configuration",
]
