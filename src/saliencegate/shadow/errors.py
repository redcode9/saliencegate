from __future__ import annotations

from typing import Final

_TRACE_REASON_CODE_ORDER: Final[tuple[str, ...]] = (
    "invalid_json",
    "input_limit_exceeded",
    "unsupported_schema",
    "profile_mismatch",
    "invalid_step",
    "invalid_timestamp",
    "partial_timestamps",
    "invalid_tool_call",
    "duplicate_tool_call_id",
    "orphan_result",
    "invalid_outcome_metadata",
    "no_supported_action",
    "digest_mismatch",
)
_TRACE_REASON_CODES: Final[frozenset[str]] = frozenset(_TRACE_REASON_CODE_ORDER)


class ShadowInputError(ValueError):
    """A value-free failure at the public Shadow input boundary."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("shadow input is invalid")


class ShadowTraceInputError(ShadowInputError):
    """A content-free trace failure with stable structural coordinates."""

    __slots__ = ("call_ordinal", "reason_code", "result_ordinal", "step_ordinal")

    def __init__(
        self,
        reason_code: str,
        *,
        step_ordinal: int | None = None,
        call_ordinal: int | None = None,
        result_ordinal: int | None = None,
    ) -> None:
        if type(reason_code) is not str or reason_code not in _TRACE_REASON_CODES:
            raise TypeError("trace reason code is invalid")
        coordinates = (step_ordinal, call_ordinal, result_ordinal)
        if any(
            value is not None and (type(value) is not int or value < 1) for value in coordinates
        ):
            raise TypeError("trace error coordinate is invalid")
        super().__init__()
        self.reason_code = reason_code
        self.step_ordinal = step_ordinal
        self.call_ordinal = call_ordinal
        self.result_ordinal = result_ordinal

    def __repr__(self) -> str:
        return f"ShadowTraceInputError(reason_code={self.reason_code!r})"


class ShadowConfigurationError(ValueError):
    """A value-free failure while binding a Shadow configuration."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("shadow configuration is invalid")


class ShadowStateError(RuntimeError):
    """A value-free failure while authenticating Shadow state."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("shadow state is invalid")


class ShadowInvariantError(RuntimeError):
    """A value-free failure of a built-in Shadow contract."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("shadow invariant is invalid")


__all__ = [
    "ShadowConfigurationError",
    "ShadowInputError",
    "ShadowInvariantError",
    "ShadowStateError",
    "ShadowTraceInputError",
]
