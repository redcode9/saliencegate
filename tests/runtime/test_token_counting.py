from __future__ import annotations

from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from saliencegate.runtime.token_counting import (
    APPROXIMATE_TOKEN_ALGORITHM_VERSION,
    DeterministicTokenCounter,
    TextSize,
    TokenCountingInputError,
)


@pytest.mark.parametrize(
    ("text", "utf8_bytes", "code_points", "approximate_tokens"),
    (
        ("", 0, 0, 0),
        ("a", 1, 1, 1),
        ("abcd", 4, 4, 1),
        ("abcde", 5, 5, 2),
        ("line one\nline two\r\n", 19, 19, 5),
        ("café", 5, 4, 2),
        ("e\N{COMBINING ACUTE ACCENT}", 3, 2, 1),
        ("\N{ROCKET}", 4, 1, 1),
        ("\N{WOMAN}\N{ZERO WIDTH JOINER}\N{PERSONAL COMPUTER}", 11, 3, 3),
    ),
)
def test_measurement_has_exact_utf8_and_versioned_approximate_counts(
    text: str,
    utf8_bytes: int,
    code_points: int,
    approximate_tokens: int,
) -> None:
    measured = DeterministicTokenCounter().measure(text)

    assert measured == TextSize(
        utf8_bytes=utf8_bytes,
        code_points=code_points,
        approximate_tokens=approximate_tokens,
        algorithm_version="utf8-bytes-ceil-div-4-v1",
    )
    assert measured.algorithm_version == APPROXIMATE_TOKEN_ALGORITHM_VERSION


def test_text_size_is_strict_immutable_and_forbids_extra_fields() -> None:
    measured = TextSize(utf8_bytes=1, code_points=1, approximate_tokens=1)

    with pytest.raises(ValidationError, match="frozen"):
        measured.__setattr__("utf8_bytes", 2)
    with pytest.raises(ValidationError):
        TextSize.model_validate(
            {
                "utf8_bytes": True,
                "code_points": 1,
                "approximate_tokens": 1,
            }
        )
    with pytest.raises(ValidationError):
        TextSize.model_validate(
            {
                "utf8_bytes": 1,
                "code_points": 1,
                "approximate_tokens": 1,
                "unexpected": 1,
            }
        )
    with pytest.raises(ValidationError):
        TextSize.model_validate(
            {
                "utf8_bytes": 1,
                "code_points": 1,
                "approximate_tokens": 1,
                "algorithm_version": "future-algorithm",
            }
        )


@pytest.mark.parametrize(
    "values",
    (
        {"utf8_bytes": -1, "code_points": 0, "approximate_tokens": 0},
        {"utf8_bytes": 0, "code_points": 1, "approximate_tokens": 0},
        {"utf8_bytes": 1, "code_points": 0, "approximate_tokens": 1},
        {"utf8_bytes": 1, "code_points": 2, "approximate_tokens": 1},
        {"utf8_bytes": 5, "code_points": 1, "approximate_tokens": 2},
        {"utf8_bytes": 4, "code_points": 4, "approximate_tokens": 2},
    ),
)
def test_text_size_rejects_internally_inconsistent_counts(values: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        TextSize.model_validate(values)


class _StringSubclass(str):
    pass


@pytest.mark.parametrize(
    "invalid",
    (
        b"bytes are not text",
        _StringSubclass("subclasses are not accepted"),
        object(),
        None,
        1,
    ),
)
def test_non_exact_strings_are_rejected_without_echoing_input(invalid: object) -> None:
    secret = "fixture-secret-must-not-echo"
    candidate = _StringSubclass(secret) if isinstance(invalid, _StringSubclass) else invalid

    with pytest.raises(TokenCountingInputError) as error:
        DeterministicTokenCounter().measure(cast(str, candidate))

    assert secret not in str(error.value)


def test_lone_surrogates_are_rejected_without_exposing_text() -> None:
    invalid = "prefix-fixture-secret-\ud800-suffix"

    with pytest.raises(TokenCountingInputError) as error:
        DeterministicTokenCounter().measure(invalid)

    assert "fixture-secret" not in str(error.value)


@given(left=st.text(), right=st.text())
def test_exact_utf8_and_code_point_measurements_are_deterministic_and_additive(
    left: str,
    right: str,
) -> None:
    counter = DeterministicTokenCounter()
    left_size = counter.measure(left)
    right_size = counter.measure(right)
    combined = counter.measure(left + right)

    assert counter.measure(left) == left_size
    assert combined.utf8_bytes == left_size.utf8_bytes + right_size.utf8_bytes
    assert combined.code_points == left_size.code_points + right_size.code_points
    assert combined.approximate_tokens == (combined.utf8_bytes + 3) // 4
