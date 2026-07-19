from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

APPROXIMATE_TOKEN_ALGORITHM_VERSION: Literal["utf8-bytes-ceil-div-4-v1"] = (
    "utf8-bytes-ceil-div-4-v1"
)


class TokenCountingInputError(ValueError):
    """Raised when input cannot be measured without retaining or echoing it."""

    def __init__(self) -> None:
        super().__init__("text must be an exact UTF-8-encodable string")


class TextSize(BaseModel):
    """An immutable measurement produced by a versioned counting algorithm."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
        validate_default=True,
    )

    utf8_bytes: Annotated[int, Field(ge=0)]
    code_points: Annotated[int, Field(ge=0)]
    approximate_tokens: Annotated[int, Field(ge=0)]
    algorithm_version: Literal["utf8-bytes-ceil-div-4-v1"] = APPROXIMATE_TOKEN_ALGORITHM_VERSION

    @model_validator(mode="after")
    def counts_are_consistent(self) -> Self:
        if (self.utf8_bytes == 0) != (self.code_points == 0):
            raise ValueError("UTF-8 byte and code-point counts must be empty together")
        if self.code_points > self.utf8_bytes:
            raise ValueError("code-point count cannot exceed the UTF-8 byte count")
        if self.utf8_bytes > self.code_points * 4:
            raise ValueError("UTF-8 byte count cannot exceed four bytes per code point")
        if self.approximate_tokens != (self.utf8_bytes + 3) // 4:
            raise ValueError("approximate token count does not match the algorithm version")
        return self


class DeterministicTokenCounter:
    """Measure text without a model-specific tokenizer.

    Version ``utf8-bytes-ceil-div-4-v1`` counts exact UTF-8 bytes and Python
    Unicode code points, then estimates one token for every started four-byte
    block. The estimate is a stable batching heuristic, not a native tokenizer
    count for any model.
    """

    __slots__ = ()

    def measure(self, text: str) -> TextSize:
        if type(text) is not str:
            raise TokenCountingInputError()
        try:
            encoded = text.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            raise TokenCountingInputError() from None

        utf8_bytes = len(encoded)
        return TextSize(
            utf8_bytes=utf8_bytes,
            code_points=len(text),
            approximate_tokens=(utf8_bytes + 3) // 4,
        )
