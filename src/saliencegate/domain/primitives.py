"""Lightweight validated scalar aliases shared with domain records."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated

from pydantic import AfterValidator, StringConstraints


def _require_exact_string(value: str) -> str:
    if type(value) is not str:
        raise ValueError("string subclasses are not accepted")
    return value


ComponentIdentifier = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9_][A-Za-z0-9._:/+\-]*$",
    ),
    AfterValidator(_require_exact_string),
]
Sha256Digest = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
    AfterValidator(_require_exact_string),
]


def _require_utc(value: datetime) -> datetime:
    if type(value) is not datetime:
        raise ValueError("datetime subclasses are not accepted")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware UTC")
    if value.utcoffset() != timedelta(0):
        raise ValueError("timestamp must use UTC rather than a non-zero offset")
    return value.astimezone(UTC)


UtcDatetime = Annotated[datetime, AfterValidator(_require_utc)]


__all__ = ["ComponentIdentifier", "Sha256Digest", "UtcDatetime"]
