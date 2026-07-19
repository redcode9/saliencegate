from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from uuid import UUID

from saliencegate.repository.sqlite import SQLiteRunRepository
from saliencegate.security import InstallationKey

KEY = InstallationKey(b"q" * 32)


def identifiers(*, start: int = 0x800) -> Iterator[UUID]:
    for value in range(start, start + 10_000):
        yield UUID(f"00000000-0000-4000-8000-{value:012x}")


def repository(
    path: Path,
    *,
    key: InstallationKey = KEY,
    id_start: int = 0x800,
    busy_timeout_ms: int = 5_000,
) -> SQLiteRunRepository:
    generated = identifiers(start=id_start)
    return SQLiteRunRepository(
        path,
        installation_key=key,
        id_factory=lambda: next(generated),
        busy_timeout_ms=busy_timeout_ms,
    )


__all__ = ["KEY", "identifiers", "repository"]
