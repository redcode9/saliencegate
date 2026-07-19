from __future__ import annotations

import itertools
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID

import pytest
from tests.repository.conformance import (
    CycleRepositoryConformance,
    RepositoryConformance,
    RepositoryFactory,
)

from saliencegate.repository.sqlite import SQLiteRunRepository
from saliencegate.security import InstallationKey

KEY = InstallationKey(b"s" * 32)


def deterministic_ids() -> Iterator[UUID]:
    for value in range(1, 10_000):
        yield UUID(f"00000000-0000-4000-8000-{value:012x}")


class TestSQLiteRunRepository(RepositoryConformance, CycleRepositoryConformance):
    @pytest.fixture
    def repository_factory(
        self,
        tmp_path: Path,
        request: pytest.FixtureRequest,
    ) -> RepositoryFactory:
        repositories: list[SQLiteRunRepository] = []
        paths = itertools.count(1)

        def create() -> SQLiteRunRepository:
            identifiers = deterministic_ids()
            repository = SQLiteRunRepository(
                tmp_path / f"repository-{next(paths)}.sqlite3",
                installation_key=KEY,
                id_factory=lambda: next(identifiers),
            )
            repositories.append(repository)
            return repository

        def close_repositories() -> None:
            for repository in repositories:
                repository.close()

        request.addfinalizer(close_repositories)
        return create
