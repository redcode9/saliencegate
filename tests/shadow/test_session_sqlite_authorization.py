from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

import saliencegate.shadow.session as session_module
from saliencegate.security import (
    InstallationKey,
    StableFileAuthorization,
    authorize_private_sqlite_path,
)
from saliencegate.shadow.errors import ShadowConfigurationError, ShadowInputError
from saliencegate.shadow.session import ShadowSession

from .conftest import NOW, RUN_ID


def _assert_configuration_error_is_sanitized(error: BaseException) -> None:
    assert type(error) is ShadowConfigurationError
    assert str(error) == "shadow configuration is invalid"
    assert error.__cause__ is None
    assert error.__context__ is None
    assert vars(error) == {}


@pytest.mark.asyncio
async def test_authorized_sqlite_factory_never_authorizes_the_path_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = authorize_private_sqlite_path(tmp_path / "authorized.sqlite3")

    def forbidden_authorization(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the authorized factory must not authorize a path")

    monkeypatch.setattr(
        session_module,
        "authorize_private_sqlite_path",
        forbidden_authorization,
    )

    session = ShadowSession._from_sqlite_authorization(
        authorization,
        run_id=RUN_ID,
        installation_key=InstallationKey(b"a" * 32),
    )
    repository = session._repository

    async with session:
        started = await session.start(source_event_id="start", occurred_at=NOW)

    assert started.ref.sequence == 1
    assert repository._file_authorization is authorization
    assert repository._closed is True
    await session.aclose()
    with pytest.raises(ShadowInputError, match=r"^shadow input is invalid$"):
        await session.start(source_event_id="start", occurred_at=NOW)


def test_authorized_sqlite_factory_rejects_non_authorizations_before_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construction_calls: list[object] = []

    def forbidden_repository(*args: object, **kwargs: object) -> object:
        construction_calls.append((args, kwargs))
        raise AssertionError("repository construction must not run")

    monkeypatch.setattr(
        session_module.SQLiteRunRepository,
        "_from_file_authorization",
        forbidden_repository,
    )

    with pytest.raises(ShadowConfigurationError) as raised:
        ShadowSession._from_sqlite_authorization(
            cast(StableFileAuthorization, object()),
            run_id=RUN_ID,
            installation_key=InstallationKey(b"b" * 32),
        )

    _assert_configuration_error_is_sanitized(raised.value)
    assert construction_calls == []


def test_authorized_sqlite_factory_closes_repository_if_session_binding_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = authorize_private_sqlite_path(tmp_path / "binding.sqlite3")
    closed = False

    class IncompleteRepository:
        def close(self) -> None:
            nonlocal closed
            closed = True

    def construct_repository(
        _authorization: StableFileAuthorization,
        **_kwargs: Any,
    ) -> IncompleteRepository:
        assert _authorization is authorization
        return IncompleteRepository()

    def fail_binding(
        _cls: type[ShadowSession],
        _repository: object,
        _options: object,
    ) -> ShadowSession:
        raise RuntimeError("caller-controlled-detail")

    monkeypatch.setattr(
        session_module.SQLiteRunRepository,
        "_from_file_authorization",
        construct_repository,
    )
    monkeypatch.setattr(ShadowSession, "_bind", classmethod(fail_binding))

    with pytest.raises(ShadowConfigurationError) as raised:
        ShadowSession._from_sqlite_authorization(
            authorization,
            run_id=RUN_ID,
            installation_key=InstallationKey(b"c" * 32),
        )

    _assert_configuration_error_is_sanitized(raised.value)
    assert "caller-controlled-detail" not in repr(raised.value)
    assert closed is True


def test_authorized_sqlite_factory_closes_repository_before_propagating_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = authorize_private_sqlite_path(tmp_path / "interrupt.sqlite3")
    closed = False

    class IncompleteRepository:
        def close(self) -> None:
            nonlocal closed
            closed = True

    def construct_repository(
        _authorization: StableFileAuthorization,
        **_kwargs: Any,
    ) -> IncompleteRepository:
        return IncompleteRepository()

    def interrupt_binding(
        _cls: type[ShadowSession],
        _repository: object,
        _options: object,
    ) -> ShadowSession:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        session_module.SQLiteRunRepository,
        "_from_file_authorization",
        construct_repository,
    )
    monkeypatch.setattr(ShadowSession, "_bind", classmethod(interrupt_binding))

    with pytest.raises(KeyboardInterrupt):
        ShadowSession._from_sqlite_authorization(
            authorization,
            run_id=RUN_ID,
            installation_key=InstallationKey(b"e" * 32),
        )

    assert closed is True


@pytest.mark.asyncio
async def test_cli_snapshot_returns_one_authenticated_initial_event_prefix() -> None:
    session = ShadowSession.in_memory(
        run_id=RUN_ID,
        installation_key=InstallationKey(b"d" * 32),
    )

    empty_head, empty_events = await session._snapshot_for_cli()
    started = await session.start(source_event_id="start", occurred_at=NOW)
    head, events = await session._snapshot_for_cli()

    assert empty_head is None
    assert empty_events == ()
    assert head is not None
    assert head.run_id == RUN_ID
    assert head.entry_count >= 1
    assert tuple(event.event_id for event in events) == (started.ref.event_id,)

    await session.aclose()
    with pytest.raises(ShadowInputError, match=r"^shadow input is invalid$"):
        await session._snapshot_for_cli()
