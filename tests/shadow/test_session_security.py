from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

import saliencegate.repository.sqlite as sqlite_module
import saliencegate.shadow.session as session_module
from saliencegate.domain import TraceEvent
from saliencegate.security import InstallationKey, RedactionPolicy
from saliencegate.shadow.errors import (
    ShadowConfigurationError,
    ShadowInputError,
    ShadowStateError,
)
from saliencegate.shadow.inputs import ShadowObservationSource
from saliencegate.shadow.session import ShadowSession

from .conftest import NOW, RUN_ID


def _assert_sanitized(error: Exception, expected: str) -> None:
    assert str(error) == expected
    assert error.__cause__ is None
    assert error.__context__ is None


def _private_directory(path: Path) -> Path:
    path.mkdir()
    path.chmod(0o700)
    return path


def _private_file(path: Path, data: bytes = b"") -> Path:
    path.write_bytes(data)
    path.chmod(0o600)
    return path


@pytest.mark.parametrize(
    "policy",
    (
        RedactionPolicy(literal_secrets=("shadow-run/v1",)),
        RedactionPolicy(literal_secrets=("deadbeef",)),
        RedactionPolicy(structured_field_names=("schema_version",)),
        RedactionPolicy(structured_field_names=("capture_scope",)),
        RedactionPolicy(structured_field_names=("detector_version",)),
    ),
)
def test_internal_record_redaction_conflicts_fail_before_repository_construction(
    policy: RedactionPolicy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def forbidden_repository(**kwargs: Any) -> object:
        calls.append(kwargs)
        raise AssertionError("repository construction must not run")

    monkeypatch.setattr(session_module, "MemoryRunRepository", forbidden_repository)

    with pytest.raises(ShadowConfigurationError) as raised:
        ShadowSession.in_memory(
            run_id=RUN_ID,
            installation_key=InstallationKey(b"p" * 32),
            redaction_policy=policy,
        )

    _assert_sanitized(raised.value, "shadow configuration is invalid")
    assert calls == []


@pytest.mark.asyncio
async def test_typed_event_redaction_is_preflighted_before_append() -> None:
    policy = RedactionPolicy(structured_field_names=("environment_digest",))
    async with ShadowSession.in_memory(
        run_id=RUN_ID,
        installation_key=InstallationKey(b"e" * 32),
        redaction_policy=policy,
    ) as session:
        await session.start(source_event_id="start", occurred_at=NOW)

        with pytest.raises(ShadowInputError) as raised:
            await session.action(
                source_event_id="unsafe-action",
                occurred_at=NOW + timedelta(seconds=1),
                command="pwd",
                working_directory="/project",
                environment_digest="a" * 64,
            )

        _assert_sanitized(raised.value, "shadow input is invalid")
        entries = await session._repository.ledger(RUN_ID)
        events = tuple(entry.record for entry in entries if isinstance(entry.record, TraceEvent))
        assert tuple(event.source_event_id for event in events) == ("start",)

        accepted = await session.observation(
            source_event_id="still-healthy",
            occurred_at=NOW + timedelta(seconds=2),
            source=ShadowObservationSource.TASK_INPUT,
            payload={"bounded": True},
        )
        assert accepted.ref.sequence == 2


@pytest.mark.asyncio
async def test_dynamic_signal_redaction_conflict_fails_before_event_append() -> None:
    policy = RedactionPolicy(literal_secrets=(NOW.strftime("%Y-%m-%dT%H"),))
    async with ShadowSession.in_memory(
        run_id=RUN_ID,
        installation_key=InstallationKey(b"d" * 32),
        redaction_policy=policy,
    ) as session:
        await session.start(source_event_id="start", occurred_at=NOW)

        with pytest.raises(ShadowConfigurationError) as raised:
            await session.action(
                source_event_id="signal-unsafe-action",
                occurred_at=NOW + timedelta(seconds=1),
                command="pwd",
                working_directory="/project",
                environment_digest="a" * 64,
            )

        _assert_sanitized(raised.value, "shadow configuration is invalid")
        entries = await session._repository.ledger(RUN_ID)
        events = tuple(entry.record for entry in entries if isinstance(entry.record, TraceEvent))
        assert tuple(event.source_event_id for event in events) == ("start",)


@pytest.mark.skipif(os.name != "posix", reason="private SQLite boundaries require POSIX")
def test_sqlite_target_replacement_between_authorization_and_connect_is_not_modified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = parent / "shadow.sqlite3"
    displaced = parent / "authorized-displaced"
    replacement = b"replacement-must-remain-byte-identical"
    real_connect = sqlite_module.sqlite3.connect

    def replace_before_connect(*args: Any, **kwargs: Any) -> Any:
        target.rename(displaced)
        _private_file(target, replacement)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite_module.sqlite3, "connect", replace_before_connect)

    with pytest.raises(ShadowConfigurationError) as raised:
        ShadowSession.sqlite(
            target,
            run_id=RUN_ID,
            installation_key=InstallationKey(b"r" * 32),
        )

    _assert_sanitized(raised.value, "shadow configuration is invalid")
    assert target.read_bytes() == replacement
    assert displaced.read_bytes() == b""


@pytest.mark.skipif(os.name != "posix", reason="private SQLite boundaries require POSIX")
def test_sqlite_target_disappearance_during_connect_never_recreates_the_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = parent / "shadow.sqlite3"
    real_connect = sqlite_module.sqlite3.connect

    def remove_before_connect(*args: Any, **kwargs: Any) -> Any:
        target.unlink()
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite_module.sqlite3, "connect", remove_before_connect)

    with pytest.raises(ShadowStateError) as raised:
        ShadowSession.sqlite(
            target,
            run_id=RUN_ID,
            installation_key=InstallationKey(b"m" * 32),
        )

    _assert_sanitized(raised.value, "shadow state is invalid")
    assert not target.exists()


@pytest.mark.skipif(os.name != "posix", reason="private SQLite boundaries require POSIX")
def test_sqlite_journal_replacement_is_rejected_before_the_first_statement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = parent / "shadow.sqlite3"
    journal = Path(f"{target}-journal")
    displaced = parent / "authorized-journal-displaced"
    replacement = b"private-journal-victim-must-remain-byte-identical"
    real_connect = sqlite_module.sqlite3.connect

    def replace_journal_during_connect(*args: Any, **kwargs: Any) -> Any:
        journal.rename(displaced)
        _private_file(journal, replacement)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite_module.sqlite3, "connect", replace_journal_during_connect)

    with pytest.raises(ShadowConfigurationError) as raised:
        ShadowSession.sqlite(
            target,
            run_id=RUN_ID,
            installation_key=InstallationKey(b"j" * 32),
        )

    _assert_sanitized(raised.value, "shadow configuration is invalid")
    assert journal.read_bytes() == replacement
    assert displaced.read_bytes() == b""
    assert target.read_bytes() == b""


@pytest.mark.skipif(os.name != "posix", reason="private SQLite boundaries require POSIX")
@pytest.mark.parametrize("suffix", ("-wal", "-shm", "-journal"))
@pytest.mark.parametrize("attack", ("symlink", "hardlink"))
def test_sqlite_sidecar_aliases_fail_before_repository_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    attack: str,
) -> None:
    parent = _private_directory(tmp_path / "store")
    target = _private_file(parent / "shadow.sqlite3")
    victim = _private_file(parent / "victim", b"victim-must-remain-byte-identical")
    sidecar = Path(f"{target}{suffix}")
    if attack == "symlink":
        sidecar.symlink_to(victim)
    else:
        sidecar.hardlink_to(victim)
    repository_calls: list[object] = []

    def forbidden_repository(*args: Any, **kwargs: Any) -> object:
        repository_calls.append((args, kwargs))
        raise AssertionError("repository construction must not run")

    monkeypatch.setattr(
        session_module.SQLiteRunRepository,
        "_from_file_authorization",
        forbidden_repository,
    )

    with pytest.raises(ShadowConfigurationError) as raised:
        ShadowSession.sqlite(
            target,
            run_id=RUN_ID,
            installation_key=InstallationKey(b"s" * 32),
        )

    _assert_sanitized(raised.value, "shadow configuration is invalid")
    assert repository_calls == []
    assert victim.read_bytes() == b"victim-must-remain-byte-identical"
    assert target.read_bytes() == b""


@pytest.mark.skipif(os.name != "posix", reason="private SQLite boundaries require POSIX")
@pytest.mark.asyncio
async def test_sqlite_rw_uri_quotes_reserved_filename_characters(tmp_path: Path) -> None:
    parent = _private_directory(tmp_path / "store")
    target = parent / "shadow ?#%.sqlite3"

    async with ShadowSession.sqlite(
        target,
        run_id=RUN_ID,
        installation_key=InstallationKey(b"u" * 32),
    ) as session:
        started = await session.start(source_event_id="start", occurred_at=NOW)

    assert started.ref.sequence == 1
    assert target.is_file()
