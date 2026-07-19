from __future__ import annotations

from pathlib import Path

import pytest
from tests.shadow.test_trace import build_trace

import saliencegate.shadow.session as session_module
from saliencegate.repository import SQLiteRunRepository
from saliencegate.security import InstallationKey, SecureFileError
from saliencegate.security.files import StableFileAuthorization
from saliencegate.shadow import (
    ShadowConfigurationError,
    ShadowSession,
    ShadowStateError,
)

_KEY = InstallationKey(b"a" * 32)
_SQLITE_SUFFIXES = ("", "-wal", "-shm", "-journal")


def _sqlite_slots(path: Path) -> tuple[Path, ...]:
    return tuple(Path(f"{path}{suffix}") for suffix in _SQLITE_SUFFIXES)


def _assert_sanitized(error: Exception) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None


@pytest.mark.parametrize("invalid_key", (None, b"a" * 32, object()))
def test_sqlite_trace_factory_rejects_invalid_keys_before_lookup_or_path_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_key: object,
) -> None:
    trace = build_trace()
    path = tmp_path / "invalid-key.sqlite3"
    key_lookups = 0
    path_inspections = 0

    def counted_key_lookup() -> InstallationKey:
        nonlocal key_lookups
        key_lookups += 1
        return _KEY

    def counted_path_inspection(_path: object) -> StableFileAuthorization:
        nonlocal path_inspections
        path_inspections += 1
        raise AssertionError("invalid key reached the filesystem boundary")

    monkeypatch.setattr(
        session_module,
        "load_or_create_installation_key",
        counted_key_lookup,
    )
    monkeypatch.setattr(
        session_module,
        "inspect_private_file_location",
        counted_path_inspection,
    )

    with pytest.raises(ShadowConfigurationError) as captured:
        ShadowSession.sqlite_for_trace(
            path,
            run_id=trace.run_id,
            trace_binding=trace.binding,
            installation_key=invalid_key,  # type: ignore[arg-type]
        )

    _assert_sanitized(captured.value)
    assert key_lookups == 0
    assert path_inspections == 0
    assert all(not slot.exists() for slot in _sqlite_slots(path))


@pytest.mark.parametrize("mutation", ("extra", "serializer"))
def test_trace_factories_reject_forged_binding_state_without_dispatching_callbacks(
    mutation: str,
) -> None:
    trace = build_trace()
    binding = trace.binding
    serializer_calls = 0

    class PoisonedSerializer:
        def to_json(self, *_args: object, **_kwargs: object) -> bytes:
            nonlocal serializer_calls
            serializer_calls += 1
            raise AssertionError("forged serializer callback was dispatched")

    if mutation == "extra":
        object.__setattr__(binding, "unexpected_state", "fixture-secret-binding-state")
    else:
        object.__setattr__(binding, "__pydantic_serializer__", PoisonedSerializer())

    with pytest.raises(ShadowConfigurationError) as captured:
        ShadowSession.in_memory_for_trace(
            run_id=trace.run_id,
            trace_binding=binding,
            installation_key=_KEY,
        )

    _assert_sanitized(captured.value)
    assert serializer_calls == 0
    assert "fixture-secret" not in str(captured.value)
    assert "fixture-secret" not in repr(captured.value)


@pytest.mark.asyncio
async def test_sqlite_trace_factory_rejects_a_post_inspection_swap_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    path = tmp_path / "swapped.sqlite3"
    path.write_bytes(b"original-database")
    path.chmod(0o600)
    session = ShadowSession.sqlite_for_trace(
        path,
        run_id=trace.run_id,
        trace_binding=trace.binding,
        installation_key=_KEY,
    )
    displaced = tmp_path / "displaced.sqlite3"
    path.rename(displaced)
    path.write_bytes(b"replacement-database")
    path.chmod(0o600)
    repository_opens = 0

    def forbidden_repository_open(
        _repository_type: type[SQLiteRunRepository],
        _authorization: StableFileAuthorization,
        **_kwargs: object,
    ) -> SQLiteRunRepository:
        nonlocal repository_opens
        repository_opens += 1
        raise AssertionError("stale authorization reached SQLite")

    monkeypatch.setattr(
        SQLiteRunRepository,
        "_from_file_authorization",
        classmethod(forbidden_repository_open),
    )

    async with session:
        with pytest.raises(ShadowConfigurationError) as captured:
            await session._snapshot_for_trace()

    _assert_sanitized(captured.value)
    assert repository_opens == 0
    assert path.read_bytes() == b"replacement-database"
    assert displaced.read_bytes() == b"original-database"
    assert all(not slot.exists() for slot in _sqlite_slots(path)[1:])


@pytest.mark.asyncio
async def test_sqlite_trace_constructor_failure_is_terminal_and_cleans_placeholders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    path = tmp_path / "constructor-failure.sqlite3"
    session = ShadowSession.sqlite_for_trace(
        path,
        run_id=trace.run_id,
        trace_binding=trace.binding,
        installation_key=_KEY,
    )
    constructor_calls: list[StableFileAuthorization] = []

    def fail_repository_open(
        _repository_type: type[SQLiteRunRepository],
        authorization: StableFileAuthorization,
        **_kwargs: object,
    ) -> SQLiteRunRepository:
        constructor_calls.append(authorization)
        assert all(slot.exists() for slot in _sqlite_slots(path))
        raise RuntimeError("fixture-secret-constructor-failure")

    monkeypatch.setattr(
        SQLiteRunRepository,
        "_from_file_authorization",
        classmethod(fail_repository_open),
    )

    with pytest.raises(ShadowStateError) as first_failure:
        await session._snapshot_for_trace()

    _assert_sanitized(first_failure.value)
    assert "fixture-secret" not in str(first_failure.value)
    assert "fixture-secret" not in repr(first_failure.value)
    assert len(constructor_calls) == 1
    assert path.exists()
    assert all(not slot.exists() for slot in _sqlite_slots(path)[1:])

    with pytest.raises(ShadowStateError) as terminal_failure:
        await session._snapshot_for_trace()

    _assert_sanitized(terminal_failure.value)
    assert len(constructor_calls) == 1
    assert session._repository is None
    assert session._lazy_sqlite_location is None
    await session.aclose()
    assert path.exists()
    assert all(not slot.exists() for slot in _sqlite_slots(path)[1:])


@pytest.mark.asyncio
async def test_sqlite_trace_secure_file_constructor_failure_is_configuration_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    path = tmp_path / "secure-file-constructor-failure.sqlite3"
    session = ShadowSession.sqlite_for_trace(
        path,
        run_id=trace.run_id,
        trace_binding=trace.binding,
        installation_key=_KEY,
    )
    constructor_calls: list[StableFileAuthorization] = []

    def fail_repository_open(
        _repository_type: type[SQLiteRunRepository],
        authorization: StableFileAuthorization,
        **_kwargs: object,
    ) -> SQLiteRunRepository:
        constructor_calls.append(authorization)
        assert all(slot.exists() for slot in _sqlite_slots(path))
        raise SecureFileError()

    monkeypatch.setattr(
        SQLiteRunRepository,
        "_from_file_authorization",
        classmethod(fail_repository_open),
    )

    with pytest.raises(ShadowConfigurationError) as first_failure:
        await session._snapshot_for_trace()

    _assert_sanitized(first_failure.value)
    assert len(constructor_calls) == 1
    assert path.exists()
    assert all(not slot.exists() for slot in _sqlite_slots(path)[1:])

    with pytest.raises(ShadowStateError) as terminal_failure:
        await session._snapshot_for_trace()

    _assert_sanitized(terminal_failure.value)
    assert len(constructor_calls) == 1
    assert session._repository is None
    assert session._lazy_sqlite_location is None
    await session.aclose()
    assert path.exists()
    assert all(not slot.exists() for slot in _sqlite_slots(path)[1:])
