from __future__ import annotations

import hmac
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import ValidationError

import saliencegate.shadow.session as session_module
import saliencegate.signals.base as signal_base_module
from saliencegate.domain import TraceEvent, canonical_json
from saliencegate.ports.repository import RepositoryError
from saliencegate.security import InstallationKey, RedactionPolicy
from saliencegate.shadow.config import ShadowConfig
from saliencegate.shadow.errors import (
    ShadowConfigurationError,
    ShadowInputError,
    ShadowInvariantError,
    ShadowStateError,
)
from saliencegate.shadow.inputs import ShadowEventRef, ShadowObservationSource
from saliencegate.shadow.observation import ShadowEventResult
from saliencegate.shadow.session import ShadowSession
from saliencegate.signals import DetectionContext

from .conftest import NOW, OTHER_RUN_ID, RUN_ID

Backend = Literal["memory", "sqlite"]

_KEY_MATERIAL = b"k" * 32
_TASK_DIGEST = "1" * 64
_LINEAGE_DIGEST = "2" * 64
_MANIFEST_DIGEST = "3" * 64
_SOURCE_ADAPTER = "shadow-contract/v1"
_POLICY_TAG_DOMAIN = b"saliencegate:shadow:redaction-policy:v1"


class _InstallationKeySubclass(InstallationKey):
    pass


class _RedactionPolicySubclass(RedactionPolicy):
    pass


class _ShadowConfigSubclass(ShadowConfig):
    pass


def _factory(
    backend: Backend,
    path: Path,
    **kwargs: Any,
) -> ShadowSession:
    if backend == "memory":
        return ShadowSession.in_memory(**kwargs)
    return ShadowSession.sqlite(path, **kwargs)


def _fixed_options() -> dict[str, object]:
    return {
        "run_id": RUN_ID,
        "config": ShadowConfig.reference(),
        "installation_key": InstallationKey(_KEY_MATERIAL),
        "redaction_policy": RedactionPolicy(
            literal_secrets=("caller-secret",),
            structured_field_names=("private_note",),
        ),
        "capture_scope": "complete_run_declared",
        "task_scope_digest": _TASK_DIGEST,
        "lineage_scope_digest": _LINEAGE_DIGEST,
        "capture_manifest_digest": _MANIFEST_DIGEST,
        "source_adapter": _SOURCE_ADAPTER,
    }


def _policy_tag(material: bytes, policy: RedactionPolicy) -> str:
    configuration = canonical_json(
        {
            "literal_secrets": policy.literal_secrets,
            "structured_field_names": policy.structured_field_names,
        }
    )
    framed = (
        len(_POLICY_TAG_DOMAIN).to_bytes(8, byteorder="big", signed=False)
        + _POLICY_TAG_DOMAIN
        + len(configuration).to_bytes(8, byteorder="big", signed=False)
        + configuration
    )
    return hmac.new(material, framed, sha256).hexdigest()


def _serialized(result: ShadowEventResult) -> bytes:
    return ShadowEventResult.__pydantic_serializer__.to_json(result, warnings=False)


def _assert_sanitized(
    error: BaseException,
    error_type: type[BaseException],
    message: str,
    secret: str,
) -> None:
    assert type(error) is error_type
    assert str(error) == message
    assert secret not in str(error)
    assert secret not in repr(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert vars(error) == {}


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize(
    "invalid_owner",
    ("config_mapping", "config_subclass", "key_subclass", "policy_subclass"),
)
def test_factories_reject_non_exact_owned_types(
    backend: Backend,
    invalid_owner: str,
    tmp_path: Path,
) -> None:
    options = _fixed_options()
    reference = options["config"]
    assert type(reference) is ShadowConfig

    if invalid_owner == "config_mapping":
        options["config"] = reference.model_dump(mode="python")
    elif invalid_owner == "config_subclass":
        options["config"] = _ShadowConfigSubclass.model_validate(
            reference.model_dump(mode="python")
        )
    elif invalid_owner == "key_subclass":
        options["installation_key"] = _InstallationKeySubclass(_KEY_MATERIAL)
    else:
        options["redaction_policy"] = _RedactionPolicySubclass(
            literal_secrets=("caller-secret",),
            structured_field_names=("private_note",),
        )

    path = tmp_path / f"invalid-{invalid_owner}.sqlite3"
    with pytest.raises(ShadowConfigurationError) as raised:
        _factory(backend, path, **options)

    _assert_sanitized(
        raised.value,
        ShadowConfigurationError,
        "shadow configuration is invalid",
        "caller-secret",
    )
    if backend == "sqlite":
        assert not path.exists()


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.asyncio
async def test_factories_copy_owned_values_and_bind_complete_authenticated_markers(
    backend: Backend,
    tmp_path: Path,
) -> None:
    options = _fixed_options()
    config = options["config"]
    key = options["installation_key"]
    policy = options["redaction_policy"]
    assert type(config) is ShadowConfig
    assert type(key) is InstallationKey
    assert type(policy) is RedactionPolicy

    session = _factory(backend, tmp_path / "markers.sqlite3", **options)

    assert type(session) is ShadowSession
    assert session._run_id == RUN_ID
    assert session._run_id is not RUN_ID
    assert session._config == config
    assert session._config is not config
    assert session._config.detectors[0] is not config.detectors[0]
    assert session._installation_key == key
    assert session._installation_key is not key
    assert session._redaction_policy == policy
    assert session._redaction_policy is not policy

    async with session:
        started = await session.start(source_event_id="start", occurred_at=NOW)
        await session.finish(
            source_event_id="finish",
            occurred_at=NOW + timedelta(seconds=1),
        )
        entries = await session._repository.ledger(RUN_ID)

    events = tuple(entry.record for entry in entries if type(entry.record) is TraceEvent)
    assert len(events) == 2
    expected_start = {
        "schema_version": "shadow-run/v1",
        "detector_profile_digest": config.detector_profile_digest,
        "evaluator_configuration_digest": config.evaluator_configuration_digest,
        "redaction_policy_tag": {
            "algorithm": "hmac_sha256",
            "value": _policy_tag(_KEY_MATERIAL, policy),
        },
        "source_adapter": _SOURCE_ADAPTER,
        "capture_scope": "complete_run_declared",
        "task_scope_digest": _TASK_DIGEST,
        "lineage_scope_digest": _LINEAGE_DIGEST,
        "capture_manifest_digest": _MANIFEST_DIGEST,
        "split_metadata_complete": True,
    }
    expected_finish = {
        **expected_start,
        "schema_version": "shadow-run-end/v1",
        "start_event_id": str(started.ref.event_id),
    }

    assert canonical_json(events[0].payload["shadow_run"]) == canonical_json(expected_start)
    assert canonical_json(events[1].payload["shadow_run_end"]) == canonical_json(expected_finish)
    serialized_markers = canonical_json((events[0].payload, events[1].payload))
    assert b"caller-secret" not in serialized_markers
    assert b"private_note" not in serialized_markers


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.asyncio
async def test_unique_event_ceiling_is_exact_and_duplicates_do_not_consume_it(
    backend: Backend,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert session_module._MAX_SHADOW_EVENTS == 10_000
    monkeypatch.setattr(session_module, "_MAX_SHADOW_EVENTS", 3)
    session = _factory(
        backend,
        tmp_path / "ceiling.sqlite3",
        run_id=RUN_ID,
        installation_key=InstallationKey(_KEY_MATERIAL),
    )

    async with session:
        await session.start(source_event_id="start", occurred_at=NOW)
        await session.observation(
            source_event_id="observation-1",
            occurred_at=NOW + timedelta(seconds=1),
            source=ShadowObservationSource.TASK_INPUT,
            payload={"ordinal": 1},
        )
        last = await session.observation(
            source_event_id="observation-2",
            occurred_at=NOW + timedelta(seconds=2),
            source=ShadowObservationSource.TASK_INPUT,
            payload={"ordinal": 2},
        )
        duplicate = await session.observation(
            source_event_id="observation-2",
            occurred_at=NOW + timedelta(seconds=2),
            source=ShadowObservationSource.TASK_INPUT,
            payload={"ordinal": 2},
        )
        assert duplicate == last

        with pytest.raises(ShadowInputError) as raised:
            await session.observation(
                source_event_id="observation-3",
                occurred_at=NOW + timedelta(seconds=3),
                source=ShadowObservationSource.TASK_INPUT,
                payload={"ordinal": 3},
            )
        _assert_sanitized(
            raised.value,
            ShadowInputError,
            "shadow input is invalid",
            "observation-3",
        )

        entries = await session._repository.ledger(RUN_ID)
        assert sum(type(entry.record) is TraceEvent for entry in entries) == 3


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.asyncio
async def test_maximal_context_suffix_survives_backend_round_trip_and_duplicate(
    backend: Backend,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "truncated.sqlite3"
    options = {
        "run_id": RUN_ID,
        "installation_key": InstallationKey(_KEY_MATERIAL),
    }
    session = _factory(backend, path, **options)
    await session.start(source_event_id="start", occurred_at=NOW)
    monkeypatch.setattr(signal_base_module, "_MAX_CONTEXT_SIZE_UPPER_BOUND", 18_000)
    blob = "x" * 1_100

    try:
        await session.observation(
            source_event_id="large-1",
            occurred_at=NOW + timedelta(seconds=1),
            source=ShadowObservationSource.EXTERNAL_MEMORY,
            payload={"blob": blob},
        )
        original = await session.observation(
            source_event_id="large-2",
            occurred_at=NOW + timedelta(seconds=2),
            source=ShadowObservationSource.EXTERNAL_MEMORY,
            payload={"blob": blob},
        )
        entries = await session._repository.ledger(RUN_ID)
        events = tuple(entry.record for entry in entries if type(entry.record) is TraceEvent)

        assert original.observation.context_first_sequence == 2
        assert original.observation.context_last_sequence == 3
        assert original.observation.context_event_count == 2
        assert original.observation.context_truncated is True
        assert DetectionContext(run_id=RUN_ID, events=events[-2:]).events == events[-2:]
        with pytest.raises(ValidationError):
            DetectionContext(run_id=RUN_ID, events=events)

        if backend == "sqlite":
            await session.aclose()
            session = _factory(backend, path, **options)
        duplicate = await session.observation(
            source_event_id="large-2",
            occurred_at=NOW + timedelta(seconds=2),
            source=ShadowObservationSource.EXTERNAL_MEMORY,
            payload={"blob": blob},
        )
        assert _serialized(duplicate) == _serialized(original)
    finally:
        await session.aclose()


async def _fixed_result_bytes(session: ShadowSession) -> tuple[bytes, ...]:
    async with session:
        started = await session.start(source_event_id="start", occurred_at=NOW)
        action = await session.action(
            source_event_id="action-1",
            occurred_at=NOW + timedelta(seconds=1),
            argv=("pytest", "-q"),
            working_directory="/project",
            environment_digest="a" * 64,
        )
        result = await session.tool_result(
            source_event_id="tool-1",
            occurred_at=NOW + timedelta(seconds=2),
            action=action.ref,
            status="failed",
            exit_status=1,
            exception_type="AssertionError caller-secret",
        )
    return tuple(_serialized(item) for item in (started, action, result))


@pytest.mark.asyncio
async def test_memory_and_sqlite_emit_byte_identical_results_for_fixed_inputs(
    tmp_path: Path,
) -> None:
    memory = _factory("memory", tmp_path / "unused.sqlite3", **_fixed_options())
    sqlite = _factory("sqlite", tmp_path / "parity.sqlite3", **_fixed_options())

    assert await _fixed_result_bytes(memory) == await _fixed_result_bytes(sqlite)


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.asyncio
async def test_cross_run_and_non_action_parents_fail_before_detector_evaluation(
    backend: Backend,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _factory(
        backend,
        tmp_path / "parents.sqlite3",
        run_id=RUN_ID,
        installation_key=InstallationKey(_KEY_MATERIAL),
    )
    async with session:
        started = await session.start(source_event_id="start", occurred_at=NOW)
        action = await session.action(
            source_event_id="action-1",
            occurred_at=NOW + timedelta(seconds=1),
            command="pytest -q",
            working_directory="/project",
            environment_digest="a" * 64,
        )

        def forbidden_detector(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("detector evaluation must not run")

        monkeypatch.setattr(
            type(session._extractor),
            "extract_report",
            forbidden_detector,
        )
        cross_run = ShadowEventRef(
            run_id=OTHER_RUN_ID,
            event_id=action.ref.event_id,
            sequence=action.ref.sequence,
        )
        for source_event_id, parent in (
            ("cross-run-parent", cross_run),
            ("non-action-parent", started.ref),
        ):
            with pytest.raises(ShadowInputError) as raised:
                await session.tool_result(
                    source_event_id=source_event_id,
                    occurred_at=NOW + timedelta(seconds=2),
                    action=parent,
                    status="failed",
                )
            _assert_sanitized(
                raised.value,
                ShadowInputError,
                "shadow input is invalid",
                source_event_id,
            )

        entries = await session._repository.ledger(RUN_ID)
        assert sum(type(entry.record) is TraceEvent for entry in entries) == 2


@pytest.mark.asyncio
async def test_session_maps_all_four_failure_families_without_values_or_chaining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration_secret = "configuration secret"
    with pytest.raises(ShadowConfigurationError) as configuration:
        ShadowSession.in_memory(
            run_id=RUN_ID,
            installation_key=InstallationKey(_KEY_MATERIAL),
            source_adapter=configuration_secret,
        )
    _assert_sanitized(
        configuration.value,
        ShadowConfigurationError,
        "shadow configuration is invalid",
        configuration_secret,
    )

    input_secret = "input secret"
    async with ShadowSession.in_memory(
        run_id=RUN_ID,
        installation_key=InstallationKey(_KEY_MATERIAL),
    ) as input_session:
        with pytest.raises(ShadowInputError) as input_failure:
            await input_session.start(source_event_id=input_secret, occurred_at=NOW)
    _assert_sanitized(
        input_failure.value,
        ShadowInputError,
        "shadow input is invalid",
        input_secret,
    )

    state_secret = "state-secret"
    async with ShadowSession.in_memory(
        run_id=RUN_ID,
        installation_key=InstallationKey(_KEY_MATERIAL),
    ) as state_session:
        repository_type = type(state_session._repository)

        async def broken_ledger(_repository: object, _run_id: object) -> object:
            raise RepositoryError(state_secret)

        with monkeypatch.context() as scoped:
            scoped.setattr(repository_type, "ledger", broken_ledger)
            with pytest.raises(ShadowStateError) as state_failure:
                await state_session.start(source_event_id="start", occurred_at=NOW)
    _assert_sanitized(
        state_failure.value,
        ShadowStateError,
        "shadow state is invalid",
        state_secret,
    )

    invariant_secret = "invariant-secret"
    async with ShadowSession.in_memory(
        run_id=RUN_ID,
        installation_key=InstallationKey(_KEY_MATERIAL),
    ) as invariant_session:
        extractor_type = type(invariant_session._extractor)

        def broken_extractor(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError(invariant_secret)

        with monkeypatch.context() as scoped:
            scoped.setattr(extractor_type, "extract_report", broken_extractor)
            with pytest.raises(ShadowInvariantError) as invariant_failure:
                await invariant_session.start(source_event_id="start", occurred_at=NOW)
    _assert_sanitized(
        invariant_failure.value,
        ShadowInvariantError,
        "shadow invariant is invalid",
        invariant_secret,
    )
