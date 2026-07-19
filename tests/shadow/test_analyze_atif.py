from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from uuid import UUID

import pytest

import saliencegate.shadow.session as session_module
from saliencegate.domain import canonical_json
from saliencegate.security import InstallationKey
from saliencegate.shadow import (
    ATIFProfile,
    ATIFShadowAdapter,
    ShadowAnalyzer,
    ShadowConfigurationError,
    ShadowEnvironmentBinding,
    ShadowSession,
    ShadowStateError,
    ShadowTraceInputError,
    ShadowTraceReport,
    analyze_atif_bytes,
    encode_shadow_trace_report,
)

_RUN_ID = UUID("55555555-5555-4555-8555-555555555555")
_KEY = InstallationKey(b"k" * 32)
_TASK_SCOPE_DIGEST = "1" * 64
_LINEAGE_SCOPE_DIGEST = "2" * 64
_CAPTURE_MANIFEST_DIGEST = "3" * 64
_FIXTURE = Path("tests/fixtures/shadow/atif/codex-bundled-synthetic.trajectory.json")
_TERMINUS_FIXTURE = Path("tests/fixtures/shadow/atif/terminus-timeout-sanitized.trajectory.json")


def _environment() -> ShadowEnvironmentBinding:
    return ShadowEnvironmentBinding(
        default_working_directory="/synthetic/one-call-default",
        environment_digest="e" * 64,
    )


def _source() -> bytes:
    return _FIXTURE.read_bytes()


def _invalid_call_source() -> bytes:
    value = json.loads(_source())
    value["steps"][1]["tool_calls"][0]["arguments"]["cmd"] = 7
    return canonical_json(value)


def _assert_sanitized(error: BaseException) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None
    assert "synthetic" not in str(error)
    assert "synthetic" not in repr(error)


def test_analyze_atif_bytes_has_the_frozen_one_call_signature() -> None:
    signature = inspect.signature(analyze_atif_bytes)

    assert tuple(signature.parameters) == (
        "source_bytes",
        "run_id",
        "profile",
        "environment",
        "installation_key",
        "redaction_policy",
        "task_scope_digest",
        "lineage_scope_digest",
        "capture_manifest_digest",
    )
    assert signature.parameters["source_bytes"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in tuple(signature.parameters.values())[1:]
    )
    assert signature.parameters["run_id"].default is inspect.Parameter.empty
    assert signature.parameters["profile"].default is inspect.Parameter.empty
    assert signature.parameters["environment"].default is inspect.Parameter.empty
    assert all(
        signature.parameters[name].default is None
        for name in (
            "installation_key",
            "redaction_policy",
            "task_scope_digest",
            "lineage_scope_digest",
            "capture_manifest_digest",
        )
    )
    assert "config" not in signature.parameters


@pytest.mark.asyncio
async def test_one_call_report_matches_the_explicit_adapter_session_path() -> None:
    source = _source()
    environment = _environment()
    adapter = ATIFShadowAdapter(
        profile=ATIFProfile.HARBOR_CODEX_V1,
        environment=environment,
    )
    trace = adapter.adapt_bytes(
        source,
        run_id=_RUN_ID,
        task_scope_digest=_TASK_SCOPE_DIGEST,
        lineage_scope_digest=_LINEAGE_SCOPE_DIGEST,
        capture_manifest_digest=_CAPTURE_MANIFEST_DIGEST,
    )
    async with ShadowSession.in_memory_for_trace(
        run_id=_RUN_ID,
        trace_binding=trace.binding,
        installation_key=_KEY,
    ) as session:
        explicit = await ShadowAnalyzer(session).analyze(trace)

    one_call = await analyze_atif_bytes(
        source,
        run_id=_RUN_ID,
        profile=ATIFProfile.HARBOR_CODEX_V1,
        environment=environment,
        installation_key=_KEY,
        task_scope_digest=_TASK_SCOPE_DIGEST,
        lineage_scope_digest=_LINEAGE_SCOPE_DIGEST,
        capture_manifest_digest=_CAPTURE_MANIFEST_DIGEST,
    )

    assert type(one_call) is ShadowTraceReport
    assert one_call == explicit
    assert encode_shadow_trace_report(one_call) == encode_shadow_trace_report(explicit)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("profile", "fixture"),
    (
        (ATIFProfile.HARBOR_CODEX_V1, _FIXTURE),
        (ATIFProfile.HARBOR_TERMINUS_2_V1, _TERMINUS_FIXTURE),
    ),
)
async def test_one_call_supports_every_sealed_atif_profile(
    profile: ATIFProfile,
    fixture: Path,
) -> None:
    report = await analyze_atif_bytes(
        fixture.read_bytes(),
        run_id=_RUN_ID,
        profile=profile,
        environment=_environment(),
        installation_key=_KEY,
    )

    assert report.binding.adapter_profile_id == profile.value
    assert report.binding.capture_scope == "selected_events"


@pytest.mark.asyncio
async def test_one_call_omitted_key_is_ephemeral_and_the_owned_session_is_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[ShadowSession] = []
    original_factory = ShadowSession.in_memory_for_trace

    def forbidden_key_lookup(*_args: object, **_kwargs: object) -> InstallationKey:
        raise AssertionError("one-call analysis reached the persistent key boundary")

    def capture_session(
        _session_type: type[ShadowSession],
        **kwargs: object,
    ) -> ShadowSession:
        session = original_factory(**kwargs)  # type: ignore[arg-type]
        captured.append(session)
        return session

    monkeypatch.setattr(
        session_module,
        "load_or_create_installation_key",
        forbidden_key_lookup,
    )
    monkeypatch.setattr(
        ShadowSession,
        "in_memory_for_trace",
        classmethod(capture_session),
    )

    report = await analyze_atif_bytes(
        _source(),
        run_id=_RUN_ID,
        profile=ATIFProfile.HARBOR_CODEX_V1,
        environment=_environment(),
    )

    assert type(report) is ShadowTraceReport
    assert len(captured) == 1
    assert captured[0]._closed is True


@pytest.mark.asyncio
async def test_one_call_preserves_structured_adapter_errors_before_key_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_key_generation() -> InstallationKey:
        raise AssertionError("invalid ATIF reached session creation")

    monkeypatch.setattr(
        session_module,
        "generate_installation_key",
        forbidden_key_generation,
    )

    with pytest.raises(ShadowTraceInputError) as captured:
        await analyze_atif_bytes(
            _invalid_call_source(),
            run_id=_RUN_ID,
            profile=ATIFProfile.HARBOR_CODEX_V1,
            environment=_environment(),
        )

    error = captured.value
    assert error.reason_code == "invalid_tool_call"
    assert error.step_ordinal == 2
    assert error.call_ordinal == 1
    assert error.result_ordinal is None
    _assert_sanitized(error)


@pytest.mark.asyncio
async def test_one_call_rejects_non_exact_adapter_configuration_without_type_leaks() -> None:
    with pytest.raises(ShadowConfigurationError) as captured:
        await analyze_atif_bytes(
            _source(),
            run_id=_RUN_ID,
            profile="harbor-codex/v1",  # type: ignore[arg-type]
            environment=_environment(),
        )

    _assert_sanitized(captured.value)


@pytest.mark.asyncio
async def test_one_call_closes_its_session_when_analysis_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[ShadowSession] = []
    original_factory = ShadowSession.in_memory_for_trace

    def capture_session(
        _session_type: type[ShadowSession],
        **kwargs: object,
    ) -> ShadowSession:
        session = original_factory(**kwargs)  # type: ignore[arg-type]
        captured.append(session)
        return session

    async def fail_analysis(
        _analyzer: ShadowAnalyzer,
        _trace: object,
    ) -> ShadowTraceReport:
        raise ShadowStateError()

    monkeypatch.setattr(
        ShadowSession,
        "in_memory_for_trace",
        classmethod(capture_session),
    )
    monkeypatch.setattr(ShadowAnalyzer, "analyze", fail_analysis)

    with pytest.raises(ShadowStateError) as failure:
        await analyze_atif_bytes(
            _source(),
            run_id=_RUN_ID,
            profile=ATIFProfile.HARBOR_CODEX_V1,
            environment=_environment(),
            installation_key=_KEY,
        )

    _assert_sanitized(failure.value)
    assert len(captured) == 1
    assert captured[0]._closed is True


@pytest.mark.asyncio
async def test_one_call_propagates_cancellation_after_closing_its_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[ShadowSession] = []
    started = asyncio.Event()
    never = asyncio.Event()
    original_factory = ShadowSession.in_memory_for_trace

    def capture_session(
        _session_type: type[ShadowSession],
        **kwargs: object,
    ) -> ShadowSession:
        session = original_factory(**kwargs)  # type: ignore[arg-type]
        captured.append(session)
        return session

    async def wait_forever(
        _analyzer: ShadowAnalyzer,
        _trace: object,
    ) -> ShadowTraceReport:
        started.set()
        await never.wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(
        ShadowSession,
        "in_memory_for_trace",
        classmethod(capture_session),
    )
    monkeypatch.setattr(ShadowAnalyzer, "analyze", wait_forever)

    task = asyncio.create_task(
        analyze_atif_bytes(
            _source(),
            run_id=_RUN_ID,
            profile=ATIFProfile.HARBOR_CODEX_V1,
            environment=_environment(),
            installation_key=_KEY,
        )
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(captured) == 1
    assert captured[0]._closed is True
