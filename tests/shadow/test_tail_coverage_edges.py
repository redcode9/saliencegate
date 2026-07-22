"""Coverage-oriented regressions for otherwise rare contract edges."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from tests.shadow.conftest import OTHER_RUN_ID, RUN_ID, TraceEventFactory
from tests.shadow.test_analyzer import _memory_session
from tests.shadow.test_analyzer_fail_closed_edges import _persist_prepared, _prepared_pair
from tests.shadow.test_atif_codex import (
    adapt as adapt_codex,
)
from tests.shadow.test_atif_codex import (
    call as codex_call,
)
from tests.shadow.test_atif_codex import (
    result as codex_result,
)
from tests.shadow.test_atif_codex import (
    single_call_source,
)
from tests.shadow.test_atif_codex import (
    source_bytes as codex_source_bytes,
)
from tests.shadow.test_atif_codex import (
    step as codex_step,
)
from tests.shadow.test_io import KEY as IO_KEY
from tests.shadow.test_io import TAG as IO_TAG
from tests.shadow.test_io import _complete_trace, _private_file, _read
from tests.shadow.test_trace import build_trace
from tests.shadow.test_trace_report import _matching_shadow_report, _trace
from tests.shadow.test_trusted_report_paths import _run_report_kwargs, _trusted_run_report

import saliencegate.shadow.analyzer as analyzer_module
import saliencegate.shadow.atif as atif_module
import saliencegate.shadow.io as io_module
import saliencegate.shadow.report as report_module
import saliencegate.shadow.trace as trace_module
import saliencegate.shadow.trace_report as trace_report_module
from saliencegate.domain import canonical_json, length_prefixed_sha256
from saliencegate.ports.repository import ConditionalBatchReceipt
from saliencegate.security import RedactionPolicy
from saliencegate.shadow import ShadowConfig, ShadowSession
from saliencegate.shadow.atif import ATIFProfile, ShadowEnvironmentBinding
from saliencegate.shadow.errors import (
    ShadowInvariantError,
    ShadowStateError,
    ShadowTraceInputError,
)
from saliencegate.shadow.inputs import ShadowFinishInput, ShadowInputKind, ShadowStartInput
from saliencegate.shadow.trace import ShadowRecordDiagnostics, ShadowTrace


def test_analyzer_rechecks_events_after_detection_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    session = _memory_session(trace)

    def damaged_sequence(events: tuple[Any, ...]) -> Any:
        first = events[0].model_copy(update={"event_id": uuid4()})
        return SimpleNamespace(events=(first, *events[1:]))

    monkeypatch.setattr(analyzer_module, "_admit_detection_sequence", damaged_sequence)
    monkeypatch.setattr(
        analyzer_module,
        "_admit_shadow_observation_sequence",
        lambda *_args, **_kwargs: object(),
    )

    with pytest.raises(ShadowInvariantError):
        analyzer_module._prepare_analysis(session, trace)


def test_analyzer_rejects_signal_outside_the_prepared_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_trace()
    session = _memory_session(trace)
    invalid = SimpleNamespace(
        run_id=OTHER_RUN_ID,
        evidence_event_ids=(),
        signal_id=uuid4(),
    )
    monkeypatch.setattr(
        analyzer_module,
        "_extract_trusted_report",
        lambda *_args: SimpleNamespace(report=SimpleNamespace(signals=(invalid,))),
    )
    monkeypatch.setattr(
        analyzer_module,
        "_build_shadow_observation_trusted",
        lambda *_args, **_kwargs: object(),
    )

    with pytest.raises(ShadowInvariantError):
        analyzer_module._prepare_analysis(session, trace)


@pytest.mark.parametrize("drift", (False, True))
def test_analyzer_reconciles_repeated_signal_identities(
    monkeypatch: pytest.MonkeyPatch,
    drift: bool,
) -> None:
    trace = build_trace()
    session = _memory_session(trace)
    signal_id = uuid4()
    calls = 0

    def extraction(_extractor: object, trusted_context: Any) -> Any:
        nonlocal calls
        calls += 1
        signal = SimpleNamespace(
            run_id=session._run_id,
            evidence_event_ids=(trusted_context.sequence.events[0].event_id,),
            signal_id=signal_id,
            drift=drift and calls > 1,
        )
        return SimpleNamespace(report=SimpleNamespace(signals=(signal,)))

    monkeypatch.setattr(analyzer_module, "_extract_trusted_report", extraction)
    monkeypatch.setattr(
        analyzer_module,
        "_build_shadow_observation_trusted",
        lambda *_args, **_kwargs: object(),
    )

    expected = ShadowInvariantError if drift else Exception
    with pytest.raises(expected):
        analyzer_module._prepare_analysis(session, trace)


def test_batch_receipt_rejects_untrusted_shape() -> None:
    _session, prepared = _prepared_pair()

    with pytest.raises(ShadowStateError):
        analyzer_module._validate_batch_receipt(
            (),
            prepared,
            initial_head=None,
            receipt=object(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_batch_receipt_rejects_defensive_round_trip_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, prepared = _prepared_pair()
    receipt, _state = await _persist_prepared(session, prepared)
    drifted = receipt.model_copy(update={"receipts": receipt.receipts[:-1]})
    monkeypatch.setattr(
        ConditionalBatchReceipt,
        "model_validate_json",
        classmethod(lambda _cls, _value: drifted),
    )

    with pytest.raises(ShadowStateError):
        analyzer_module._validate_batch_receipt(
            prepared.full_operations,
            prepared,
            initial_head=None,
            receipt=receipt,
        )


@pytest.mark.asyncio
async def test_legacy_retry_requires_a_preflighted_observation(tmp_path: Path) -> None:
    trace = _read(_private_file(tmp_path / "legacy.ndjson", _complete_trace()))
    retry = replace(trace.rows[0], retry_target_ordinal=999)
    damaged = replace(trace, rows=(retry,))
    session = ShadowSession.in_memory(run_id=trace.run_id, installation_key=IO_KEY)

    with pytest.raises(ShadowInvariantError):
        await analyzer_module._analyze_legacy_preflighted(session, damaged)


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ("missing_state", "wrong_suffix"))
async def test_analyzer_rechecks_post_append_state(
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
    session, prepared = _prepared_pair()
    receipt, _state = await _persist_prepared(session, prepared)
    loads = iter(
        (
            None,
            None
            if scenario == "missing_state"
            else SimpleNamespace(head=receipt.final_head, entries=()),
        )
    )

    async def load(_session: ShadowSession) -> Any:
        return next(loads)

    async def append(
        _session: ShadowSession,
        _operations: object,
        *,
        expected_head: object,
    ) -> ConditionalBatchReceipt:
        assert expected_head is None
        return receipt

    monkeypatch.setattr(analyzer_module, "_load_trace_state", load)
    monkeypatch.setattr(ShadowSession, "_append_trace_batch_locked", append)

    with pytest.raises(ShadowStateError):
        await analyzer_module._analyze_prepared(session, prepared)


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ("missing_final", "evidence_mismatch"))
async def test_analyzer_rechecks_final_state(
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
    session, prepared = _prepared_pair()
    state = (
        None
        if scenario == "missing_final"
        else SimpleNamespace(head=None, entries=(), events=(), signals=())
    )

    async def load(_session: ShadowSession) -> Any:
        return state

    monkeypatch.setattr(analyzer_module, "_load_trace_state", load)
    monkeypatch.setattr(analyzer_module, "_validate_state", lambda *_args: None)
    monkeypatch.setattr(analyzer_module, "_missing_operations", lambda *_args: ())

    with pytest.raises(ShadowStateError):
        await analyzer_module._analyze_prepared(session, prepared)


def _io_options() -> Any:
    return io_module._prepare_options(
        run_id=RUN_ID,
        config=ShadowConfig.reference(),
        installation_key=IO_KEY,
        redaction_policy=RedactionPolicy(),
        redaction_policy_tag=IO_TAG,
        capture_scope="unknown",
        task_scope_digest=None,
        lineage_scope_digest=None,
        capture_manifest_digest=None,
        source_adapter="test-shadow/v1",
    )


def test_io_redaction_policy_copy_rejects_equality_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = RedactionPolicy()
    monkeypatch.setattr(RedactionPolicy, "__eq__", lambda _self, _other: False)

    with pytest.raises(ValueError, match="redaction policy is invalid"):
        io_module._copy_redaction_policy(policy)


def test_io_timestamp_rejects_naive_parser_result(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDateTime:
        @staticmethod
        def fromisoformat(_value: str) -> datetime:
            return datetime(2026, 1, 1)

    monkeypatch.setattr(io_module, "datetime", FakeDateTime)

    with pytest.raises(ValueError, match="timestamp is not UTC"):
        io_module._parse_canonical_timestamp("2026-01-01T00:00:00Z")


def test_io_timestamp_rejects_noncanonical_fraction() -> None:
    with pytest.raises(ValueError, match="timestamp is not canonical"):
        io_module._parse_canonical_timestamp("2026-01-01T00:00:00.1Z")


def test_io_timestamp_rechecks_parser_canonicalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Parsed:
        tzinfo = UTC
        microsecond = 0

        def utcoffset(self) -> timedelta:
            return timedelta(0)

        def astimezone(self, _zone: object) -> Parsed:
            return self

        def isoformat(self, *, timespec: str) -> str:
            assert timespec == "seconds"
            return "2026-01-02T00:00:00+00:00"

    class FakeDateTime:
        @staticmethod
        def fromisoformat(_value: str) -> Parsed:
            return Parsed()

    monkeypatch.setattr(io_module, "datetime", FakeDateTime)

    with pytest.raises(ValueError, match="timestamp is not canonical"):
        io_module._parse_canonical_timestamp("2026-01-01T00:00:00Z")


@pytest.mark.parametrize(
    "payload",
    (
        {},
        {
            "schema_version": "1.0",
            "framework": "pytest",
            "status": "failed",
            "failures": [],
        },
    ),
)
def test_io_test_payload_requires_exact_shape(payload: dict[str, object]) -> None:
    event = SimpleNamespace(payload={"test_report": payload})

    assert (
        io_module._event_payload_is_valid(
            _io_options(),
            event,  # type: ignore[arg-type]
            ShadowInputKind.TEST_RESULT,
        )
        is False
    )


def test_io_payload_validation_rejects_equal_non_enum_kind() -> None:
    event = SimpleNamespace(payload={"shadow_run": {}})

    assert (
        io_module._event_payload_is_valid(
            _io_options(),
            event,  # type: ignore[arg-type]
            str(ShadowInputKind.START.value),  # type: ignore[arg-type]
        )
        is False
    )


@pytest.mark.parametrize("kind", (ShadowInputKind.START, ShadowInputKind.FINISH))
def test_io_marker_preflight_requires_redaction_stable_payload(kind: ShadowInputKind) -> None:
    record = (
        ShadowStartInput(source_event_id="io-start", occurred_at=datetime(2026, 1, 1, tzinfo=UTC))
        if kind is ShadowInputKind.START
        else ShadowFinishInput(
            source_event_id="io-finish",
            occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )

    with pytest.raises(ValueError, match="marker is not redaction-stable"):
        io_module._preflight_redacted_event(
            _io_options(),
            object(),  # type: ignore[arg-type]
            record,
            kind=kind,
            event_id=uuid4(),
            sequence=1,
            start_payload=None,
            finish_payload=None,
        )


def test_io_preflight_rechecks_start_before_finish_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finish = ShadowFinishInput(
        source_event_id="io-finish",
        occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    monkeypatch.setattr(
        io_module,
        "_parse_input_record",
        lambda *_args, **_kwargs: (ShadowInputKind.FINISH, finish),
    )

    with pytest.raises(ValueError, match="run_end has no run_start"):
        io_module._preflight_record_values(({"kind": "run_start"},), _io_options())


def test_trace_binding_can_defensively_fall_through_profile_identity_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AlwaysContains:
        def __contains__(self, _value: object) -> bool:
            return True

    class FlippingIdentity(str):
        equality_calls = 0

        def __eq__(self, other: object) -> bool:
            self.equality_calls += 1
            return self.equality_calls == 1 and other == "profile_content_addressed"

        def __ne__(self, _other: object) -> bool:
            return False

        __hash__ = str.__hash__

    binding = build_trace().binding.model_copy(
        update={"identity_mode": FlippingIdentity("profile_content_addressed")}
    )
    monkeypatch.setattr(trace_module, "_IDENTITY_MODES", AlwaysContains())

    assert binding.validate_identity_and_digest() is binding


def test_trace_exact_json_addition_enforces_byte_budget() -> None:
    with pytest.raises(trace_module._ExactJSONLimitError):
        trace_module._copy_exact_json(
            [1, 1],
            max_bytes=3,
            max_depth=1,
            max_container_items=2,
            max_nodes=3,
            max_string_bytes=1,
        )


def test_trace_timestamp_rejects_noncanonical_fraction() -> None:
    with pytest.raises(ShadowTraceInputError):
        trace_module._parse_timestamp("2026-01-01T00:00:00.1Z", ordinal=1)


def test_trace_timestamp_rechecks_parser_canonicalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Parsed:
        microsecond = 0

        def astimezone(self, _zone: object) -> Parsed:
            return self

        def isoformat(self, *, timespec: str) -> str:
            assert timespec == "seconds"
            return "2026-01-02T00:00:00+00:00"

    class FakeDateTime:
        min = datetime.min

        @staticmethod
        def fromisoformat(_value: str) -> Parsed:
            return Parsed()

    monkeypatch.setattr(trace_module, "datetime", FakeDateTime)

    with pytest.raises(ShadowTraceInputError):
        trace_module._parse_timestamp("2026-01-01T00:00:00Z", ordinal=1)


def test_trace_canonicalizer_handles_an_exhausted_ordinal_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trace_module, "range", lambda *_args: (), raising=False)

    with pytest.raises(ShadowTraceInputError):
        trace_module._canonical_records(
            (),
            run_id=RUN_ID,
            capture_scope="unknown",
            timestamp_mode="record_declared",
        )


@pytest.mark.parametrize("field", ("mapped_shadow_record_count", "source_record_count"))
def test_trace_exactness_rechecks_direct_diagnostic_counts(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    trace = build_trace()
    diagnostics = trace.diagnostics
    assert type(diagnostics) is ShadowRecordDiagnostics
    diagnostics.__dict__[field] = getattr(diagnostics, field) + 1
    object.__setattr__(trace, "_diagnostics_bytes", canonical_json(diagnostics))
    monkeypatch.setattr(
        ShadowRecordDiagnostics,
        "model_validate",
        classmethod(lambda _cls, value: value),
    )

    assert trace._is_exact() is False


def _trusted_trace_case(trace_event_factory: TraceEventFactory) -> tuple[Any, Any, Any]:
    trace = _trace()
    public = _matching_shadow_report(trace_event_factory, trace)
    return trace, public, _trusted_run_report(public)


def test_public_trace_report_builder_rechecks_mapped_digest(
    trace_event_factory: TraceEventFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace, public, _trusted = _trusted_trace_case(trace_event_factory)
    object.__setattr__(trace, "mapped_record_digest", "0" * 64)
    monkeypatch.setattr(ShadowTrace, "_is_exact", lambda _self: True)

    with pytest.raises(ShadowInvariantError):
        trace_report_module._build_shadow_trace_report(
            trace=trace,
            shadow_report=public,
            session_binding=trace.binding,
            authenticated_start_source_adapter=trace.binding.source_adapter,
        )


def test_trusted_trace_report_builder_rejects_non_trace() -> None:
    with pytest.raises(ShadowInvariantError):
        trace_report_module._build_shadow_trace_report_trusted(
            trace=object(),  # type: ignore[arg-type]
            shadow_report=object(),  # type: ignore[arg-type]
            session_binding=object(),  # type: ignore[arg-type]
            authenticated_start_source_adapter="adapter",
        )


def test_trusted_trace_report_builder_rejects_session_binding_drift(
    trace_event_factory: TraceEventFactory,
) -> None:
    trace, _public, trusted = _trusted_trace_case(trace_event_factory)
    other_binding = _trace(adapter_descriptor={"schema_version": "other/v1"}).binding

    with pytest.raises(ShadowInvariantError):
        trace_report_module._build_shadow_trace_report_trusted(
            trace=trace,
            shadow_report=trusted,
            session_binding=other_binding,
            authenticated_start_source_adapter=trace.binding.source_adapter,
        )


def test_trusted_trace_report_builder_rejects_run_drift(
    trace_event_factory: TraceEventFactory,
) -> None:
    _trace_value, _public, trusted = _trusted_trace_case(trace_event_factory)
    other = _trace(run_id=OTHER_RUN_ID)

    with pytest.raises(ShadowInvariantError):
        trace_report_module._build_shadow_trace_report_trusted(
            trace=other,
            shadow_report=trusted,
            session_binding=other.binding,
            authenticated_start_source_adapter=other.binding.source_adapter,
        )


def test_trusted_trace_report_builder_rechecks_capture_provenance(
    trace_event_factory: TraceEventFactory,
) -> None:
    trace, public, _trusted = _trusted_trace_case(trace_event_factory)
    kwargs = _run_report_kwargs(public)
    kwargs["input_byte_digest"] = "f" * 64
    changed = report_module._build_shadow_run_report_trusted(**kwargs)  # type: ignore[arg-type]

    with pytest.raises(ShadowInvariantError):
        trace_report_module._build_shadow_trace_report_trusted(
            trace=trace,
            shadow_report=changed,
            session_binding=trace.binding,
            authenticated_start_source_adapter=trace.binding.source_adapter,
        )


def test_trusted_trace_report_builder_rechecks_mapped_digest(
    trace_event_factory: TraceEventFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace, _public, trusted = _trusted_trace_case(trace_event_factory)
    object.__setattr__(trace, "mapped_record_digest", "0" * 64)
    monkeypatch.setattr(ShadowTrace, "_is_exact", lambda _self: True)

    with pytest.raises(ShadowInvariantError):
        trace_report_module._build_shadow_trace_report_trusted(
            trace=trace,
            shadow_report=trusted,
            session_binding=trace.binding,
            authenticated_start_source_adapter=trace.binding.source_adapter,
        )


def test_trusted_trace_report_builder_rejects_authenticated_adapter_drift(
    trace_event_factory: TraceEventFactory,
) -> None:
    trace, _public, trusted = _trusted_trace_case(trace_event_factory)

    with pytest.raises(ShadowInvariantError):
        trace_report_module._build_shadow_trace_report_trusted(
            trace=trace,
            shadow_report=trusted,
            session_binding=trace.binding,
            authenticated_start_source_adapter="different-adapter",
        )


def test_trusted_trace_report_builder_rejects_constructed_state_drift(
    trace_event_factory: TraceEventFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace, _public, trusted = _trusted_trace_case(trace_event_factory)
    original = trace_report_module._model_state_is_exact

    def reject_body(model_type: type[Any], value: object) -> bool:
        if model_type is trace_report_module._ShadowTraceReportBody:
            return False
        return original(model_type, value)

    monkeypatch.setattr(trace_report_module, "_model_state_is_exact", reject_body)

    with pytest.raises(ShadowInvariantError):
        trace_report_module._build_shadow_trace_report_trusted(
            trace=trace,
            shadow_report=trusted,
            session_binding=trace.binding,
            authenticated_start_source_adapter=trace.binding.source_adapter,
        )


def test_trace_report_json_preflight_handles_top_level_comma_branch() -> None:
    trace_report_module._preflight_canonical_json_structure(b",")


def _atif_topology_case(source: bytes | None = None) -> tuple[Any, Any, Any, Any, Any]:
    trace = adapt_codex(single_call_source(0) if source is None else source)
    configuration = json.loads(trace._configuration_preimage())
    environment = ShadowEnvironmentBinding.model_validate(configuration["environment"])
    contexts = configuration["mapped_action_contexts"]
    contract = atif_module._PROFILE_CONTRACTS[ATIFProfile.HARBOR_CODEX_V1]
    return trace, contract, environment, contexts, configuration


def _changed_atif_records(trace: Any, index: int, **changes: object) -> tuple[object, ...]:
    records = [dict(record) for record in trace.records]
    records[index].update(changes)
    return tuple(atif_module._freeze_json(record) for record in records)


def test_atif_timestamp_rejects_parser_canonicalization_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Parsed:
        microsecond = 0

        def astimezone(self, _zone: object) -> Parsed:
            return self

        def isoformat(self, *, timespec: str) -> str:
            assert timespec == "seconds"
            return "2026-01-02T00:00:00+00:00"

    class FakeDateTime:
        @staticmethod
        def fromisoformat(_value: str) -> Parsed:
            return Parsed()

    monkeypatch.setattr(atif_module, "datetime", FakeDateTime)

    with pytest.raises(ShadowTraceInputError):
        atif_module._normalize_timestamp("2026-01-01T00:00:00Z", step=1)


@pytest.mark.parametrize("scenario", ("shape", "semantics", "diagnostic_equation"))
def test_atif_topology_rechecks_result_and_diagnostic_contracts(scenario: str) -> None:
    trace, contract, environment, contexts, _configuration = _atif_topology_case()
    records = trace.records
    diagnostics = trace.diagnostics
    if scenario == "shape":
        changed = dict(records[2])
        changed.pop("status")
        records = (records[0], records[1], atif_module._freeze_json(changed), records[3])
    elif scenario == "semantics":
        records = _changed_atif_records(trace, 2, status="failed")
    else:
        diagnostics = diagnostics.model_copy(
            update={"ignored_message_step_count": diagnostics.total_step_count}
        )

    assert not atif_module._records_match_atif_topology(
        contract=contract,
        environment=environment,
        contexts=contexts,
        records=records,
        timestamp_mode=trace.binding.timestamp_mode,
        diagnostics=diagnostics,
    )


def test_atif_topology_rejects_decreasing_action_timestamps() -> None:
    metadata = {"tool_metadata": {"exit_code": 0, "status": "completed"}}
    source = codex_source_bytes(
        [
            codex_step(
                1,
                calls=[codex_call(1, tool_call_id="tail-call-1")],
                results=[codex_result("tail-call-1")],
                extra=metadata,
                timestamp="2026-01-01T00:00:01Z",
            ),
            codex_step(
                2,
                calls=[codex_call(2, tool_call_id="tail-call-2")],
                results=[codex_result("tail-call-2")],
                extra=metadata,
                timestamp="2026-01-01T00:00:02Z",
            ),
        ]
    )
    trace, contract, environment, contexts, _configuration = _atif_topology_case(source)
    records = list(trace.records)
    second_action = next(
        index
        for index, record in enumerate(records)
        if record.get("kind") == "action" and index > 1
    )
    records[second_action] = atif_module._freeze_json(
        {**dict(records[second_action]), "occurred_at": "2026-01-01T00:00:00Z"}
    )
    records[second_action + 1] = atif_module._freeze_json(
        {**dict(records[second_action + 1]), "occurred_at": "2026-01-01T00:00:00Z"}
    )

    assert not atif_module._records_match_atif_topology(
        contract=contract,
        environment=environment,
        contexts=contexts,
        records=tuple(records),
        timestamp_mode="source_utc",
        diagnostics=trace.diagnostics,
    )


def test_atif_configuration_contract_rejects_non_tuple_records() -> None:
    trace, _contract, _environment, _contexts, _configuration = _atif_topology_case()

    assert not atif_module._matches_sealed_configuration_contract(
        profile_id=trace.binding.adapter_profile_id,
        profile_digest=trace.binding.adapter_profile_digest,
        configuration_bytes=trace._configuration_preimage(),
        configuration_digest=trace.binding.adapter_configuration_digest,
        source_schema_version=trace.binding.source_schema_version,
        timestamp_mode=trace.binding.timestamp_mode,
        capture_scope=trace.binding.capture_scope,
        records=list(trace.records),  # type: ignore[arg-type]
        diagnostics=trace.diagnostics,
    )


def test_atif_configuration_contract_rejects_exact_shape_field_drift() -> None:
    trace, _contract, _environment, _contexts, configuration = _atif_topology_case()
    configuration["selection_scope"] = "different-selection"
    encoded = canonical_json(configuration)
    digest = length_prefixed_sha256(
        encoded,
        domain="saliencegate:shadow:adapter-configuration:v1",
    )

    assert not atif_module._matches_sealed_configuration_contract(
        profile_id=trace.binding.adapter_profile_id,
        profile_digest=trace.binding.adapter_profile_digest,
        configuration_bytes=encoded,
        configuration_digest=digest,
        source_schema_version=trace.binding.source_schema_version,
        timestamp_mode=trace.binding.timestamp_mode,
        capture_scope=trace.binding.capture_scope,
        records=trace.records,
        diagnostics=trace.diagnostics,
    )


def test_atif_builder_rechecks_new_trace_exactness(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(_is_exact=lambda: False)
    monkeypatch.setattr(atif_module, "_new_shadow_trace", lambda **_kwargs: fake)

    with pytest.raises(ShadowInvariantError):
        adapt_codex(single_call_source(0))
