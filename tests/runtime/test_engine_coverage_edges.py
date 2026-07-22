"""Coverage-oriented regressions for otherwise rare contract edges."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from tests.runtime import test_engine as fixtures

import saliencegate.runtime.engine as engine_module
from saliencegate.domain import (
    DeliveryTarget,
    ReasonCode,
    Signal,
    SignalType,
    canonical_digest,
    canonical_json,
)
from saliencegate.ports.models import ModelCallStatus, ModelResult, ModelUsage
from saliencegate.repository import MemoryRunRepository
from saliencegate.runtime.engine import (
    ReplayEngine,
    ReplayEngineInputError,
    ReplayEngineInvariantError,
    ReplayEngineModelError,
    ReplayRoutingBinding,
)
from saliencegate.signals import DetectionContext


@pytest.fixture(scope="module")
def replay_result():
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=fixtures.repository_ids(),
    )
    result, *_rest = asyncio.run(fixtures.run_replay(repository))
    return result


def _signal(event: Any, *, suffix: int, signal_type: SignalType) -> Signal:
    return Signal(
        signal_id=UUID(f"00000000-0000-4000-8000-{suffix:012x}"),
        run_id=event.run_id,
        created_at=event.timestamp,
        signal_type=signal_type,
        strength=1.0,
        evidence_event_ids=(event.event_id,),
        detector_version="engine-coverage-edge/v1",
        reason_code=ReasonCode(signal_type.value),
    )


def _valid_model_result() -> ModelResult:
    return ModelResult(
        status=ModelCallStatus.COMPLETED,
        request_digest="a" * 64,
        output=fixtures._silent_output(),
        usage=ModelUsage(
            input_tokens=1,
            output_tokens=1,
            canonical_token_equivalents=2,
            latency_us=1,
        ),
    )


def test_small_value_guards_cover_partial_routing_and_non_mapping_digest() -> None:
    with pytest.raises(ValueError, match="routing binding is partial"):
        ReplayRoutingBinding(
            ordinal=1,
            target=DeliveryTarget.NEXT_MODEL_CALL,
        )

    with pytest.raises(ValueError, match="not a mapping"):
        engine_module._result_digest(object())


def test_event_validator_rejects_delivery_metadata_drift(replay_result: Any) -> None:
    reminder = replay_result.events[-1]
    assert reminder.delivery is not None
    forged = reminder.delivery.model_copy(
        update={"updated_at": reminder.delivery.updated_at + timedelta(microseconds=1)}
    )

    with pytest.raises(ValueError, match="delivery belongs"):
        fixtures._validate_event_result(reminder.model_copy(update={"delivery": forged}))


def test_run_validator_rejects_event_digest_and_duplicate_request_identity(
    replay_result: Any,
) -> None:
    with pytest.raises(ValueError, match="event digest"):
        fixtures._validate_run_result(replay_result.model_copy(update={"events_digest": "f" * 64}))

    first_digest = replay_result.events[1].model_request_digest
    duplicated = replay_result.events[2].model_copy(update={"model_request_digest": first_digest})
    with pytest.raises(ValueError, match="request identities"):
        fixtures._validate_run_result(
            replay_result.model_copy(
                update={
                    "events": (
                        replay_result.events[0],
                        replay_result.events[1],
                        duplicated,
                        replay_result.events[3],
                    )
                }
            )
        )


def test_run_validator_rejects_duplicate_decisions_and_signal_order(
    replay_result: Any,
) -> None:
    duplicate_decision = replay_result.events[1].decision.model_copy(
        update={"decision_id": replay_result.events[0].decision.decision_id}
    )
    duplicate_event = replay_result.events[1].model_copy(update={"decision": duplicate_decision})
    duplicate_events = (
        replay_result.events[0],
        duplicate_event,
        *replay_result.events[2:],
    )
    decisions = tuple(item.decision.model_dump(mode="json") for item in duplicate_events)
    with pytest.raises(ValueError, match="unique ordered trace"):
        fixtures._validate_run_result(
            replay_result.model_copy(
                update={
                    "events": duplicate_events,
                    "decisions_json": canonical_json(decisions).decode(),
                    "decisions_digest": canonical_digest(decisions),
                }
            )
        )

    event = replay_result.events[0].event
    signals = (
        _signal(event, suffix=0xD092, signal_type=SignalType.TOOL_ERROR),
        _signal(event, suffix=0xD091, signal_type=SignalType.TEST_FAILURE),
    )
    ordered = tuple(
        sorted(
            signals,
            key=lambda signal: (
                signal.signal_type.value,
                signal.detector_version,
                str(signal.signal_id),
            ),
            reverse=True,
        )
    )
    assert ordered != tuple(reversed(ordered))
    forged_event = replay_result.events[0].model_copy(update={"signals": ordered})
    with pytest.raises(ValueError, match="signal order"):
        fixtures._validate_run_result(
            replay_result.model_copy(update={"events": (forged_event, *replay_result.events[1:])})
        )


def test_run_validator_rejects_cycle_intervention_and_memory_identity(
    replay_result: Any,
) -> None:
    first_cycle_event = replay_result.events[1]
    assert first_cycle_event.cycle is not None
    bad_range = first_cycle_event.cycle.model_copy(
        update={"first_event_sequence": first_cycle_event.cycle.first_event_sequence + 1}
    )
    with pytest.raises(ValueError, match="cycle identity or range"):
        fixtures._validate_run_result(
            replay_result.model_copy(
                update={
                    "events": (
                        replay_result.events[0],
                        first_cycle_event.model_copy(update={"cycle": bad_range}),
                        *replay_result.events[2:],
                    )
                }
            )
        )

    assert first_cycle_event.cycle.intervention is not None
    bad_intervention = first_cycle_event.cycle.intervention.model_copy(
        update={"intervention_id": UUID("00000000-0000-4000-8000-00000000d090")}
    )
    with pytest.raises(ValueError, match="intervention identity"):
        fixtures._validate_run_result(
            replay_result.model_copy(
                update={
                    "events": (
                        replay_result.events[0],
                        first_cycle_event.model_copy(
                            update={
                                "cycle": first_cycle_event.cycle.model_copy(
                                    update={"intervention": bad_intervention}
                                )
                            }
                        ),
                        *replay_result.events[2:],
                    )
                }
            )
        )

    assert first_cycle_event.cycle.memory_id_assignments
    assignment = first_cycle_event.cycle.memory_id_assignments[0].model_copy(
        update={"memory_id": UUID("00000000-0000-4000-8000-00000000d089")}
    )
    bad_memory = first_cycle_event.cycle.model_copy(update={"memory_id_assignments": (assignment,)})
    with pytest.raises(ValueError, match="memory identity"):
        fixtures._validate_run_result(
            replay_result.model_copy(
                update={
                    "events": (
                        replay_result.events[0],
                        first_cycle_event.model_copy(update={"cycle": bad_memory}),
                        *replay_result.events[2:],
                    )
                }
            )
        )


def test_run_validator_rejects_routing_and_delivery_identity(replay_result: Any) -> None:
    bad_binding = replay_result.routing_bindings[-1].model_copy(
        update={"target_request_id_digest": "f" * 64}
    )
    bindings = (*replay_result.routing_bindings[:-1], bad_binding)
    with pytest.raises(ValueError, match="routing binding"):
        fixtures._validate_run_result(
            replay_result.model_copy(
                update={
                    "routing_bindings": bindings,
                    "routing_digest": canonical_digest(
                        tuple(item.model_dump(mode="json") for item in bindings)
                    ),
                }
            )
        )

    reminder = replay_result.events[-1]
    assert reminder.delivery is not None
    forged_delivery = reminder.delivery.model_copy(
        update={"delivery_id": UUID("00000000-0000-4000-8000-00000000d088")}
    )
    with pytest.raises(ValueError, match="delivery identity"):
        fixtures._validate_run_result(
            replay_result.model_copy(
                update={
                    "events": (
                        *replay_result.events[:-1],
                        reminder.model_copy(update={"delivery": forged_delivery}),
                    )
                }
            )
        )


def test_normalized_trace_and_signal_byte_guards(
    replay_result: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ReplayEngineInputError):
        engine_module.normalized_trace_digest(())

    monkeypatch.setattr(engine_module, "normalized_trace_event_draft_is_bounded", lambda _: False)
    with pytest.raises(ReplayEngineInputError):
        engine_module.normalized_trace_digest((fixtures._draft(1),))

    event = replay_result.events[0].event
    context = DetectionContext(run_id=event.run_id, events=(event,))
    signal = _signal(event, suffix=0xD087, signal_type=SignalType.TOOL_ERROR)
    monkeypatch.setattr(engine_module, "_MAX_SIGNAL_BYTES_PER_EVENT", 0)
    with pytest.raises(ReplayEngineInputError):
        engine_module._validated_signals((signal,), context)


def test_model_result_and_ledger_position_guards() -> None:
    limits = fixtures.engine_config().reservation
    with pytest.raises(ReplayEngineModelError):
        engine_module._bounded_model_result_output_bytes(
            object(),
            byte_limit=100,
            usage_limits=limits,
        )

    valid = _valid_model_result()
    without_output = valid.model_copy(update={"output": None})
    with pytest.raises(ReplayEngineModelError):
        engine_module._bounded_model_result_output_bytes(
            without_output,
            byte_limit=100,
            usage_limits=limits,
        )
    with pytest.raises(ReplayEngineModelError):
        engine_module._bounded_model_result_output_bytes(
            valid,
            byte_limit=0,
            usage_limits=limits,
        )

    with pytest.raises(ReplayEngineInvariantError):
        engine_module._advance_ledger_position(0, 2)


def test_selected_grounding_state_covers_missing_current_skip_and_reminder(
    replay_result: Any,
) -> None:
    with pytest.raises(ReplayEngineInvariantError):
        engine_module._selected_grounding_state(
            SimpleNamespace(
                events_by_sequence={},
                events_by_id={},
                memories={},
                reminder_interventions_by_sequence={},
            ),
            SimpleNamespace(claims=()),
            current_sequence=1,
            grounding=fixtures.grounding_pipeline(),
        )

    current = replay_result.events[-1].event
    cycle = replay_result.events[-1].cycle
    assert cycle is not None and cycle.intervention is not None
    configuration = fixtures.grounding_pipeline().configuration.model_copy(
        update={"duplicate_window_events": 2}
    )
    state = engine_module._selected_grounding_state(
        SimpleNamespace(
            events_by_sequence={current.sequence: current},
            events_by_id={current.event_id: current},
            memories={},
            reminder_interventions_by_sequence={current.sequence - 1: cycle.intervention},
        ),
        SimpleNamespace(claims=()),
        current_sequence=current.sequence,
        grounding=SimpleNamespace(configuration=configuration),
    )

    assert len(state.reminder_history) == 1


class _NativeAdapter:
    def __init__(self, event_ids: tuple[object, ...]) -> None:
        self.event_ids = event_ids

    def normalize(self, native_event: object):
        return native_event

    def resolve_event_id(self, _native_event: object, ordinal: int):
        return self.event_ids[ordinal - 1]

    def resolve_target_request_id(self, _native_event: object, _target: object) -> None:
        return None


class _AttestedAdapter(_NativeAdapter):
    def __init__(
        self,
        events: tuple[Any, ...],
        event_ids: tuple[object, ...],
        *,
        trace_digest: str,
    ) -> None:
        super().__init__(event_ids)
        self.events = events
        self.expected_event_ids = event_ids
        self.trace_digest = trace_digest


def _preflight_engine(adapter: Any) -> ReplayEngine:
    return fixtures._engine(
        MemoryRunRepository(synthetic_benchmark=True),
        adapter,
    )


@pytest.mark.asyncio
async def test_attested_preflight_rejects_event_id_and_record_digest_drift() -> None:
    source = fixtures.trace_adapter().events[0]
    digest = "a" * 64
    cases = (
        (
            _AttestedAdapter((source,), (fixtures.EVENT_1_ID,), trace_digest=digest),
            (source.model_copy(update={"ordinal": 2}),),
        ),
        (_AttestedAdapter((source,), (object(),), trace_digest=digest), (source,)),
        (
            _AttestedAdapter(
                (source.model_copy(update={"record_digest": "bad"}),),
                (fixtures.EVENT_1_ID,),
                trace_digest=digest,
            ),
            (source.model_copy(update={"record_digest": "bad"}),),
        ),
    )

    for adapter, native_events in cases:
        with pytest.raises(ReplayEngineInputError):
            await _preflight_engine(adapter)._preflight(native_events, trace_digest=digest)


@pytest.mark.asyncio
async def test_native_preflight_rejects_resolver_bounds_duplicates_and_parents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = fixtures._draft(1)
    adapter = _NativeAdapter((fixtures.EVENT_1_ID,))
    adapter.resolve_event_id = 1  # type: ignore[method-assign]
    with pytest.raises(ReplayEngineInputError):
        await _preflight_engine(adapter)._preflight((draft,), trace_digest="a" * 64)

    bounded = _NativeAdapter((fixtures.EVENT_1_ID,))
    monkeypatch.setattr(engine_module, "normalized_trace_event_draft_is_bounded", lambda _: False)
    with pytest.raises(ReplayEngineInputError):
        await _preflight_engine(bounded)._preflight((draft,), trace_digest="a" * 64)
    monkeypatch.undo()

    invalid_id = _NativeAdapter((object(),))
    with pytest.raises(ReplayEngineInputError):
        await _preflight_engine(invalid_id)._preflight((draft,), trace_digest="a" * 64)

    duplicate_ids = _NativeAdapter((fixtures.EVENT_1_ID, fixtures.EVENT_1_ID))
    with pytest.raises(ReplayEngineInputError):
        await _preflight_engine(duplicate_ids)._preflight(
            (draft, fixtures._draft(2)),
            trace_digest="a" * 64,
        )

    bad_parent = fixtures._draft(2, parent_ids=(fixtures.EVENT_4_ID,))
    parents = _NativeAdapter((fixtures.EVENT_1_ID, fixtures.EVENT_2_ID))
    with pytest.raises(ReplayEngineInputError):
        await _preflight_engine(parents)._preflight(
            (draft, bad_parent),
            trace_digest="a" * 64,
        )


def test_owned_prefix_rejects_missing_event(replay_result: Any) -> None:
    with pytest.raises(ReplayEngineInvariantError):
        ReplayEngine._assert_owned_prefix(
            SimpleNamespace(ingestion_cursor=1, events_by_sequence={}),
            (replay_result.events[0].event,),
        )


class _PartialFixtureModel:
    replay_id = "partial-fixture/v1"
    fixture_digest = "b" * 64
    total_responses = 2
    remaining_responses = 1

    async def generate(self, _request: object) -> None:
        return None


@pytest.mark.asyncio
async def test_run_rejects_partially_consumed_initial_fixture() -> None:
    trace = fixtures.one_event_trace()
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=trace.event_id_factory(),
    )
    engine = fixtures._engine(repository, trace, model=_PartialFixtureModel())

    with pytest.raises(ReplayEngineModelError):
        await engine.run(trace.events, trace_digest=trace.manifest.trace_digest)


class _BeginCycleDriftRepository:
    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    async def begin_cycle(self, command: Any):
        receipt = await self.inner.begin_cycle(command)
        cycle = receipt.cycle.model_copy(update={"cycle_id": "f" * 64})
        return receipt.model_copy(update={"cycle": cycle})


@pytest.mark.asyncio
async def test_run_rejects_repository_cycle_identity_drift() -> None:
    trace = fixtures.one_event_trace()
    inner = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=trace.event_id_factory(),
    )

    with pytest.raises(ReplayEngineInvariantError):
        await fixtures._engine(_BeginCycleDriftRepository(inner), trace).run(
            trace.events,
            trace_digest=trace.manifest.trace_digest,
        )


class _EmergingFixtureModel(fixtures.CompletedModel):
    async def generate(self, request: Any) -> ModelResult:
        result = await super().generate(request)
        self.replay_id = "emerging-fixture/v1"
        self.fixture_digest = "c" * 64
        self.total_responses = 1
        self.remaining_responses = 0
        return result


@pytest.mark.asyncio
async def test_run_rejects_fixture_presence_drift() -> None:
    trace = fixtures.one_event_trace()
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=trace.event_id_factory(),
    )
    model = _EmergingFixtureModel(fixtures._silent_output())

    with pytest.raises(ReplayEngineModelError):
        await fixtures._engine(repository, trace, model=model).run(
            trace.events,
            trace_digest=trace.manifest.trace_digest,
        )


class _LedgerHeadDriftRepository:
    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    async def ledger_head(self, run_id: UUID):
        head = await self.inner.ledger_head(run_id)
        return head.model_copy(update={"entry_count": head.entry_count + 1})


@pytest.mark.asyncio
async def test_run_rejects_ledger_head_drift() -> None:
    trace = fixtures.one_event_trace()
    inner = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=trace.event_id_factory(),
    )

    with pytest.raises(ReplayEngineInvariantError):
        await fixtures._engine(_LedgerHeadDriftRepository(inner), trace).run(
            trace.events,
            trace_digest=trace.manifest.trace_digest,
        )
