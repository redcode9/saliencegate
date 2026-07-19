from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from pydantic import ValidationError

from saliencegate.domain import (
    EventPhase,
    EventType,
    NormalizedTraceEventDraft,
    PayloadDigest,
    PayloadDigestAlgorithm,
    TextSpan,
    TraceEvent,
    TrustLabel,
    canonical_json,
)
from saliencegate.ports.repository import LedgerEntry, RunRepository
from saliencegate.ports.trajectory import (
    ActionStepBinding,
    AttestedTrajectoryEvent,
    AttestedTrajectoryPrefix,
    EventTextSelector,
    LogicalMessageBinding,
    LogicalMessageRole,
    TrajectoryError,
    TrajectoryErrorCode,
    TrajectoryPrefixRequest,
    _resolve_attested_payload_value,
    _validated_attested_trajectory_prefix_structure,
    bind_persisted_trajectory_event,
    resolve_trajectory_prefix,
)
from saliencegate.repository import MemoryRunRepository
from saliencegate.runtime.scheduling import (
    FixedStepDecision,
    FixedStepReason,
    FixedStepSchedule,
    project_fixed_step_schedule,
    validated_fixed_step_schedule_for_prefix,
)

RUN_ID = UUID("00000000-0000-4000-8000-000000002501")
OTHER_RUN_ID = UUID("00000000-0000-4000-8000-000000002502")
NOW = datetime(2026, 7, 12, 8, 0, tzinfo=UTC)


def _repository() -> MemoryRunRepository:
    identifiers = itertools.count(0x2600)
    return MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=lambda: UUID(f"00000000-0000-4000-8000-{next(identifiers):012x}"),
    )


def _draft(
    sequence: int,
    event_type: EventType,
    *,
    step: int | None,
) -> NormalizedTraceEventDraft:
    payload: dict[str, object] = {"message": f"message-{sequence}"}
    if sequence == 1:
        payload["task"] = "Keep the deployment reversible."
    if step is not None:
        payload["step"] = step
    return NormalizedTraceEventDraft(
        run_id=RUN_ID,
        source_event_id=f"trajectory-{sequence}",
        timestamp=NOW + timedelta(seconds=sequence),
        event_type=event_type,
        phase=(
            EventPhase.INITIALIZATION
            if event_type is EventType.RUN_START
            else EventPhase.ACTION_EXECUTION
        ),
        payload=payload,
        source_adapter="trajectory-fixture/v1",
        trust_label=TrustLabel.SYNTHETIC_FIXTURE,
    )


async def _append_trace(repository: RunRepository) -> tuple[LedgerEntry, ...]:
    shapes = (
        (EventType.RUN_START, 1),
        (EventType.MODEL_OUTPUT, 1),
        (EventType.ACTION_PROPOSAL, 1),
        (EventType.TOOL_START, 1),
        (EventType.TOOL_COMPLETION, 1),
        (EventType.OBSERVATION, 1),
        (EventType.ACTION_PROPOSAL, 2),
        (EventType.CONTROLLER_ERROR, None),
    )
    for sequence, (event_type, step) in enumerate(shapes, start=1):
        await repository.append(_draft(sequence, event_type, step=step))
    return tuple(
        entry for entry in await repository.ledger(RUN_ID) if type(entry.record) is TraceEvent
    )


def _bindings(entries: tuple[LedgerEntry, ...]):
    result = []
    for entry in entries:
        event = entry.record
        assert type(event) is TraceEvent
        result.append(
            bind_persisted_trajectory_event(
                entry,
                task_description=(
                    EventTextSelector(field_path="/payload/task") if event.sequence == 1 else None
                ),
                logical_messages=(
                    LogicalMessageBinding(
                        role=LogicalMessageRole.USER,
                        selector=EventTextSelector(field_path="/payload/message"),
                    ),
                ),
                action_step=(
                    ActionStepBinding(field_path="/payload/step")
                    if "step" in event.payload
                    else None
                ),
            )
        )
    return tuple(result)


@pytest.mark.asyncio
async def test_bootstrap_consumes_first_step_and_technical_events_stay_silent() -> None:
    repository = _repository()
    entries = await _append_trace(repository)
    bindings = _bindings(entries)
    prefix = await resolve_trajectory_prefix(
        repository,
        TrajectoryPrefixRequest(
            schema_version="trajectory-prefix-request/v1",
            run_id=RUN_ID,
            boundary_event_sequence=8,
            bindings=bindings,
        ),
    )

    schedule = await project_fixed_step_schedule(repository, prefix)

    assert tuple(decision.invoke for decision in schedule.decisions) == (
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
    )
    assert tuple(decision.invocation_ordinal for decision in schedule.decisions) == (
        1,
        None,
        None,
        None,
        None,
        None,
        2,
        None,
    )
    assert tuple(decision.reason for decision in schedule.decisions) == (
        FixedStepReason.BOOTSTRAP,
        FixedStepReason.CURRENT_ACTION_STEP,
        FixedStepReason.CURRENT_ACTION_STEP,
        FixedStepReason.CURRENT_ACTION_STEP,
        FixedStepReason.CURRENT_ACTION_STEP,
        FixedStepReason.CURRENT_ACTION_STEP,
        FixedStepReason.ACTION_STEP,
        FixedStepReason.NO_ACTION_STEP,
    )
    assert schedule.invocation_count == 2
    assert await validated_fixed_step_schedule_for_prefix(repository, prefix, schedule) == schedule
    assert canonical_json(await project_fixed_step_schedule(repository, prefix)) == canonical_json(
        schedule
    )


@pytest.mark.asyncio
async def test_schedule_is_prefix_stable() -> None:
    repository = _repository()
    entries = await _append_trace(repository)
    bindings = _bindings(entries)
    short_prefix = await resolve_trajectory_prefix(
        repository,
        TrajectoryPrefixRequest(
            schema_version="trajectory-prefix-request/v1",
            run_id=RUN_ID,
            boundary_event_sequence=6,
            bindings=bindings[:6],
        ),
    )
    full_prefix = await resolve_trajectory_prefix(
        repository,
        TrajectoryPrefixRequest(
            schema_version="trajectory-prefix-request/v1",
            run_id=RUN_ID,
            boundary_event_sequence=8,
            bindings=bindings,
        ),
    )

    short_schedule = await project_fixed_step_schedule(repository, short_prefix)
    full_schedule = await project_fixed_step_schedule(repository, full_prefix)
    assert short_schedule.decisions == full_schedule.decisions[:6]
    with pytest.raises(TrajectoryError) as error:
        await validated_fixed_step_schedule_for_prefix(repository, short_prefix, full_schedule)
    assert error.value.code is TrajectoryErrorCode.UNATTESTED_REFERENCE
    with pytest.raises(TrajectoryError) as error:
        await validated_fixed_step_schedule_for_prefix(
            repository,
            short_prefix.model_copy(update={"prefix_digest": "0" * 64}),
            short_schedule,
        )
    assert error.value.code is TrajectoryErrorCode.INVALID_INPUT


@pytest.mark.asyncio
async def test_retrograde_step_rejects_the_entire_projection() -> None:
    repository = _repository()
    entries = await _append_trace(repository)
    bindings = list(_bindings(entries))
    seventh = entries[6]
    bindings[6] = bind_persisted_trajectory_event(
        seventh,
        logical_messages=(
            LogicalMessageBinding(
                role=LogicalMessageRole.USER,
                selector=EventTextSelector(field_path="/payload/message"),
            ),
        ),
        action_step=ActionStepBinding(field_path="/payload/retrograde"),
    )
    # The binding remains authoritative; only the persisted step value is adversarial.
    values = seventh.record.model_dump(mode="python")
    values["payload"] = {**dict(seventh.record.payload), "retrograde": 0}
    forged_event = TraceEvent.model_validate(values)
    forged_entry = seventh.model_copy(update={"record": forged_event})

    forged_binding = bind_persisted_trajectory_event(
        forged_entry,
        action_step=ActionStepBinding(field_path="/payload/retrograde"),
    )
    forged_bindings = (*bindings[:6], forged_binding, *bindings[7:])
    persisted_prefix = await resolve_trajectory_prefix(
        repository,
        TrajectoryPrefixRequest(
            schema_version="trajectory-prefix-request/v1",
            run_id=RUN_ID,
            boundary_event_sequence=8,
            bindings=forged_bindings,
        ),
    )
    with pytest.raises(TrajectoryError) as error:
        await project_fixed_step_schedule(repository, persisted_prefix)
    assert error.value.code is TrajectoryErrorCode.INVALID_POINTER

    # A real retrograde value is introduced by an additional persisted event.
    bindings = list(_bindings(entries))
    await repository.append(_draft(9, EventType.ACTION_PROPOSAL, step=1))
    ninth_entry = next(
        entry
        for entry in await repository.ledger(RUN_ID)
        if type(entry.record) is TraceEvent and entry.record.sequence == 9
    )
    ninth_binding = bind_persisted_trajectory_event(
        ninth_entry,
        action_step=ActionStepBinding(field_path="/payload/step"),
    )
    prefix = await resolve_trajectory_prefix(
        repository,
        TrajectoryPrefixRequest(
            schema_version="trajectory-prefix-request/v1",
            run_id=RUN_ID,
            boundary_event_sequence=9,
            bindings=(*bindings, ninth_binding),
        ),
    )
    with pytest.raises(TrajectoryError) as error:
        await project_fixed_step_schedule(repository, prefix)
    assert error.value.code is TrajectoryErrorCode.RETROGRADE_BINDING


@pytest.mark.asyncio
async def test_prefix_rejects_missing_future_cross_run_and_unattested_bindings() -> None:
    repository = _repository()
    entries = await _append_trace(repository)
    bindings = _bindings(entries)

    cases = (
        (bindings[:-1], 8, TrajectoryErrorCode.MISSING_REFERENCE),
        (bindings, 7, TrajectoryErrorCode.FUTURE_REFERENCE),
    )
    for candidate, boundary, expected in cases:
        with pytest.raises(TrajectoryError) as error:
            await resolve_trajectory_prefix(
                repository,
                TrajectoryPrefixRequest.model_construct(
                    schema_version="trajectory-prefix-request/v1",
                    run_id=RUN_ID,
                    boundary_event_sequence=boundary,
                    bindings=candidate,
                    request_digest="0" * 64,
                ),
            )
        assert error.value.code is expected

    cross_run = bindings[0].model_copy(update={"run_id": OTHER_RUN_ID})
    with pytest.raises(TrajectoryError) as error:
        await resolve_trajectory_prefix(
            repository,
            TrajectoryPrefixRequest.model_construct(
                schema_version="trajectory-prefix-request/v1",
                run_id=RUN_ID,
                boundary_event_sequence=8,
                bindings=(cross_run, *bindings[1:]),
                request_digest="0" * 64,
            ),
        )
    assert error.value.code is TrajectoryErrorCode.CROSS_RUN_REFERENCE

    unattested = bindings[0].model_copy(update={"record_tag": bindings[1].record_tag})
    with pytest.raises(TrajectoryError) as error:
        await resolve_trajectory_prefix(
            repository,
            TrajectoryPrefixRequest.model_construct(
                schema_version="trajectory-prefix-request/v1",
                run_id=RUN_ID,
                boundary_event_sequence=8,
                bindings=(unattested, *bindings[1:]),
                request_digest="0" * 64,
            ),
        )
    assert error.value.code is TrajectoryErrorCode.UNATTESTED_REFERENCE


@pytest.mark.asyncio
async def test_binding_contract_is_strict_bounded_and_content_addressed() -> None:
    repository = _repository()
    entries = await _append_trace(repository)
    binding = _bindings(entries)[0]

    for pointer in (
        "/message",
        "/payload/" + "/".join("x" for _ in range(32)),
        "/payload/" + "x" * 1_025,
    ):
        with pytest.raises(ValidationError):
            EventTextSelector(field_path=pointer)

    with pytest.raises(ValidationError):
        EventTextSelector(
            field_path="/payload/message",
            span=TextSpan.model_construct(start_byte=-1, end_byte=0),
        )

    duplicate = LogicalMessageBinding(
        role=LogicalMessageRole.USER,
        selector=EventTextSelector(field_path="/payload/message"),
    )
    with pytest.raises(TrajectoryError) as error:
        bind_persisted_trajectory_event(
            entries[0],
            task_description=EventTextSelector(field_path="/payload/task"),
            logical_messages=(duplicate, duplicate),
        )
    assert error.value.code is TrajectoryErrorCode.INVALID_INPUT

    with pytest.raises(TrajectoryError):
        bind_persisted_trajectory_event(
            entries[0],
            task_description=EventTextSelector(field_path="/payload/task"),
            action_step=ActionStepBinding(field_path="/payload/task"),
        )
    with pytest.raises(TrajectoryError):
        bind_persisted_trajectory_event(object())  # type: ignore[arg-type]

    values = binding.model_dump(mode="python")
    values["event_sequence"] = binding.ledger_position + 1
    with pytest.raises(ValidationError):
        type(binding).model_validate(values)

    values = binding.model_dump(mode="python")
    values["payload_digest"] = PayloadDigest.model_construct(
        algorithm="not-an-algorithm",
        value="not-a-digest",
    )
    with pytest.raises(ValidationError):
        type(binding).model_validate(values)

    values = binding.model_dump(mode="python")
    values["payload_digest"] = {
        "algorithm": PayloadDigestAlgorithm.HMAC_SHA256,
        "value": binding.payload_digest.value,
    }
    with pytest.raises(ValidationError):
        type(binding).model_validate(values)

    values = binding.model_dump(mode="python")
    values["binding_digest"] = "0" * 64
    with pytest.raises(ValidationError):
        type(binding).model_validate(values)

    with pytest.raises(ValidationError):
        TrajectoryPrefixRequest(
            schema_version="trajectory-prefix-request/v1",
            run_id=RUN_ID,
            boundary_event_sequence=(1 << 63) - 1,
            bindings=(binding,),
        )


@pytest.mark.asyncio
async def test_prefix_preflight_has_typed_value_free_failure_modes() -> None:
    repository = _repository()
    entries = await _append_trace(repository)
    bindings = _bindings(entries)

    malformed_requests = (
        (
            TrajectoryPrefixRequest.model_construct(
                schema_version="trajectory-prefix-request/v1",
                run_id=RUN_ID,
                boundary_event_sequence=10_001,
                bindings=bindings,
                request_digest="0" * 64,
            ),
            TrajectoryErrorCode.LIMIT_EXCEEDED,
        ),
        (
            TrajectoryPrefixRequest.model_construct(
                schema_version="trajectory-prefix-request/v1",
                run_id="not-a-uuid",
                boundary_event_sequence=8,
                bindings=bindings,
                request_digest="0" * 64,
            ),
            TrajectoryErrorCode.INVALID_INPUT,
        ),
        (
            TrajectoryPrefixRequest.model_construct(
                schema_version="trajectory-prefix-request/v1",
                run_id=RUN_ID,
                boundary_event_sequence=8,
                bindings=list(bindings),
                request_digest="0" * 64,
            ),
            TrajectoryErrorCode.LIMIT_EXCEEDED,
        ),
        (
            TrajectoryPrefixRequest.model_construct(
                schema_version="trajectory-prefix-request/v1",
                run_id=RUN_ID,
                boundary_event_sequence=8,
                bindings=(object(), *bindings[1:]),
                request_digest="0" * 64,
            ),
            TrajectoryErrorCode.INVALID_INPUT,
        ),
        (
            TrajectoryPrefixRequest.model_construct(
                schema_version="trajectory-prefix-request/v1",
                run_id=RUN_ID,
                boundary_event_sequence=8,
                bindings=(
                    bindings[0],
                    bindings[1].model_copy(update={"event_id": bindings[0].event_id}),
                    *bindings[2:],
                ),
                request_digest="0" * 64,
            ),
            TrajectoryErrorCode.DUPLICATE_BINDING,
        ),
        (
            TrajectoryPrefixRequest.model_construct(
                schema_version="trajectory-prefix-request/v1",
                run_id=RUN_ID,
                boundary_event_sequence=8,
                bindings=(
                    bindings[0].model_copy(update={"event_sequence": 2}),
                    bindings[1].model_copy(update={"event_sequence": 1}),
                    *bindings[2:],
                ),
                request_digest="0" * 64,
            ),
            TrajectoryErrorCode.RETROGRADE_BINDING,
        ),
        (
            TrajectoryPrefixRequest.model_construct(
                schema_version="trajectory-prefix-request/v1",
                run_id=RUN_ID,
                boundary_event_sequence=8,
                bindings=(
                    bindings[0],
                    bindings[1].model_copy(update={"ledger_position": bindings[0].ledger_position}),
                    *bindings[2:],
                ),
                request_digest="0" * 64,
            ),
            TrajectoryErrorCode.DUPLICATE_BINDING,
        ),
        (
            TrajectoryPrefixRequest.model_construct(
                schema_version="trajectory-prefix-request/v1",
                run_id=RUN_ID,
                boundary_event_sequence=8,
                bindings=(
                    bindings[0].model_copy(update={"ledger_position": bindings[1].ledger_position}),
                    bindings[1].model_copy(update={"ledger_position": bindings[0].ledger_position}),
                    *bindings[2:],
                ),
                request_digest="0" * 64,
            ),
            TrajectoryErrorCode.RETROGRADE_BINDING,
        ),
    )
    for request, expected in malformed_requests:
        with pytest.raises(TrajectoryError) as error:
            await resolve_trajectory_prefix(repository, request)
        assert error.value.code is expected
        assert str(error.value) == f"trajectory projection failed: {expected.value}"

    with pytest.raises(TrajectoryError) as error:
        await resolve_trajectory_prefix(repository, object())  # type: ignore[arg-type]
    assert error.value.code is TrajectoryErrorCode.INVALID_INPUT

    oversized = TrajectoryPrefixRequest.model_construct(
        schema_version="trajectory-prefix-request/v1",
        run_id=RUN_ID,
        boundary_event_sequence=10_000,
        bindings=(bindings[0],) * 10_001,
        request_digest="0" * 64,
    )
    with pytest.raises(TrajectoryError) as error:
        await resolve_trajectory_prefix(repository, oversized)
    assert error.value.code is TrajectoryErrorCode.LIMIT_EXCEEDED


@pytest.mark.asyncio
async def test_prefix_requires_authoritative_ledger_task_and_request_digest() -> None:
    repository = _repository()
    entries = await _append_trace(repository)
    bindings = _bindings(entries)

    class FailingRepository:
        async def ledger(self, _run_id: UUID) -> tuple[LedgerEntry, ...]:
            raise RuntimeError("sensitive backend detail")

    with pytest.raises(TrajectoryError) as error:
        await resolve_trajectory_prefix(
            FailingRepository(),  # type: ignore[arg-type]
            TrajectoryPrefixRequest(
                schema_version="trajectory-prefix-request/v1",
                run_id=RUN_ID,
                boundary_event_sequence=8,
                bindings=bindings,
            ),
        )
    assert error.value.code is TrajectoryErrorCode.REPOSITORY_UNAVAILABLE
    assert "sensitive" not in str(error.value)

    class MalformedRepository:
        async def ledger(self, _run_id: UUID) -> object:
            return None

    with pytest.raises(TrajectoryError) as error:
        await resolve_trajectory_prefix(
            MalformedRepository(),  # type: ignore[arg-type]
            TrajectoryPrefixRequest(
                schema_version="trajectory-prefix-request/v1",
                run_id=RUN_ID,
                boundary_event_sequence=8,
                bindings=bindings,
            ),
        )
    assert error.value.code is TrajectoryErrorCode.REPOSITORY_UNAVAILABLE

    class ShortRepository:
        async def ledger(self, _run_id: UUID) -> tuple[LedgerEntry, ...]:
            return entries[:-1]

    with pytest.raises(TrajectoryError) as error:
        await resolve_trajectory_prefix(
            ShortRepository(),  # type: ignore[arg-type]
            TrajectoryPrefixRequest(
                schema_version="trajectory-prefix-request/v1",
                run_id=RUN_ID,
                boundary_event_sequence=8,
                bindings=bindings,
            ),
        )
    assert error.value.code is TrajectoryErrorCode.MISSING_REFERENCE

    no_task = tuple(
        bind_persisted_trajectory_event(
            entry,
            action_step=(
                ActionStepBinding(field_path="/payload/step")
                if "step" in entry.record.payload
                else None
            ),
        )
        for entry in entries
        if type(entry.record) is TraceEvent
    )
    with pytest.raises(TrajectoryError) as error:
        await resolve_trajectory_prefix(
            repository,
            TrajectoryPrefixRequest.model_construct(
                schema_version="trajectory-prefix-request/v1",
                run_id=RUN_ID,
                boundary_event_sequence=8,
                bindings=no_task,
                request_digest="0" * 64,
            ),
        )
    assert error.value.code is TrajectoryErrorCode.MISSING_REFERENCE

    second_with_task = bind_persisted_trajectory_event(
        entries[1],
        task_description=EventTextSelector(field_path="/payload/message"),
        action_step=ActionStepBinding(field_path="/payload/step"),
    )
    with pytest.raises(TrajectoryError) as error:
        await resolve_trajectory_prefix(
            repository,
            TrajectoryPrefixRequest.model_construct(
                schema_version="trajectory-prefix-request/v1",
                run_id=RUN_ID,
                boundary_event_sequence=8,
                bindings=(bindings[0], second_with_task, *bindings[2:]),
                request_digest="0" * 64,
            ),
        )
    assert error.value.code is TrajectoryErrorCode.DUPLICATE_BINDING

    with pytest.raises(TrajectoryError) as error:
        await resolve_trajectory_prefix(
            repository,
            TrajectoryPrefixRequest.model_construct(
                schema_version="trajectory-prefix-request/v1",
                run_id=RUN_ID,
                boundary_event_sequence=8,
                bindings=bindings,
                request_digest="0" * 64,
            ),
        )
    assert error.value.code is TrajectoryErrorCode.INVALID_INPUT


@pytest.mark.asyncio
async def test_prefix_and_pointer_outputs_reject_forged_models_and_scalar_traversal() -> None:
    repository = _repository()
    entries = await _append_trace(repository)
    prefix = await resolve_trajectory_prefix(
        repository,
        TrajectoryPrefixRequest(
            schema_version="trajectory-prefix-request/v1",
            run_id=RUN_ID,
            boundary_event_sequence=8,
            bindings=_bindings(entries),
        ),
    )

    with pytest.raises(TrajectoryError):
        _validated_attested_trajectory_prefix_structure(object())
    with pytest.raises(TrajectoryError):
        _validated_attested_trajectory_prefix_structure(
            prefix.model_copy(update={"prefix_digest": "0" * 64})
        )
    with pytest.raises(TrajectoryError) as error:
        _resolve_attested_payload_value(object(), "/payload/task")  # type: ignore[arg-type]
    assert error.value.code is TrajectoryErrorCode.INVALID_POINTER
    for pointer in ("/payload/missing", "/payload/task/0", "/payload/\ud800"):
        with pytest.raises(TrajectoryError) as error:
            _resolve_attested_payload_value(prefix.items[0], pointer)
        assert error.value.code is TrajectoryErrorCode.INVALID_POINTER

    mismatched = AttestedTrajectoryEvent.model_construct(
        event=prefix.items[0].event,
        binding=prefix.items[1].binding,
    )
    with pytest.raises(ValidationError):
        AttestedTrajectoryEvent.model_validate_json(mismatched.model_dump_json(warnings=False))

    malformed_event = TraceEvent.model_construct(
        **(prefix.items[0].event.model_dump(mode="python") | {"payload": object()})
    )
    with pytest.raises(ValidationError):
        AttestedTrajectoryEvent(event=malformed_event, binding=prefix.items[0].binding)


@pytest.mark.asyncio
async def test_public_schedule_reauthenticates_payload_before_reading_step() -> None:
    repository = _repository()
    entries = await _append_trace(repository)
    prefix = await resolve_trajectory_prefix(
        repository,
        TrajectoryPrefixRequest(
            schema_version="trajectory-prefix-request/v1",
            run_id=RUN_ID,
            boundary_event_sequence=8,
            bindings=_bindings(entries),
        ),
    )
    first = prefix.items[0]
    event_values = first.event.model_dump(mode="python")
    event_values["payload"] = {"task": "raw caller text", "message": "raw", "step": 9_999}
    forged_event = TraceEvent.model_validate(event_values)
    forged_item = AttestedTrajectoryEvent(event=forged_event, binding=first.binding)
    forged_prefix = AttestedTrajectoryPrefix(
        schema_version="attested-trajectory-prefix/v1",
        run_id=prefix.run_id,
        boundary_event_sequence=prefix.boundary_event_sequence,
        request_digest=prefix.request_digest,
        items=(forged_item, *prefix.items[1:]),
    )

    with pytest.raises(TrajectoryError) as error:
        await project_fixed_step_schedule(repository, forged_prefix)
    assert error.value.code is TrajectoryErrorCode.UNATTESTED_REFERENCE

    no_task_binding = bind_persisted_trajectory_event(
        entries[0],
        logical_messages=first.binding.logical_messages,
        action_step=first.binding.action_step,
    )
    no_task_item = AttestedTrajectoryEvent(event=first.event, binding=no_task_binding)
    no_task_prefix = AttestedTrajectoryPrefix(
        schema_version="attested-trajectory-prefix/v1",
        run_id=prefix.run_id,
        boundary_event_sequence=prefix.boundary_event_sequence,
        request_digest=prefix.request_digest,
        items=(no_task_item, *prefix.items[1:]),
    )
    with pytest.raises(TrajectoryError) as error:
        await project_fixed_step_schedule(repository, no_task_prefix)
    assert error.value.code is TrajectoryErrorCode.UNATTESTED_REFERENCE


@pytest.mark.asyncio
async def test_prefix_model_validators_reject_every_noncanonical_relationship() -> None:
    repository = _repository()
    entries = await _append_trace(repository)
    bindings = _bindings(entries)
    request = TrajectoryPrefixRequest(
        schema_version="trajectory-prefix-request/v1",
        run_id=RUN_ID,
        boundary_event_sequence=8,
        bindings=bindings,
    )
    invalid_request_bindings = (
        (bindings[0].model_copy(update={"run_id": OTHER_RUN_ID}), *bindings[1:]),
        bindings[:-1],
        (
            bindings[0],
            bindings[1].model_copy(update={"ledger_position": bindings[0].ledger_position}),
            *bindings[2:],
        ),
        (
            bindings[0],
            bindings[1].model_copy(update={"event_id": bindings[0].event_id}),
            *bindings[2:],
        ),
        tuple(binding.model_copy(update={"task_description": None}) for binding in bindings),
    )
    for candidate_bindings in invalid_request_bindings:
        candidate = request.model_copy(update={"bindings": candidate_bindings})
        with pytest.raises(ValueError):
            candidate.bindings_cover_one_exact_prefix()

    prefix = await resolve_trajectory_prefix(repository, request)
    first, second, *remaining = prefix.items
    cross_run_event = first.event.model_copy(update={"run_id": OTHER_RUN_ID})
    cross_run_item = AttestedTrajectoryEvent.model_construct(
        event=cross_run_event,
        binding=first.binding,
    )
    gap_event = second.event.model_copy(update={"sequence": 3})
    gap_item = AttestedTrajectoryEvent.model_construct(event=gap_event, binding=second.binding)
    non_start_event = first.event.model_copy(update={"event_type": EventType.OBSERVATION})
    non_start_item = AttestedTrajectoryEvent.model_construct(
        event=non_start_event,
        binding=first.binding,
    )
    invalid_prefixes = (
        prefix.model_copy(update={"boundary_event_sequence": 9}),
        prefix.model_copy(update={"items": (cross_run_item, second, *remaining)}),
        prefix.model_copy(update={"items": (first, gap_item, *remaining)}),
        prefix.model_copy(update={"items": (non_start_item, second, *remaining)}),
    )
    for candidate in invalid_prefixes:
        with pytest.raises(ValueError):
            candidate.items_form_the_requested_run_prefix()


@pytest.mark.asyncio
async def test_resolver_rechecks_event_order_run_and_binding_serialization() -> None:
    repository = _repository()
    entries = await _append_trace(repository)
    bindings = _bindings(entries)
    request = TrajectoryPrefixRequest(
        schema_version="trajectory-prefix-request/v1",
        run_id=RUN_ID,
        boundary_event_sequence=8,
        bindings=bindings,
    )

    class ForgedLedger:
        def __init__(self, forged: LedgerEntry) -> None:
            self._forged = forged

        async def ledger(self, _run_id: UUID) -> tuple[LedgerEntry, ...]:
            return (self._forged, *entries[1:])

    wrong_sequence = entries[0].model_copy(
        update={"record": entries[0].record.model_copy(update={"sequence": 2})}
    )
    with pytest.raises(TrajectoryError) as error:
        await resolve_trajectory_prefix(ForgedLedger(wrong_sequence), request)  # type: ignore[arg-type]
    assert error.value.code is TrajectoryErrorCode.MISSING_REFERENCE

    cross_run = entries[0].model_copy(
        update={"record": entries[0].record.model_copy(update={"run_id": OTHER_RUN_ID})}
    )
    with pytest.raises(TrajectoryError) as error:
        await resolve_trajectory_prefix(ForgedLedger(cross_run), request)  # type: ignore[arg-type]
    assert error.value.code is TrajectoryErrorCode.REPOSITORY_UNAVAILABLE

    invalid_binding = bindings[0].model_copy(update={"binding_digest": "0" * 64})
    with pytest.raises(TrajectoryError) as error:
        await resolve_trajectory_prefix(
            repository,
            TrajectoryPrefixRequest.model_construct(
                schema_version="trajectory-prefix-request/v1",
                run_id=RUN_ID,
                boundary_event_sequence=8,
                bindings=(invalid_binding, *bindings[1:]),
                request_digest="0" * 64,
            ),
        )
    assert error.value.code is TrajectoryErrorCode.UNATTESTED_REFERENCE

    no_start_repository = _repository()
    await no_start_repository.append(_draft(1, EventType.OBSERVATION, step=None))
    no_start_entry = next(
        item for item in await no_start_repository.ledger(RUN_ID) if type(item.record) is TraceEvent
    )
    no_start_binding = bind_persisted_trajectory_event(
        no_start_entry,
        task_description=EventTextSelector(field_path="/payload/task"),
    )
    with pytest.raises(TrajectoryError) as error:
        await resolve_trajectory_prefix(
            no_start_repository,
            TrajectoryPrefixRequest(
                schema_version="trajectory-prefix-request/v1",
                run_id=RUN_ID,
                boundary_event_sequence=1,
                bindings=(no_start_binding,),
            ),
        )
    assert error.value.code is TrajectoryErrorCode.INVALID_INPUT


@pytest.mark.asyncio
async def test_bootstrap_without_step_then_gapped_step_invokes_once_each() -> None:
    repository = _repository()
    await repository.append(_draft(1, EventType.RUN_START, step=None))
    await repository.append(_draft(2, EventType.OBSERVATION, step=None))
    await repository.append(_draft(3, EventType.ACTION_PROPOSAL, step=4))
    entries = tuple(
        entry for entry in await repository.ledger(RUN_ID) if type(entry.record) is TraceEvent
    )
    bindings = tuple(
        bind_persisted_trajectory_event(
            entry,
            task_description=(
                EventTextSelector(field_path="/payload/task")
                if entry.record.sequence == 1
                else None
            ),
            action_step=(
                ActionStepBinding(field_path="/payload/step")
                if "step" in entry.record.payload
                else None
            ),
        )
        for entry in entries
        if type(entry.record) is TraceEvent
    )
    prefix = await resolve_trajectory_prefix(
        repository,
        TrajectoryPrefixRequest(
            schema_version="trajectory-prefix-request/v1",
            run_id=RUN_ID,
            boundary_event_sequence=3,
            bindings=bindings,
        ),
    )

    schedule = await project_fixed_step_schedule(repository, prefix)

    assert tuple(decision.reason for decision in schedule.decisions) == (
        FixedStepReason.BOOTSTRAP,
        FixedStepReason.NO_ACTION_STEP,
        FixedStepReason.ACTION_STEP,
    )
    assert schedule.decisions[-1].action_step_ordinal == 4


@pytest.mark.asyncio
async def test_invalid_step_values_and_tampered_schedule_fail_closed() -> None:
    for step in (0, True, "1"):
        repository = _repository()
        await repository.append(
            NormalizedTraceEventDraft(
                run_id=RUN_ID,
                source_event_id="invalid-step",
                timestamp=NOW,
                event_type=EventType.RUN_START,
                phase=EventPhase.INITIALIZATION,
                payload={"task": "task", "step": step},
                source_adapter="trajectory-fixture/v1",
                trust_label=TrustLabel.SYNTHETIC_FIXTURE,
            )
        )
        entry = next(
            item for item in await repository.ledger(RUN_ID) if type(item.record) is TraceEvent
        )
        binding = bind_persisted_trajectory_event(
            entry,
            task_description=EventTextSelector(field_path="/payload/task"),
            action_step=ActionStepBinding(field_path="/payload/step"),
        )
        prefix = await resolve_trajectory_prefix(
            repository,
            TrajectoryPrefixRequest(
                schema_version="trajectory-prefix-request/v1",
                run_id=RUN_ID,
                boundary_event_sequence=1,
                bindings=(binding,),
            ),
        )
        with pytest.raises(TrajectoryError) as error:
            await project_fixed_step_schedule(repository, prefix)
        assert error.value.code is TrajectoryErrorCode.INVALID_POINTER

    repository = _repository()
    entries = await _append_trace(repository)
    prefix = await resolve_trajectory_prefix(
        repository,
        TrajectoryPrefixRequest(
            schema_version="trajectory-prefix-request/v1",
            run_id=RUN_ID,
            boundary_event_sequence=8,
            bindings=_bindings(entries),
        ),
    )
    schedule = await project_fixed_step_schedule(repository, prefix)
    with pytest.raises(TrajectoryError):
        await validated_fixed_step_schedule_for_prefix(repository, prefix, object())
    with pytest.raises(TrajectoryError):
        await validated_fixed_step_schedule_for_prefix(
            repository,
            prefix,
            schedule.model_copy(update={"schedule_digest": "0" * 64}),
        )


def test_schedule_models_reject_internally_inconsistent_decisions_and_results() -> None:
    event_id = UUID("00000000-0000-4000-8000-0000000025ff")
    invalid_decisions = (
        {
            "event_id": event_id,
            "event_sequence": 1,
            "invoke": False,
            "invocation_ordinal": 1,
            "reason": FixedStepReason.BOOTSTRAP,
        },
        {
            "event_id": event_id,
            "event_sequence": 2,
            "invoke": True,
            "invocation_ordinal": 1,
            "reason": FixedStepReason.BOOTSTRAP,
        },
        {
            "event_id": event_id,
            "event_sequence": 1,
            "invoke": True,
            "invocation_ordinal": 1,
            "reason": FixedStepReason.ACTION_STEP,
        },
        {
            "event_id": event_id,
            "event_sequence": 1,
            "invoke": True,
            "invocation_ordinal": 1,
            "reason": FixedStepReason.CURRENT_ACTION_STEP,
            "action_step_ordinal": 1,
        },
        {
            "event_id": event_id,
            "event_sequence": 1,
            "invoke": False,
            "reason": FixedStepReason.NO_ACTION_STEP,
            "action_step_ordinal": 1,
        },
    )
    for values in invalid_decisions:
        with pytest.raises(ValidationError):
            FixedStepDecision.model_validate(values)

    bootstrap = {
        "event_id": event_id,
        "event_sequence": 1,
        "invoke": True,
        "invocation_ordinal": 1,
        "reason": FixedStepReason.BOOTSTRAP,
        "action_step_ordinal": 1,
    }
    base = {
        "schedule_version": "first-and-every-action-step/v1",
        "run_id": RUN_ID,
        "boundary_event_sequence": 1,
        "trajectory_prefix_digest": "1" * 64,
        "decisions": (bootstrap,),
        "invocation_count": 1,
    }
    invalid_schedules = (
        {**base, "boundary_event_sequence": 2},
        {
            **base,
            "boundary_event_sequence": 2,
            "decisions": (
                bootstrap,
                {
                    "event_id": UUID("00000000-0000-4000-8000-0000000025fe"),
                    "event_sequence": 3,
                    "invoke": False,
                    "reason": FixedStepReason.NO_ACTION_STEP,
                },
            ),
        },
        {
            **base,
            "boundary_event_sequence": 2,
            "decisions": (
                bootstrap,
                {
                    "event_id": event_id,
                    "event_sequence": 2,
                    "invoke": False,
                    "reason": FixedStepReason.NO_ACTION_STEP,
                },
            ),
        },
        {**base, "invocation_count": 2},
        {
            **base,
            "decisions": (
                {
                    **bootstrap,
                    "reason": FixedStepReason.ACTION_STEP,
                },
            ),
        },
        {**base, "schedule_digest": "0" * 64},
        {**base, "boundary_event_sequence": 10_001},
        {**base, "invocation_count": 10_001},
        {**base, "decisions": (bootstrap,) * 10_001},
    )
    for values in invalid_schedules:
        with pytest.raises(ValidationError):
            FixedStepSchedule.model_validate(values)

    backwards = FixedStepSchedule.model_construct(
        schedule_version="first-and-every-action-step/v1",
        run_id=RUN_ID,
        boundary_event_sequence=2,
        trajectory_prefix_digest="1" * 64,
        decisions=(
            FixedStepDecision.model_validate(bootstrap | {"action_step_ordinal": 2}),
            FixedStepDecision(
                event_id=UUID("00000000-0000-4000-8000-0000000025fe"),
                event_sequence=2,
                invoke=False,
                reason=FixedStepReason.CURRENT_ACTION_STEP,
                action_step_ordinal=1,
            ),
        ),
        invocation_count=1,
        schedule_digest="0" * 64,
    )
    duplicate_new_step = backwards.model_copy(
        update={
            "decisions": (
                FixedStepDecision.model_validate(bootstrap),
                FixedStepDecision(
                    event_id=UUID("00000000-0000-4000-8000-0000000025fe"),
                    event_sequence=2,
                    invoke=True,
                    invocation_ordinal=2,
                    reason=FixedStepReason.ACTION_STEP,
                    action_step_ordinal=1,
                ),
            ),
            "invocation_count": 2,
        }
    )
    mismatched_current = backwards.model_copy(
        update={
            "decisions": (
                FixedStepDecision.model_validate(bootstrap),
                FixedStepDecision(
                    event_id=UUID("00000000-0000-4000-8000-0000000025fe"),
                    event_sequence=2,
                    invoke=False,
                    reason=FixedStepReason.CURRENT_ACTION_STEP,
                    action_step_ordinal=2,
                ),
            )
        }
    )
    for candidate in (backwards, duplicate_new_step, mismatched_current):
        with pytest.raises(ValueError):
            candidate.decisions_cover_the_prefix_exactly()
