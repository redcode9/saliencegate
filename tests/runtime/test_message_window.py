from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from pydantic import ValidationError

from saliencegate.domain import (
    EventPhase,
    EventType,
    EvidenceReference,
    EvidenceSource,
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
    AttestedTrajectoryEvent,
    AttestedTrajectoryPrefix,
    EventTextSelector,
    LogicalMessageBinding,
    LogicalMessageRole,
    TrajectoryBinding,
    TrajectoryError,
    TrajectoryErrorCode,
    TrajectoryPrefixRequest,
    bind_persisted_trajectory_event,
    resolve_trajectory_prefix,
)
from saliencegate.repository import MemoryRunRepository, SQLiteRunRepository
from saliencegate.runtime.message_window import (
    MAX_MESSAGE_WINDOW_CANONICAL_BYTES,
    MESSAGE_WINDOW_VERSION,
    AttestedTaskDescription,
    MessageWindow,
    MessageWindowError,
    MessageWindowMessage,
    MessageWindowPayload,
    TrajectoryTextSource,
    project_message_window,
    validated_message_window_for_prefix,
)
from saliencegate.security import InstallationKey, RedactionPolicy

RUN_ID = UUID("00000000-0000-4000-8000-000000002701")
NOW = datetime(2026, 7, 12, 9, 0, tzinfo=UTC)


def _id_factory():
    identifiers = itertools.count(0x2800)
    return lambda: UUID(f"00000000-0000-4000-8000-{next(identifiers):012x}")


def _draft(
    sequence: int,
    *,
    payload: dict[str, object],
    event_type: EventType = EventType.MODEL_OUTPUT,
    trust_label: TrustLabel = TrustLabel.UNTRUSTED_MODEL_OUTPUT,
) -> NormalizedTraceEventDraft:
    return NormalizedTraceEventDraft(
        run_id=RUN_ID,
        source_event_id=f"window-{sequence}",
        timestamp=NOW + timedelta(seconds=sequence),
        event_type=event_type,
        phase=(
            EventPhase.INITIALIZATION
            if event_type is EventType.RUN_START
            else EventPhase.POST_ACTION
        ),
        payload=payload,
        source_adapter="window-fixture/v1",
        trust_label=trust_label,
    )


async def _entry_for(repository: RunRepository, sequence: int) -> LedgerEntry:
    return next(
        entry
        for entry in await repository.ledger(RUN_ID)
        if type(entry.record) is TraceEvent and entry.record.sequence == sequence
    )


@pytest.mark.asyncio
async def test_latest_eight_messages_ignore_technical_events_and_keep_task_separate() -> None:
    repository = MemoryRunRepository(
        redaction_policy=RedactionPolicy(literal_secrets=("fixture-secret",)),
        installation_key=InstallationKey(b"w" * 32),
        id_factory=_id_factory(),
    )
    await repository.append(
        _draft(
            1,
            payload={
                "task": "Deploy without exposing fixture-secret.",
                "messages": ["user-1", "assistant-1"],
            },
            event_type=EventType.RUN_START,
            trust_label=TrustLabel.UNTRUSTED_TASK_INPUT,
        )
    )
    await repository.append(
        _draft(
            2,
            payload={"detail": "technical"},
            event_type=EventType.TOOL_START,
            trust_label=TrustLabel.UNTRUSTED_TOOL_OUTPUT,
        )
    )
    for sequence in range(3, 13):
        message = f"message-{sequence}"
        if sequence == 12:
            message += " fixture-secret"
        await repository.append(
            _draft(
                sequence,
                payload={"message": message},
                event_type=(EventType.TOOL_COMPLETION if sequence == 7 else EventType.MODEL_OUTPUT),
                trust_label=(
                    TrustLabel.UNTRUSTED_TOOL_OUTPUT
                    if sequence == 7
                    else TrustLabel.UNTRUSTED_MODEL_OUTPUT
                ),
            )
        )

    entries = tuple(
        entry for entry in await repository.ledger(RUN_ID) if type(entry.record) is TraceEvent
    )
    roles = (
        LogicalMessageRole.USER,
        LogicalMessageRole.ASSISTANT,
        LogicalMessageRole.TOOL,
        LogicalMessageRole.CONTROLLER,
    )
    bindings = []
    for entry in entries:
        event = entry.record
        assert type(event) is TraceEvent
        if event.sequence == 1:
            messages = (
                LogicalMessageBinding(
                    role=LogicalMessageRole.USER,
                    selector=EventTextSelector(field_path="/payload/messages/0"),
                ),
                LogicalMessageBinding(
                    role=LogicalMessageRole.ASSISTANT,
                    selector=EventTextSelector(field_path="/payload/messages/1"),
                ),
            )
        elif event.sequence == 2:
            messages = ()
        else:
            messages = (
                LogicalMessageBinding(
                    role=roles[(event.sequence - 3) % len(roles)],
                    selector=EventTextSelector(field_path="/payload/message"),
                ),
            )
        bindings.append(
            bind_persisted_trajectory_event(
                entry,
                task_description=(
                    EventTextSelector(field_path="/payload/task") if event.sequence == 1 else None
                ),
                logical_messages=messages,
            )
        )
    prefix = await resolve_trajectory_prefix(
        repository,
        TrajectoryPrefixRequest(
            schema_version="trajectory-prefix-request/v1",
            run_id=RUN_ID,
            boundary_event_sequence=12,
            bindings=tuple(bindings),
        ),
    )

    window = await project_message_window(repository, prefix)

    assert window.payload.version == MESSAGE_WINDOW_VERSION
    assert tuple(message.content for message in window.payload.messages) == (
        *(f"message-{sequence}" for sequence in range(5, 12)),
        "message-12 [REDACTED]",
    )
    assert tuple(message.role for message in window.payload.messages) == tuple(
        roles[(sequence - 3) % len(roles)] for sequence in range(5, 13)
    )
    assert tuple(message.evidence.source_id for message in window.payload.messages) == tuple(
        entries[sequence - 1].record.event_id for sequence in range(5, 13)
    )
    assert window.task_description.content == "Deploy without exposing [REDACTED]."
    assert "fixture-secret" not in canonical_json(window).decode()
    assert len(window.payload.messages) == 8
    assert window.payload_canonical_utf8_bytes == len(canonical_json(window.payload))
    assert window.payload_canonical_utf8_bytes <= MAX_MESSAGE_WINDOW_CANONICAL_BYTES
    assert await validated_message_window_for_prefix(repository, prefix, window) == window
    assert "fixture-secret" not in repr(window)
    assert "message-12" not in repr(window)


@pytest.mark.asyncio
async def test_message_pointer_supports_rfc6901_and_utf8_byte_spans() -> None:
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=_id_factory(),
    )
    await repository.append(
        _draft(
            1,
            payload={"task": "Résumé", "a/b": {"~key": ["préfixe café suffixe"]}},
            event_type=EventType.RUN_START,
        )
    )
    entry = await _entry_for(repository, 1)
    text = "préfixe café suffixe"
    encoded = text.encode("utf-8")
    start = encoded.index("café".encode())
    binding = bind_persisted_trajectory_event(
        entry,
        task_description=EventTextSelector(field_path="/payload/task"),
        logical_messages=(
            LogicalMessageBinding(
                role=LogicalMessageRole.USER,
                selector=EventTextSelector(
                    field_path="/payload/a~1b/~0key/0",
                    span=TextSpan(start_byte=start, end_byte=start + len("café".encode())),
                ),
            ),
        ),
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

    assert (await project_message_window(repository, prefix)).payload.messages[0].content == "café"


@pytest.mark.asyncio
async def test_invalid_pointer_span_duplicate_task_and_overflow_fail_closed() -> None:
    repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=_id_factory(),
    )
    await repository.append(
        _draft(
            1,
            payload={
                "task": "task",
                "message": "é",
                "empty": "",
                "not_text": 42,
                "array": ["only"],
            },
            event_type=EventType.RUN_START,
        )
    )
    entry = await _entry_for(repository, 1)

    invalid_selectors = (
        EventTextSelector(field_path="/payload/missing"),
        EventTextSelector(field_path="/payload/not_text"),
        EventTextSelector(field_path="/payload/empty"),
        EventTextSelector(field_path="/payload/array/01"),
        EventTextSelector(field_path="/payload/array/1"),
        EventTextSelector(field_path="/payload/not_text/value"),
        EventTextSelector(
            field_path="/payload/message",
            span=TextSpan(start_byte=0, end_byte=1),
        ),
        EventTextSelector(
            field_path="/payload/message",
            span=TextSpan(start_byte=0, end_byte=3),
        ),
    )
    for selector in invalid_selectors:
        binding = bind_persisted_trajectory_event(
            entry,
            task_description=EventTextSelector(field_path="/payload/task"),
            logical_messages=(
                LogicalMessageBinding(role=LogicalMessageRole.USER, selector=selector),
            ),
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
            await project_message_window(repository, prefix)
        assert error.value.code in {
            TrajectoryErrorCode.INVALID_POINTER,
            TrajectoryErrorCode.INVALID_SPAN,
        }

    duplicate_task = bind_persisted_trajectory_event(
        entry,
        task_description=EventTextSelector(field_path="/payload/task"),
    )
    forged = duplicate_task.model_copy(update={"event_sequence": 2})
    with pytest.raises(TrajectoryError) as error:
        await resolve_trajectory_prefix(
            repository,
            TrajectoryPrefixRequest.model_construct(
                schema_version="trajectory-prefix-request/v1",
                run_id=RUN_ID,
                boundary_event_sequence=2,
                bindings=(duplicate_task, forged),
                request_digest="0" * 64,
            ),
        )
    assert error.value.code in {
        TrajectoryErrorCode.DUPLICATE_BINDING,
        TrajectoryErrorCode.UNATTESTED_REFERENCE,
    }

    huge = "x" * MAX_MESSAGE_WINDOW_CANONICAL_BYTES
    second_repository = MemoryRunRepository(
        synthetic_benchmark=True,
        id_factory=_id_factory(),
    )
    await second_repository.append(
        _draft(
            1,
            payload={"task": "task", "message": huge},
            event_type=EventType.RUN_START,
        )
    )
    huge_entry = await _entry_for(second_repository, 1)
    huge_binding = bind_persisted_trajectory_event(
        huge_entry,
        task_description=EventTextSelector(field_path="/payload/task"),
        logical_messages=(
            LogicalMessageBinding(
                role=LogicalMessageRole.USER,
                selector=EventTextSelector(field_path="/payload/message"),
            ),
        ),
    )
    huge_prefix = await resolve_trajectory_prefix(
        second_repository,
        TrajectoryPrefixRequest(
            schema_version="trajectory-prefix-request/v1",
            run_id=RUN_ID,
            boundary_event_sequence=1,
            bindings=(huge_binding,),
        ),
    )
    with pytest.raises(MessageWindowError):
        await project_message_window(second_repository, huge_prefix)


async def _single_message_window(
    repository: RunRepository,
    *,
    task: str = "Keep the task separate.",
    message: str | None = "logical message",
    role: LogicalMessageRole = LogicalMessageRole.USER,
) -> MessageWindow:
    payload: dict[str, object] = {"task": task}
    if message is not None:
        payload["message"] = message
    await repository.append(
        _draft(
            1,
            payload=payload,
            event_type=EventType.RUN_START,
            trust_label=TrustLabel.UNTRUSTED_TASK_INPUT,
        )
    )
    entry = await _entry_for(repository, 1)
    binding = bind_persisted_trajectory_event(
        entry,
        task_description=EventTextSelector(field_path="/payload/task"),
        logical_messages=(
            ()
            if message is None
            else (
                LogicalMessageBinding(
                    role=role,
                    selector=EventTextSelector(field_path="/payload/message"),
                ),
            )
        ),
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
    return await project_message_window(repository, prefix)


@pytest.mark.asyncio
async def test_empty_window_is_explicit_and_backend_outputs_are_byte_identical(
    tmp_path,
) -> None:
    empty = await _single_message_window(
        MemoryRunRepository(synthetic_benchmark=True, id_factory=_id_factory()),
        message=None,
    )
    assert empty.payload.messages == ()
    assert empty.source_attestations == ()
    assert empty.task_description.content == "Keep the task separate."

    memory = MemoryRunRepository(synthetic_benchmark=True, id_factory=_id_factory())
    sqlite = SQLiteRunRepository(
        tmp_path / "trajectory-window.sqlite3",
        synthetic_benchmark=True,
        id_factory=_id_factory(),
    )
    try:
        memory_window = await _single_message_window(memory)
        sqlite_window = await _single_message_window(sqlite)
    finally:
        sqlite.close()
    assert canonical_json(memory_window) == canonical_json(sqlite_window)


@pytest.mark.asyncio
async def test_provider_visible_payload_enforces_exact_canonical_byte_ceiling() -> None:
    probe = await _single_message_window(
        MemoryRunRepository(synthetic_benchmark=True, id_factory=_id_factory()),
        message="x",
    )
    exact_content_bytes = 1 + (
        MAX_MESSAGE_WINDOW_CANONICAL_BYTES - probe.payload_canonical_utf8_bytes
    )
    exact = await _single_message_window(
        MemoryRunRepository(synthetic_benchmark=True, id_factory=_id_factory()),
        message="x" * exact_content_bytes,
    )
    assert exact.payload_canonical_utf8_bytes == MAX_MESSAGE_WINDOW_CANONICAL_BYTES

    repository = MemoryRunRepository(synthetic_benchmark=True, id_factory=_id_factory())
    with pytest.raises(MessageWindowError) as error:
        await _single_message_window(
            repository,
            message="x" * (exact_content_bytes + 1),
        )
    assert error.value.code is TrajectoryErrorCode.LIMIT_EXCEEDED


@pytest.mark.asyncio
async def test_task_description_has_an_independent_byte_ceiling() -> None:
    repository = MemoryRunRepository(synthetic_benchmark=True, id_factory=_id_factory())
    with pytest.raises(MessageWindowError) as error:
        await _single_message_window(
            repository,
            task="t" * 32_001,
            message=None,
        )
    assert error.value.code is TrajectoryErrorCode.LIMIT_EXCEEDED


@pytest.mark.asyncio
async def test_window_digest_binds_content_role_order_and_source() -> None:
    user = await _single_message_window(
        MemoryRunRepository(synthetic_benchmark=True, id_factory=_id_factory()),
        message="same text",
        role=LogicalMessageRole.USER,
    )
    assistant = await _single_message_window(
        MemoryRunRepository(synthetic_benchmark=True, id_factory=_id_factory()),
        message="same text",
        role=LogicalMessageRole.ASSISTANT,
    )
    changed = await _single_message_window(
        MemoryRunRepository(synthetic_benchmark=True, id_factory=_id_factory()),
        message="changed text",
        role=LogicalMessageRole.USER,
    )

    assert len({user.window_digest, assistant.window_digest, changed.window_digest}) == 3
    assert user.window_digest == "531b981dcad3d0d09f18f334dec6fafd533028c974e9480683d39d2496eecfdb"
    assert user.payload_canonical_utf8_bytes == len(canonical_json(user.payload))
    assert "content" not in TrajectoryBinding.model_fields

    async def ordered_window(order: tuple[int, int]) -> MessageWindow:
        repository = MemoryRunRepository(synthetic_benchmark=True, id_factory=_id_factory())
        await repository.append(
            _draft(
                1,
                payload={"task": "task", "messages": ["first", "second"]},
                event_type=EventType.RUN_START,
            )
        )
        entry = await _entry_for(repository, 1)
        binding = bind_persisted_trajectory_event(
            entry,
            task_description=EventTextSelector(field_path="/payload/task"),
            logical_messages=tuple(
                LogicalMessageBinding(
                    role=LogicalMessageRole.USER,
                    selector=EventTextSelector(field_path=f"/payload/messages/{index}"),
                )
                for index in order
            ),
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
        return await project_message_window(repository, prefix)

    forward = await ordered_window((0, 1))
    reverse = await ordered_window((1, 0))
    assert tuple(item.content for item in forward.payload.messages) == ("first", "second")
    assert tuple(item.content for item in reverse.payload.messages) == ("second", "first")
    assert forward.window_digest != reverse.window_digest


@pytest.mark.asyncio
async def test_window_records_reject_cross_domain_provenance_and_digest_tampering() -> None:
    repository = MemoryRunRepository(synthetic_benchmark=True, id_factory=_id_factory())
    window = await _single_message_window(repository)
    source = window.source_attestations[0]
    message = window.payload.messages[0]

    memory_evidence = EvidenceReference(
        source=EvidenceSource.MEMORY,
        source_id=message.evidence.source_id,
        revision=1,
        field_path="/content",
    )
    source_values = source.model_dump(mode="python")
    source_values["evidence"] = memory_evidence
    with pytest.raises(ValidationError):
        TrajectoryTextSource.model_validate(source_values)

    forged_evidence = EvidenceReference.model_construct(
        source=EvidenceSource.EVENT,
        source_id=message.evidence.source_id,
        revision=None,
        field_path="not-a-pointer",
        span=None,
    )
    source_values = source.model_dump(mode="python")
    source_values["evidence"] = forged_evidence
    with pytest.raises(ValidationError):
        TrajectoryTextSource.model_validate(source_values)
    with pytest.raises(ValidationError):
        MessageWindowMessage(
            role=LogicalMessageRole.USER,
            content="content",
            evidence=forged_evidence,
            trust_label=TrustLabel.UNTRUSTED_MODEL_OUTPUT,
        )

    source_values = source.model_dump(mode="python")
    source_values["record_tag"] = {
        "algorithm": PayloadDigestAlgorithm.HMAC_SHA256,
        "value": source.record_tag.value,
    }
    with pytest.raises(ValidationError):
        TrajectoryTextSource.model_validate(source_values)

    with pytest.raises(ValidationError):
        MessageWindowMessage(
            role=LogicalMessageRole.USER,
            content="content",
            evidence=memory_evidence,
            trust_label=TrustLabel.UNTRUSTED_EXTERNAL_MEMORY,
        )

    task_values = window.task_description.model_dump(mode="python")
    task_values["task_digest"] = "0" * 64
    with pytest.raises(ValidationError):
        AttestedTaskDescription.model_validate(task_values)
    with pytest.raises(ValidationError):
        AttestedTaskDescription(
            version="attested-task-description/v1",
            content="t" * 32_001,
            source=source,
        )
    with pytest.raises(ValidationError):
        AttestedTaskDescription(
            version="attested-task-description/v1",
            content="\ud800",
            source=source,
        )
    with pytest.raises(ValidationError):
        MessageWindowMessage(
            role=LogicalMessageRole.USER,
            content="\ud800",
            evidence=message.evidence,
            trust_label=message.trust_label,
        )

    window_values = window.model_dump(mode="python")
    window_values["source_attestations"] = ()
    with pytest.raises(ValidationError):
        MessageWindow.model_validate(window_values)

    other_source = source.model_copy(update={"trust_label": TrustLabel.TRUSTED_CONTROLLER})
    window_values = window.model_dump(mode="python")
    window_values["source_attestations"] = (other_source,)
    with pytest.raises(ValidationError):
        MessageWindow.model_validate(window_values)

    window_values = window.model_dump(mode="python")
    window_values["payload_canonical_utf8_bytes"] -= 1
    with pytest.raises(ValidationError):
        MessageWindow.model_validate(window_values)

    window_values = window.model_dump(mode="python")
    window_values["window_digest"] = "0" * 64
    with pytest.raises(ValidationError):
        MessageWindow.model_validate(window_values)

    payload_values = window.payload.model_dump(mode="python")
    payload_values["messages"] = tuple(window.payload.messages) * 9
    with pytest.raises(ValidationError):
        MessageWindowPayload.model_validate(payload_values)

    window_values = window.model_dump(mode="python")
    window_values["boundary_chain_tag"] = PayloadDigest.model_construct(
        algorithm="bad",
        value="bad",
    )
    with pytest.raises(ValidationError):
        MessageWindow.model_validate(window_values)


@pytest.mark.asyncio
async def test_window_verifier_rejects_wrong_types_and_constructed_tampering() -> None:
    repository = MemoryRunRepository(synthetic_benchmark=True, id_factory=_id_factory())
    await repository.append(
        _draft(
            1,
            payload={"task": "task", "message": "message"},
            event_type=EventType.RUN_START,
        )
    )
    entry = await _entry_for(repository, 1)
    binding = bind_persisted_trajectory_event(
        entry,
        task_description=EventTextSelector(field_path="/payload/task"),
        logical_messages=(
            LogicalMessageBinding(
                role=LogicalMessageRole.USER,
                selector=EventTextSelector(field_path="/payload/message"),
            ),
        ),
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
    window = await project_message_window(repository, prefix)

    original = prefix.items[0]
    event_values = original.event.model_dump(mode="python")
    event_values["payload"] = {"task": "raw task", "message": "raw unredacted message"}
    forged_event = TraceEvent.model_validate(event_values)
    forged_item = AttestedTrajectoryEvent(event=forged_event, binding=original.binding)
    forged_prefix = AttestedTrajectoryPrefix(
        schema_version="attested-trajectory-prefix/v1",
        run_id=prefix.run_id,
        boundary_event_sequence=1,
        request_digest=prefix.request_digest,
        items=(forged_item,),
    )
    with pytest.raises(TrajectoryError) as error:
        await project_message_window(repository, forged_prefix)
    assert error.value.code is TrajectoryErrorCode.UNATTESTED_REFERENCE

    with pytest.raises(TrajectoryError) as error:
        await validated_message_window_for_prefix(repository, prefix, object())
    assert error.value.code is TrajectoryErrorCode.UNATTESTED_REFERENCE
    with pytest.raises(TrajectoryError) as error:
        await validated_message_window_for_prefix(
            repository,
            prefix,
            window.model_copy(update={"window_digest": "0" * 64}),
        )
    assert error.value.code is TrajectoryErrorCode.UNATTESTED_REFERENCE

    other = await _single_message_window(
        MemoryRunRepository(synthetic_benchmark=True, id_factory=_id_factory()),
        message="different but valid",
    )
    with pytest.raises(TrajectoryError) as error:
        await validated_message_window_for_prefix(repository, prefix, other)
    assert error.value.code is TrajectoryErrorCode.UNATTESTED_REFERENCE
    with pytest.raises(TrajectoryError) as error:
        await validated_message_window_for_prefix(
            repository,
            prefix.model_copy(update={"prefix_digest": "0" * 64}),
            window,
        )
    assert error.value.code is TrajectoryErrorCode.INVALID_INPUT
