from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from saliencegate.domain import (
    EventPhase,
    EventType,
    PayloadDigest,
    PayloadDigestAlgorithm,
    TraceEvent,
    TrustLabel,
)
from saliencegate.shadow.inputs import derive_shadow_event_id

RUN_ID = UUID("b35f05f3-555b-4f09-8996-a7b3693bb54a")
OTHER_RUN_ID = UUID("b35f05f3-555b-4f09-8996-a7b3693bb54b")
NOW = datetime(2026, 7, 16, 10, 0, tzinfo=UTC)
SOURCE_ADAPTER = "saliencegate-shadow/v1"

TraceEventFactory = Callable[..., TraceEvent]


def identifier(value: int) -> UUID:
    return UUID(f"00000000-0000-4000-8000-{value:012x}")


@pytest.fixture
def trace_event_factory() -> TraceEventFactory:
    def build(
        sequence: int,
        *,
        event_type: EventType = EventType.OBSERVATION,
        phase: EventPhase = EventPhase.POST_ACTION,
        payload: dict[str, object] | None = None,
        parent_ids: tuple[UUID, ...] = (),
        trust_label: TrustLabel = TrustLabel.UNTRUSTED_TASK_INPUT,
        run_id: UUID = RUN_ID,
    ) -> TraceEvent:
        source_event_id = f"shadow-source-{sequence}"
        return TraceEvent(
            event_id=derive_shadow_event_id(run_id, source_event_id),
            run_id=run_id,
            sequence=sequence,
            source_event_id=source_event_id,
            timestamp=NOW + timedelta(seconds=sequence),
            event_type=event_type,
            phase=phase,
            payload={} if payload is None else payload,
            payload_digest=PayloadDigest(
                algorithm=PayloadDigestAlgorithm.SYNTHETIC_SHA256,
                value=f"{sequence % 16:x}" * 64,
            ),
            parent_ids=parent_ids,
            source_adapter=SOURCE_ADAPTER,
            trust_label=trust_label,
        )

    return build
