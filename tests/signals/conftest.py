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

RUN_ID = UUID("00000000-0000-4000-8000-000000002001")
OTHER_RUN_ID = UUID("00000000-0000-4000-8000-000000002002")
NOW = datetime(2026, 7, 11, 20, 0, tzinfo=UTC)

EventFactory = Callable[..., TraceEvent]


def identifier(value: int) -> UUID:
    return UUID(f"00000000-0000-4000-8000-{value:012x}")


@pytest.fixture
def event_factory() -> EventFactory:
    def build(
        sequence: int,
        *,
        event_type: EventType = EventType.OBSERVATION,
        phase: EventPhase = EventPhase.POST_ACTION,
        payload: dict[str, object] | None = None,
        parent_ids: tuple[UUID, ...] = (),
        run_id: UUID = RUN_ID,
    ) -> TraceEvent:
        return TraceEvent(
            event_id=identifier(0x2100 + sequence),
            run_id=run_id,
            sequence=sequence,
            source_event_id=f"signal-source-{sequence}",
            timestamp=NOW + timedelta(seconds=sequence),
            event_type=event_type,
            phase=phase,
            payload={} if payload is None else payload,
            payload_digest=PayloadDigest(
                algorithm=PayloadDigestAlgorithm.SYNTHETIC_SHA256,
                value=f"{sequence % 16:x}" * 64,
            ),
            parent_ids=parent_ids,
            source_adapter="signal-fixture/1",
            trust_label=TrustLabel.SYNTHETIC_FIXTURE,
        )

    return build
