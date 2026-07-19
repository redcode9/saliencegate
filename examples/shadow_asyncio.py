"""Minimal provider-free SalienceGate Shadow Mode integration."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from hashlib import sha256
from uuid import UUID

from saliencegate.security import InstallationKey
from saliencegate.shadow import ShadowSession

RUN_ID = UUID("b35f05f3-555b-4f09-8996-a7b3693bb54a")
ENVIRONMENT_DIGEST = sha256(b"saliencegate-shadow-example-environment-v1").hexdigest()
EXAMPLE_ONLY_KEY = InstallationKey(bytes(32))


async def main() -> None:
    async with ShadowSession.in_memory(
        run_id=RUN_ID,
        installation_key=EXAMPLE_ONLY_KEY,
    ) as session:
        await session.start(
            source_event_id="run-start",
            occurred_at=datetime(2026, 7, 16, 10, 0, tzinfo=UTC),
        )
        action = await session.action(
            source_event_id="action-1",
            occurred_at=datetime(2026, 7, 16, 10, 1, tzinfo=UTC),
            argv=("example-tool", "--check"),
            working_directory="/example",
            environment_digest=ENVIRONMENT_DIGEST,
        )
        failed_result = await session.tool_result(
            source_event_id="tool-result-1",
            occurred_at=datetime(2026, 7, 16, 10, 2, tzinfo=UTC),
            action=action.ref,
            status="failed",
            exit_status=1,
            exception_type="ExampleToolFailure",
        )

    observation = failed_result.observation
    heuristic = observation.heuristic_evaluations[0]
    assert heuristic.disposition.value == "flagged"
    detected = ", ".join(signal.signal_type.value for signal in observation.detected_signals)

    print("SalienceGate Shadow Mode example")
    print(f"evaluated events: {observation.sequence}")
    detector_count = len(observation.supported_signal_types) + len(
        observation.unsupported_signal_types
    )
    print(f"supported detectors: {len(observation.supported_signal_types)} of {detector_count}")
    print(f"failed-result disposition: {heuristic.disposition.value}")
    print(f"detected signals: {detected}")
    print("evidence: descriptive observational; no decision authority")
    print(f"model calls: {observation.model_calls}")


if __name__ == "__main__":
    asyncio.run(main())
