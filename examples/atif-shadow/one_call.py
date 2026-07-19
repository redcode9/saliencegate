"""Analyze both synthetic ATIF examples through the provider-free one-call API."""

from __future__ import annotations

import asyncio
from hashlib import sha256
from pathlib import Path
from uuid import UUID

from saliencegate.shadow import (
    ATIFProfile,
    ShadowEnvironmentBinding,
    analyze_atif_bytes,
)

EXAMPLE_DIRECTORY = Path(__file__).resolve().parent
ENVIRONMENT = ShadowEnvironmentBinding(
    default_working_directory="/synthetic/workspace",
    environment_digest=sha256(b"saliencegate-atif-shadow-example-v1").hexdigest(),
)
EXAMPLES = (
    (
        "Codex",
        EXAMPLE_DIRECTORY / "codex-minimal.trajectory.json",
        ATIFProfile.HARBOR_CODEX_V1,
        UUID("c0de0000-0000-4000-8000-000000000001"),
    ),
    (
        "Terminus 2",
        EXAMPLE_DIRECTORY / "terminus-minimal.trajectory.json",
        ATIFProfile.HARBOR_TERMINUS_2_V1,
        UUID("7e2a0000-0000-4000-8000-000000000001"),
    ),
)


async def main() -> None:
    for label, source_path, profile, run_id in EXAMPLES:
        report = await analyze_atif_bytes(
            source_path.read_bytes(),
            run_id=run_id,
            profile=profile,
            environment=ENVIRONMENT,
        )
        counts = dict(report.shadow_report.heuristic_disposition_counts)
        print(
            f"{label}: profile={report.binding.adapter_profile_id}; "
            f"flagged={counts['flagged']}; report={report.report_digest}"
        )


if __name__ == "__main__":
    asyncio.run(main())
