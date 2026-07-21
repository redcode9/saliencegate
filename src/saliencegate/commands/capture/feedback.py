"""Project-bound local feedback for authenticated passive capture sessions."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from saliencegate.capture import (
    CaptureStoreError,
    CaptureStoreIntegrityError,
    CaptureStoreStateError,
)
from saliencegate.capture.feedback import CaptureFeedbackLabel, CaptureFeedbackReceipt
from saliencegate.commands.capture.common import (
    CaptureCommandConfigurationError,
    CaptureCommandInputError,
    CaptureCommandIntegrityError,
    capture_project_digest,
    resolve_capture_project,
)
from saliencegate.commands.capture.runtime import open_capture_runtime
from saliencegate.domain import canonical_json


def run_capture_feedback(
    *,
    session_id: str,
    label: str,
    project: str | os.PathLike[str] | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> CaptureFeedbackReceipt:
    """Record one bounded label for a session in the selected current project."""

    if type(session_id) is not str or type(label) is not str:
        raise CaptureCommandInputError()
    try:
        selected_label = CaptureFeedbackLabel(label)
    except (TypeError, ValueError):
        raise CaptureCommandInputError() from None
    resolved = resolve_capture_project(project)
    try:
        with open_capture_runtime(project=resolved, environ=environ, drain=False) as runtime:
            project_id = capture_project_digest(
                runtime.project,
                installation_key=runtime.installation_key,
            )
            return runtime.store.record_feedback(
                session_id,
                selected_label,
                project_digest=project_id,
            )
    except CaptureStoreStateError:
        raise CaptureCommandInputError() from None
    except CaptureStoreIntegrityError:
        raise CaptureCommandIntegrityError() from None
    except CaptureStoreError:
        raise CaptureCommandConfigurationError() from None


def render_capture_feedback_json(report: CaptureFeedbackReceipt) -> str:
    """Render one compact canonical feedback receipt."""

    checked = CaptureFeedbackReceipt.model_validate(report)
    return canonical_json(checked.model_dump(mode="json", warnings=False)).decode("utf-8") + "\n"


def render_capture_feedback_human(report: CaptureFeedbackReceipt) -> str:
    """Render feedback state without paths, digests, provider IDs, or event content."""

    checked = CaptureFeedbackReceipt.model_validate(report)
    return (
        f"Capture feedback {checked.session_id}: {checked.disposition.value}; "
        f"label={checked.label.value}; revisions={checked.revision_count}.\n"
    )


__all__ = [
    "render_capture_feedback_human",
    "render_capture_feedback_json",
    "run_capture_feedback",
]
