"""Project-bound listing for authenticated passive capture sessions."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from saliencegate.capture import (
    CaptureProfile,
    CaptureSessionState,
    CaptureStoreError,
    CaptureStoreIntegrityError,
    CaptureStoreStateError,
)
from saliencegate.capture.sessions import CaptureHumanSessionId
from saliencegate.commands.capture.common import (
    CaptureCommandConfigurationError,
    CaptureCommandInputError,
    CaptureCommandIntegrityError,
    CaptureCommandUnavailableError,
    capture_project_digest,
    resolve_capture_project,
)
from saliencegate.commands.capture.runtime import open_capture_runtime
from saliencegate.domain import canonical_json
from saliencegate.domain.records import UtcDatetime

CaptureProviderAlias: TypeAlias = Literal["codex", "claude-code", "opencode", "pi"]

_PROVIDER_PROFILES: dict[CaptureProviderAlias, CaptureProfile] = {
    "codex": CaptureProfile.CODEX_HOOKS_V1,
    "claude-code": CaptureProfile.CLAUDE_CODE_HOOKS_V1,
    "opencode": CaptureProfile.OPENCODE_PLUGIN_V1,
    "pi": CaptureProfile.PI_EXTENSION_V1,
}
_PROFILE_PROVIDERS = {profile: alias for alias, profile in _PROVIDER_PROFILES.items()}
_CLI_SESSION_STATES = {
    "open": CaptureSessionState.OPEN,
    "closed": CaptureSessionState.CLOSED,
    "quarantined": CaptureSessionState.QUARANTINED,
}


class _SessionsModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(<redacted>)"

    __str__ = __repr__


class CaptureSessionListItem(_SessionsModel):
    schema_version: Literal["capture-session-list-item/v1"] = "capture-session-list-item/v1"
    session_id: Annotated[CaptureHumanSessionId, Field(repr=False)]
    provider: CaptureProviderAlias
    state: CaptureSessionState
    event_count: Annotated[int, Field(ge=0, le=1_000)]
    coverage_degraded: bool
    opened_at: Annotated[UtcDatetime, Field(repr=False)]
    updated_at: Annotated[UtcDatetime, Field(repr=False)]
    closed_at: Annotated[UtcDatetime | None, Field(repr=False)]


class CaptureSessionsReport(_SessionsModel):
    schema_version: Literal["capture-sessions/v1"] = "capture-sessions/v1"
    sessions: Annotated[tuple[CaptureSessionListItem, ...], Field(max_length=100)]


def _provider(value: str | None) -> CaptureProviderAlias | None:
    if value is None:
        return None
    if type(value) is not str or value not in _PROVIDER_PROFILES:
        raise CaptureCommandInputError()
    return value


def run_sessions(
    *,
    project: str | os.PathLike[str] | Path | None = None,
    provider: str | None = None,
    state: str | None = None,
    limit: int = 20,
    environ: Mapping[str, str] | None = None,
) -> CaptureSessionsReport:
    """Drain and list only sessions belonging to the selected current project."""

    alias = _provider(provider)
    if type(limit) is not int or not 1 <= limit <= 100:
        raise CaptureCommandInputError()
    if state is None:
        selected_state = None
    elif type(state) is str and state in _CLI_SESSION_STATES:
        selected_state = _CLI_SESSION_STATES[state]
    else:
        raise CaptureCommandInputError()
    resolved = resolve_capture_project(project)
    try:
        with open_capture_runtime(project=resolved, environ=environ, drain=True) as runtime:
            project_id = capture_project_digest(
                runtime.project,
                installation_key=runtime.installation_key,
            )
            summaries = runtime.store.list_sessions(
                project_digest=project_id,
                profile_id=None if alias is None else _PROVIDER_PROFILES[alias],
                state=selected_state,
                limit=limit,
            )
    except CaptureCommandUnavailableError:
        summaries = ()
    except CaptureStoreIntegrityError:
        raise CaptureCommandIntegrityError() from None
    except (CaptureStoreError, CaptureStoreStateError):
        raise CaptureCommandConfigurationError() from None
    try:
        rows = tuple(
            CaptureSessionListItem(
                session_id=item.human_id,
                provider=_PROFILE_PROVIDERS[item.profile_id],
                state=item.state,
                event_count=item.event_count,
                coverage_degraded=item.coverage_degraded,
                opened_at=item.opened_at,
                updated_at=item.updated_at,
                closed_at=item.closed_at,
            )
            for item in summaries
        )
        return CaptureSessionsReport(sessions=rows)
    except Exception:
        raise CaptureCommandIntegrityError() from None


def render_sessions_json(report: CaptureSessionsReport) -> str:
    checked = CaptureSessionsReport.model_validate(report)
    return canonical_json(checked.model_dump(mode="json", warnings=False)).decode("utf-8") + "\n"


def render_sessions_human(report: CaptureSessionsReport) -> str:
    checked = CaptureSessionsReport.model_validate(report)
    if not checked.sessions:
        return "No captured sessions.\n"
    lines = ["Captured sessions"]
    for item in checked.sessions:
        degraded = "; degraded" if item.coverage_degraded else ""
        lines.append(
            f"{item.session_id}  {item.provider}  {item.state.value}  "
            f"events={item.event_count}{degraded}"
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "CaptureSessionListItem",
    "CaptureSessionsReport",
    "render_sessions_human",
    "render_sessions_json",
    "run_sessions",
]
