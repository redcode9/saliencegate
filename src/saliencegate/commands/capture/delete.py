"""Explicit local capture deletion command."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.capture import (
    CaptureSpool,
    CaptureSpoolError,
    CaptureSpoolIntegrityError,
    CaptureStoreError,
    CaptureStoreIntegrityError,
    CaptureStoreStateError,
)
from saliencegate.capture.delete import (
    CaptureDeleteDisposition,
    delete_capture_project,
    delete_capture_session,
)
from saliencegate.capture.sessions import CaptureHumanSessionId
from saliencegate.commands.capture.common import (
    CaptureCommandConfigurationError,
    CaptureCommandInputError,
    CaptureCommandIntegrityError,
    CaptureCommandRequiresDisconnectError,
    capture_project_digest,
    resolve_capture_project,
)
from saliencegate.commands.capture.runtime import open_capture_runtime
from saliencegate.domain import canonical_json


class CaptureDeleteReport(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    schema_version: Literal["capture-delete/v1"] = "capture-delete/v1"
    scope: Literal["session", "project"]
    disposition: CaptureDeleteDisposition
    session_id: Annotated[CaptureHumanSessionId | None, Field(repr=False)] = None
    deleted_connections: Annotated[int, Field(ge=0)] = 0
    deleted_sessions: Annotated[int, Field(ge=0)] = 0
    deleted_tombstones: Annotated[int, Field(ge=0)] = 0
    secure_delete: bool
    wal_checkpointed: bool

    @model_validator(mode="after")
    def scope_matches_fields(self) -> Self:
        if (self.scope == "session") != (self.session_id is not None):
            raise ValueError("capture delete scope is inconsistent")
        if self.scope == "session" and any(
            (self.deleted_connections, self.deleted_sessions, self.deleted_tombstones)
        ):
            raise ValueError("session delete cannot contain project counts")
        return self

    def __repr__(self) -> str:
        return "CaptureDeleteReport(<redacted>)"

    __str__ = __repr__


def run_delete(
    *,
    session_id: str | None = None,
    delete_all: bool = False,
    confirm: bool = False,
    project: str | os.PathLike[str] | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> CaptureDeleteReport:
    """Delete exactly one session or one explicitly confirmed disabled project."""

    if (
        type(delete_all) is not bool
        or type(confirm) is not bool
        or delete_all == (session_id is not None)
        or (delete_all and (not confirm or project is None))
        or (not delete_all and confirm)
        or (session_id is not None and type(session_id) is not str)
    ):
        raise CaptureCommandInputError()
    resolved = resolve_capture_project(project)
    try:
        with open_capture_runtime(project=resolved, environ=environ, drain=False) as runtime:
            project_id = capture_project_digest(
                runtime.project,
                installation_key=runtime.installation_key,
            )
            if session_id is not None and not runtime.store.session_belongs_to_project(
                session_id,
                project_id,
                include_deleted=True,
            ):
                raise CaptureStoreStateError()

            def execute(drain: Callable[[], object]) -> CaptureDeleteReport:
                if delete_all:
                    project_receipt = delete_capture_project(
                        runtime.store,
                        project_id,
                        confirm=True,
                        drain=drain,
                    )
                    return CaptureDeleteReport(
                        scope="project",
                        disposition=project_receipt.disposition,
                        deleted_connections=project_receipt.deleted_connections,
                        deleted_sessions=project_receipt.deleted_sessions,
                        deleted_tombstones=project_receipt.deleted_tombstones,
                        secure_delete=project_receipt.secure_delete,
                        wal_checkpointed=project_receipt.wal_checkpointed,
                    )
                assert session_id is not None
                if not runtime.store.session_belongs_to_project(
                    session_id,
                    project_id,
                    include_deleted=True,
                ):
                    raise CaptureStoreStateError()
                session_receipt = delete_capture_session(runtime.store, session_id, drain=drain)
                return CaptureDeleteReport(
                    scope="session",
                    disposition=session_receipt.disposition,
                    session_id=session_receipt.human_id,
                    secure_delete=session_receipt.secure_delete,
                    wal_checkpointed=session_receipt.wal_checkpointed,
                )

            spool = (
                runtime.spool
                if runtime.spool is not None
                else CaptureSpool.open(runtime.locations, runtime.installation_key)
            )
            with spool.maintenance() as maintenance:
                report = execute(lambda: maintenance.drain(runtime.store))
                if delete_all and not runtime.store.list_connections():
                    maintenance.clear_drop_health_if_empty()
                return report
    except CaptureStoreStateError:
        if delete_all:
            raise CaptureCommandRequiresDisconnectError() from None
        raise CaptureCommandInputError() from None
    except (CaptureSpoolIntegrityError, CaptureStoreIntegrityError):
        raise CaptureCommandIntegrityError() from None
    except (CaptureSpoolError, CaptureStoreError):
        raise CaptureCommandConfigurationError() from None


def render_delete_json(report: CaptureDeleteReport) -> str:
    checked = CaptureDeleteReport.model_validate(report)
    return canonical_json(checked.model_dump(mode="json", warnings=False)).decode("utf-8") + "\n"


def render_delete_human(report: CaptureDeleteReport) -> str:
    checked = CaptureDeleteReport.model_validate(report)
    if checked.scope == "session":
        assert checked.session_id is not None
        return f"Capture session {checked.session_id}: {checked.disposition.value}.\n"
    return (
        f"Project capture: {checked.disposition.value}; "
        f"connections={checked.deleted_connections}, sessions={checked.deleted_sessions}.\n"
    )


__all__ = [
    "CaptureDeleteReport",
    "render_delete_human",
    "render_delete_json",
    "run_delete",
]
