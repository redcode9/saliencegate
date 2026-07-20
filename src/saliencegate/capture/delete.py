"""Verified, retry-safe capture deletion coordination."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from threading import Lock
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.capture.sessions import CaptureHumanSessionId
from saliencegate.capture.spool import CaptureSpoolError
from saliencegate.capture.store import CaptureStore, CaptureStoreError, CaptureStoreStateError
from saliencegate.domain.records import Sha256Digest

CaptureDrain = Callable[[], object]


class CaptureDeleteDisposition(StrEnum):
    DELETED = "deleted"
    ALREADY_DELETED = "already_deleted"


class _CaptureDeleteModel(BaseModel):
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


class CaptureSessionDeleteReceipt(_CaptureDeleteModel):
    """Content-free outcome for one durable session deletion."""

    schema_version: Literal["capture-session-delete-receipt/v1"] = (
        "capture-session-delete-receipt/v1"
    )
    disposition: CaptureDeleteDisposition
    human_id: Annotated[CaptureHumanSessionId, Field(repr=False)]
    secure_delete: Literal[True] = True
    wal_checkpointed: Literal[True] = True


class CaptureProjectDeleteReceipt(_CaptureDeleteModel):
    """Content-free outcome for one project-scoped deletion."""

    schema_version: Literal["capture-project-delete-receipt/v1"] = (
        "capture-project-delete-receipt/v1"
    )
    disposition: CaptureDeleteDisposition
    project_digest: Annotated[Sha256Digest, Field(repr=False)]
    deleted_connections: Annotated[int, Field(ge=0)]
    deleted_sessions: Annotated[int, Field(ge=0)]
    deleted_tombstones: Annotated[int, Field(ge=0)]
    secure_delete: Literal[True] = True
    wal_checkpointed: Literal[True] = True

    @model_validator(mode="after")
    def disposition_matches_counts(self) -> Self:
        counts = (
            self.deleted_connections,
            self.deleted_sessions,
            self.deleted_tombstones,
        )
        if (self.disposition is CaptureDeleteDisposition.ALREADY_DELETED and any(counts)) or (
            self.disposition is CaptureDeleteDisposition.DELETED and self.deleted_connections == 0
        ):
            raise ValueError("capture project deletion receipt is inconsistent")
        return self


_DELETE_COORDINATION = Lock()


def _drain(callback: CaptureDrain) -> None:
    if not callable(callback):
        raise CaptureStoreStateError()
    try:
        callback()
    except (KeyboardInterrupt, SystemExit):
        raise
    except (CaptureSpoolError, CaptureStoreError):
        raise
    except Exception:
        raise CaptureStoreError() from None


def delete_capture_session(
    store: CaptureStore,
    human_id: str,
    *,
    drain: CaptureDrain,
) -> CaptureSessionDeleteReceipt:
    """Drain producers and durably delete one human-addressed session."""

    if type(store) is not CaptureStore:
        raise CaptureStoreStateError()
    with _DELETE_COORDINATION:
        store._require_maintenance()
        store._validate_human_id(human_id)
        if store._session_delete_requires_drain(human_id):
            _drain(drain)
        return store._delete_session(human_id)


def delete_capture_project(
    store: CaptureStore,
    project_digest: str,
    *,
    confirm: bool,
    drain: CaptureDrain,
) -> CaptureProjectDeleteReceipt:
    """Delete all app records for one project after explicit confirmation."""

    if type(store) is not CaptureStore or type(confirm) is not bool or not confirm:
        raise CaptureStoreStateError()
    with _DELETE_COORDINATION:
        store._require_maintenance()
        if store._require_project_connections_disabled(project_digest):
            _drain(drain)
        return store._delete_project(project_digest)


__all__ = [
    "CaptureDeleteDisposition",
    "CaptureDrain",
    "CaptureProjectDeleteReceipt",
    "CaptureSessionDeleteReceipt",
    "delete_capture_project",
    "delete_capture_session",
]
