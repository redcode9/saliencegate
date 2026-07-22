"""Content-free contracts for bounded bridge transport chunks."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.domain.primitives import ComponentIdentifier, Sha256Digest

MAX_CAPTURE_TRANSPORT_CHUNKS_PER_SESSION = 1_000


class CaptureTransportError(ValueError):
    """A content-free transport contract failure."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("capture transport is invalid")


class CaptureTransportDisposition(StrEnum):
    """Durable outcome of one chunk admission attempt."""

    ADMITTED = "admitted"
    REPLAYED = "replayed"
    QUARANTINED = "quarantined"
    OVERFLOW = "overflow"


class _CaptureTransportModel(BaseModel):
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


class CaptureTransportChunk(_CaptureTransportModel):
    """Only pseudonymous routing and keyed commitments for one native chunk."""

    schema_version: Literal["capture-transport-chunk/v1"] = "capture-transport-chunk/v1"
    connection_id: Annotated[ComponentIdentifier, Field(repr=False)]
    session_id: Annotated[Sha256Digest, Field(repr=False)]
    batch_ref: Annotated[Sha256Digest, Field(repr=False)]
    chunk_index: Annotated[
        int,
        Field(ge=0, lt=MAX_CAPTURE_TRANSPORT_CHUNKS_PER_SESSION),
    ]
    chunk_count: Annotated[
        int,
        Field(ge=1, le=MAX_CAPTURE_TRANSPORT_CHUNKS_PER_SESSION),
    ]
    chunk_digest: Annotated[Sha256Digest, Field(repr=False)]

    @model_validator(mode="after")
    def index_is_within_declared_batch(self) -> Self:
        if self.chunk_index >= self.chunk_count:
            raise ValueError("capture transport chunk index is invalid")
        return self


class CaptureTransportReceipt(_CaptureTransportModel):
    """Content-free receipt returned after one atomic transport attempt."""

    disposition: CaptureTransportDisposition
    connection_id: Annotated[ComponentIdentifier, Field(repr=False)]
    session_id: Annotated[Sha256Digest, Field(repr=False)]
    batch_ref: Annotated[Sha256Digest, Field(repr=False)]
    chunk_index: Annotated[
        int,
        Field(ge=0, lt=MAX_CAPTURE_TRANSPORT_CHUNKS_PER_SESSION),
    ]
    chunk_count: Annotated[
        int,
        Field(ge=1, le=MAX_CAPTURE_TRANSPORT_CHUNKS_PER_SESSION),
    ]
    intake_count: Annotated[int, Field(ge=0, le=MAX_CAPTURE_TRANSPORT_CHUNKS_PER_SESSION)]
    transport_ordinal: Annotated[
        int | None,
        Field(ge=1, le=MAX_CAPTURE_TRANSPORT_CHUNKS_PER_SESSION),
    ]
    previous_receipt_tag: Annotated[Sha256Digest | None, Field(repr=False)]
    receipt_tag: Annotated[Sha256Digest | None, Field(repr=False)]
    incomplete_batch_count: Annotated[
        int,
        Field(ge=0, le=MAX_CAPTURE_TRANSPORT_CHUNKS_PER_SESSION),
    ]
    event_count: Annotated[int, Field(ge=0, le=MAX_CAPTURE_TRANSPORT_CHUNKS_PER_SESSION)]

    @model_validator(mode="after")
    def receipt_coordinates_match_disposition(self) -> Self:
        admitted = self.disposition in {
            CaptureTransportDisposition.ADMITTED,
            CaptureTransportDisposition.REPLAYED,
        }
        if self.chunk_index >= self.chunk_count:
            raise ValueError("capture transport receipt is inconsistent")
        if admitted:
            if self.transport_ordinal is None or self.receipt_tag is None:
                raise ValueError("capture transport receipt is inconsistent")
            if (self.transport_ordinal == 1) is not (self.previous_receipt_tag is None):
                raise ValueError("capture transport receipt is inconsistent")
        elif any(
            value is not None
            for value in (
                self.transport_ordinal,
                self.previous_receipt_tag,
                self.receipt_tag,
            )
        ):
            raise ValueError("capture transport receipt is inconsistent")
        return self


def validate_capture_transport_chunk(value: object) -> CaptureTransportChunk:
    """Defensively revalidate one transport descriptor."""

    try:
        payload = (
            value.model_dump(mode="python", warnings="error")
            if type(value) is CaptureTransportChunk
            else value
        )
        return CaptureTransportChunk.model_validate(payload)
    except Exception:
        raise CaptureTransportError() from None


__all__ = [
    "MAX_CAPTURE_TRANSPORT_CHUNKS_PER_SESSION",
    "CaptureTransportChunk",
    "CaptureTransportDisposition",
    "CaptureTransportError",
    "CaptureTransportReceipt",
    "validate_capture_transport_chunk",
]
