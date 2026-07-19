from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Annotated, Literal, Self, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.domain import (
    EvidenceReference,
    EvidenceSource,
    PayloadDigest,
    TrustLabel,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.domain.records import UUID4, PositiveSigned64Offset, Sha256Digest
from saliencegate.ports.repository import RunRepository
from saliencegate.ports.trajectory import (
    AttestedTrajectoryEvent,
    AttestedTrajectoryPrefix,
    EventTextSelector,
    LogicalMessageRole,
    TrajectoryError,
    TrajectoryErrorCode,
    _resolve_attested_payload_value,
    verify_attested_trajectory_prefix,
)

MESSAGE_WINDOW_VERSION: Literal["latest-eight-logical-messages/v1"] = (
    "latest-eight-logical-messages/v1"
)
TASK_DESCRIPTION_VERSION: Literal["attested-task-description/v1"] = "attested-task-description/v1"
MAX_MESSAGE_WINDOW_ITEMS = 8
MAX_MESSAGE_WINDOW_CANONICAL_BYTES = 32_000
MAX_TASK_DESCRIPTION_UTF8_BYTES = 32_000

_TASK_DIGEST_DOMAIN = "saliencegate:trajectory:task-description:v1"
_WINDOW_DIGEST_DOMAIN = "saliencegate:trajectory:message-window:latest-eight-logical-messages:v1"
_ModelT = TypeVar("_ModelT", bound=BaseModel)


class MessageWindowError(TrajectoryError):
    """A value-free failure specific to the provider-visible message window."""


class _WindowModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


def _is_exact_model(value: object, model: type[_ModelT]) -> bool:
    try:
        if type(value) is not model:  # pragma: no cover - typed fields reject before validators
            return False
        serialized = cast(BaseModel, value).model_dump_json(warnings=False)
        return model.model_validate_json(serialized) == value
    except Exception:
        return False


class TrajectoryTextSource(_WindowModel):
    evidence: EvidenceReference
    event_sequence: PositiveSigned64Offset
    ledger_position: PositiveSigned64Offset
    trust_label: TrustLabel
    payload_digest: PayloadDigest
    record_tag: PayloadDigest
    chain_tag: PayloadDigest
    binding_digest: Sha256Digest

    @model_validator(mode="after")
    def provenance_is_one_event_attestation(self) -> Self:
        if not _is_exact_model(self.evidence, EvidenceReference) or any(
            not _is_exact_model(value, PayloadDigest)
            for value in (self.payload_digest, self.record_tag, self.chain_tag)
        ):
            raise ValueError("trajectory text provenance failed exact validation")
        if self.evidence.source is not EvidenceSource.EVENT or self.evidence.revision is not None:
            raise ValueError("trajectory text provenance must identify an event")
        if (
            len(
                {
                    self.payload_digest.algorithm,
                    self.record_tag.algorithm,
                    self.chain_tag.algorithm,
                }
            )
            != 1
        ):
            raise ValueError("trajectory source integrity algorithms must match")
        return self


def _task_digest(values: Mapping[str, object]) -> str:
    return length_prefixed_sha256(
        canonical_json(
            {
                "version": values["version"],
                "content": values["content"],
                "source": values["source"],
            }
        ),
        domain=_TASK_DIGEST_DOMAIN,
    )


class AttestedTaskDescription(_WindowModel):
    version: Literal["attested-task-description/v1"]
    content: Annotated[str, Field(min_length=1, repr=False)]
    source: TrajectoryTextSource
    task_digest: Sha256Digest = Field(default_factory=_task_digest)

    @model_validator(mode="after")
    def content_and_digest_are_bounded(self) -> Self:
        try:
            size = len(self.content.encode("utf-8", errors="strict"))
        except Exception:  # pragma: no cover - exact Pydantic strings encode before this hook
            raise ValueError("task description failed UTF-8 validation") from None
        if size > MAX_TASK_DESCRIPTION_UTF8_BYTES:
            raise ValueError("task description exceeds its byte ceiling")
        values = self.model_dump(mode="json", exclude={"task_digest"})
        if self.task_digest != _task_digest(values):
            raise ValueError("task description digest does not match")
        return self


class MessageWindowMessage(_WindowModel):
    """One provider-visible logical message with non-authoritative role metadata."""

    role: LogicalMessageRole
    content: Annotated[str, Field(min_length=1, repr=False)]
    evidence: EvidenceReference
    trust_label: TrustLabel

    @model_validator(mode="after")
    def evidence_is_an_event_pointer(self) -> Self:
        if not _is_exact_model(self.evidence, EvidenceReference):
            raise ValueError("logical message evidence failed exact validation")
        if self.evidence.source is not EvidenceSource.EVENT or self.evidence.revision is not None:
            raise ValueError("logical message evidence must identify an event")
        try:
            self.content.encode("utf-8", errors="strict")
        except Exception:  # pragma: no cover - exact Pydantic strings encode before this hook
            raise ValueError("logical message failed UTF-8 validation") from None
        return self


class MessageWindowPayload(_WindowModel):
    """The exact bounded data payload consumed by the fixed-step prompt builder."""

    version: Literal["latest-eight-logical-messages/v1"]
    messages: Annotated[
        tuple[MessageWindowMessage, ...],
        Field(max_length=MAX_MESSAGE_WINDOW_ITEMS, repr=False),
    ] = ()


def _window_digest(values: Mapping[str, object]) -> str:
    return length_prefixed_sha256(
        canonical_json(
            {
                "version": values["version"],
                "run_id": str(values["run_id"]),
                "boundary_event_id": str(values["boundary_event_id"]),
                "boundary_event_sequence": values["boundary_event_sequence"],
                "boundary_ledger_position": values["boundary_ledger_position"],
                "boundary_chain_tag": values["boundary_chain_tag"],
                "trajectory_prefix_digest": values["trajectory_prefix_digest"],
                "task_description": values["task_description"],
                "payload": values["payload"],
                "payload_canonical_utf8_bytes": values["payload_canonical_utf8_bytes"],
                "source_attestations": values["source_attestations"],
            }
        ),
        domain=_WINDOW_DIGEST_DOMAIN,
    )


class MessageWindow(_WindowModel):
    """Latest logical messages plus separate task and authoritative provenance."""

    version: Literal["latest-eight-logical-messages/v1"]
    run_id: UUID4
    boundary_event_id: UUID4
    boundary_event_sequence: PositiveSigned64Offset
    boundary_ledger_position: PositiveSigned64Offset
    boundary_chain_tag: PayloadDigest
    trajectory_prefix_digest: Sha256Digest
    task_description: AttestedTaskDescription = Field(repr=False)
    payload: MessageWindowPayload = Field(repr=False)
    payload_canonical_utf8_bytes: Annotated[int, Field(ge=0, le=MAX_MESSAGE_WINDOW_CANONICAL_BYTES)]
    source_attestations: Annotated[
        tuple[TrajectoryTextSource, ...],
        Field(max_length=MAX_MESSAGE_WINDOW_ITEMS, repr=False),
    ] = ()
    window_digest: Sha256Digest = Field(default_factory=_window_digest)

    @model_validator(mode="after")
    def payload_and_attestations_match_exactly(self) -> Self:
        if not _is_exact_model(self.boundary_chain_tag, PayloadDigest):
            raise ValueError("message window boundary tag failed exact validation")
        if len(self.payload.messages) != len(self.source_attestations):
            raise ValueError("every logical message requires one source attestation")
        for message, source in zip(self.payload.messages, self.source_attestations, strict=True):
            if message.evidence != source.evidence or message.trust_label is not source.trust_label:
                raise ValueError("logical message and source attestation differ")
        size = len(canonical_json(self.payload))
        if size != self.payload_canonical_utf8_bytes or size > MAX_MESSAGE_WINDOW_CANONICAL_BYTES:
            raise ValueError("message window payload exceeds its canonical byte ceiling")
        values = self.model_dump(mode="json", exclude={"window_digest"})
        if self.window_digest != _window_digest(values):
            raise ValueError("message window digest does not match")
        return self


def _source(
    item: AttestedTrajectoryEvent,
    selector: EventTextSelector,
) -> TrajectoryTextSource:
    return TrajectoryTextSource(
        evidence=EvidenceReference(
            source=EvidenceSource.EVENT,
            source_id=item.event.event_id,
            field_path=selector.field_path,
            span=selector.span,
        ),
        event_sequence=item.event.sequence,
        ledger_position=item.binding.ledger_position,
        trust_label=item.event.trust_label,
        payload_digest=item.event.payload_digest,
        record_tag=item.binding.record_tag,
        chain_tag=item.binding.chain_tag,
        binding_digest=item.binding.binding_digest,
    )


def _selected_text(
    item: AttestedTrajectoryEvent,
    selector: EventTextSelector,
) -> str:
    value = _resolve_attested_payload_value(item, selector.field_path)
    if type(value) is not str or not value:
        raise TrajectoryError(TrajectoryErrorCode.INVALID_POINTER)
    try:
        encoded = value.encode("utf-8", errors="strict")
    except Exception:  # pragma: no cover - persisted JSON strings are UTF-8 bounded
        raise TrajectoryError(TrajectoryErrorCode.INVALID_POINTER) from None
    if selector.span is None:
        return value
    span = selector.span
    if span.end_byte > len(encoded):
        raise TrajectoryError(TrajectoryErrorCode.INVALID_SPAN)
    try:
        selected = encoded[span.start_byte : span.end_byte].decode("utf-8", errors="strict")
    except Exception:
        raise TrajectoryError(TrajectoryErrorCode.INVALID_SPAN) from None
    return selected


def _project_verified_message_window(
    validated: AttestedTrajectoryPrefix,
) -> MessageWindow:
    """Project only text from a repository-verified prefix."""

    task_item = validated.items[0]
    task_selector = task_item.binding.task_description
    if task_selector is None:  # pragma: no cover - prefix invariant
        raise TrajectoryError(TrajectoryErrorCode.MISSING_REFERENCE)
    task_content = _selected_text(task_item, task_selector)
    task_source = _source(task_item, task_selector)
    try:
        task = AttestedTaskDescription(
            version=TASK_DESCRIPTION_VERSION,
            content=task_content,
            source=task_source,
        )
    except Exception:
        raise MessageWindowError(TrajectoryErrorCode.LIMIT_EXCEEDED) from None

    selected_messages: deque[tuple[MessageWindowMessage, TrajectoryTextSource]] = deque(
        maxlen=MAX_MESSAGE_WINDOW_ITEMS
    )
    for item in validated.items:
        for binding in item.binding.logical_messages:
            content = _selected_text(item, binding.selector)
            source = _source(item, binding.selector)
            message = MessageWindowMessage(
                role=binding.role,
                content=content,
                evidence=source.evidence,
                trust_label=source.trust_label,
            )
            selected_messages.append((message, source))

    messages = tuple(message for message, _source_item in selected_messages)
    sources = tuple(source for _message, source in selected_messages)
    payload = MessageWindowPayload(version=MESSAGE_WINDOW_VERSION, messages=messages)
    payload_size = len(canonical_json(payload))
    if payload_size > MAX_MESSAGE_WINDOW_CANONICAL_BYTES:
        raise MessageWindowError(TrajectoryErrorCode.LIMIT_EXCEEDED)
    boundary = validated.items[-1]
    try:
        return MessageWindow(
            version=MESSAGE_WINDOW_VERSION,
            run_id=validated.run_id,
            boundary_event_id=boundary.event.event_id,
            boundary_event_sequence=boundary.event.sequence,
            boundary_ledger_position=boundary.binding.ledger_position,
            boundary_chain_tag=boundary.binding.chain_tag,
            trajectory_prefix_digest=validated.prefix_digest,
            task_description=task,
            payload=payload,
            payload_canonical_utf8_bytes=payload_size,
            source_attestations=sources,
        )
    except Exception:  # pragma: no cover - constructed entirely from validated values
        raise TrajectoryError(TrajectoryErrorCode.INVALID_INPUT) from None


async def project_message_window(
    repository: RunRepository,
    prefix: AttestedTrajectoryPrefix,
) -> MessageWindow:
    """Verify against the ledger, then resolve the latest eight logical messages."""

    validated = await verify_attested_trajectory_prefix(repository, prefix)
    return _project_verified_message_window(validated)


async def validated_message_window_for_prefix(
    repository: RunRepository,
    prefix: AttestedTrajectoryPrefix,
    value: object,
) -> MessageWindow:
    """Reproject and require byte-equivalent window output."""

    try:
        if type(value) is not MessageWindow:
            raise TypeError
        validated = MessageWindow.model_validate_json(value.model_dump_json(warnings=False))
        expected = await project_message_window(repository, prefix)
        if validated != value or canonical_json(validated) != canonical_json(expected):
            raise ValueError
        return validated
    except TrajectoryError:
        raise
    except Exception:
        raise TrajectoryError(TrajectoryErrorCode.UNATTESTED_REFERENCE) from None


__all__ = [
    "MAX_MESSAGE_WINDOW_CANONICAL_BYTES",
    "MAX_MESSAGE_WINDOW_ITEMS",
    "MAX_TASK_DESCRIPTION_UTF8_BYTES",
    "MESSAGE_WINDOW_VERSION",
    "TASK_DESCRIPTION_VERSION",
    "AttestedTaskDescription",
    "MessageWindow",
    "MessageWindowError",
    "MessageWindowMessage",
    "MessageWindowPayload",
    "TrajectoryTextSource",
    "project_message_window",
    "validated_message_window_for_prefix",
]
