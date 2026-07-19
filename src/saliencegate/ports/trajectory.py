from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum
from itertools import pairwise
from typing import Annotated, Literal, Self, TypeVar, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from saliencegate.domain import (
    EventType,
    PayloadDigest,
    TextSpan,
    TraceEvent,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.domain.records import UUID4, JsonPointer, PositiveSigned64Offset, Sha256Digest
from saliencegate.ports.repository import LedgerEntry, RunRepository

TRAJECTORY_BINDING_SCHEMA_VERSION: Literal["trajectory-binding/v1"] = "trajectory-binding/v1"
TRAJECTORY_PREFIX_REQUEST_SCHEMA_VERSION: Literal["trajectory-prefix-request/v1"] = (
    "trajectory-prefix-request/v1"
)
ATTESTED_TRAJECTORY_PREFIX_SCHEMA_VERSION: Literal["attested-trajectory-prefix/v1"] = (
    "attested-trajectory-prefix/v1"
)
MAX_TRAJECTORY_POINTER_SEGMENTS = 32
MAX_TRAJECTORY_POINTER_UTF8_BYTES = 1_024
MAX_LOGICAL_MESSAGES_PER_BINDING = 64
MAX_TRAJECTORY_EVENTS = 10_000

_BINDING_DIGEST_DOMAIN = "saliencegate:trajectory:binding:v1"
_PREFIX_REQUEST_DIGEST_DOMAIN = "saliencegate:trajectory:prefix-request:v1"
_PREFIX_DIGEST_DOMAIN = "saliencegate:trajectory:attested-prefix:v1"
_POINTER_INDEX = re.compile(r"^(?:0|[1-9][0-9]*)$")
_ModelT = TypeVar("_ModelT", bound=BaseModel)


class TrajectoryErrorCode(StrEnum):
    INVALID_INPUT = "invalid_input"
    DUPLICATE_BINDING = "duplicate_binding"
    RETROGRADE_BINDING = "retrograde_binding"
    MISSING_REFERENCE = "missing_reference"
    FUTURE_REFERENCE = "future_reference"
    CROSS_RUN_REFERENCE = "cross_run_reference"
    UNATTESTED_REFERENCE = "unattested_reference"
    INVALID_POINTER = "invalid_pointer"
    INVALID_SPAN = "invalid_span"
    LIMIT_EXCEEDED = "limit_exceeded"
    REPOSITORY_UNAVAILABLE = "repository_unavailable"


class TrajectoryError(ValueError):
    """A typed, value-free trajectory boundary failure."""

    def __init__(self, code: TrajectoryErrorCode) -> None:
        self.code = code
        super().__init__(f"trajectory projection failed: {code.value}")


class LogicalMessageRole(StrEnum):
    """A trajectory data label; it never grants provider instruction authority."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    CONTROLLER = "controller"


class _TrajectoryModel(BaseModel):
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


def _bounded_payload_pointer(value: str) -> str:
    try:
        encoded = value.encode("utf-8", errors="strict")
        segments = value.split("/")[1:]
    except Exception:  # pragma: no cover - exact Pydantic strings encode before this hook
        raise ValueError("trajectory pointer failed UTF-8 validation") from None
    if (
        not value.startswith("/payload/")
        or not segments
        or len(segments) > MAX_TRAJECTORY_POINTER_SEGMENTS
        or len(encoded) > MAX_TRAJECTORY_POINTER_UTF8_BYTES
    ):
        raise ValueError("trajectory pointer exceeds its bounded payload namespace")
    return value


class EventTextSelector(_TrajectoryModel):
    field_path: JsonPointer
    span: TextSpan | None = None

    _bound_field_path = field_validator("field_path")(_bounded_payload_pointer)

    @field_validator("span")
    @classmethod
    def span_is_an_exact_domain_record(cls, value: TextSpan | None) -> TextSpan | None:
        if value is not None and not _is_exact_model(value, TextSpan):
            raise ValueError("trajectory span failed exact validation")
        return value


class LogicalMessageBinding(_TrajectoryModel):
    role: LogicalMessageRole
    selector: EventTextSelector


class ActionStepBinding(_TrajectoryModel):
    field_path: JsonPointer

    _bound_field_path = field_validator("field_path")(_bounded_payload_pointer)


def _selector_identity(selector: EventTextSelector) -> tuple[str, int | None, int | None]:
    span = selector.span
    return (
        selector.field_path,
        None if span is None else span.start_byte,
        None if span is None else span.end_byte,
    )


def _binding_digest(values: Mapping[str, object]) -> str:
    return length_prefixed_sha256(
        canonical_json(
            {
                "schema_version": values["schema_version"],
                "run_id": str(values["run_id"]),
                "event_id": str(values["event_id"]),
                "event_sequence": values["event_sequence"],
                "ledger_position": values["ledger_position"],
                "payload_digest": values["payload_digest"],
                "record_tag": values["record_tag"],
                "chain_tag": values["chain_tag"],
                "task_description": values["task_description"],
                "logical_messages": values["logical_messages"],
                "action_step": values["action_step"],
            }
        ),
        domain=_BINDING_DIGEST_DOMAIN,
    )


class TrajectoryBinding(_TrajectoryModel):
    """Repository-attested selectors for one persisted, redacted trace event.

    A binding deliberately contains no task, message, or step value. Those values
    are resolved later from the exact ledger entry named by the integrity fields.
    The SHA-256 binding digest provides content addressing, not authentication;
    authentication remains the repository's trust boundary.
    """

    schema_version: Literal["trajectory-binding/v1"]
    run_id: UUID4
    event_id: UUID4
    event_sequence: PositiveSigned64Offset
    ledger_position: PositiveSigned64Offset
    payload_digest: PayloadDigest
    record_tag: PayloadDigest
    chain_tag: PayloadDigest
    task_description: EventTextSelector | None = None
    logical_messages: Annotated[
        tuple[LogicalMessageBinding, ...],
        Field(max_length=MAX_LOGICAL_MESSAGES_PER_BINDING),
    ] = ()
    action_step: ActionStepBinding | None = None
    binding_digest: Sha256Digest = Field(default_factory=_binding_digest)

    @model_validator(mode="after")
    def selectors_and_attestation_are_canonical(self) -> Self:
        if any(
            not _is_exact_model(value, PayloadDigest)
            for value in (self.payload_digest, self.record_tag, self.chain_tag)
        ):
            raise ValueError("trajectory integrity records failed exact validation")
        if self.event_sequence > self.ledger_position:
            raise ValueError("event sequence cannot exceed its ledger position")
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
            raise ValueError("trajectory integrity algorithms must match")
        message_selectors = tuple(
            _selector_identity(message.selector) for message in self.logical_messages
        )
        if len(set(message_selectors)) != len(message_selectors):
            raise ValueError("logical message selectors must be unique")
        reserved_paths: list[str] = []
        if self.task_description is not None:
            reserved_paths.append(self.task_description.field_path)
        if self.action_step is not None:
            reserved_paths.append(self.action_step.field_path)
        reserved_paths.extend(message.selector.field_path for message in self.logical_messages)
        if len(set(reserved_paths)) != len(reserved_paths):
            raise ValueError("trajectory selector paths must be unique")
        values = self.model_dump(mode="json", exclude={"binding_digest"})
        if self.binding_digest != _binding_digest(values):
            raise ValueError("trajectory binding digest does not match")
        return self


def bind_persisted_trajectory_event(
    entry: LedgerEntry,
    *,
    task_description: EventTextSelector | None = None,
    logical_messages: tuple[LogicalMessageBinding, ...] = (),
    action_step: ActionStepBinding | None = None,
) -> TrajectoryBinding:
    """Build selectors only from a repository-returned trace ledger entry.

    This factory copies the repository-owned integrity anchors. The caller must
    obtain ``entry`` from ``RunRepository.ledger``; the later resolver reads that
    ledger again and verifies the copied anchors before exposing any payload data.
    """

    try:
        if type(entry) is not LedgerEntry or type(entry.record) is not TraceEvent:
            raise TypeError
        validated_entry = LedgerEntry.model_validate_json(entry.model_dump_json(warnings=False))
        if validated_entry != entry:  # pragma: no cover - exact JSON round-trip invariant
            raise ValueError
        event = validated_entry.record
        assert type(event) is TraceEvent
        return TrajectoryBinding(
            schema_version=TRAJECTORY_BINDING_SCHEMA_VERSION,
            run_id=event.run_id,
            event_id=event.event_id,
            event_sequence=event.sequence,
            ledger_position=validated_entry.position,
            payload_digest=event.payload_digest,
            record_tag=validated_entry.record_tag,
            chain_tag=validated_entry.chain_tag,
            task_description=task_description,
            logical_messages=logical_messages,
            action_step=action_step,
        )
    except Exception:
        raise TrajectoryError(TrajectoryErrorCode.INVALID_INPUT) from None


def _prefix_request_digest(values: Mapping[str, object]) -> str:
    bindings = values["bindings"]
    if not isinstance(bindings, (tuple, list)):  # pragma: no cover - default-factory invariant
        raise ValueError("trajectory bindings must be a sequence")
    return length_prefixed_sha256(
        canonical_json(
            {
                "schema_version": values["schema_version"],
                "run_id": str(values["run_id"]),
                "boundary_event_sequence": values["boundary_event_sequence"],
                "binding_digests": [
                    (
                        binding.binding_digest
                        if isinstance(binding, TrajectoryBinding)
                        else binding["binding_digest"]
                    )
                    for binding in bindings
                ],
            }
        ),
        domain=_PREFIX_REQUEST_DIGEST_DOMAIN,
    )


class TrajectoryPrefixRequest(_TrajectoryModel):
    schema_version: Literal["trajectory-prefix-request/v1"]
    run_id: UUID4
    boundary_event_sequence: Annotated[int, Field(ge=1, le=MAX_TRAJECTORY_EVENTS)]
    bindings: Annotated[
        tuple[TrajectoryBinding, ...],
        Field(min_length=1, max_length=MAX_TRAJECTORY_EVENTS, repr=False),
    ]
    request_digest: Sha256Digest = Field(default_factory=_prefix_request_digest)

    @model_validator(mode="after")
    def bindings_cover_one_exact_prefix(self) -> Self:
        if any(binding.run_id != self.run_id for binding in self.bindings):
            raise ValueError("trajectory bindings belong to another run")
        if len(self.bindings) != self.boundary_event_sequence or any(
            binding.event_sequence != expected
            for expected, binding in enumerate(self.bindings, start=1)
        ):
            raise ValueError("trajectory bindings must cover one ordered event prefix")
        positions = tuple(binding.ledger_position for binding in self.bindings)
        if any(later <= earlier for earlier, later in pairwise(positions)):
            raise ValueError("trajectory ledger positions must be strictly increasing")
        event_ids = tuple(binding.event_id for binding in self.bindings)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("trajectory bindings must identify unique events")
        task_sources = tuple(
            binding.event_sequence
            for binding in self.bindings
            if binding.task_description is not None
        )
        if task_sources != (1,):
            raise ValueError("the run task must be bound exactly once at run start")
        values = self.model_dump(mode="json", exclude={"request_digest"})
        if self.request_digest != _prefix_request_digest(values):
            raise ValueError("trajectory prefix request digest does not match")
        return self


class AttestedTrajectoryEvent(_TrajectoryModel):
    event: TraceEvent = Field(repr=False)
    binding: TrajectoryBinding

    @model_validator(mode="after")
    def binding_names_the_event(self) -> Self:
        if not _is_exact_model(self.event, TraceEvent):
            raise ValueError("trajectory event failed exact validation")
        if (
            self.binding.run_id != self.event.run_id
            or self.binding.event_id != self.event.event_id
            or self.binding.event_sequence != self.event.sequence
            or self.binding.payload_digest != self.event.payload_digest
        ):
            raise ValueError("trajectory binding does not identify its event")
        return self


def _attested_prefix_digest(values: Mapping[str, object]) -> str:
    items = values["items"]
    if not isinstance(items, (tuple, list)):  # pragma: no cover - default-factory invariant
        raise ValueError("trajectory items must be a sequence")
    return length_prefixed_sha256(
        canonical_json(
            {
                "schema_version": values["schema_version"],
                "run_id": str(values["run_id"]),
                "boundary_event_sequence": values["boundary_event_sequence"],
                "request_digest": values["request_digest"],
                "event_attestations": [
                    {
                        "event_id": str(
                            item.event.event_id
                            if isinstance(item, AttestedTrajectoryEvent)
                            else item["event"]["event_id"]
                        ),
                        "event_sequence": (
                            item.event.sequence
                            if isinstance(item, AttestedTrajectoryEvent)
                            else item["event"]["sequence"]
                        ),
                        "binding_digest": (
                            item.binding.binding_digest
                            if isinstance(item, AttestedTrajectoryEvent)
                            else item["binding"]["binding_digest"]
                        ),
                    }
                    for item in items
                ],
            }
        ),
        domain=_PREFIX_DIGEST_DOMAIN,
    )


class AttestedTrajectoryPrefix(_TrajectoryModel):
    """A repository-resolved prefix claim that must be reverified before projection."""

    schema_version: Literal["attested-trajectory-prefix/v1"]
    run_id: UUID4
    boundary_event_sequence: Annotated[int, Field(ge=1, le=MAX_TRAJECTORY_EVENTS)]
    request_digest: Sha256Digest
    items: Annotated[
        tuple[AttestedTrajectoryEvent, ...],
        Field(min_length=1, max_length=MAX_TRAJECTORY_EVENTS, repr=False),
    ]
    prefix_digest: Sha256Digest = Field(default_factory=_attested_prefix_digest)

    @model_validator(mode="after")
    def items_form_the_requested_run_prefix(self) -> Self:
        if len(self.items) != self.boundary_event_sequence:
            raise ValueError("attested prefix length does not match its boundary")
        if any(item.event.run_id != self.run_id for item in self.items):
            raise ValueError("attested prefix contains a cross-run event")
        if any(
            item.event.sequence != expected for expected, item in enumerate(self.items, start=1)
        ):
            raise ValueError("attested events do not form one contiguous prefix")
        if self.items[0].event.event_type is not EventType.RUN_START:
            raise ValueError("the trajectory prefix must begin with run_start")
        values = self.model_dump(mode="json", exclude={"prefix_digest"})
        if self.prefix_digest != _attested_prefix_digest(values):
            raise ValueError("attested trajectory prefix digest does not match")
        return self


def _preflight_request(value: object) -> TrajectoryErrorCode | None:
    try:
        if type(value) is not TrajectoryPrefixRequest:
            return TrajectoryErrorCode.INVALID_INPUT
        if type(value.run_id) is not UUID or value.run_id.version != 4:
            return TrajectoryErrorCode.INVALID_INPUT
        if (
            type(value.boundary_event_sequence) is not int
            or not 1 <= value.boundary_event_sequence <= MAX_TRAJECTORY_EVENTS
            or type(value.bindings) is not tuple
        ):
            return TrajectoryErrorCode.LIMIT_EXCEEDED
        bindings = value.bindings
        if len(bindings) > MAX_TRAJECTORY_EVENTS:
            return TrajectoryErrorCode.LIMIT_EXCEEDED
        if any(type(binding) is not TrajectoryBinding for binding in bindings):
            return TrajectoryErrorCode.INVALID_INPUT
        if any(binding.run_id != value.run_id for binding in bindings):
            return TrajectoryErrorCode.CROSS_RUN_REFERENCE
        sequences = tuple(binding.event_sequence for binding in bindings)
        if len(set(sequences)) != len(sequences) or len(
            {binding.event_id for binding in bindings}
        ) != len(bindings):
            return TrajectoryErrorCode.DUPLICATE_BINDING
        if any(sequence > value.boundary_event_sequence for sequence in sequences):
            return TrajectoryErrorCode.FUTURE_REFERENCE
        if any(later < earlier for earlier, later in pairwise(sequences)):
            return TrajectoryErrorCode.RETROGRADE_BINDING
        expected = tuple(range(1, value.boundary_event_sequence + 1))
        if sequences != expected:
            return TrajectoryErrorCode.MISSING_REFERENCE
        positions = tuple(binding.ledger_position for binding in bindings)
        if len(set(positions)) != len(positions):
            return TrajectoryErrorCode.DUPLICATE_BINDING
        if any(later < earlier for earlier, later in pairwise(positions)):
            return TrajectoryErrorCode.RETROGRADE_BINDING
    except Exception:  # pragma: no cover - guarded exact fields are non-throwing
        return TrajectoryErrorCode.INVALID_INPUT
    return None


async def resolve_trajectory_prefix(
    repository: RunRepository,
    request: TrajectoryPrefixRequest,
) -> AttestedTrajectoryPrefix:
    """Resolve selectors against a freshly verified authoritative ledger snapshot."""

    failure = _preflight_request(request)
    if failure is not None:
        raise TrajectoryError(failure)
    try:
        entries = await repository.ledger(request.run_id)
    except Exception:
        raise TrajectoryError(TrajectoryErrorCode.REPOSITORY_UNAVAILABLE) from None
    try:
        if type(entries) is not tuple or any(
            not _is_exact_model(entry, LedgerEntry) for entry in entries
        ):
            raise TypeError
        event_entries = tuple(
            entry
            for entry in entries
            if type(entry.record) is TraceEvent
            and entry.record.sequence <= request.boundary_event_sequence
        )
    except Exception:
        raise TrajectoryError(TrajectoryErrorCode.REPOSITORY_UNAVAILABLE) from None
    if len(event_entries) != request.boundary_event_sequence:
        raise TrajectoryError(TrajectoryErrorCode.MISSING_REFERENCE)

    items: list[AttestedTrajectoryEvent] = []
    for expected_sequence, (entry, binding) in enumerate(
        zip(event_entries, request.bindings, strict=True), start=1
    ):
        event = entry.record
        if type(event) is not TraceEvent or event.sequence != expected_sequence:
            raise TrajectoryError(TrajectoryErrorCode.MISSING_REFERENCE)
        if (  # pragma: no cover - exact ledger and request preflight establish one run
            event.run_id != request.run_id or binding.run_id != request.run_id
        ):
            raise TrajectoryError(TrajectoryErrorCode.CROSS_RUN_REFERENCE)
        if (
            binding.event_id != event.event_id
            or binding.event_sequence != event.sequence
            or binding.ledger_position != entry.position
            or binding.payload_digest != event.payload_digest
            or binding.record_tag != entry.record_tag
            or binding.chain_tag != entry.chain_tag
        ):
            raise TrajectoryError(TrajectoryErrorCode.UNATTESTED_REFERENCE)
        try:
            validated_binding = TrajectoryBinding.model_validate_json(
                binding.model_dump_json(warnings=False)
            )
        except Exception:
            raise TrajectoryError(TrajectoryErrorCode.UNATTESTED_REFERENCE) from None
        if validated_binding != binding:  # pragma: no cover - exact JSON round-trip invariant
            raise TrajectoryError(TrajectoryErrorCode.UNATTESTED_REFERENCE)
        items.append(AttestedTrajectoryEvent(event=event, binding=validated_binding))

    task_sources = tuple(
        item.event.sequence for item in items if item.binding.task_description is not None
    )
    if task_sources != (1,):
        code = (
            TrajectoryErrorCode.MISSING_REFERENCE
            if not task_sources
            else TrajectoryErrorCode.DUPLICATE_BINDING
        )
        raise TrajectoryError(code)
    try:
        validated_request = TrajectoryPrefixRequest.model_validate_json(
            request.model_dump_json(warnings=False)
        )
        if validated_request != request:  # pragma: no cover - exact JSON round-trip invariant
            raise ValueError
        return AttestedTrajectoryPrefix(
            schema_version=ATTESTED_TRAJECTORY_PREFIX_SCHEMA_VERSION,
            run_id=request.run_id,
            boundary_event_sequence=request.boundary_event_sequence,
            request_digest=request.request_digest,
            items=tuple(items),
        )
    except Exception:
        raise TrajectoryError(TrajectoryErrorCode.INVALID_INPUT) from None


def _validated_attested_trajectory_prefix_structure(
    value: object,
) -> AttestedTrajectoryPrefix:
    try:
        if type(value) is not AttestedTrajectoryPrefix:
            raise TypeError
        validated = AttestedTrajectoryPrefix.model_validate_json(
            value.model_dump_json(warnings=False)
        )
        if validated != value:  # pragma: no cover - exact JSON round-trip invariant
            raise ValueError
        return validated
    except Exception:
        raise TrajectoryError(TrajectoryErrorCode.INVALID_INPUT) from None


async def verify_attested_trajectory_prefix(
    repository: RunRepository,
    value: object,
) -> AttestedTrajectoryPrefix:
    """Re-resolve a prefix from the repository before any payload is consumed.

    Prefix and binding digests provide stable content identities but are not
    authenticators. This verification step reconstructs the original request,
    asks the repository for a freshly verified ledger snapshot, and requires the
    supplied prefix to be byte-exact with that authoritative reconstruction.
    """

    structural = _validated_attested_trajectory_prefix_structure(value)
    try:
        request = TrajectoryPrefixRequest(
            schema_version=TRAJECTORY_PREFIX_REQUEST_SCHEMA_VERSION,
            run_id=structural.run_id,
            boundary_event_sequence=structural.boundary_event_sequence,
            bindings=tuple(item.binding for item in structural.items),
            request_digest=structural.request_digest,
        )
    except Exception:
        raise TrajectoryError(TrajectoryErrorCode.UNATTESTED_REFERENCE) from None
    authoritative = await resolve_trajectory_prefix(repository, request)
    if canonical_json(structural) != canonical_json(authoritative):
        raise TrajectoryError(TrajectoryErrorCode.UNATTESTED_REFERENCE)
    return authoritative


def _resolve_attested_payload_value(
    item: AttestedTrajectoryEvent,
    field_path: str,
) -> object:
    """Resolve one bounded RFC 6901 pointer from an attested redacted event."""

    try:
        if type(item) is not AttestedTrajectoryEvent or type(field_path) is not str:
            raise TypeError
        field_path = _bounded_payload_pointer(field_path)
        segments = tuple(
            segment.replace("~1", "/").replace("~0", "~") for segment in field_path.split("/")[1:]
        )
        value: object = {"payload": item.event.payload}
        for segment in segments:
            if isinstance(value, Mapping):
                if segment not in value:
                    raise KeyError
                value = value[segment]
            elif type(value) in (list, tuple):
                assert isinstance(value, (list, tuple))
                if _POINTER_INDEX.fullmatch(segment) is None:
                    raise KeyError
                index = int(segment)
                if index >= len(value):
                    raise KeyError
                value = value[index]
            else:
                raise KeyError
        return value
    except Exception:
        raise TrajectoryError(TrajectoryErrorCode.INVALID_POINTER) from None


__all__ = [
    "ATTESTED_TRAJECTORY_PREFIX_SCHEMA_VERSION",
    "MAX_LOGICAL_MESSAGES_PER_BINDING",
    "MAX_TRAJECTORY_EVENTS",
    "MAX_TRAJECTORY_POINTER_SEGMENTS",
    "MAX_TRAJECTORY_POINTER_UTF8_BYTES",
    "TRAJECTORY_BINDING_SCHEMA_VERSION",
    "TRAJECTORY_PREFIX_REQUEST_SCHEMA_VERSION",
    "ActionStepBinding",
    "AttestedTrajectoryEvent",
    "AttestedTrajectoryPrefix",
    "EventTextSelector",
    "LogicalMessageBinding",
    "LogicalMessageRole",
    "TrajectoryBinding",
    "TrajectoryError",
    "TrajectoryErrorCode",
    "TrajectoryPrefixRequest",
    "bind_persisted_trajectory_event",
    "resolve_trajectory_prefix",
    "verify_attested_trajectory_prefix",
]
