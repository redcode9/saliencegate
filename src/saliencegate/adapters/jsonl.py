from __future__ import annotations

import hmac
import json
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from os import PathLike
from typing import Annotated, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.domain import (
    DeliveryTarget,
    NormalizedTraceEventDraft,
    canonical_digest,
    canonical_json,
    validate_normalized_trace_event_draft,
)
from saliencegate.ports.adapters import AdapterNormalizationError, AdapterTargetResolutionError
from saliencegate.ports.repository import (
    UUID4,
    ComponentIdentifier,
    PositiveInt,
    Sha256Digest,
)
from saliencegate.security import (
    SecureFileBoundError,
    SecureFileError,
    StableReadPolicy,
    read_stable_file,
)

JSONL_SCHEMA_VERSION: Literal["1.0"] = "1.0"
MAX_TRACE_BYTES = 8 * 1024 * 1024
MAX_LINE_BYTES = 256 * 1024
MAX_TRACE_LINES = 100_001

_SCHEMA_VERSION = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


class JsonlReplayError(ValueError):
    """Stable parse failure that never includes source bytes or callback text."""

    def __init__(self, code: str, *, line_number: int | None = None) -> None:
        self.code = code
        self.line_number = line_number
        location = "" if line_number is None else f" at line {line_number}"
        super().__init__(f"invalid JSONL replay: {code}{location}")


class _ReplayModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class JsonlTraceManifest(_ReplayModel):
    record_type: Literal["trace_manifest"] = "trace_manifest"
    schema_version: Literal["1.0"] = JSONL_SCHEMA_VERSION
    run_id: UUID4
    record_count: Annotated[int, Field(ge=1, le=MAX_TRACE_LINES - 1)]
    trace_digest: Sha256Digest


class JsonlReplayEvent(_ReplayModel):
    record_type: Literal["trace_event_draft"] = "trace_event_draft"
    schema_version: Literal["1.0"] = JSONL_SCHEMA_VERSION
    ordinal: PositiveInt
    expected_event_id: UUID4
    draft: NormalizedTraceEventDraft
    next_model_call_target_request_id: ComponentIdentifier | None = None
    pre_action_target_request_id: ComponentIdentifier | None = None
    record_digest: Sha256Digest

    @model_validator(mode="after")
    def digest_matches_canonical_record(self) -> Self:
        expected = _record_digest(self)
        if not hmac.compare_digest(self.record_digest, expected):
            raise ValueError("record digest mismatch")
        return self

    @classmethod
    def create(
        cls,
        *,
        ordinal: int,
        expected_event_id: UUID4,
        draft: object,
        next_model_call_target_request_id: str | None = None,
        pre_action_target_request_id: str | None = None,
    ) -> JsonlReplayEvent:
        normalized = _copy_draft(draft)
        body: dict[str, object] = {
            "record_type": "trace_event_draft",
            "schema_version": JSONL_SCHEMA_VERSION,
            "ordinal": ordinal,
            "expected_event_id": str(expected_event_id),
            "draft": normalized.model_dump(mode="json", warnings=False),
            "next_model_call_target_request_id": next_model_call_target_request_id,
            "pre_action_target_request_id": pre_action_target_request_id,
        }
        try:
            return cls.model_validate_json(
                canonical_json({**body, "record_digest": canonical_digest(body)})
            )
        except Exception:
            raise JsonlReplayError("invalid_record") from None


def _record_body(event: JsonlReplayEvent) -> dict[str, object]:
    return event.model_dump(mode="json", exclude={"record_digest"}, warnings=False)


def _record_digest(event: JsonlReplayEvent) -> str:
    return canonical_digest(_record_body(event))


def _trace_digest(
    *,
    schema_version: str,
    run_id: UUID4,
    record_digests: Sequence[str],
) -> str:
    return canonical_digest(
        {
            "schema_version": schema_version,
            "run_id": str(run_id),
            "record_digests": tuple(record_digests),
        }
    )


def _copy_draft(value: object) -> NormalizedTraceEventDraft:
    validated: NormalizedTraceEventDraft | None = None
    try:
        validated = validate_normalized_trace_event_draft(value)
    except Exception:
        try:
            validated = NormalizedTraceEventDraft.model_validate_json(canonical_json(value))
        except Exception:
            raise AdapterNormalizationError() from None
    try:
        return NormalizedTraceEventDraft.model_validate_json(
            validated.model_dump_json(warnings=False)
        )
    except Exception:
        raise AdapterNormalizationError() from None


def _copy_event(value: object) -> JsonlReplayEvent:
    if type(value) is not JsonlReplayEvent:
        raise JsonlReplayError("invalid_record")
    try:
        return JsonlReplayEvent.model_validate_json(value.model_dump_json(warnings=False))
    except Exception:
        raise JsonlReplayError("invalid_record") from None


def _copy_manifest(value: object) -> JsonlTraceManifest:
    if type(value) is not JsonlTraceManifest:
        raise JsonlReplayError("invalid_manifest")
    try:
        return JsonlTraceManifest.model_validate_json(value.model_dump_json(warnings=False))
    except Exception:
        raise JsonlReplayError("invalid_manifest") from None


def _copy_bounded_events(
    values: object,
) -> tuple[tuple[JsonlReplayEvent, ...], tuple[bytes, ...]]:
    """Copy an untrusted iterable without allowing it to grow memory past trace limits."""

    try:
        iterator = iter(values)  # type: ignore[call-overload]
    except Exception:
        raise JsonlReplayError("invalid_record") from None

    copied: list[JsonlReplayEvent] = []
    encoded: list[bytes] = []
    cumulative_bytes = 0
    while True:
        try:
            candidate = next(iterator)
        except StopIteration:
            break
        except Exception:
            raise JsonlReplayError("invalid_record") from None
        if len(copied) >= MAX_TRACE_LINES - 1:
            raise JsonlReplayError("too_many_lines")
        try:
            event = _copy_event(candidate)
            line = canonical_json(event)
        except Exception:
            raise JsonlReplayError("invalid_record") from None
        if len(line) > MAX_LINE_BYTES:
            raise JsonlReplayError("line_too_large", line_number=len(copied) + 2)
        cumulative_bytes += len(line) + 1
        if cumulative_bytes > MAX_TRACE_BYTES:
            raise JsonlReplayError("trace_too_large")
        copied.append(event)
        encoded.append(line)
    return tuple(copied), tuple(encoded)


def _validate_encoded_bound(manifest: JsonlTraceManifest, event_lines: tuple[bytes, ...]) -> bytes:
    try:
        manifest_line = canonical_json(manifest)
    except Exception:
        raise JsonlReplayError("invalid_manifest") from None
    if len(manifest_line) > MAX_LINE_BYTES:
        raise JsonlReplayError("line_too_large", line_number=1)
    encoded = b"\n".join((manifest_line, *event_lines)) + b"\n"
    if len(encoded) > MAX_TRACE_BYTES:
        raise JsonlReplayError("trace_too_large")
    return encoded


def _read_regular_trace(value: object) -> bytes:
    if type(value) is not str and not isinstance(value, PathLike):
        raise JsonlReplayError("invalid_input_type")

    try:
        stable = read_stable_file(
            value,
            maximum_bytes=MAX_TRACE_BYTES,
            policy=StableReadPolicy.LEGACY_COMPATIBILITY,
        )
    except SecureFileBoundError:
        raise JsonlReplayError("trace_too_large") from None
    except SecureFileError:
        raise JsonlReplayError("read_failed") from None
    except Exception:
        raise JsonlReplayError("read_failed") from None
    return stable.data


def _preflight(
    manifest: JsonlTraceManifest,
    events: tuple[JsonlReplayEvent, ...],
) -> None:
    if manifest.record_count != len(events):
        raise JsonlReplayError("record_count_mismatch")

    seen_event_ids: set[UUID4] = set()
    seen_source_ids: set[str] = set()
    previous_timestamp = None
    for expected_ordinal, event in enumerate(events, start=1):
        if event.ordinal != expected_ordinal:
            raise JsonlReplayError("non_contiguous_ordinal", line_number=expected_ordinal + 1)
        if event.draft.run_id != manifest.run_id:
            raise JsonlReplayError("multiple_runs", line_number=expected_ordinal + 1)
        if event.expected_event_id in seen_event_ids:
            raise JsonlReplayError("duplicate_expected_event_id", line_number=expected_ordinal + 1)
        if event.draft.source_event_id in seen_source_ids:
            raise JsonlReplayError("duplicate_source_event_id", line_number=expected_ordinal + 1)
        if any(parent_id not in seen_event_ids for parent_id in event.draft.parent_ids):
            raise JsonlReplayError("parent_not_preceding", line_number=expected_ordinal + 1)
        if previous_timestamp is not None and event.draft.timestamp < previous_timestamp:
            raise JsonlReplayError(
                "non_monotonic_timestamp",
                line_number=expected_ordinal + 1,
            )
        seen_event_ids.add(event.expected_event_id)
        seen_source_ids.add(event.draft.source_event_id)
        previous_timestamp = event.draft.timestamp

    expected_digest = _trace_digest(
        schema_version=manifest.schema_version,
        run_id=manifest.run_id,
        record_digests=tuple(event.record_digest for event in events),
    )
    if not hmac.compare_digest(manifest.trace_digest, expected_digest):
        raise JsonlReplayError("trace_digest_mismatch", line_number=1)


class _InvalidJsonError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidJsonError()
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise _InvalidJsonError()


def _parse_line(line: str, *, line_number: int) -> dict[str, object]:
    if not line:
        raise JsonlReplayError("blank_line", line_number=line_number)
    try:
        value = json.loads(
            line,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, _InvalidJsonError, RecursionError, ValueError):
        raise JsonlReplayError("invalid_json", line_number=line_number) from None
    if type(value) is not dict:
        raise JsonlReplayError("object_required", line_number=line_number)
    return cast(dict[str, object], value)


def _require_schema(payload: Mapping[str, object], *, line_number: int) -> None:
    version = payload.get("schema_version")
    if type(version) is not str or _SCHEMA_VERSION.fullmatch(version) is None:
        raise JsonlReplayError("unsupported_schema", line_number=line_number)
    if version != JSONL_SCHEMA_VERSION:
        raise JsonlReplayError("unsupported_schema", line_number=line_number)


def _load_manifest(payload: dict[str, object]) -> JsonlTraceManifest:
    if payload.get("record_type") != "trace_manifest":
        raise JsonlReplayError("invalid_manifest", line_number=1)
    _require_schema(payload, line_number=1)
    try:
        return JsonlTraceManifest.model_validate_json(canonical_json(payload))
    except Exception:
        raise JsonlReplayError("invalid_manifest", line_number=1) from None


def _load_event(payload: dict[str, object], *, line_number: int) -> JsonlReplayEvent:
    if payload.get("record_type") != "trace_event_draft":
        raise JsonlReplayError("invalid_record", line_number=line_number)
    _require_schema(payload, line_number=line_number)
    draft = payload.get("draft")
    if isinstance(draft, Mapping):
        _require_schema(draft, line_number=line_number)
    try:
        return JsonlReplayEvent.model_validate_json(canonical_json(payload))
    except Exception:
        raise JsonlReplayError("invalid_record", line_number=line_number) from None


def _decode_trace(value: bytes) -> tuple[JsonlTraceManifest, tuple[JsonlReplayEvent, ...]]:
    if len(value) > MAX_TRACE_BYTES:
        raise JsonlReplayError("trace_too_large")
    try:
        text = value.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise JsonlReplayError("invalid_encoding") from None
    lines = text.splitlines()
    if len(lines) > MAX_TRACE_LINES:
        raise JsonlReplayError("too_many_lines")
    if not lines:
        raise JsonlReplayError("missing_manifest")
    for number, line in enumerate(lines, start=1):
        byte_length = len(line.encode("utf-8", errors="strict"))
        if byte_length > MAX_LINE_BYTES:
            raise JsonlReplayError("line_too_large", line_number=number)

    payloads = tuple(
        _parse_line(line, line_number=number) for number, line in enumerate(lines, start=1)
    )
    manifest = _load_manifest(payloads[0])
    events = tuple(
        _load_event(payload, line_number=number)
        for number, payload in enumerate(payloads[1:], start=2)
    )
    _preflight(manifest, events)
    return manifest, events


class JSONLReplayAdapter:
    """Fully validates a frozen JSONL trace before exposing its first event."""

    __slots__ = ("_events", "_manifest")

    def __init__(
        self,
        manifest: JsonlTraceManifest,
        events: Sequence[JsonlReplayEvent],
    ) -> None:
        copied_manifest = _copy_manifest(manifest)
        copied_events, event_lines = _copy_bounded_events(events)
        _preflight(copied_manifest, copied_events)
        _validate_encoded_bound(copied_manifest, event_lines)
        self._manifest = copied_manifest
        self._events = copied_events

    @classmethod
    def from_bytes(cls, value: bytes) -> JSONLReplayAdapter:
        if type(value) is not bytes:
            raise JsonlReplayError("invalid_input_type")
        manifest, events = _decode_trace(value)
        return cls(manifest, events)

    @classmethod
    def from_text(cls, value: str) -> JSONLReplayAdapter:
        if type(value) is not str:
            raise JsonlReplayError("invalid_input_type")
        try:
            encoded = value.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            raise JsonlReplayError("invalid_encoding") from None
        return cls.from_bytes(encoded)

    @classmethod
    def from_path(cls, value: str | PathLike[str]) -> JSONLReplayAdapter:
        return cls.from_bytes(_read_regular_trace(value))

    @property
    def manifest(self) -> JsonlTraceManifest:
        return _copy_manifest(self._manifest)

    @property
    def events(self) -> tuple[JsonlReplayEvent, ...]:
        return tuple(_copy_event(event) for event in self._events)

    @property
    def run_id(self) -> UUID4:
        return self._manifest.run_id

    @property
    def trace_digest(self) -> str:
        return self._manifest.trace_digest

    @property
    def expected_event_ids(self) -> tuple[UUID4, ...]:
        return tuple(event.expected_event_id for event in self._events)

    def event_id_factory(self) -> Callable[[], UUID4]:
        """Return a fresh repository event-ID stream for this frozen trace."""

        return iter(self.expected_event_ids).__next__

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[JsonlReplayEvent]:
        return iter(self.events)

    def _member(self, value: object) -> JsonlReplayEvent:
        try:
            event = _copy_event(value)
        except JsonlReplayError:
            raise AdapterNormalizationError() from None
        index = event.ordinal - 1
        if index < 0 or index >= len(self._events) or event != self._events[index]:
            raise AdapterNormalizationError()
        return event

    def normalize(self, native_event: object) -> NormalizedTraceEventDraft:
        return _copy_draft(self._member(native_event).draft)

    def resolve_event_id(self, native_event: object, ordinal: int) -> UUID4 | None:
        event = self._member(native_event)
        if type(ordinal) is not int or event.ordinal != ordinal:
            raise AdapterNormalizationError()
        return event.expected_event_id

    def resolve_target_request_id(
        self,
        native_event: object,
        target: DeliveryTarget,
    ) -> str | None:
        if type(target) is not DeliveryTarget:
            raise AdapterTargetResolutionError()
        try:
            event = self._member(native_event)
        except AdapterNormalizationError:
            raise AdapterTargetResolutionError() from None
        if target is DeliveryTarget.NEXT_MODEL_CALL:
            return event.next_model_call_target_request_id
        return event.pre_action_target_request_id


def encode_jsonl_trace(events: Sequence[JsonlReplayEvent]) -> bytes:
    copied, event_lines = _copy_bounded_events(events)
    if not copied:
        raise JsonlReplayError("empty_trace")
    first = copied[0]
    manifest = JsonlTraceManifest(
        run_id=first.draft.run_id,
        record_count=len(copied),
        trace_digest=_trace_digest(
            schema_version=JSONL_SCHEMA_VERSION,
            run_id=first.draft.run_id,
            record_digests=tuple(event.record_digest for event in copied),
        ),
    )
    _preflight(manifest, copied)
    return _validate_encoded_bound(manifest, event_lines)


__all__ = [
    "JSONL_SCHEMA_VERSION",
    "MAX_LINE_BYTES",
    "MAX_TRACE_BYTES",
    "MAX_TRACE_LINES",
    "JSONLReplayAdapter",
    "JsonlReplayError",
    "JsonlReplayEvent",
    "JsonlTraceManifest",
    "encode_jsonl_trace",
]
