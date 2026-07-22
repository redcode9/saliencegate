from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Never, cast

from pydantic import BaseModel

from saliencegate.domain.errors import (
    CanonicalJSONError,
    InvalidSchemaVersionError,
    UnknownRecordTypeError,
    UnsupportedSchemaVersionError,
)
from saliencegate.domain.ids import content_digest

if TYPE_CHECKING:
    from saliencegate.domain.records import RuntimeRecord

_SCHEMA_VERSION = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


def _normalize_json(value: object) -> object:
    if isinstance(value, BaseModel):
        return _normalize_json(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalJSONError("JSON object keys must be strings")
            normalized[key] = _normalize_json(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalJSONError("JSON numbers must be finite")
        return value
    raise CanonicalJSONError(f"unsupported JSON value type: {type(value).__name__}")


def canonical_json(value: object) -> bytes:
    """Serialize a record or JSON-compatible value into stable UTF-8 bytes."""

    normalized = _normalize_json(value)
    try:
        text = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return text.encode("utf-8")
    except (TypeError, UnicodeEncodeError, ValueError) as error:
        raise CanonicalJSONError(str(error)) from error


def canonical_digest(value: object) -> str:
    return content_digest(canonical_json(value))


def _reject_non_finite(token: str) -> Never:
    raise CanonicalJSONError(f"JSON numbers must be finite, got {token}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalJSONError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _decode_json(data: bytes | str) -> tuple[str, dict[str, object]]:
    try:
        text = data.decode("utf-8") if isinstance(data, bytes) else data
        parsed: object = json.loads(
            text,
            parse_constant=_reject_non_finite,
            object_pairs_hook=_unique_object,
        )
    except CanonicalJSONError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CanonicalJSONError(str(error)) from error
    if not isinstance(parsed, dict):
        raise CanonicalJSONError("a runtime record must be a JSON object")
    return text, parsed


def _validate_version(payload: Mapping[str, object]) -> str:
    from saliencegate.domain.records import SUPPORTED_SCHEMA_VERSIONS

    version = payload.get("schema_version")
    if not isinstance(version, str) or _SCHEMA_VERSION.fullmatch(version) is None:
        raise InvalidSchemaVersionError(version)
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise UnsupportedSchemaVersionError(version, SUPPORTED_SCHEMA_VERSIONS)
    return version


def load_record(data: bytes | str) -> RuntimeRecord:
    """Validate version metadata, then dispatch a serialized runtime record."""

    from saliencegate.domain.records import (
        CycleRecord,
        DeliveryRecord,
        InterventionDecision,
        InterventionOutcome,
        InvocationDecision,
        MemoryDelta,
        MemoryRecord,
        RuntimeRecord,
        Signal,
        TraceEvent,
        VersionedRecord,
    )

    record_types: dict[str, type[VersionedRecord]] = {
        "trace_event": TraceEvent,
        "signal": Signal,
        "memory_record": MemoryRecord,
        "invocation_decision": InvocationDecision,
        "memory_delta": MemoryDelta,
        "intervention_decision": InterventionDecision,
        "intervention_outcome": InterventionOutcome,
        "cycle_record": CycleRecord,
        "delivery_record": DeliveryRecord,
    }
    text, payload = _decode_json(data)
    _validate_version(payload)
    record_type = payload.get("record_type")
    if not isinstance(record_type, str) or record_type not in record_types:
        raise UnknownRecordTypeError(record_type)
    model = record_types[record_type]
    return cast(RuntimeRecord, model.model_validate_json(text))
