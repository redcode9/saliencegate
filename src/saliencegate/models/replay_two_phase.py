from __future__ import annotations

import asyncio
import json
import os
import re
import stat
from collections.abc import Mapping
from contextlib import suppress
from os import PathLike
from types import MappingProxyType
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.domain import PayloadDigestAlgorithm, canonical_json, length_prefixed_sha256
from saliencegate.domain.records import ComponentIdentifier, Sha256Digest
from saliencegate.models.replay import (
    ReplayFixtureError,
    ReplayIntegrityError,
    ReplayMissingResponseError,
    ReplayResponseReusedError,
)
from saliencegate.ports.model_calls import (
    INTERVENTION_OUTPUT_SCHEMA_VERSION,
    MEMORY_EDIT_OUTPUT_SCHEMA_VERSION,
    CanonicalUsageProvenance,
    ProviderUsageProvenance,
    StructuredCallBoundaryError,
    StructuredCallPhase,
    StructuredCallRequest,
    StructuredCallResult,
    StructuredCallStatus,
    StructuredResponseSchemaVersion,
    validated_result_for_request,
    validated_structured_call_request,
    validated_structured_call_result,
)
from saliencegate.ports.two_phase import CallReceipt

TWO_PHASE_REPLAY_VERSION: Literal["two-phase-replay/v1"] = "two-phase-replay/v1"
TWO_PHASE_REPLAY_RECORD_SCHEMA_VERSION: Literal["two-phase-replay-record/v1"] = (
    "two-phase-replay-record/v1"
)

_FIXTURE_DIGEST_DOMAIN = "saliencegate:model:two-phase-replay-fixture:v1"
_RECORD_DIGEST_DOMAIN = "saliencegate:model:two-phase-replay-record:v1"
_REPLAY_ID = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._:/+\-]{0,255}$")
_MAX_RECORDS = 100_000
_MAX_LINE_BYTES = 16 * 1024 * 1024
_MAX_FIXTURE_BYTES = 64 * 1024 * 1024
_MAX_JSON_NODES = 100_000
_MAX_JSON_DEPTH = 64

_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "record_type",
        "replay_version",
        "replay_id",
        "fixture_digest",
        "ordinal",
        "request_digest",
        "model_call_index",
        "phase",
        "attempt",
        "response_schema_version",
        "result",
        "record_digest",
    }
)
_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "request_digest",
        "model_call_index",
        "phase",
        "attempt",
        "response_schema_version",
        "status",
        "parse_status",
        "output",
        "completion_digest",
        "completion_byte_count",
        "usage",
        "call_digest",
    }
)
_USAGE_KEYS = frozenset(
    {
        "schema_version",
        "provider_input_tokens",
        "provider_output_tokens",
        "provider_usage_provenance",
        "latency_us",
    }
)
_OPTIONAL_CANONICAL_USAGE_KEYS = frozenset(
    {
        "canonical_input_tokens",
        "canonical_output_tokens",
        "canonical_usage_provenance",
        "local_counter_id",
        "local_counter_version",
        "local_counter_configuration_digest",
        "local_counter_model_id",
    }
)
_MAX_SIGNED_64 = (1 << 63) - 1


class _TwoPhaseReplayModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


def _result_json(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", warnings=False)
    return value


def _record_digest_from_values(values: Mapping[str, object]) -> str:
    return length_prefixed_sha256(
        canonical_json(
            {
                "schema_version": values["schema_version"],
                "record_type": values["record_type"],
                "replay_version": values["replay_version"],
                "replay_id": values["replay_id"],
                "fixture_digest": values["fixture_digest"],
                "ordinal": values["ordinal"],
                "request_digest": values["request_digest"],
                "model_call_index": values["model_call_index"],
                "phase": values["phase"],
                "attempt": values["attempt"],
                "response_schema_version": values["response_schema_version"],
                "result": _result_json(values["result"]),
            }
        ),
        domain=_RECORD_DIGEST_DOMAIN,
    )


class TwoPhaseReplayRecord(_TwoPhaseReplayModel):
    """One sealed member of a canonical structured-call replay fixture."""

    schema_version: Literal["two-phase-replay-record/v1"] = TWO_PHASE_REPLAY_RECORD_SCHEMA_VERSION
    record_type: Literal["two_phase_replay_response"] = "two_phase_replay_response"
    replay_version: Literal["two-phase-replay/v1"] = TWO_PHASE_REPLAY_VERSION
    replay_id: ComponentIdentifier
    fixture_digest: Sha256Digest
    ordinal: Annotated[int, Field(ge=1, le=_MAX_RECORDS)]
    request_digest: Sha256Digest
    model_call_index: Annotated[int, Field(ge=0, le=_MAX_SIGNED_64)]
    phase: StructuredCallPhase
    attempt: Annotated[int, Field(ge=0, le=_MAX_SIGNED_64)]
    response_schema_version: StructuredResponseSchemaVersion
    result: StructuredCallResult = Field(repr=False)
    record_digest: Sha256Digest = Field(default_factory=_record_digest_from_values)

    @model_validator(mode="after")
    def result_metadata_and_digests_match(self) -> Self:
        try:
            result = validated_structured_call_result(self.result)
        except StructuredCallBoundaryError:
            raise ValueError("two-phase replay result failed validation") from None
        if (
            result.request_digest != self.request_digest
            or result.model_call_index != self.model_call_index
            or result.phase is not self.phase
            or result.attempt != self.attempt
            or result.response_schema_version != self.response_schema_version
        ):
            raise ValueError("two-phase replay result metadata does not match")
        if not _call_evidence_is_replay_native(result):
            raise ValueError("two-phase replay result evidence is not replay-native")
        values = self.model_dump(mode="json", exclude={"record_digest"})
        if self.record_digest != _record_digest_from_values(values):
            raise ValueError("two-phase replay record digest does not match")
        return self


def _validated_record(value: object) -> TwoPhaseReplayRecord:
    if type(value) is not TwoPhaseReplayRecord:
        raise ReplayFixtureError()
    try:
        return TwoPhaseReplayRecord.model_validate_json(value.model_dump_json(warnings=False))
    except Exception:
        raise ReplayFixtureError() from None


def _call_evidence_is_replay_native(value: StructuredCallResult | CallReceipt) -> bool:
    usage = value.usage
    provider_replay_attested = (
        usage.provider_usage_provenance is ProviderUsageProvenance.REPLAY_ATTESTED
    )
    canonical_replay_attested = (
        usage.canonical_usage_provenance is CanonicalUsageProvenance.REPLAY_ATTESTED
        and usage.canonical_input_tokens is not None
        and usage.canonical_output_tokens is not None
    )
    return not (
        usage.provider_usage_provenance is ProviderUsageProvenance.PROVIDER_REPORTED
        or usage.canonical_usage_provenance is CanonicalUsageProvenance.LOCAL_COUNTER
        or (
            value.status is not StructuredCallStatus.COMPLETED
            and usage.canonical_output_tokens is not None
        )
        or (
            value.status is StructuredCallStatus.COMPLETED
            and not provider_replay_attested
            and not canonical_replay_attested
        )
        or (
            value.completion_digest is not None
            and value.completion_digest.algorithm is not PayloadDigestAlgorithm.SYNTHETIC_SHA256
        )
    )


def two_phase_receipts_are_replay_native(receipts: tuple[CallReceipt, ...]) -> bool:
    """Return whether every exact receipt carries replay-native call evidence."""

    if (
        type(receipts) is not tuple
        or not receipts
        or len(receipts) > _MAX_RECORDS
        or any(type(receipt) is not CallReceipt for receipt in receipts)
    ):
        return False
    try:
        checked = tuple(
            CallReceipt.model_validate_json(receipt.model_dump_json(warnings=False))
            for receipt in receipts
        )
    except Exception:
        return False
    return all(_call_evidence_is_replay_native(receipt) for receipt in checked)


def _fixture_material(
    *,
    ordinal: int,
    request_digest: str,
    model_call_index: int,
    phase: StructuredCallPhase,
    attempt: int,
    response_schema_version: StructuredResponseSchemaVersion,
    call_digest: str,
) -> dict[str, object]:
    return {
        "schema_version": TWO_PHASE_REPLAY_RECORD_SCHEMA_VERSION,
        "record_type": "two_phase_replay_response",
        "replay_version": TWO_PHASE_REPLAY_VERSION,
        "ordinal": ordinal,
        "request_digest": request_digest,
        "model_call_index": model_call_index,
        "phase": phase.value,
        "attempt": attempt,
        "response_schema_version": response_schema_version,
        "call_digest": call_digest,
    }


def _fixture_digest_from_material(
    material: tuple[dict[str, object], ...],
    *,
    replay_id: str,
) -> str:
    return length_prefixed_sha256(
        replay_id,
        canonical_json(material),
        domain=_FIXTURE_DIGEST_DOMAIN,
    )


def _fixture_digest(records: tuple[TwoPhaseReplayRecord, ...], *, replay_id: str) -> str:
    return _fixture_digest_from_material(
        tuple(
            _fixture_material(
                ordinal=record.ordinal,
                request_digest=record.request_digest,
                model_call_index=record.model_call_index,
                phase=record.phase,
                attempt=record.attempt,
                response_schema_version=record.response_schema_version,
                call_digest=record.result.call_digest,
            )
            for record in records
        ),
        replay_id=replay_id,
    )


def two_phase_replay_fixture_digest_from_receipts(
    receipts: tuple[CallReceipt, ...],
    *,
    replay_id: str,
) -> str:
    """Reconstruct the order-sensitive replay fixture digest from call receipts."""

    if (
        type(receipts) is not tuple
        or len(receipts) > _MAX_RECORDS
        or any(type(receipt) is not CallReceipt for receipt in receipts)
        or type(replay_id) is not str
        or _REPLAY_ID.fullmatch(replay_id) is None
    ):
        raise ReplayFixtureError()
    try:
        checked = tuple(
            CallReceipt.model_validate_json(receipt.model_dump_json(warnings=False))
            for receipt in receipts
        )
        if any(
            len({getattr(receipt, field_name) for receipt in checked}) != len(checked)
            for field_name in ("request_digest", "call_digest", "receipt_digest")
        ):
            raise ReplayFixtureError()
        return _fixture_digest_from_material(
            tuple(
                _fixture_material(
                    ordinal=ordinal,
                    request_digest=receipt.request_digest,
                    model_call_index=receipt.model_call_index,
                    phase=receipt.phase,
                    attempt=receipt.attempt,
                    response_schema_version=(
                        MEMORY_EDIT_OUTPUT_SCHEMA_VERSION
                        if receipt.phase is StructuredCallPhase.MEMORY_EDIT
                        else INTERVENTION_OUTPUT_SCHEMA_VERSION
                    ),
                    call_digest=receipt.call_digest,
                )
                for ordinal, receipt in enumerate(checked, start=1)
            ),
            replay_id=replay_id,
        )
    except ReplayFixtureError:
        raise
    except Exception:
        raise ReplayFixtureError() from None


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReplayFixtureError()
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ReplayFixtureError()


def _json_is_bounded(value: object) -> bool:
    try:
        stack: list[tuple[object, int]] = [(value, 0)]
        nodes = 0
        while stack:
            item, depth = stack.pop()
            nodes += 1
            if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
                return False
            if type(item) in (dict, MappingProxyType):
                assert isinstance(item, (dict, MappingProxyType))
                if len(item) > _MAX_JSON_NODES - nodes - len(stack):
                    return False
                for key, nested in item.items():
                    if type(key) is not str:
                        return False
                    stack.append((nested, depth + 1))
            elif type(item) in (list, tuple):
                assert isinstance(item, (list, tuple))
                if len(item) > _MAX_JSON_NODES - nodes - len(stack):
                    return False
                stack.extend((nested, depth + 1) for nested in item)
            elif item is not None and type(item) not in (str, bool, int, float):
                return False
        return True
    except Exception:
        return False


def _load_line(line: bytes) -> TwoPhaseReplayRecord:
    if not line or len(line) > _MAX_LINE_BYTES:
        raise ReplayFixtureError()
    try:
        payload = json.loads(
            line.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        if (
            type(payload) is not dict
            or not _json_is_bounded(payload)
            or payload.keys() != _RECORD_KEYS
        ):
            raise ReplayFixtureError()
        result = payload.get("result")
        if not isinstance(result, Mapping) or result.keys() != _RESULT_KEYS:
            raise ReplayFixtureError()
        usage = result.get("usage")
        if (
            not isinstance(usage, Mapping)
            or not _USAGE_KEYS.issubset(usage.keys())
            or not usage.keys() <= _USAGE_KEYS | _OPTIONAL_CANONICAL_USAGE_KEYS
        ):
            raise ReplayFixtureError()
        record = TwoPhaseReplayRecord.model_validate_json(line)
        checked = _validated_record(record)
        if canonical_json(checked) != line:
            raise ReplayFixtureError()
        return checked
    except ReplayFixtureError:
        raise
    except Exception:
        raise ReplayFixtureError() from None


class TwoPhaseReplayClient:
    """A preflighted, one-shot offline client for structured two-phase calls."""

    __slots__ = (
        "_consumed",
        "_fixture_digest",
        "_lock",
        "_record_digests",
        "_records",
        "_replay_id",
    )

    def __init__(
        self,
        records: tuple[TwoPhaseReplayRecord, ...],
        *,
        replay_id: str = TWO_PHASE_REPLAY_VERSION,
        expected_fixture_digest: str | None = None,
    ) -> None:
        if (
            type(records) is not tuple
            or len(records) > _MAX_RECORDS
            or type(replay_id) is not str
            or _REPLAY_ID.fullmatch(replay_id) is None
            or (
                expected_fixture_digest is not None
                and (
                    type(expected_fixture_digest) is not str
                    or re.fullmatch(r"[0-9a-f]{64}", expected_fixture_digest) is None
                )
            )
        ):
            raise ReplayFixtureError()

        validated: list[TwoPhaseReplayRecord] = []
        seen: set[str] = set()
        total_bytes = 0
        for ordinal, candidate in enumerate(records, start=1):
            record = _validated_record(candidate)
            line_bytes = len(canonical_json(record))
            total_bytes += line_bytes + 1
            if (
                line_bytes > _MAX_LINE_BYTES
                or total_bytes > _MAX_FIXTURE_BYTES
                or record.ordinal != ordinal
                or record.request_digest in seen
                or record.replay_id != replay_id
            ):
                raise ReplayFixtureError()
            seen.add(record.request_digest)
            validated.append(record)

        validated_records = tuple(validated)
        fixture_digest = _fixture_digest(validated_records, replay_id=replay_id)
        if any(record.fixture_digest != fixture_digest for record in validated_records) or (
            expected_fixture_digest is not None and expected_fixture_digest != fixture_digest
        ):
            raise ReplayFixtureError()

        records_by_request = {record.request_digest: record for record in validated_records}
        self._records: Mapping[str, TwoPhaseReplayRecord] = MappingProxyType(records_by_request)
        self._record_digests: Mapping[str, str] = MappingProxyType(
            {
                request_digest: record.record_digest
                for request_digest, record in records_by_request.items()
            }
        )
        self._consumed: set[str] = set()
        self._lock = asyncio.Lock()
        self._replay_id = replay_id
        self._fixture_digest = fixture_digest

    @classmethod
    def from_path(
        cls,
        path: str | PathLike[str],
        *,
        replay_id: str = TWO_PHASE_REPLAY_VERSION,
        expected_fixture_digest: str | None = None,
    ) -> TwoPhaseReplayClient:
        if type(path) is not str and not isinstance(path, PathLike):
            raise ReplayFixtureError()
        records: list[TwoPhaseReplayRecord] = []
        descriptor: int | None = None
        try:
            raw_path = os.fspath(path)
            if type(raw_path) is not str:
                raise TypeError
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(raw_path, flags)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size > _MAX_FIXTURE_BYTES
            ):
                raise ReplayFixtureError()
            stream = os.fdopen(descriptor, "rb", closefd=True)
            descriptor = None
            with stream as source:
                line_number = 0
                total_bytes = 0
                while True:
                    remaining = _MAX_FIXTURE_BYTES - total_bytes
                    read_limit = max(1, min(_MAX_LINE_BYTES + 2, remaining + 1))
                    raw_line = source.readline(read_limit)
                    if not raw_line:
                        break
                    total_bytes += len(raw_line)
                    line_number += 1
                    if (
                        total_bytes > _MAX_FIXTURE_BYTES
                        or line_number > _MAX_RECORDS
                        or not raw_line.endswith(b"\n")
                        or raw_line.endswith(b"\r\n")
                    ):
                        raise ReplayFixtureError()
                    record = _load_line(raw_line[:-1])
                    if record.ordinal != line_number:
                        raise ReplayFixtureError()
                    records.append(record)
                after = os.fstat(source.fileno())
                current = os.stat(raw_path, follow_symlinks=False)
                if (
                    not stat.S_ISREG(after.st_mode)
                    or not stat.S_ISREG(current.st_mode)
                    or after.st_nlink != 1
                    or current.st_nlink != 1
                    or metadata.st_dev != after.st_dev
                    or metadata.st_dev != current.st_dev
                    or metadata.st_ino != after.st_ino
                    or metadata.st_ino != current.st_ino
                    or metadata.st_size != after.st_size
                    or metadata.st_size != current.st_size
                    or metadata.st_mtime_ns != after.st_mtime_ns
                    or metadata.st_mtime_ns != current.st_mtime_ns
                    or metadata.st_ctime_ns != after.st_ctime_ns
                    or metadata.st_ctime_ns != current.st_ctime_ns
                ):
                    raise ReplayFixtureError()
        except Exception:
            raise ReplayFixtureError() from None
        finally:
            if descriptor is not None:
                with suppress(Exception):
                    os.close(descriptor)
        return cls(
            tuple(records),
            replay_id=replay_id,
            expected_fixture_digest=expected_fixture_digest,
        )

    @property
    def replay_id(self) -> str:
        return self._replay_id

    @property
    def fixture_digest(self) -> str:
        return self._fixture_digest

    @property
    def total_responses(self) -> int:
        return len(self._records)

    @property
    def remaining_responses(self) -> int:
        return len(self._records) - len(self._consumed)

    async def generate(self, request: StructuredCallRequest) -> StructuredCallResult:
        try:
            checked_request = validated_structured_call_request(request)
        except StructuredCallBoundaryError:
            raise ReplayIntegrityError() from None

        async with self._lock:
            digest = checked_request.request_digest
            record = self._records.get(digest)
            if record is None:
                raise ReplayMissingResponseError()
            if digest in self._consumed:
                raise ReplayResponseReusedError()
            try:
                checked_record = _validated_record(record)
                if (
                    checked_record.request_digest != digest
                    or checked_record.replay_id != self._replay_id
                    or checked_record.fixture_digest != self._fixture_digest
                    or checked_record.record_digest != self._record_digests.get(digest)
                ):
                    raise ReplayFixtureError()
                result = validated_result_for_request(checked_request, checked_record.result)
            except (ReplayFixtureError, StructuredCallBoundaryError):
                raise ReplayIntegrityError() from None
            self._consumed.add(digest)
            return result


__all__ = [
    "TWO_PHASE_REPLAY_RECORD_SCHEMA_VERSION",
    "TWO_PHASE_REPLAY_VERSION",
    "TwoPhaseReplayClient",
    "TwoPhaseReplayRecord",
    "two_phase_receipts_are_replay_native",
    "two_phase_replay_fixture_digest_from_receipts",
]
