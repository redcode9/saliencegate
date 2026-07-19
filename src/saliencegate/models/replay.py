from __future__ import annotations

import asyncio
import json
import os
import re
import stat
from collections.abc import Mapping
from contextlib import suppress
from os import PathLike
from typing import Annotated, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.domain import canonical_json, length_prefixed_sha256
from saliencegate.domain.records import Sha256Digest
from saliencegate.ports.models import (
    ModelBoundaryError,
    ModelRequest,
    ModelResult,
    validated_model_request,
    validated_model_result,
)

REPLAY_MODEL_VERSION: Literal["replay-model/v1"] = "replay-model/v1"
REPLAY_RECORD_SCHEMA_VERSION: Literal["replay-record/v1"] = "replay-record/v1"

_RECORD_DIGEST_DOMAIN = "saliencegate:model:replay-record:v1"
_FIXTURE_DIGEST_DOMAIN = "saliencegate:model:replay-fixture:v1"
_REPLAY_ID = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._:/+\-]{0,255}$")
_MAX_RECORDS = 100_000
_MAX_LINE_BYTES = 16 * 1024 * 1024
_MAX_FIXTURE_BYTES = 64 * 1024 * 1024
_REPLAY_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "record_type",
        "replay_version",
        "ordinal",
        "request_digest",
        "result",
        "record_digest",
    }
)
_MODEL_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "request_digest",
        "output",
        "usage",
        "call_digest",
    }
)


class ReplayError(RuntimeError):
    """Base class for failures that never disclose request or response content."""


class ReplayFixtureError(ReplayError):
    def __init__(self) -> None:
        super().__init__("replay fixture failed preflight validation")


class ReplayIntegrityError(ReplayError):
    def __init__(self) -> None:
        super().__init__("replay request or response failed integrity validation")


class ReplayMissingResponseError(ReplayError):
    def __init__(self) -> None:
        super().__init__("no replay response is registered for the request")


class ReplayResponseReusedError(ReplayError):
    def __init__(self) -> None:
        super().__init__("the replay response for this request was already consumed")


class _ReplayRecordModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


def _record_digest_from_values(values: dict[str, object]) -> str:
    result = values["result"]
    if isinstance(result, BaseModel):
        result = result.model_dump(mode="json", warnings=False)
    return length_prefixed_sha256(
        canonical_json(
            {
                "schema_version": values["schema_version"],
                "record_type": values["record_type"],
                "replay_version": values["replay_version"],
                "ordinal": values["ordinal"],
                "request_digest": values["request_digest"],
                "result": result,
            }
        ),
        domain=_RECORD_DIGEST_DOMAIN,
    )


class ReplayRecord(_ReplayRecordModel):
    """One content-bound, one-shot replay response."""

    schema_version: Literal["replay-record/v1"] = REPLAY_RECORD_SCHEMA_VERSION
    record_type: Literal["replay_model_response"] = "replay_model_response"
    replay_version: Literal["replay-model/v1"] = REPLAY_MODEL_VERSION
    ordinal: Annotated[int, Field(ge=1, le=_MAX_RECORDS)]
    request_digest: Sha256Digest
    result: ModelResult = Field(repr=False)
    record_digest: Sha256Digest = Field(default_factory=_record_digest_from_values)

    @model_validator(mode="after")
    def bindings_and_digest_match(self) -> Self:
        if self.result.request_digest != self.request_digest:
            raise ValueError("replay response belongs to a different request")
        expected = _record_digest_from_values(
            cast(dict[str, object], self.model_dump(mode="json", exclude={"record_digest"}))
        )
        if self.record_digest != expected:
            raise ValueError("replay record digest does not match")
        return self


def _validated_record(value: object) -> ReplayRecord:
    if type(value) is not ReplayRecord:
        raise ReplayFixtureError()
    try:
        candidate = ReplayRecord.model_validate_json(value.model_dump_json(warnings=False))
    except Exception:
        raise ReplayFixtureError() from None
    return candidate


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReplayFixtureError()
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ReplayFixtureError()


def _load_line(line: bytes) -> ReplayRecord:
    if not line or len(line) > _MAX_LINE_BYTES:
        raise ReplayFixtureError()
    try:
        payload = json.loads(
            line.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        if not isinstance(payload, Mapping) or payload.keys() != _REPLAY_RECORD_KEYS:
            raise ReplayFixtureError()
        result_payload = payload.get("result")
        if not isinstance(result_payload, Mapping) or result_payload.keys() != _MODEL_RESULT_KEYS:
            raise ReplayFixtureError()
        record = ReplayRecord.model_validate_json(line)
    except ReplayFixtureError:
        raise
    except Exception:
        raise ReplayFixtureError() from None
    return _validated_record(record)


class ReplayModel:
    """A deterministic, in-process structured model backed by frozen responses.

    Construction performs a complete preflight. ``generate`` does no file or network I/O and
    atomically consumes each registered response at most once, even under concurrent calls.
    """

    __slots__ = ("_consumed", "_fixture_digest", "_lock", "_records", "_replay_id")

    def __init__(
        self,
        records: tuple[ReplayRecord, ...],
        *,
        replay_id: str = REPLAY_MODEL_VERSION,
    ) -> None:
        if type(records) is not tuple or len(records) > _MAX_RECORDS:
            raise ReplayFixtureError()
        if type(replay_id) is not str or _REPLAY_ID.fullmatch(replay_id) is None:
            raise ReplayFixtureError()

        validated: list[ReplayRecord] = []
        seen: set[str] = set()
        total_bytes = 0
        for ordinal, candidate in enumerate(records, start=1):
            record = _validated_record(candidate)
            total_bytes += len(canonical_json(record)) + 1
            if record.ordinal != ordinal or record.request_digest in seen:
                raise ReplayFixtureError()
            if total_bytes > _MAX_FIXTURE_BYTES:
                raise ReplayFixtureError()
            seen.add(record.request_digest)
            validated.append(record)

        self._records = {record.request_digest: record for record in validated}
        self._consumed: set[str] = set()
        self._lock = asyncio.Lock()
        self._replay_id = replay_id
        self._fixture_digest = length_prefixed_sha256(
            replay_id,
            canonical_json(tuple(record.model_dump(mode="json") for record in validated)),
            domain=_FIXTURE_DIGEST_DOMAIN,
        )

    @classmethod
    def from_path(
        cls,
        path: str | PathLike[str],
        *,
        replay_id: str = REPLAY_MODEL_VERSION,
    ) -> ReplayModel:
        if type(path) is not str and not isinstance(path, PathLike):
            raise ReplayFixtureError()
        records: list[ReplayRecord] = []
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
                    remaining_bytes = _MAX_FIXTURE_BYTES - total_bytes
                    read_limit = max(1, min(_MAX_LINE_BYTES + 3, remaining_bytes + 1))
                    raw_line = source.readline(read_limit)
                    if not raw_line:
                        break
                    total_bytes += len(raw_line)
                    if total_bytes > _MAX_FIXTURE_BYTES:
                        raise ReplayFixtureError()
                    line_number += 1
                    if line_number > _MAX_RECORDS:
                        raise ReplayFixtureError()
                    line = raw_line.removesuffix(b"\n").removesuffix(b"\r")
                    record = _load_line(line)
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
        return cls(tuple(records), replay_id=replay_id)

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

    async def generate(self, request: ModelRequest) -> ModelResult:
        try:
            validated_request = validated_model_request(request)
        except ModelBoundaryError:
            raise ReplayIntegrityError() from None

        async with self._lock:
            digest = validated_request.request_digest
            record = self._records.get(digest)
            if record is None:
                raise ReplayMissingResponseError()
            if digest in self._consumed:
                raise ReplayResponseReusedError()
            try:
                validated_record = _validated_record(record)
                result = validated_model_result(validated_record.result)
            except (ModelBoundaryError, ReplayFixtureError):
                raise ReplayIntegrityError() from None
            if result.request_digest != digest:
                raise ReplayIntegrityError()
            self._consumed.add(digest)
            return result


__all__ = [
    "REPLAY_MODEL_VERSION",
    "REPLAY_RECORD_SCHEMA_VERSION",
    "ReplayError",
    "ReplayFixtureError",
    "ReplayIntegrityError",
    "ReplayMissingResponseError",
    "ReplayModel",
    "ReplayRecord",
    "ReplayResponseReusedError",
]
