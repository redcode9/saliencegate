from __future__ import annotations

import json
import math
import os
import re
import stat
from collections.abc import Mapping
from contextlib import suppress
from itertools import pairwise
from os import PathLike
from types import MappingProxyType
from typing import Annotated, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.domain import (
    EventType,
    TrustLabel,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.domain.records import UUID4, ComponentIdentifier, Sha256Digest
from saliencegate.runtime.fixed_step import FixedStepEventInput

STAGE2_TRAJECTORY_VERSION: Literal["stage2-trajectory/v1"] = "stage2-trajectory/v1"
STAGE2_TRAJECTORY_RECORD_SCHEMA_VERSION: Literal["stage2-trajectory-record/v1"] = (
    "stage2-trajectory-record/v1"
)
PAPER_TWO_PHASE_BASIC_TRAJECTORY_ID: Literal["paper-two-phase-basic/v1"] = (
    "paper-two-phase-basic/v1"
)

_INPUT_DIGEST_DOMAIN = "saliencegate:experiment:stage2-trajectory-input:v1"
_RECORD_DIGEST_DOMAIN = "saliencegate:experiment:stage2-trajectory-record:v1"
_FIXTURE_DIGEST_DOMAIN = "saliencegate:experiment:stage2-trajectory-fixture:v1"
_COMPONENT_ID = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._:/+\-]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_RECORDS = 10_000
_MAX_LINE_BYTES = 2 * 1024 * 1024
_MAX_FIXTURE_BYTES = 64 * 1024 * 1024
_MAX_JSON_NODES = 100_000
_MAX_JSON_DEPTH = 72
_MAX_SIGNED_64 = (1 << 63) - 1

_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "record_type",
        "trajectory_version",
        "fixture_id",
        "fixture_digest",
        "ordinal",
        "event_input",
        "input_digest",
        "record_digest",
    }
)


class Stage2TrajectoryFixtureError(ValueError):
    """A value-free failure at the closed offline trajectory boundary."""

    def __init__(self) -> None:
        super().__init__("offline experiment trajectory fixture failed validation")


class _Stage2TrajectoryModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


def _copy_event_input(value: object) -> FixedStepEventInput:
    if type(value) is not FixedStepEventInput:
        raise Stage2TrajectoryFixtureError()
    try:
        checked = FixedStepEventInput.model_validate_json(value.model_dump_json(warnings=False))
        if checked != value:
            raise ValueError
        return checked
    except Exception:
        raise Stage2TrajectoryFixtureError() from None


def _event_input_json(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", warnings=False)
    return value


def _input_digest(value: FixedStepEventInput) -> str:
    return length_prefixed_sha256(
        canonical_json(value),
        domain=_INPUT_DIGEST_DOMAIN,
    )


def _record_digest(values: Mapping[str, object]) -> str:
    return length_prefixed_sha256(
        canonical_json(
            {
                "schema_version": values["schema_version"],
                "record_type": values["record_type"],
                "trajectory_version": values["trajectory_version"],
                "fixture_id": values["fixture_id"],
                "fixture_digest": values["fixture_digest"],
                "ordinal": values["ordinal"],
                "event_input": _event_input_json(values["event_input"]),
                "input_digest": values["input_digest"],
            }
        ),
        domain=_RECORD_DIGEST_DOMAIN,
    )


class Stage2TrajectoryRecord(_Stage2TrajectoryModel):
    """One content-addressed input in a canonical offline JSONL trajectory."""

    schema_version: Literal["stage2-trajectory-record/v1"] = STAGE2_TRAJECTORY_RECORD_SCHEMA_VERSION
    record_type: Literal["stage2_trajectory_input"] = "stage2_trajectory_input"
    trajectory_version: Literal["stage2-trajectory/v1"] = STAGE2_TRAJECTORY_VERSION
    fixture_id: ComponentIdentifier
    fixture_digest: Sha256Digest
    ordinal: Annotated[int, Field(ge=1, le=_MAX_RECORDS)]
    event_input: FixedStepEventInput = Field(repr=False)
    input_digest: Sha256Digest
    record_digest: Sha256Digest = Field(default_factory=_record_digest)

    @model_validator(mode="after")
    def input_and_digests_are_exact(self) -> Self:
        checked = _copy_event_input(self.event_input)
        if self.input_digest != _input_digest(checked):
            raise ValueError("offline trajectory input digest does not match")
        values = self.model_dump(mode="json", exclude={"record_digest"}, warnings=False)
        if self.record_digest != _record_digest(values):
            raise ValueError("offline trajectory record digest does not match")
        return self


def _resolve_pointer(item: FixedStepEventInput, field_path: str) -> object:
    value: object = {"payload": item.draft.payload}
    try:
        for encoded_segment in field_path.split("/")[1:]:
            segment = encoded_segment.replace("~1", "/").replace("~0", "~")
            if isinstance(value, Mapping):
                value = value[segment]
            elif type(value) in (list, tuple):
                if not segment.isdigit() or (len(segment) > 1 and segment.startswith("0")):
                    raise KeyError
                sequence = cast(list[object] | tuple[object, ...], value)
                value = sequence[int(segment)]
            else:
                raise KeyError
        return value
    except Exception:
        raise Stage2TrajectoryFixtureError() from None


def _validate_selectors(inputs: tuple[FixedStepEventInput, ...]) -> None:
    last_step: int | None = None
    logical_message_count = 0
    for item in inputs:
        selectors = []
        selector_paths: list[str] = []
        if item.task_description is not None:
            selectors.append(item.task_description)
            selector_paths.append(item.task_description.field_path)
        selectors.extend(binding.selector for binding in item.logical_messages)
        selector_paths.extend(binding.selector.field_path for binding in item.logical_messages)
        logical_message_count += len(item.logical_messages)
        if item.action_step is not None:
            selector_paths.append(item.action_step.field_path)
        if len(set(selector_paths)) != len(selector_paths):
            raise Stage2TrajectoryFixtureError()
        for selector in selectors:
            selected = _resolve_pointer(item, selector.field_path)
            if type(selected) is not str or not selected:
                raise Stage2TrajectoryFixtureError()
            try:
                encoded = selected.encode("utf-8", errors="strict")
                if selector.span is not None:
                    if selector.span.end_byte > len(encoded):
                        raise ValueError
                    selected = encoded[selector.span.start_byte : selector.span.end_byte].decode(
                        "utf-8", errors="strict"
                    )
            except Exception:
                raise Stage2TrajectoryFixtureError() from None
        if item.action_step is not None:
            step = _resolve_pointer(item, item.action_step.field_path)
            if type(step) is not int or not 1 <= step <= _MAX_SIGNED_64:
                raise Stage2TrajectoryFixtureError()
            if last_step is not None and step < last_step:
                raise Stage2TrajectoryFixtureError()
            last_step = step
    if logical_message_count == 0 or last_step is None:
        raise Stage2TrajectoryFixtureError()


def _validate_inputs(value: object) -> tuple[FixedStepEventInput, ...]:
    if type(value) is not tuple or not value or len(value) > _MAX_RECORDS:
        raise Stage2TrajectoryFixtureError()
    checked = tuple(_copy_event_input(item) for item in cast(tuple[object, ...], value))
    first = checked[0]
    run_id = first.draft.run_id
    expected_order = {
        item.expected_event_id: ordinal for ordinal, item in enumerate(checked, start=1)
    }
    target_request_ids = tuple(
        item.target_request_id for item in checked if item.target_request_id is not None
    )
    if (
        first.draft.event_type is not EventType.RUN_START
        or first.task_description is None
        or any(item.draft.event_type is EventType.RUN_START for item in checked[1:])
        or any(item.draft.run_id != run_id for item in checked)
        or any(item.draft.trust_label is not TrustLabel.SYNTHETIC_FIXTURE for item in checked)
        or any(item.task_description is not None for item in checked[1:])
        or any(
            later.draft.timestamp < earlier.draft.timestamp for earlier, later in pairwise(checked)
        )
        or len(expected_order) != len(checked)
        or run_id in expected_order
        or len({item.draft.source_event_id for item in checked}) != len(checked)
        or not target_request_ids
        or len(set(target_request_ids)) != len(target_request_ids)
        or any(
            parent_id not in expected_order or expected_order[parent_id] >= ordinal
            for ordinal, item in enumerate(checked, start=1)
            for parent_id in item.draft.parent_ids
        )
    ):
        raise Stage2TrajectoryFixtureError()
    _validate_selectors(checked)
    return checked


def _fixture_material(
    inputs: tuple[FixedStepEventInput, ...],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "schema_version": STAGE2_TRAJECTORY_RECORD_SCHEMA_VERSION,
            "record_type": "stage2_trajectory_input",
            "trajectory_version": STAGE2_TRAJECTORY_VERSION,
            "ordinal": ordinal,
            "input_digest": _input_digest(item),
        }
        for ordinal, item in enumerate(inputs, start=1)
    )


def _fixture_digest(inputs: tuple[FixedStepEventInput, ...], *, fixture_id: str) -> str:
    return length_prefixed_sha256(
        fixture_id,
        canonical_json(_fixture_material(inputs)),
        domain=_FIXTURE_DIGEST_DOMAIN,
    )


def _copy_record(value: object) -> Stage2TrajectoryRecord:
    if type(value) is not Stage2TrajectoryRecord:
        raise Stage2TrajectoryFixtureError()
    try:
        checked = Stage2TrajectoryRecord.model_validate_json(value.model_dump_json(warnings=False))
        if checked != value:
            raise ValueError
        return checked
    except Exception:
        raise Stage2TrajectoryFixtureError() from None


class Stage2Trajectory(_Stage2TrajectoryModel):
    """A preflighted offline experiment fixture with no deferred file access."""

    schema_version: Literal["stage2-trajectory/v1"] = STAGE2_TRAJECTORY_VERSION
    fixture_id: ComponentIdentifier
    fixture_digest: Sha256Digest
    run_id: UUID4
    records: Annotated[
        tuple[Stage2TrajectoryRecord, ...],
        Field(min_length=1, max_length=_MAX_RECORDS, repr=False),
    ]

    @model_validator(mode="after")
    def records_form_one_exact_fixture(self) -> Self:
        try:
            records = tuple(_copy_record(record) for record in self.records)
            inputs = _validate_inputs(tuple(record.event_input for record in records))
            expected_digest = _fixture_digest(inputs, fixture_id=self.fixture_id)
            total_bytes = 0
            for ordinal, record in enumerate(records, start=1):
                line = canonical_json(record)
                total_bytes += len(line) + 1
                if (
                    record.ordinal != ordinal
                    or record.fixture_id != self.fixture_id
                    or record.fixture_digest != expected_digest
                    or len(line) > _MAX_LINE_BYTES
                    or total_bytes > _MAX_FIXTURE_BYTES
                ):
                    raise ValueError
            if self.run_id != inputs[0].draft.run_id or self.fixture_digest != expected_digest:
                raise ValueError
            return self
        except Exception:
            raise ValueError("offline trajectory records failed validation") from None

    @property
    def inputs(self) -> tuple[FixedStepEventInput, ...]:
        return tuple(record.event_input for record in self.records)

    @property
    def canonical_bytes(self) -> bytes:
        return b"".join(canonical_json(record) + b"\n" for record in self.records)


def _validate_fixture_id(value: object) -> str:
    if type(value) is not str or _COMPONENT_ID.fullmatch(value) is None:
        raise Stage2TrajectoryFixtureError()
    return value


def build_stage2_trajectory(
    inputs: tuple[FixedStepEventInput, ...],
    *,
    fixture_id: str = PAPER_TWO_PHASE_BASIC_TRAJECTORY_ID,
) -> Stage2Trajectory:
    """Purely seal reviewed event inputs into records and canonical JSONL bytes."""

    try:
        checked_id = _validate_fixture_id(fixture_id)
        checked_inputs = _validate_inputs(inputs)
        fixture_digest = _fixture_digest(checked_inputs, fixture_id=checked_id)
        records = tuple(
            Stage2TrajectoryRecord(
                fixture_id=checked_id,
                fixture_digest=fixture_digest,
                ordinal=ordinal,
                event_input=item,
                input_digest=_input_digest(item),
            )
            for ordinal, item in enumerate(checked_inputs, start=1)
        )
        trajectory = Stage2Trajectory(
            fixture_id=checked_id,
            fixture_digest=fixture_digest,
            run_id=checked_inputs[0].draft.run_id,
            records=records,
        )
        return trajectory
    except Stage2TrajectoryFixtureError:
        raise
    except Exception:
        raise Stage2TrajectoryFixtureError() from None


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, item in pairs:
        if key in result:
            raise Stage2TrajectoryFixtureError()
        result[key] = item
    return result


def _reject_constant(_value: str) -> None:
    raise Stage2TrajectoryFixtureError()


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
            elif type(item) is float:
                if not math.isfinite(item):
                    return False
            elif item is not None and type(item) not in (str, bool, int):
                return False
        return True
    except Exception:
        return False


def _load_line(line: bytes) -> Stage2TrajectoryRecord:
    if not line or len(line) > _MAX_LINE_BYTES:
        raise Stage2TrajectoryFixtureError()
    try:
        payload = json.loads(
            line.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        if type(payload) is not dict or payload.keys() != _RECORD_KEYS:
            raise Stage2TrajectoryFixtureError()
        if not _json_is_bounded(payload):
            raise Stage2TrajectoryFixtureError()
        record = Stage2TrajectoryRecord.model_validate_json(line)
        checked = _copy_record(record)
        if canonical_json(checked) != line:
            raise Stage2TrajectoryFixtureError()
        return checked
    except Stage2TrajectoryFixtureError:
        raise
    except Exception:
        raise Stage2TrajectoryFixtureError() from None


def _same_file_identity(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        stat.S_ISREG(after.st_mode)
        and after.st_nlink == 1
        and before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
        and before.st_ctime_ns == after.st_ctime_ns
    )


def load_stage2_trajectory(
    path: str | PathLike[str],
    *,
    expected_fixture_digest: str | None = None,
    expected_fixture_id: str = PAPER_TWO_PHASE_BASIC_TRAJECTORY_ID,
) -> Stage2Trajectory:
    """Read, fully attest, and close one regular canonical JSONL fixture."""

    if type(path) is not str and not isinstance(path, PathLike):
        raise Stage2TrajectoryFixtureError()
    checked_id = _validate_fixture_id(expected_fixture_id)
    if expected_fixture_digest is not None and (
        type(expected_fixture_digest) is not str
        or _SHA256.fullmatch(expected_fixture_digest) is None
    ):
        raise Stage2TrajectoryFixtureError()

    records: list[Stage2TrajectoryRecord] = []
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
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_FIXTURE_BYTES
        ):
            raise Stage2TrajectoryFixtureError()
        stream = os.fdopen(descriptor, "rb", closefd=True)
        descriptor = None
        with stream as source:
            total_bytes = 0
            line_number = 0
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
                    raise Stage2TrajectoryFixtureError()
                record = _load_line(raw_line[:-1])
                if record.ordinal != line_number:
                    raise Stage2TrajectoryFixtureError()
                records.append(record)
            after = os.fstat(source.fileno())
            current = os.stat(raw_path, follow_symlinks=False)
            if not _same_file_identity(before, after) or not _same_file_identity(before, current):
                raise Stage2TrajectoryFixtureError()
        inputs = tuple(record.event_input for record in records)
        trajectory = Stage2Trajectory(
            fixture_id=checked_id,
            fixture_digest=records[0].fixture_digest,
            run_id=inputs[0].draft.run_id,
            records=tuple(records),
        )
        if (
            trajectory.fixture_id != checked_id
            or (
                expected_fixture_digest is not None
                and trajectory.fixture_digest != expected_fixture_digest
            )
            or trajectory.canonical_bytes
            != b"".join(canonical_json(record) + b"\n" for record in records)
        ):
            raise Stage2TrajectoryFixtureError()
        return trajectory
    except Stage2TrajectoryFixtureError:
        raise
    except Exception:
        raise Stage2TrajectoryFixtureError() from None
    finally:
        if descriptor is not None:
            with suppress(Exception):
                os.close(descriptor)


__all__ = [
    "PAPER_TWO_PHASE_BASIC_TRAJECTORY_ID",
    "STAGE2_TRAJECTORY_RECORD_SCHEMA_VERSION",
    "STAGE2_TRAJECTORY_VERSION",
    "Stage2Trajectory",
    "Stage2TrajectoryFixtureError",
    "Stage2TrajectoryRecord",
    "build_stage2_trajectory",
    "load_stage2_trajectory",
]
