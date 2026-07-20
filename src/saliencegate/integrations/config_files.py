"""Opaque, byte-preserving edits for provider-owned configuration spans."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import tomllib
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from pathlib import Path, PureWindowsPath
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from saliencegate.domain import canonical_json
from saliencegate.security.windows import (
    NativeWindowsSecurityOperations,
    WindowsPathKind,
    authorize_windows_managed_path,
)

MAX_PROVIDER_CONFIG_BYTES = 2 * 1_024 * 1_024
MAX_OWNED_CONFIG_BYTES = 256 * 1_024
_MARKER = re.compile(r"^saliencegate-owned:[a-z0-9][a-z0-9._-]{0,95}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOML_KEY = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ConfigFileError(ValueError):
    """A content-free configuration boundary failure."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("provider configuration edit failed")


class ConfigSyntax(StrEnum):
    JSON_OBJECT = "json_object"
    OPAQUE_TEXT = "opaque_text"
    TOML_DOCUMENT = "toml_document"


class _ConfigModel(BaseModel):
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


class TomlBooleanConstraint(_ConfigModel):
    """Require one existing TOML boolean path to have a safe value."""

    path: Annotated[
        tuple[
            Annotated[
                str,
                StringConstraints(min_length=1, max_length=64, pattern=_TOML_KEY.pattern),
            ],
            ...,
        ],
        Field(min_length=1, max_length=8),
    ]
    expected: bool


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigFileError()
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise ConfigFileError()


def _strict_json_loads(value: bytes) -> object:
    try:
        return json.loads(
            value,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except ConfigFileError:
        raise
    except Exception:
        raise ConfigFileError() from None


def _strict_toml_loads(value: bytes) -> dict[str, object]:
    try:
        return tomllib.loads(value.decode("utf-8", errors="strict"))
    except Exception:
        raise ConfigFileError() from None


class OwnedConfigSpec(_ConfigModel):
    """One provider-owned fragment and its schema-neutral ownership marker."""

    syntax: ConfigSyntax
    marker: Annotated[
        str,
        StringConstraints(min_length=20, max_length=114, pattern=_MARKER.pattern),
    ]
    owned_fragment: Annotated[bytes, Field(min_length=1, max_length=MAX_OWNED_CONFIG_BYTES)] = (
        Field(repr=False)
    )
    toml_boolean_constraints: Annotated[
        tuple[TomlBooleanConstraint, ...],
        Field(max_length=16),
    ] = ()
    bind_json_paths: bool = Field(default=False, exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def fragment_is_closed_and_uniquely_marked(self) -> Self:
        marker = self.marker.encode("ascii")
        if self.owned_fragment.count(marker) != 1 or b"\x00" in self.owned_fragment:
            raise ValueError("owned configuration marker is invalid")
        try:
            self.owned_fragment.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("owned configuration fragment is invalid") from None
        if self.syntax is ConfigSyntax.JSON_OBJECT:
            decoded = _strict_json_loads(b"{" + self.owned_fragment + b"}")
            if type(decoded) is not dict or len(decoded) != 1:
                raise ValueError("owned JSON configuration fragment is invalid")
        elif self.syntax is ConfigSyntax.TOML_DOCUMENT:
            _strict_toml_loads(self.owned_fragment)
        if self.toml_boolean_constraints:
            paths = tuple(item.path for item in self.toml_boolean_constraints)
            if self.syntax is not ConfigSyntax.TOML_DOCUMENT or paths != tuple(sorted(set(paths))):
                raise ValueError("owned TOML constraints are invalid")
        if self.bind_json_paths and self.syntax is not ConfigSyntax.JSON_OBJECT:
            raise ValueError("owned JSON path binding is invalid")
        return self


class OwnedConfigSpan(_ConfigModel):
    """One bounded insertion span containing only integration-owned bytes."""

    start: Annotated[int, Field(ge=0, le=MAX_PROVIDER_CONFIG_BYTES)]
    end: Annotated[int, Field(ge=1, le=MAX_PROVIDER_CONFIG_BYTES)]
    owned_span_digest: Annotated[str, StringConstraints(pattern=_SHA256.pattern)]
    owned_span_base64: Annotated[str, StringConstraints(min_length=4, max_length=400_000)] = Field(
        repr=False
    )
    json_path: Annotated[
        tuple[Annotated[str, StringConstraints(min_length=1, max_length=1_024)], ...] | None,
        Field(max_length=8, exclude_if=lambda value: value is None),
    ] = None

    @model_validator(mode="after")
    def span_is_self_consistent(self) -> Self:
        try:
            span = base64.b64decode(self.owned_span_base64, validate=True)
        except Exception:
            raise ValueError("owned configuration span is invalid") from None
        if (
            self.end <= self.start
            or self.end - self.start != len(span)
            or len(span) > MAX_OWNED_CONFIG_BYTES + 8
            or hashlib.sha256(span).hexdigest() != self.owned_span_digest
        ):
            raise ValueError("owned configuration span is invalid")
        return self

    def owned_span(self) -> bytes:
        try:
            return base64.b64decode(self.owned_span_base64, validate=True)
        except Exception:
            raise ConfigFileError() from None


class OwnedConfigReverseEdit(OwnedConfigSpan):
    """Minimal reverse edit; it never contains the foreign configuration preimage."""

    schema_version: Literal["owned-config-reverse-edit/v1"] = "owned-config-reverse-edit/v1"
    syntax: ConfigSyntax
    target_existed: bool
    preimage_digest: Annotated[str, StringConstraints(pattern=_SHA256.pattern)]
    installed_digest: Annotated[str, StringConstraints(pattern=_SHA256.pattern)]
    marker: Annotated[str, StringConstraints(pattern=_MARKER.pattern)]
    additional_spans: Annotated[
        tuple[OwnedConfigSpan, ...],
        Field(max_length=15, exclude_if=lambda value: not value),
    ] = ()

    @model_validator(mode="after")
    def reverse_edit_is_self_consistent(self) -> Self:
        all_spans: tuple[OwnedConfigSpan, ...] = (self, *self.additional_spans)
        spans = tuple(sorted(all_spans, key=lambda item: item.start))
        if tuple(self.additional_spans) != tuple(
            sorted(self.additional_spans, key=lambda item: item.start)
        ):
            raise ValueError("owned configuration reverse edit is invalid")
        if any(left.end > right.start for left, right in pairwise(spans)):
            raise ValueError("owned configuration reverse edit is invalid")
        payloads = tuple(span.owned_span() for span in spans)
        marker = self.marker.encode("ascii")
        if (
            sum(len(payload) for payload in payloads) > MAX_OWNED_CONFIG_BYTES + 128
            or sum(payload.count(marker) for payload in payloads) != 1
            or self.owned_span().count(marker) != 1
            or len(set(payloads)) != len(payloads)
            or (self.additional_spans and self.json_path is None)
            or any(span.json_path is None for span in self.additional_spans)
            or (
                self.syntax is not ConfigSyntax.JSON_OBJECT
                and any(span.json_path is not None for span in spans)
            )
            or (self.syntax is not ConfigSyntax.JSON_OBJECT and self.additional_spans)
        ):
            raise ValueError("owned configuration reverse edit is invalid")
        return self

    def owned_spans(self) -> tuple[OwnedConfigSpan, ...]:
        all_spans: tuple[OwnedConfigSpan, ...] = (self, *self.additional_spans)
        return tuple(sorted(all_spans, key=lambda item: item.start))


class OwnedConfigPlan(_ConfigModel):
    installed_bytes: Annotated[
        bytes,
        Field(min_length=1, max_length=MAX_PROVIDER_CONFIG_BYTES, repr=False),
    ]
    reverse_edit: OwnedConfigReverseEdit


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _owned_span(
    start: int,
    span: bytes,
    *,
    json_path: tuple[str, ...] | None = None,
) -> OwnedConfigSpan:
    return OwnedConfigSpan(
        start=start,
        end=start + len(span),
        owned_span_digest=_digest(span),
        owned_span_base64=base64.b64encode(span).decode("ascii"),
        json_path=json_path,
    )


def _reverse_edit(
    *,
    syntax: ConfigSyntax,
    target_existed: bool,
    original: bytes,
    installed: bytes,
    marker: str,
    spans: tuple[tuple[int, bytes, tuple[str, ...]], ...],
    bind_json_paths: bool,
) -> OwnedConfigReverseEdit:
    marker_bytes = marker.encode("ascii")
    marked = tuple(item for item in spans if item[1].count(marker_bytes) == 1)
    if len(marked) != 1 or sum(span.count(marker_bytes) for _start, span, _path in spans) != 1:
        raise ConfigFileError()
    main_start, main_bytes, _main_path = marked[0]
    additional = tuple(
        _owned_span(start, span, json_path=json_path)
        for start, span, json_path in sorted(spans, key=lambda item: item[0])
        if (start, span, json_path) != marked[0]
    )
    return OwnedConfigReverseEdit(
        syntax=syntax,
        target_existed=target_existed,
        preimage_digest=_digest(original),
        installed_digest=_digest(installed),
        marker=marker,
        start=main_start,
        end=main_start + len(main_bytes),
        owned_span_digest=_digest(main_bytes),
        owned_span_base64=base64.b64encode(main_bytes).decode("ascii"),
        json_path=(_main_path if bind_json_paths or len(spans) > 1 or _main_path else None),
        additional_spans=additional,
    )


def _validated_spec(value: object) -> OwnedConfigSpec:
    try:
        payload = (
            value.model_dump(mode="python", warnings="error")
            if type(value) is OwnedConfigSpec
            else value
        )
        return OwnedConfigSpec.model_validate(payload)
    except Exception:
        raise ConfigFileError() from None


def _skip_json_whitespace(value: bytes, start: int) -> int:
    while start < len(value) and value[start] in b" \t\r\n":
        start += 1
    return start


def _json_string_end(value: bytes, start: int) -> int:
    if start >= len(value) or value[start] != ord('"'):
        raise ConfigFileError()
    cursor = start + 1
    while cursor < len(value):
        current = value[cursor]
        if current == ord('"'):
            return cursor + 1
        if current == ord("\\"):
            cursor += 2
        else:
            cursor += 1
    raise ConfigFileError()


def _json_value_end(value: bytes, start: int) -> int:
    cursor = _skip_json_whitespace(value, start)
    if cursor >= len(value):
        raise ConfigFileError()
    if value[cursor] == ord('"'):
        return _json_string_end(value, cursor)
    if value[cursor] not in b"[{":
        end = cursor
        while end < len(value) and value[end] not in b" \t\r\n,]}":
            end += 1
        if end == cursor:
            raise ConfigFileError()
        return end

    expected_closers = [ord("]") if value[cursor] == ord("[") else ord("}")]
    cursor += 1
    while cursor < len(value) and expected_closers:
        current = value[cursor]
        if current == ord('"'):
            cursor = _json_string_end(value, cursor)
            continue
        if current == ord("["):
            expected_closers.append(ord("]"))
        elif current == ord("{"):
            expected_closers.append(ord("}"))
        elif current in b"]}" and current != expected_closers.pop():
            raise ConfigFileError()
        cursor += 1
    if expected_closers:
        raise ConfigFileError()
    return cursor


def _json_object_member_value_span(
    document: bytes,
    member: str,
) -> tuple[int, int] | None:
    cursor = _skip_json_whitespace(document, 0)
    if cursor >= len(document) or document[cursor] != ord("{"):
        raise ConfigFileError()
    cursor = _skip_json_whitespace(document, cursor + 1)
    while cursor < len(document) and document[cursor] != ord("}"):
        key_start = cursor
        key_end = _json_string_end(document, key_start)
        decoded_key = _strict_json_loads(document[key_start:key_end])
        if type(decoded_key) is not str:
            raise ConfigFileError()
        cursor = _skip_json_whitespace(document, key_end)
        if cursor >= len(document) or document[cursor] != ord(":"):
            raise ConfigFileError()
        value_start = _skip_json_whitespace(document, cursor + 1)
        value_end = _json_value_end(document, value_start)
        if decoded_key == member:
            return value_start, value_end
        cursor = _skip_json_whitespace(document, value_end)
        if cursor >= len(document) or document[cursor] not in b",}":
            raise ConfigFileError()
        if document[cursor] == ord(","):
            cursor = _skip_json_whitespace(document, cursor + 1)
    return None


def _json_path_value_span(document: bytes, path: tuple[str, ...]) -> tuple[int, int]:
    start = 0
    end = len(document)
    for member in path:
        relative = _json_object_member_value_span(document[start:end], member)
        if relative is None:
            raise ConfigFileError()
        relative_start, relative_end = relative
        start, end = start + relative_start, start + relative_end
    return start, end


def _json_insertion_semantics(span: bytes, container: object) -> object:
    core = span[1:] if span.startswith(b",") else span
    if type(container) is list:
        decoded = _strict_json_loads(b"[" + core + b"]")
        if type(decoded) is not list or not decoded:
            raise ConfigFileError()
        return decoded
    if type(container) is dict:
        decoded = _strict_json_loads(b"{" + core + b"}")
        if type(decoded) is not dict or not decoded:
            raise ConfigFileError()
        return decoded
    raise ConfigFileError()


def _json_string_values(value: object) -> tuple[str, ...]:
    if type(value) is str:
        return (value,)
    if type(value) is list:
        return tuple(item for nested in value for item in _json_string_values(nested))
    if type(value) is dict:
        return tuple(item for nested in value.values() for item in _json_string_values(nested))
    return ()


def _direct_json_span_offsets(container: bytes, span: bytes) -> tuple[int, ...]:
    raw_offsets: list[int] = []
    cursor = 0
    while True:
        start = container.find(span, cursor)
        if start < 0:
            break
        raw_offsets.append(start)
        if len(raw_offsets) > 64:
            raise ConfigFileError()
        cursor = start + 1

    leading_comma = span.startswith(b",")
    core_starts = {start + int(leading_comma) for start in raw_offsets}
    ends = {start + len(span) for start in raw_offsets}
    next_nonwhitespace: dict[int, int | None] = {}
    next_index: int | None = None
    queries = core_starts | ends
    for index in range(len(container) - 1, -1, -1):
        if container[index] not in b" \t\r\n":
            next_index = index
        if index in queries:
            next_nonwhitespace[index] = next_index
    if len(container) in queries:
        next_nonwhitespace[len(container)] = None

    candidate_positions: list[tuple[int, int, int | None, int | None]] = []
    for start in raw_offsets:
        end = start + len(span)
        core_start = start + int(leading_comma)
        candidate_positions.append(
            (
                start,
                end,
                next_nonwhitespace.get(core_start),
                next_nonwhitespace.get(end),
            )
        )

    needed_states = {
        position
        for start, end, first, following in candidate_positions
        for position in (start, end, first, following)
        if position is not None
    }
    starts = set(raw_offsets)
    preceding: dict[int, int | None] = {}
    states: dict[int, tuple[int, bool]] = {}
    depth = 0
    in_string = False
    escaped = False
    previous_nonwhitespace: int | None = None
    for index, value in enumerate(container):
        if index in starts:
            preceding[index] = previous_nonwhitespace
        if index in needed_states:
            states[index] = (depth, in_string)
        if in_string:
            if escaped:
                escaped = False
            elif value == ord("\\"):
                escaped = True
            elif value == ord('"'):
                in_string = False
        elif value == ord('"'):
            in_string = True
        elif value in b"[{":
            depth += 1
        elif value in b"]}":
            depth -= 1
        if value not in b" \t\r\n":
            previous_nonwhitespace = index
    if len(container) in needed_states:
        states[len(container)] = (depth, in_string)
    if depth != 0 or in_string:
        raise ConfigFileError()

    offsets: list[int] = []
    for start, end, first, following in candidate_positions:
        previous = preceding[start]
        if (
            first is None
            or first >= end
            or following is None
            or states.get(first) != (1, False)
            or states.get(end) != (1, False)
            or states.get(following) != (1, False)
            or container[following : following + 1] not in b",]}"
            or (
                leading_comma
                and (states.get(start) != (1, False) or container[start : start + 1] != b",")
            )
            or (
                not leading_comma
                and (previous is None or container[previous : previous + 1] not in b"[{,")
            )
        ):
            continue
        offsets.append(start)
    return tuple(offsets)


def _locate_json_owned_spans(
    document: bytes,
    spans: tuple[OwnedConfigSpan, ...],
) -> tuple[tuple[int, bytes, tuple[str, ...] | None, object | None], ...]:
    located: list[tuple[int, bytes, tuple[str, ...] | None, object | None]] = []
    for owned in spans:
        span = owned.owned_span()
        if owned.json_path is not None:
            container_start, container_end = _json_path_value_span(document, owned.json_path)
            container_bytes = document[container_start:container_end]
            container = _strict_json_loads(container_bytes)
            semantics = _json_insertion_semantics(span, container)
        else:
            container_start = 0
            container_bytes = document
            container = _strict_json_loads(container_bytes)
            semantics = None
        offsets = _direct_json_span_offsets(container_bytes, span)
        if len(offsets) != 1:
            raise ConfigFileError()
        if type(container) not in {dict, list}:
            raise ConfigFileError()
        start = container_start + offsets[0]
        located.append((start, span, owned.json_path, semantics))
    ordered = tuple(sorted(located, key=lambda item: item[0], reverse=True))
    ascending = tuple(reversed(ordered))
    if any(left[0] + len(left[1]) > right[0] for left, right in pairwise(ascending)):
        raise ConfigFileError()
    return ordered


def _validate_json_owned_semantics_removed(
    document: bytes,
    located: tuple[tuple[int, bytes, tuple[str, ...] | None, object | None], ...],
) -> None:
    for _start, _span, path, semantics in located:
        if path is None or semantics is None:
            continue
        container_start, container_end = _json_path_value_span(document, path)
        container = _strict_json_loads(document[container_start:container_end])
        identity_values = tuple(
            sorted(
                (value for value in _json_string_values(semantics) if len(value) >= 20),
                key=len,
                reverse=True,
            )
        )
        if type(container) is list and type(semantics) is list:
            if any(item in container for item in semantics) or (
                identity_values and identity_values[0] in _json_string_values(container)
            ):
                raise ConfigFileError()
        elif type(container) is dict and type(semantics) is dict:
            if any(
                key in container and container[key] == value for key, value in semantics.items()
            ) or (identity_values and identity_values[0] in _json_string_values(container)):
                raise ConfigFileError()
        else:
            raise ConfigFileError()


def _owned_config_edit_matches_spec(
    edit: OwnedConfigReverseEdit,
    spec: OwnedConfigSpec,
) -> bool:
    """Match a reverse edit to either a top-level or nested owned JSON span."""

    try:
        checked = _validated_spec(spec)
        if type(edit) is not OwnedConfigReverseEdit:
            return False
        if edit.syntax is not checked.syntax or edit.marker != checked.marker:
            return False
        if checked.bind_json_paths and edit.json_path is None:
            return False
        spans = tuple(item.owned_span() for item in edit.owned_spans())
        separator = b"," if checked.syntax is ConfigSyntax.JSON_OBJECT else b"\n"
        candidates = {checked.owned_fragment, separator + checked.owned_fragment}
        if checked.syntax is not ConfigSyntax.JSON_OBJECT:
            return len(spans) == 1 and spans[0] in candidates
        document = b"{" + checked.owned_fragment + b"}"
        decoded = _strict_json_loads(document)
        if type(decoded) is not dict or len(decoded) != 1:
            return False
        _key, value = next(iter(decoded.items()))
        if type(value) is not dict or not value:
            return len(spans) == 1 and spans[0] in candidates
        value_span = _json_object_member_value_span(document, _key)
        if value_span is None:
            return False
        value_start, value_end = value_span
        members = document[value_start + 1 : value_end - 1]
        candidates.update((members, b"," + members))

        def matches_owned_piece(span: bytes) -> bool:
            if span in candidates:
                return True
            core = span[1:] if span.startswith(b",") else span
            try:
                object_piece = _strict_json_loads(b"{" + core + b"}")
            except ConfigFileError:
                object_piece = None
            if (
                type(object_piece) is dict
                and object_piece
                and set(object_piece) <= set(value)
                and all(object_piece[key] == value[key] for key in object_piece)
            ):
                return True
            try:
                array_piece = _strict_json_loads(b"[" + core + b"]")
            except ConfigFileError:
                return False
            return type(array_piece) is list and any(
                array_piece == groups for groups in value.values()
            )

        return all(matches_owned_piece(span) for span in spans)
    except Exception:
        return False


def _json_object_insertion(
    current: bytes,
    fragment: bytes,
) -> tuple[bytes, tuple[tuple[int, bytes, tuple[str, ...]], ...]]:
    decoded = _strict_json_loads(current)
    if type(decoded) is not dict:
        raise ConfigFileError()
    owned = _strict_json_loads(b"{" + fragment + b"}")
    if type(owned) is not dict or len(owned) != 1:
        raise ConfigFileError()
    owned_key, owned_value = next(iter(owned.items()))
    if owned_key in decoded:
        existing_value = decoded[owned_key]
        if type(existing_value) is not dict or type(owned_value) is not dict or not owned_value:
            raise ConfigFileError()
        current_value_span = _json_object_member_value_span(current, owned_key)
        if current_value_span is None:
            raise ConfigFileError()
        current_value_start, current_value_end = current_value_span
        if (
            current[current_value_start : current_value_start + 1] != b"{"
            or current[current_value_end - 1 : current_value_end] != b"}"
        ):
            raise ConfigFileError()
        if set(existing_value).isdisjoint(owned_value):
            owned_document = b"{" + fragment + b"}"
            owned_value_span = _json_object_member_value_span(owned_document, owned_key)
            if owned_value_span is None:
                raise ConfigFileError()
            owned_value_start, owned_value_end = owned_value_span
            owned_members = owned_document[owned_value_start + 1 : owned_value_end - 1]
            separator = b"" if not existing_value else b","
            span = separator + owned_members
            close = current_value_end - 1
            return current[:close] + span + current[close:], ((close, span, (owned_key,)),)
        current_object = current[current_value_start:current_value_end]
        operations: list[tuple[int, bytes, tuple[str, ...]]] = []
        missing: list[bytes] = []
        for member, owned_member_value in owned_value.items():
            if member not in existing_value:
                encoded_member = canonical_json({member: owned_member_value})[1:-1]
                if not encoded_member:
                    raise ConfigFileError()
                missing.append(encoded_member)
                continue
            existing_member_value = existing_value[member]
            if (
                type(existing_member_value) is not list
                or type(owned_member_value) is not list
                or not owned_member_value
            ):
                raise ConfigFileError()
            member_span = _json_object_member_value_span(current_object, member)
            if member_span is None:
                raise ConfigFileError()
            member_start, member_end = member_span
            if (
                current_object[member_start : member_start + 1] != b"["
                or current_object[member_end - 1 : member_end] != b"]"
            ):
                raise ConfigFileError()
            owned_members = canonical_json(owned_member_value)[1:-1]
            if not owned_members or owned_members in current:
                raise ConfigFileError()
            separator = b"," if existing_member_value else b""
            operations.append(
                (
                    current_value_start + member_end - 1,
                    separator + owned_members,
                    (owned_key, member),
                )
            )
        if missing:
            separator = b"," if existing_value else b""
            operations.append(
                (
                    current_value_end - 1,
                    separator + b",".join(missing),
                    (owned_key,),
                )
            )
        if not operations or len(operations) > 16:
            raise ConfigFileError()
        ordered = tuple(sorted(operations, key=lambda item: item[0]))
        installed = current
        for position, span, _json_path in reversed(ordered):
            installed = installed[:position] + span + installed[position:]
        shifted: list[tuple[int, bytes, tuple[str, ...]]] = []
        offset = 0
        for position, span, json_path in ordered:
            shifted.append((position + offset, span, json_path))
            offset += len(span)
        return installed, tuple(shifted)
    end = len(current)
    while end and current[end - 1] in b" \t\r\n":
        end -= 1
    if end < 2 or current[end - 1 : end] != b"}":
        raise ConfigFileError()
    close = end - 1
    start = 0
    while start < close and current[start] in b" \t\r\n":
        start += 1
    if current[start : start + 1] != b"{":
        raise ConfigFileError()
    separator = b"" if not decoded else b","
    span = separator + fragment
    return current[:close] + span + current[close:], ((close, span, ()),)


def _opaque_insertion(current: bytes, fragment: bytes) -> tuple[bytes, int, bytes]:
    try:
        current.decode("utf-8")
    except UnicodeDecodeError:
        raise ConfigFileError() from None
    separator = b"" if not current or current.endswith((b"\n", b"\r")) else b"\n"
    span = separator + fragment
    return current + span, len(current), span


def _validate_toml_boolean_constraints(
    document: dict[str, object],
    constraints: tuple[TomlBooleanConstraint, ...],
) -> None:
    for constraint in constraints:
        current: object = document
        missing = False
        for component in constraint.path:
            if type(current) is not dict:
                raise ConfigFileError()
            if component not in current:
                missing = True
                break
            current = current[component]
        if missing:
            continue
        if type(current) is not bool or current is not constraint.expected:
            raise ConfigFileError()


def plan_owned_config_install(
    current: bytes | None,
    spec: OwnedConfigSpec,
) -> OwnedConfigPlan:
    """Plan one insertion without normalizing or retaining any foreign bytes."""

    try:
        checked = _validated_spec(spec)
        if current is not None and type(current) is not bytes:
            raise ConfigFileError()
        target_existed = current is not None
        original = (
            b"{}"
            if current is None and checked.syntax is ConfigSyntax.JSON_OBJECT
            else (b"" if current is None else current)
        )
        if len(original) > MAX_PROVIDER_CONFIG_BYTES:
            raise ConfigFileError()
        marker = checked.marker.encode("ascii")
        if marker in original:
            raise ConfigFileError()
        if checked.syntax is ConfigSyntax.JSON_OBJECT:
            installed, spans = _json_object_insertion(original, checked.owned_fragment)
        else:
            installed, start, span = _opaque_insertion(original, checked.owned_fragment)
            spans = ((start, span, ()),)
        if checked.syntax is ConfigSyntax.TOML_DOCUMENT:
            original_document = _strict_toml_loads(original)
            installed_document = _strict_toml_loads(installed)
            _validate_toml_boolean_constraints(
                original_document,
                checked.toml_boolean_constraints,
            )
            _validate_toml_boolean_constraints(
                installed_document,
                checked.toml_boolean_constraints,
            )
        if len(installed) > MAX_PROVIDER_CONFIG_BYTES:
            raise ConfigFileError()
        reverse = _reverse_edit(
            syntax=checked.syntax,
            target_existed=target_existed,
            original=original,
            installed=installed,
            marker=checked.marker,
            spans=spans,
            bind_json_paths=checked.bind_json_paths,
        )
        return OwnedConfigPlan(installed_bytes=installed, reverse_edit=reverse)
    except ConfigFileError:
        raise
    except Exception:
        raise ConfigFileError() from None


def remove_owned_config_edit(
    current: bytes,
    reverse_edit: OwnedConfigReverseEdit,
) -> bytes | None:
    """Reverse an untouched edit exactly, or remove one unique owned span after drift."""

    try:
        if type(current) is not bytes or len(current) > MAX_PROVIDER_CONFIG_BYTES:
            raise ConfigFileError()
        payload = (
            reverse_edit.model_dump(mode="python", warnings="error")
            if type(reverse_edit) is OwnedConfigReverseEdit
            else reverse_edit
        )
        checked = OwnedConfigReverseEdit.model_validate(payload)
        spans = checked.owned_spans()
        if _digest(current) == checked.installed_digest:
            restored = current
            for owned in reversed(spans):
                span = owned.owned_span()
                if restored[owned.start : owned.end] != span:
                    raise ConfigFileError()
                restored = restored[: owned.start] + restored[owned.end :]
            if _digest(restored) != checked.preimage_digest:
                raise ConfigFileError()
            return None if not checked.target_existed else restored
        if current.count(checked.marker.encode("ascii")) != 1:
            raise ConfigFileError()
        if checked.syntax is ConfigSyntax.JSON_OBJECT:
            if type(_strict_json_loads(current)) is not dict:
                raise ConfigFileError()
            restored = current
            remaining = list(spans)
            removed: list[tuple[int, bytes, tuple[str, ...] | None, object | None]] = []
            while remaining:
                located = _locate_json_owned_spans(restored, tuple(remaining))
                start, span, path, semantics = located[0]
                before = restored[:start]
                after = restored[start + len(span) :]
                candidates = [before + after]
                following = next(
                    (index for index, value in enumerate(after) if value not in b" \t\r\n"),
                    None,
                )
                if following is not None and after[following : following + 1] == b",":
                    candidates.append(before + after[:following] + after[following + 1 :])
                preceding = next(
                    (
                        index
                        for index in range(len(before) - 1, -1, -1)
                        if before[index] not in b" \t\r\n"
                    ),
                    None,
                )
                if preceding is not None and before[preceding : preceding + 1] == b",":
                    candidates.append(before[:preceding] + before[preceding + 1 :] + after)
                valid: list[bytes] = []
                for candidate in candidates:
                    try:
                        decoded = _strict_json_loads(candidate)
                    except ConfigFileError:
                        continue
                    if type(decoded) is dict and candidate not in valid:
                        valid.append(candidate)
                if len(valid) != 1:
                    raise ConfigFileError()
                restored = valid[0]
                removed.append((start, span, path, semantics))
                remaining = [owned for owned in remaining if owned.owned_span() != span]
            _validate_json_owned_semantics_removed(restored, tuple(removed))
        elif checked.syntax is ConfigSyntax.TOML_DOCUMENT:
            span = checked.owned_span()
            if current.count(span) != 1:
                raise ConfigFileError()
            start = current.index(span)
            restored = current[:start] + current[start + len(span) :]
            _strict_toml_loads(current)
            _strict_toml_loads(restored)
        else:
            span = checked.owned_span()
            if current.count(span) != 1:
                raise ConfigFileError()
            start = current.index(span)
            restored = current[:start] + current[start + len(span) :]
        if len(restored) > MAX_PROVIDER_CONFIG_BYTES:
            raise ConfigFileError()
        return restored
    except ConfigFileError:
        raise
    except Exception:
        raise ConfigFileError() from None


@dataclass(frozen=True, slots=True, repr=False)
class _ConfigSnapshot:
    data: bytes | None
    identity: tuple[int, int, int, int, int, int] | None
    mode: int


def _safe_parent(path: Path) -> os.stat_result:
    metadata = path.parent.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or (os.name == "posix" and hasattr(os, "getuid") and metadata.st_uid != os.getuid())
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise ConfigFileError()
    return metadata


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_mode,
    )


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        metadata.st_mode,
    )


def _safe_ancestor(metadata: os.stat_result, *, leaf: bool) -> bool:
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        return False
    mode = stat.S_IMODE(metadata.st_mode)
    if leaf:
        return metadata.st_uid == os.getuid() and not mode & 0o022
    return metadata.st_uid in (0, os.getuid()) and (not mode & 0o022 or bool(mode & stat.S_ISVTX))


def _open_config_parent(path: Path) -> tuple[int, tuple[int, int, int, int]]:
    if os.name != "posix" or not hasattr(os, "getuid") or path.anchor != os.sep:
        raise ConfigFileError()
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(os.sep, flags)
    try:
        components = path.parent.parts[1:]
        for index, component in enumerate(components):
            child = os.open(component, flags, dir_fd=descriptor)
            try:
                opened = os.fstat(child)
                named = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
                if (
                    _directory_identity(opened) != _directory_identity(named)
                    or not _safe_ancestor(opened, leaf=index == len(components) - 1)
                    or not _safe_ancestor(named, leaf=index == len(components) - 1)
                ):
                    raise ConfigFileError()
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        parent_identity = _directory_identity(os.fstat(descriptor))
        return descriptor, parent_identity
    except BaseException:
        os.close(descriptor)
        raise


def _revalidate_config_parent(
    path: Path,
    descriptor: int,
    expected: tuple[int, int, int, int],
) -> None:
    if _directory_identity(os.fstat(descriptor)) != expected or not _safe_ancestor(
        os.fstat(descriptor), leaf=True
    ):
        raise ConfigFileError()
    fresh, fresh_identity = _open_config_parent(path)
    try:
        if fresh_identity != expected:
            raise ConfigFileError()
    finally:
        os.close(fresh)


def _snapshot_at(
    path: Path,
    descriptor: int,
    parent_identity: tuple[int, int, int, int],
) -> _ConfigSnapshot:
    _revalidate_config_parent(path, descriptor, parent_identity)
    try:
        named = os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        _revalidate_config_parent(path, descriptor, parent_identity)
        try:
            os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return _ConfigSnapshot(data=None, identity=None, mode=0o600)
        raise ConfigFileError() from None
    if (
        not stat.S_ISREG(named.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or named.st_nlink != 1
        or named.st_size > MAX_PROVIDER_CONFIG_BYTES
        or named.st_uid != os.getuid()
        or stat.S_IMODE(named.st_mode) & 0o022
    ):
        raise ConfigFileError()
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    opened_descriptor = os.open(path.name, flags, dir_fd=descriptor)
    try:
        opened = os.fstat(opened_descriptor)
        data = bytearray()
        while len(data) <= MAX_PROVIDER_CONFIG_BYTES:
            chunk = os.read(
                opened_descriptor,
                min(64 * 1024, MAX_PROVIDER_CONFIG_BYTES + 1 - len(data)),
            )
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(opened_descriptor)
    finally:
        os.close(opened_descriptor)
    current = os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
    expected = _identity(named)
    if (
        _identity(opened) != expected
        or _identity(after) != expected
        or _identity(current) != expected
        or len(data) != named.st_size
        or len(data) > MAX_PROVIDER_CONFIG_BYTES
    ):
        raise ConfigFileError()
    _revalidate_config_parent(path, descriptor, parent_identity)
    return _ConfigSnapshot(
        data=bytes(data),
        identity=expected,
        mode=stat.S_IMODE(named.st_mode),
    )


def _snapshot(path: Path) -> _ConfigSnapshot:
    if os.name == "posix":
        descriptor, parent_identity = _open_config_parent(path)
        try:
            return _snapshot_at(path, descriptor, parent_identity)
        finally:
            os.close(descriptor)
    _safe_parent(path)
    try:
        named = path.lstat()
    except FileNotFoundError:
        return _ConfigSnapshot(data=None, identity=None, mode=0o600)
    if (
        not stat.S_ISREG(named.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or named.st_nlink != 1
        or named.st_size > MAX_PROVIDER_CONFIG_BYTES
        or (os.name == "posix" and hasattr(os, "getuid") and named.st_uid != os.getuid())
    ):
        raise ConfigFileError()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        data = bytearray()
        while len(data) <= MAX_PROVIDER_CONFIG_BYTES:
            chunk = os.read(descriptor, min(64 * 1024, MAX_PROVIDER_CONFIG_BYTES + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = path.lstat()
    expected = _identity(named)
    if (
        _identity(opened) != expected
        or _identity(after) != expected
        or _identity(current) != expected
        or len(data) != named.st_size
        or len(data) > MAX_PROVIDER_CONFIG_BYTES
    ):
        raise ConfigFileError()
    return _ConfigSnapshot(
        data=bytes(data),
        identity=expected,
        mode=stat.S_IMODE(named.st_mode),
    )


def read_config_bytes(path: Path) -> bytes | None:
    try:
        if not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts:
            raise ConfigFileError()
        if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
            operations = NativeWindowsSecurityOperations()
            windows_path = PureWindowsPath(os.fspath(path))
            parent = authorize_windows_managed_path(
                windows_path.parent,
                kind=WindowsPathKind.DIRECTORY,
                operations=operations,
            )
            if operations.inspect_path(windows_path) is None:
                parent.revalidate()
                if operations.inspect_path(windows_path) is not None:
                    raise ConfigFileError()
                return None
            stable = operations.read_managed_file(
                windows_path,
                maximum_bytes=MAX_PROVIDER_CONFIG_BYTES,
            )
            parent.revalidate()
            stable.authorization.revalidate()
            return stable.data
        return _snapshot(path).data
    except ConfigFileError:
        raise
    except Exception:
        raise ConfigFileError() from None


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_config_bytes(path: Path, *, expected: bytes | None, data: bytes) -> None:
    """Atomically replace exactly the bytes observed by the caller."""

    temporary: str | None = None
    parent_descriptor: int | None = None
    try:
        if (
            not isinstance(path, Path)
            or not path.is_absolute()
            or ".." in path.parts
            or (expected is not None and type(expected) is not bytes)
            or type(data) is not bytes
            or len(data) > MAX_PROVIDER_CONFIG_BYTES
            or (not data and expected is None)
        ):
            raise ConfigFileError()
        if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
            operations = NativeWindowsSecurityOperations()
            published = operations.publish_managed_file(
                PureWindowsPath(os.fspath(path)),
                data,
                maximum_bytes=MAX_PROVIDER_CONFIG_BYTES,
                validate_replacement=(
                    None
                    if expected is None
                    else lambda current: hmac.compare_digest(current, expected)
                ),
                validate_published=lambda current: hmac.compare_digest(current, data),
            )
            if not hmac.compare_digest(published.data, data):
                raise ConfigFileError()
            return
        parent_descriptor, parent_identity = _open_config_parent(path)
        snapshot = _snapshot_at(path, parent_descriptor, parent_identity)
        if snapshot.data != expected:
            raise ConfigFileError()
        temporary = f".{path.name}.{secrets.token_hex(16)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(
            temporary,
            flags,
            snapshot.mode,
            dir_fd=parent_descriptor,
        )
        try:
            os.fchmod(descriptor, snapshot.mode)
            view = memoryview(data)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise ConfigFileError()
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if _snapshot_at(path, parent_descriptor, parent_identity).identity != snapshot.identity:
            raise ConfigFileError()
        if snapshot.identity is None:
            os.link(
                temporary,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            os.unlink(temporary, dir_fd=parent_descriptor)
            temporary = None
        else:
            os.replace(
                temporary,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            temporary = None
        os.fsync(parent_descriptor)
        if _snapshot_at(path, parent_descriptor, parent_identity).data != data:
            raise ConfigFileError()
        _revalidate_config_parent(path, parent_descriptor, parent_identity)
    except ConfigFileError:
        raise
    except Exception:
        raise ConfigFileError() from None
    finally:
        if temporary is not None and parent_descriptor is not None:
            with suppress(OSError):
                os.unlink(temporary, dir_fd=parent_descriptor)
        if parent_descriptor is not None:
            with suppress(OSError):
                os.close(parent_descriptor)


def delete_config_bytes(path: Path, *, expected: bytes) -> None:
    """Delete only the exact user-owned config bytes supplied by the caller."""

    try:
        if type(expected) is not bytes:
            raise ConfigFileError()
        if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
            operations = NativeWindowsSecurityOperations()
            windows_path = PureWindowsPath(os.fspath(path))
            parent = authorize_windows_managed_path(
                windows_path.parent,
                kind=WindowsPathKind.DIRECTORY,
                operations=operations,
            )
            stable = operations.read_managed_file(
                windows_path,
                maximum_bytes=MAX_PROVIDER_CONFIG_BYTES,
            )
            if not hmac.compare_digest(stable.data, expected):
                raise ConfigFileError()
            parent.revalidate()
            operations.delete_authorized_file(stable.authorization)
            parent.revalidate()
            if operations.inspect_path(windows_path) is not None:
                raise ConfigFileError()
            return
        if os.name == "posix":
            descriptor, parent_identity = _open_config_parent(path)
            try:
                if _snapshot_at(path, descriptor, parent_identity).data != expected:
                    raise ConfigFileError()
                os.unlink(path.name, dir_fd=descriptor)
                os.fsync(descriptor)
                if _snapshot_at(path, descriptor, parent_identity).data is not None:
                    raise ConfigFileError()
                _revalidate_config_parent(path, descriptor, parent_identity)
                return
            finally:
                os.close(descriptor)
        if _snapshot(path).data != expected:
            raise ConfigFileError()
        path.unlink()
        _fsync_directory(path.parent)
        if path.exists() or path.is_symlink():
            raise ConfigFileError()
    except ConfigFileError:
        raise
    except Exception:
        raise ConfigFileError() from None


__all__ = [
    "MAX_PROVIDER_CONFIG_BYTES",
    "ConfigFileError",
    "ConfigSyntax",
    "OwnedConfigPlan",
    "OwnedConfigReverseEdit",
    "OwnedConfigSpec",
    "delete_config_bytes",
    "plan_owned_config_install",
    "publish_config_bytes",
    "read_config_bytes",
    "remove_owned_config_edit",
]
