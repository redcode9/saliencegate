from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest

import saliencegate.adapters.jsonl as jsonl_module
from saliencegate.adapters.jsonl import (
    JSONLReplayAdapter,
    JsonlReplayError,
    JsonlReplayEvent,
    encode_jsonl_trace,
)
from saliencegate.domain import (
    DeliveryTarget,
    EventPhase,
    EventType,
    NormalizedTraceEventDraft,
    TrustLabel,
    canonical_digest,
    canonical_json,
)
from saliencegate.ports.adapters import AdapterNormalizationError, AdapterTargetResolutionError
from saliencegate.security import SecureFileError, StableReadPolicy

RUN_ID = UUID("00000000-0000-4000-8000-000000007101")
EVENT_1_ID = UUID("00000000-0000-4000-8000-000000007102")
EVENT_2_ID = UUID("00000000-0000-4000-8000-000000007103")
EVENT_3_ID = UUID("00000000-0000-4000-8000-000000007104")
NOW = datetime(2026, 7, 11, 15, 0, tzinfo=UTC)


def _draft(
    source_event_id: str,
    *,
    run_id: UUID = RUN_ID,
    parent_ids: tuple[UUID, ...] = (),
    timestamp_offset: int = 0,
    payload: object | None = None,
) -> NormalizedTraceEventDraft:
    return NormalizedTraceEventDraft.model_validate(
        {
            "run_id": run_id,
            "source_event_id": source_event_id,
            "timestamp": NOW + timedelta(seconds=timestamp_offset),
            "event_type": EventType.OBSERVATION,
            "phase": EventPhase.POST_ACTION,
            "payload": {"message": source_event_id} if payload is None else payload,
            "parent_ids": parent_ids,
            "source_adapter": "jsonl.fixture/1",
            "trust_label": TrustLabel.SYNTHETIC_FIXTURE,
        }
    )


def _events() -> tuple[JsonlReplayEvent, ...]:
    return (
        JsonlReplayEvent.create(
            ordinal=1,
            expected_event_id=EVENT_1_ID,
            draft=_draft("event-1"),
            next_model_call_target_request_id="request-next-1",
        ),
        JsonlReplayEvent.create(
            ordinal=2,
            expected_event_id=EVENT_2_ID,
            draft=_draft("event-2", parent_ids=(EVENT_1_ID,), timestamp_offset=1),
            pre_action_target_request_id="request-action-1",
        ),
    )


def _lines(encoded: bytes) -> list[dict[str, object]]:
    return [cast(dict[str, object], json.loads(line)) for line in encoded.splitlines()]


def _rewrite(lines: list[dict[str, object]]) -> bytes:
    return b"\n".join(canonical_json(line) for line in lines) + b"\n"


def _refresh_record_digest(record: dict[str, object]) -> None:
    record["record_digest"] = canonical_digest(
        {key: value for key, value in record.items() if key != "record_digest"}
    )


def _refresh_manifest(lines: list[dict[str, object]]) -> None:
    manifest = lines[0]
    manifest["trace_digest"] = canonical_digest(
        {
            "schema_version": manifest["schema_version"],
            "run_id": manifest["run_id"],
            "record_digests": [record["record_digest"] for record in lines[1:]],
        }
    )


def test_jsonl_round_trip_is_deterministic_and_preflighted_from_all_inputs(
    tmp_path: Path,
) -> None:
    encoded = encode_jsonl_trace(_events())
    path = tmp_path / "trace.jsonl"
    path.write_bytes(encoded)

    from_bytes = JSONLReplayAdapter.from_bytes(encoded)
    from_text = JSONLReplayAdapter.from_text(encoded.decode("utf-8"))
    from_path = JSONLReplayAdapter.from_path(path)

    assert encoded == encode_jsonl_trace(from_bytes.events)
    assert from_bytes.manifest == from_text.manifest == from_path.manifest
    assert from_bytes.events == from_text.events == from_path.events == _events()
    assert from_bytes.run_id == RUN_ID
    assert from_bytes.trace_digest == from_bytes.manifest.trace_digest
    assert from_bytes.expected_event_ids == (EVENT_1_ID, EVENT_2_ID)
    event_id_factory = from_bytes.event_id_factory()
    assert (event_id_factory(), event_id_factory()) == (EVENT_1_ID, EVENT_2_ID)
    with pytest.raises(StopIteration):
        event_id_factory()
    assert len(from_bytes) == 2
    assert tuple(from_bytes) == _events()
    assert from_bytes.events is not from_bytes.events
    assert from_bytes.manifest is not from_bytes.manifest


def test_jsonl_adapter_normalizes_copies_and_resolves_frozen_target_ids() -> None:
    adapter = JSONLReplayAdapter.from_bytes(encode_jsonl_trace(_events()))
    first, second = adapter.events

    normalized = adapter.normalize(first)

    assert normalized == first.draft
    assert normalized is not first.draft
    assert (
        adapter.resolve_target_request_id(first, DeliveryTarget.NEXT_MODEL_CALL) == "request-next-1"
    )
    assert adapter.resolve_target_request_id(first, DeliveryTarget.PRE_ACTION_REPLAN) is None
    assert (
        adapter.resolve_target_request_id(second, DeliveryTarget.PRE_ACTION_REPLAN)
        == "request-action-1"
    )


def test_jsonl_adapter_rejects_non_wrappers_and_invalid_targets() -> None:
    adapter = JSONLReplayAdapter.from_bytes(encode_jsonl_trace(_events()))

    with pytest.raises(AdapterNormalizationError):
        adapter.normalize(_events()[0].draft)
    with pytest.raises(AdapterTargetResolutionError):
        adapter.resolve_target_request_id(_events()[0], cast(DeliveryTarget, "next_model_call"))
    with pytest.raises(AdapterTargetResolutionError):
        adapter.resolve_target_request_id(object(), DeliveryTarget.NEXT_MODEL_CALL)

    foreign = JsonlReplayEvent.create(
        ordinal=1,
        expected_event_id=EVENT_3_ID,
        draft=_draft("foreign-event"),
    )
    with pytest.raises(AdapterNormalizationError):
        adapter.normalize(foreign)


def test_event_builder_revalidates_forged_drafts_and_its_own_fields() -> None:
    forged = _draft("event-1").model_copy(update={"source_event_id": "invalid id"})
    with pytest.raises(AdapterNormalizationError):
        JsonlReplayEvent.create(
            ordinal=1,
            expected_event_id=EVENT_1_ID,
            draft=forged,
        )

    with pytest.raises(JsonlReplayError, match="invalid_record"):
        JsonlReplayEvent.create(
            ordinal=0,
            expected_event_id=EVENT_1_ID,
            draft=_draft("event-1"),
        )


def test_record_digest_tampering_is_rejected_without_exposing_payload() -> None:
    secret = "sk-raw-jsonl-secret"
    lines = _lines(encode_jsonl_trace(_events()))
    cast(dict[str, object], cast(dict[str, object], lines[2]["draft"])["payload"])["message"] = (
        secret
    )

    with pytest.raises(JsonlReplayError) as raised:
        JSONLReplayAdapter.from_bytes(_rewrite(lines))

    assert raised.value.code == "invalid_record"
    assert secret not in str(raised.value)
    assert raised.value.__cause__ is None


def test_trace_digest_tampering_is_rejected() -> None:
    lines = _lines(encode_jsonl_trace(_events()))
    lines[0]["trace_digest"] = "f" * 64

    with pytest.raises(JsonlReplayError) as raised:
        JSONLReplayAdapter.from_bytes(_rewrite(lines))

    assert raised.value.code == "trace_digest_mismatch"


def test_trace_timestamps_cannot_move_backwards() -> None:
    lines = _lines(encode_jsonl_trace(_events()))
    cast(dict[str, object], lines[2]["draft"])["timestamp"] = (
        (NOW - timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    )
    _refresh_record_digest(lines[2])
    _refresh_manifest(lines)

    with pytest.raises(JsonlReplayError) as raised:
        JSONLReplayAdapter.from_bytes(_rewrite(lines))

    assert raised.value.code == "non_monotonic_timestamp"


@pytest.mark.parametrize("version", ("2.0", "garbage", 1))
def test_unknown_or_invalid_manifest_schema_is_rejected(version: object) -> None:
    lines = _lines(encode_jsonl_trace(_events()))
    lines[0]["schema_version"] = version

    with pytest.raises(JsonlReplayError) as raised:
        JSONLReplayAdapter.from_bytes(_rewrite(lines))

    assert raised.value.code == "unsupported_schema"


@pytest.mark.parametrize("version", ("2.0", "1.1"))
def test_unknown_event_wrapper_schema_is_rejected(version: str) -> None:
    lines = _lines(encode_jsonl_trace(_events()))
    lines[1]["schema_version"] = version
    _refresh_record_digest(lines[1])
    _refresh_manifest(lines)

    with pytest.raises(JsonlReplayError) as raised:
        JSONLReplayAdapter.from_bytes(_rewrite(lines))

    assert raised.value.code == "unsupported_schema"


@pytest.mark.parametrize(
    ("mutation", "code"),
    (
        (lambda lines: lines[1].__setitem__("ordinal", 2), "non_contiguous_ordinal"),
        (lambda lines: lines[0].__setitem__("record_count", 3), "record_count_mismatch"),
        (
            lambda lines: cast(dict[str, object], lines[2]["draft"]).__setitem__(
                "run_id", str(EVENT_3_ID)
            ),
            "multiple_runs",
        ),
        (
            lambda lines: cast(dict[str, object], lines[2]["draft"]).__setitem__(
                "source_event_id", "event-1"
            ),
            "duplicate_source_event_id",
        ),
        (
            lambda lines: lines[2].__setitem__("expected_event_id", str(EVENT_1_ID)),
            "duplicate_expected_event_id",
        ),
        (
            lambda lines: cast(dict[str, object], lines[1]["draft"]).__setitem__(
                "parent_ids", [str(EVENT_2_ID)]
            ),
            "parent_not_preceding",
        ),
    ),
)
def test_whole_trace_invariants_are_checked_before_exposure(
    mutation: object,
    code: str,
) -> None:
    lines = _lines(encode_jsonl_trace(_events()))
    cast(object, mutation)(lines)  # type: ignore[operator]
    for record in lines[1:]:
        _refresh_record_digest(record)
    _refresh_manifest(lines)

    with pytest.raises(JsonlReplayError) as raised:
        JSONLReplayAdapter.from_bytes(_rewrite(lines))

    assert raised.value.code == code


@pytest.mark.parametrize(
    ("encoded", "code"),
    (
        (b"", "missing_manifest"),
        (b"\n", "blank_line"),
        (b"not-json\n", "invalid_json"),
        (b"[]\n", "object_required"),
        (b'{"record_type":"trace_manifest","record_type":"other"}\n', "invalid_json"),
    ),
)
def test_malformed_jsonl_is_rejected_with_stable_errors(encoded: bytes, code: str) -> None:
    with pytest.raises(JsonlReplayError) as raised:
        JSONLReplayAdapter.from_bytes(encoded)
    assert raised.value.code == code
    assert raised.value.__cause__ is None


def test_non_finite_json_and_invalid_manifest_or_draft_shapes_are_rejected() -> None:
    with pytest.raises(JsonlReplayError) as raised:
        JSONLReplayAdapter.from_bytes(b'{"value":NaN}\n')
    assert raised.value.code == "invalid_json"

    lines = _lines(encode_jsonl_trace(_events()))
    lines[0].pop("trace_digest")
    with pytest.raises(JsonlReplayError) as raised:
        JSONLReplayAdapter.from_bytes(_rewrite(lines))
    assert raised.value.code == "invalid_manifest"

    lines = _lines(encode_jsonl_trace(_events()))
    lines[1]["draft"] = "not-an-event"
    _refresh_record_digest(lines[1])
    _refresh_manifest(lines)
    with pytest.raises(JsonlReplayError) as raised:
        JSONLReplayAdapter.from_bytes(_rewrite(lines))
    assert raised.value.code == "invalid_record"


def test_wrong_record_types_and_blank_suffix_are_rejected() -> None:
    lines = _lines(encode_jsonl_trace(_events()))
    lines[0]["record_type"] = "trace_event_draft"
    with pytest.raises(JsonlReplayError) as raised:
        JSONLReplayAdapter.from_bytes(_rewrite(lines))
    assert raised.value.code == "invalid_manifest"

    lines = _lines(encode_jsonl_trace(_events()))
    lines[1]["record_type"] = "trace_manifest"
    _refresh_record_digest(lines[1])
    _refresh_manifest(lines)
    with pytest.raises(JsonlReplayError) as raised:
        JSONLReplayAdapter.from_bytes(_rewrite(lines))
    assert raised.value.code == "invalid_record"

    with pytest.raises(JsonlReplayError) as raised:
        JSONLReplayAdapter.from_bytes(encode_jsonl_trace(_events()) + b"\n")
    assert raised.value.code == "blank_line"


def test_input_and_resource_bounds_fail_before_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(JsonlReplayError) as raised:
        JSONLReplayAdapter.from_bytes(cast(bytes, "text"))
    assert raised.value.code == "invalid_input_type"

    with pytest.raises(JsonlReplayError) as raised:
        JSONLReplayAdapter.from_text(cast(str, b"bytes"))
    assert raised.value.code == "invalid_input_type"

    monkeypatch.setattr(jsonl_module, "MAX_TRACE_BYTES", 8)
    with pytest.raises(JsonlReplayError) as raised:
        JSONLReplayAdapter.from_bytes(b"123456789")
    assert raised.value.code == "trace_too_large"

    monkeypatch.setattr(jsonl_module, "MAX_TRACE_BYTES", 1024)
    monkeypatch.setattr(jsonl_module, "MAX_LINE_BYTES", 4)
    with pytest.raises(JsonlReplayError) as raised:
        JSONLReplayAdapter.from_bytes(b"12345\n")
    assert raised.value.code == "line_too_large"

    monkeypatch.setattr(jsonl_module, "MAX_LINE_BYTES", 1024)
    monkeypatch.setattr(jsonl_module, "MAX_TRACE_LINES", 2)
    with pytest.raises(JsonlReplayError) as raised:
        JSONLReplayAdapter.from_bytes(b"{}\n{}\n{}\n")
    assert raised.value.code == "too_many_lines"


def test_invalid_utf8_and_unicode_text_are_rejected_without_raw_data() -> None:
    with pytest.raises(JsonlReplayError) as raised:
        JSONLReplayAdapter.from_bytes(b"\xff\n")
    assert raised.value.code == "invalid_encoding"

    with pytest.raises(JsonlReplayError) as raised:
        JSONLReplayAdapter.from_text("\ud800")
    assert raised.value.code == "invalid_encoding"


def test_path_read_failures_and_non_path_inputs_are_sanitized(tmp_path: Path) -> None:
    missing = tmp_path / "raw-secret-name.jsonl"
    with pytest.raises(JsonlReplayError) as raised:
        JSONLReplayAdapter.from_path(missing)
    assert raised.value.code == "read_failed"
    assert "raw-secret-name" not in str(raised.value)

    with pytest.raises(JsonlReplayError) as raised:
        JSONLReplayAdapter.from_path(cast(Path, object()))
    assert raised.value.code == "invalid_input_type"


def test_path_preflight_rejects_oversized_files_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "large.jsonl"
    path.write_bytes(b"12345")
    monkeypatch.setattr(jsonl_module, "MAX_TRACE_BYTES", 4)

    with pytest.raises(JsonlReplayError) as raised:
        JSONLReplayAdapter.from_path(path)
    assert raised.value.code == "trace_too_large"


@pytest.mark.parametrize("mode", (0o644, 0o666))
def test_path_loader_preserves_legacy_non_private_file_modes(
    tmp_path: Path,
    mode: int,
) -> None:
    encoded = encode_jsonl_trace(_events())
    path = tmp_path / "legacy-permissions.jsonl"
    path.write_bytes(encoded)
    path.chmod(mode)

    adapter = JSONLReplayAdapter.from_path(path)

    assert adapter.events == _events()


def test_path_loader_preserves_legacy_world_writable_parent_acceptance(
    tmp_path: Path,
) -> None:
    encoded = encode_jsonl_trace(_events())
    parent = tmp_path / "legacy-shared"
    parent.mkdir()
    path = parent / "trace.jsonl"
    path.write_bytes(encoded)
    parent.chmod(0o777)
    try:
        adapter = JSONLReplayAdapter.from_path(path)
    finally:
        parent.chmod(0o700)

    assert adapter.events == _events()


def test_path_loader_preserves_legacy_relative_and_dot_component_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded = encode_jsonl_trace(_events())
    (tmp_path / "trace.jsonl").write_bytes(encoded)
    (tmp_path / "nested").mkdir()
    monkeypatch.chdir(tmp_path)

    for path in ("trace.jsonl", "./trace.jsonl", "nested/../trace.jsonl"):
        adapter = JSONLReplayAdapter.from_path(path)
        assert adapter.events == _events()


def test_path_loader_preserves_legacy_symlinked_ancestor_acceptance(tmp_path: Path) -> None:
    encoded = encode_jsonl_trace(_events())
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    (real_parent / "trace.jsonl").write_bytes(encoded)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    adapter = JSONLReplayAdapter.from_path(linked_parent / "trace.jsonl")

    assert adapter.events == _events()


def test_path_loader_delegates_to_the_legacy_stable_read_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded = encode_jsonl_trace(_events())
    path = tmp_path / "trace.jsonl"
    path.write_bytes(encoded)
    delegated: list[tuple[object, int, StableReadPolicy]] = []
    original = jsonl_module.read_stable_file

    def track_delegation(
        value: object,
        *,
        maximum_bytes: int,
        policy: StableReadPolicy,
    ) -> object:
        delegated.append((value, maximum_bytes, policy))
        return original(value, maximum_bytes=maximum_bytes, policy=policy)  # type: ignore[arg-type]

    monkeypatch.setattr(jsonl_module, "read_stable_file", track_delegation)

    adapter = JSONLReplayAdapter.from_path(path)

    assert adapter.events == _events()
    assert delegated == [
        (path, jsonl_module.MAX_TRACE_BYTES, StableReadPolicy.LEGACY_COMPATIBILITY)
    ]


@pytest.mark.parametrize(
    "failure",
    (SecureFileError(), RuntimeError("raw-stable-reader-secret")),
)
def test_path_loader_maps_read_and_stability_failures_to_read_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    secret = "raw-stable-reader-secret"
    path = tmp_path / "changing.jsonl"
    path.write_bytes(encode_jsonl_trace(_events()))

    def fail_read(*_args: object, **_kwargs: object) -> object:
        raise failure

    monkeypatch.setattr(jsonl_module, "read_stable_file", fail_read)

    with pytest.raises(JsonlReplayError) as raised:
        JSONLReplayAdapter.from_path(path)

    assert raised.value.code == "read_failed"
    assert secret not in str(raised.value)
    assert raised.value.__cause__ is None


def test_path_loader_rejects_non_regular_and_hostile_pathlike_inputs(tmp_path: Path) -> None:
    secret = "raw-pathlike-secret"

    class HostilePath:
        def __fspath__(self) -> str:
            raise RuntimeError(secret)

    class BytesPath:
        def __fspath__(self) -> bytes:
            return b"raw-bytes-path"

    for value in (tmp_path, HostilePath(), BytesPath()):
        with pytest.raises(JsonlReplayError) as raised:
            JSONLReplayAdapter.from_path(value)  # type: ignore[arg-type]
        assert raised.value.code == "read_failed"
        assert secret not in str(raised.value)
        assert raised.value.__cause__ is None


def test_path_loader_rejects_symbolic_and_hard_links(tmp_path: Path) -> None:
    source = tmp_path / "trace.jsonl"
    source.write_bytes(encode_jsonl_trace(_events()))
    symbolic = tmp_path / "symbolic.jsonl"
    symbolic.symlink_to(source)
    hard = tmp_path / "hard.jsonl"
    hard.hardlink_to(source)

    for path in (symbolic, hard):
        with pytest.raises(JsonlReplayError) as raised:
            JSONLReplayAdapter.from_path(path)
        assert raised.value.code == "read_failed"
        assert raised.value.__cause__ is None


def test_create_and_encode_reject_invalid_or_empty_event_sequences() -> None:
    with pytest.raises(JsonlReplayError, match="empty_trace"):
        encode_jsonl_trace(())
    with pytest.raises(JsonlReplayError, match="invalid_record"):
        encode_jsonl_trace(cast(tuple[JsonlReplayEvent, ...], (object(),)))

    with pytest.raises(AdapterNormalizationError):
        JsonlReplayEvent.create(
            ordinal=1,
            expected_event_id=EVENT_1_ID,
            draft=cast(NormalizedTraceEventDraft, object()),
        )


def test_constructor_revalidates_forged_models() -> None:
    adapter = JSONLReplayAdapter.from_bytes(encode_jsonl_trace(_events()))
    with pytest.raises(JsonlReplayError, match="invalid_manifest"):
        JSONLReplayAdapter(cast(object, object()), adapter.events)  # type: ignore[arg-type]

    forged_event = adapter.events[0].model_copy(update={"record_digest": "f" * 64})
    with pytest.raises(JsonlReplayError, match="invalid_record"):
        JSONLReplayAdapter(adapter.manifest, (forged_event,))

    forged_manifest = adapter.manifest.model_copy(update={"record_count": 0})
    with pytest.raises(JsonlReplayError, match="invalid_manifest"):
        JSONLReplayAdapter(forged_manifest, adapter.events)


def test_encoder_enforces_bounds_and_sanitizes_iteration_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _events()
    monkeypatch.setattr(jsonl_module, "MAX_TRACE_LINES", 2)
    with pytest.raises(JsonlReplayError) as raised:
        encode_jsonl_trace(events)
    assert raised.value.code == "too_many_lines"

    monkeypatch.setattr(jsonl_module, "MAX_TRACE_LINES", 100)

    def broken_events() -> object:
        yield events[0]
        raise RuntimeError("raw iterator secret")

    with pytest.raises(JsonlReplayError) as raised:
        encode_jsonl_trace(cast(object, broken_events()))  # type: ignore[arg-type]
    assert raised.value.code == "invalid_record"
    assert "secret" not in str(raised.value)

    class HostileSequence:
        def __iter__(self) -> object:
            raise JsonlReplayError("raw-sequence-secret")

    adapter = JSONLReplayAdapter.from_bytes(encode_jsonl_trace(events))
    with pytest.raises(JsonlReplayError) as raised:
        JSONLReplayAdapter(adapter.manifest, cast(object, HostileSequence()))  # type: ignore[arg-type]
    assert raised.value.code == "invalid_record"
    assert "secret" not in str(raised.value)
    assert raised.value.__cause__ is None

    monkeypatch.setattr(jsonl_module, "MAX_TRACE_BYTES", 1)
    with pytest.raises(JsonlReplayError) as raised:
        encode_jsonl_trace((events[0],))
    assert raised.value.code == "trace_too_large"

    monkeypatch.setattr(jsonl_module, "MAX_TRACE_BYTES", 1024 * 1024)
    monkeypatch.setattr(jsonl_module, "MAX_LINE_BYTES", 1)
    with pytest.raises(JsonlReplayError) as raised:
        encode_jsonl_trace((events[0],))
    assert raised.value.code == "line_too_large"


def test_encoder_applies_the_total_byte_bound_before_advancing_the_iterator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _events()
    event_bytes = tuple(len(canonical_json(event)) + 1 for event in events)
    monkeypatch.setattr(jsonl_module, "MAX_TRACE_BYTES", sum(event_bytes) - 1)
    seen: list[int] = []

    def streamed_events() -> object:
        seen.append(1)
        yield events[0]
        seen.append(2)
        yield events[1]
        raise RuntimeError("iterator advanced past the cumulative byte bound")

    with pytest.raises(JsonlReplayError) as raised:
        encode_jsonl_trace(cast(object, streamed_events()))  # type: ignore[arg-type]

    assert raised.value.code == "trace_too_large"
    assert seen == [1, 2]
