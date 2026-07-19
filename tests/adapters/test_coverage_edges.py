from __future__ import annotations

from datetime import UTC, datetime
from os import PathLike
from typing import cast
from uuid import UUID

import pytest

import saliencegate.adapters.jsonl as jsonl_module
from saliencegate.adapters.generic import GenericHarnessAdapter
from saliencegate.adapters.jsonl import (
    JSONLReplayAdapter,
    JsonlReplayError,
    JsonlReplayEvent,
    JsonlTraceManifest,
    encode_jsonl_trace,
)
from saliencegate.domain import (
    EventPhase,
    EventType,
    NormalizedTraceEventDraft,
    TrustLabel,
)
from saliencegate.ports.adapters import AdapterNormalizationError


async def _unused_delivery(_delivery: object) -> None:
    return None


def _generic_adapter(*, event_id_callback: object) -> GenericHarnessAdapter:
    return GenericHarnessAdapter(
        normalize_callback=lambda value: value,
        capabilities_callback=lambda: None,
        target_request_id_callback=lambda _event, _target: None,
        delivery_callback=_unused_delivery,
        event_id_callback=cast(object, event_id_callback),
    )


def _jsonl_adapter() -> JSONLReplayAdapter:
    run_id = UUID("00000000-0000-4000-8000-000000008001")
    event_id = UUID("00000000-0000-4000-8000-000000008002")
    draft = NormalizedTraceEventDraft(
        run_id=run_id,
        source_event_id="coverage-event",
        timestamp=datetime(2026, 7, 12, 10, 0, tzinfo=UTC),
        event_type=EventType.OBSERVATION,
        phase=EventPhase.POST_ACTION,
        payload={"message": "coverage"},
        parent_ids=(),
        source_adapter="jsonl.coverage/1",
        trust_label=TrustLabel.SYNTHETIC_FIXTURE,
    )
    event = JsonlReplayEvent.create(
        ordinal=1,
        expected_event_id=event_id,
        draft=draft,
    )
    return JSONLReplayAdapter.from_bytes(encode_jsonl_trace((event,)))


def test_generic_constructor_rejects_a_non_callable_event_id_callback() -> None:
    with pytest.raises(TypeError, match="adapter callbacks must be callable"):
        _generic_adapter(event_id_callback=object())


def test_generic_event_id_callback_may_explicitly_decline_a_mapping() -> None:
    adapter = _generic_adapter(event_id_callback=lambda _event, _ordinal: None)

    assert adapter.resolve_event_id(object(), 1) is None


def test_jsonl_event_id_resolution_validates_the_requested_ordinal() -> None:
    adapter = _jsonl_adapter()
    event = adapter.events[0]

    assert adapter.resolve_event_id(event, 1) == event.expected_event_id
    with pytest.raises(AdapterNormalizationError):
        adapter.resolve_event_id(event, 2)


def test_jsonl_path_reader_rejects_pathlikes_returning_bytes() -> None:
    class BytesPath:
        def __fspath__(self) -> bytes:
            return b"not-a-text-path"

    with pytest.raises(JsonlReplayError, match="read_failed"):
        JSONLReplayAdapter.from_path(cast(PathLike[str], BytesPath()))


def test_jsonl_constructor_sanitizes_manifest_encoding_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _jsonl_adapter()
    real_canonical_json = jsonl_module.canonical_json

    def fail_for_manifest(value: object) -> bytes:
        if isinstance(value, JsonlTraceManifest):
            raise RuntimeError("raw secret")
        return real_canonical_json(value)

    monkeypatch.setattr(jsonl_module, "canonical_json", fail_for_manifest)

    with pytest.raises(JsonlReplayError, match="invalid_manifest") as raised:
        JSONLReplayAdapter(adapter.manifest, adapter.events)

    assert raised.value.__cause__ is None
