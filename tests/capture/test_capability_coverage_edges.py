"""Coverage-oriented regressions for otherwise rare contract edges."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import pytest

import saliencegate.capture.capabilities as capabilities_module
from saliencegate.capture.capabilities import (
    CaptureCapabilityError,
    CaptureProfile,
    _fixture_path_present,
    capture_capability_digest,
    capture_profile,
    load_capture_capability_registry,
)
from saliencegate.capture.schema import CAPTURE_NATIVE_JSON_LIMITS, read_bounded_json


class _FixtureResource:
    def __init__(self, source: bytes) -> None:
        self.source = source

    def joinpath(self, *_parts: str) -> _FixtureResource:
        return self

    def read_bytes(self) -> bytes:
        return self.source


def _first_fixture_material() -> tuple[object, object, bytes, Mapping[str, object]]:
    registry = load_capture_capability_registry()
    profile = registry.profiles[0]
    fixture = profile.fixtures[0]
    source = (
        capabilities_module.resources.files("saliencegate.integrations")
        .joinpath(*capabilities_module.PurePosixPath(fixture.path).parts)
        .read_bytes()
    )
    body = read_bounded_json(source, limits=CAPTURE_NATIVE_JSON_LIMITS)
    assert isinstance(body, Mapping)
    return registry, profile, source, body


def test_exact_manifest_string_and_fixture_path_helpers_reject_runtime_shapes() -> None:
    with pytest.raises(ValueError, match="string"):
        capabilities_module._require_exact_string(1)  # type: ignore[arg-type]
    assert not _fixture_path_present((), "events[].name")
    assert not _fixture_path_present({}, "events[].name")
    assert not _fixture_path_present({}, "missing")
    assert _fixture_path_present({"value": 1}, "value")


def test_fixture_binding_rejects_parent_traversal_even_on_defensive_revalidation() -> None:
    fixture = capture_profile(CaptureProfile.CODEX_HOOKS_V1).fixtures[0]
    forged = fixture.model_copy(update={"path": "fixtures/../hostile.json"})
    with pytest.raises(ValueError, match="path"):
        forged.fixture_path_is_relative()


def test_manifest_validator_rejects_non_manifest_and_digest_serialization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(CaptureCapabilityError):
        capabilities_module._validated_manifest(object())

    profile = capture_profile(CaptureProfile.CODEX_HOOKS_V1)
    monkeypatch.setattr(
        capabilities_module,
        "canonical_json",
        lambda _value: (_ for _ in ()).throw(RuntimeError()),
    )
    with pytest.raises(CaptureCapabilityError):
        capture_capability_digest(profile)


def test_fixture_resource_digest_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = load_capture_capability_registry()
    monkeypatch.setattr(
        capabilities_module.resources,
        "files",
        lambda _package: _FixtureResource(b"tampered"),
    )
    with pytest.raises(CaptureCapabilityError):
        capabilities_module._validate_fixture_resources(registry)


@pytest.mark.parametrize(
    "scenario",
    [
        "wrong_keys",
        "wrong_header",
        "invalid_event_shape",
        "invalid_event_types",
        "unknown_event",
        "incomplete_event_set",
    ],
)
def test_fixture_resource_shape_failures_are_all_rejected(
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
    registry, profile, source, original = _first_fixture_material()
    registry = registry.model_copy(update={"profiles": (profile,)})
    body = dict(original)
    original_events = body["events"]
    assert type(original_events) is tuple
    first = original_events[0]
    assert isinstance(first, Mapping)

    if scenario == "wrong_keys":
        body.pop("provenance")
    elif scenario == "wrong_header":
        body["schema_version"] = "capture-native-fixture/v2"
    elif scenario == "invalid_event_shape":
        body["events"] = ("not-a-mapping",)
    elif scenario == "invalid_event_types":
        changed = dict(first)
        changed["event_name"] = 1
        body["events"] = (changed,)
    elif scenario == "unknown_event":
        changed = dict(first)
        changed["event_name"] = "unknown_event"
        body["events"] = (changed,)
    else:
        final_name = profile.events[-1].event_name
        body["events"] = tuple(
            event for event in original_events if event["event_name"] != final_name
        )

    monkeypatch.setattr(
        capabilities_module.resources,
        "files",
        lambda _package: _FixtureResource(source),
    )
    monkeypatch.setattr(capabilities_module, "read_bounded_json", lambda *_a, **_k: body)
    monkeypatch.setattr(capabilities_module, "canonical_json", lambda _value: source)
    with pytest.raises(CaptureCapabilityError):
        capabilities_module._validate_fixture_resources(registry)


def test_fixture_resource_rejects_missing_critical_payload_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, profile, source, original = _first_fixture_material()
    registry = registry.model_copy(update={"profiles": (profile,)})
    events = original["events"]
    assert type(events) is tuple
    selected = next(
        event
        for event in events
        if next(
            item for item in profile.events if item.event_name == event["event_name"]
        ).critical_fields
    )
    changed = dict(selected)
    changed["payload"] = {}
    body = dict(original)
    body["events"] = (changed,)
    monkeypatch.setattr(
        capabilities_module.resources,
        "files",
        lambda _package: _FixtureResource(source),
    )
    monkeypatch.setattr(capabilities_module, "read_bounded_json", lambda *_a, **_k: body)
    monkeypatch.setattr(capabilities_module, "canonical_json", lambda _value: source)
    with pytest.raises(CaptureCapabilityError):
        capabilities_module._validate_fixture_resources(registry)


def test_verified_registry_loader_maps_digest_canonical_and_unexpected_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = capabilities_module._load_verified_capture_capability_registry
    loader.cache_clear()
    with pytest.raises(CaptureCapabilityError):
        loader("0" * 64)

    source = (
        capabilities_module.resources.files("saliencegate.integrations")
        .joinpath(capabilities_module._REGISTRY_RESOURCE)
        .read_bytes()
    )
    assert hashlib.sha256(source).hexdigest() == capabilities_module._REGISTRY_RESOURCE_SHA256
    monkeypatch.setattr(
        capabilities_module.resources,
        "files",
        lambda _package: _FixtureResource(source),
    )
    monkeypatch.setattr(capabilities_module, "canonical_json", lambda _value: b"different")
    loader.cache_clear()
    with pytest.raises(CaptureCapabilityError):
        loader(capabilities_module._REGISTRY_RESOURCE_SHA256)

    monkeypatch.setattr(
        capabilities_module.resources,
        "files",
        lambda _package: (_ for _ in ()).throw(RuntimeError()),
    )
    loader.cache_clear()
    with pytest.raises(CaptureCapabilityError):
        loader(capabilities_module._REGISTRY_RESOURCE_SHA256)
    loader.cache_clear()


def test_public_registry_and_profile_boundaries_map_unexpected_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        capabilities_module,
        "_load_verified_capture_capability_registry",
        lambda _digest: (_ for _ in ()).throw(RuntimeError()),
    )
    with pytest.raises(CaptureCapabilityError):
        load_capture_capability_registry()

    registry = capabilities_module.CaptureCapabilityRegistry.model_construct(
        schema_version="capture-capability-registry/v1",
        profiles=(),
    )
    monkeypatch.setattr(capabilities_module, "load_capture_capability_registry", lambda: registry)
    with pytest.raises(CaptureCapabilityError):
        capture_profile(CaptureProfile.CODEX_HOOKS_V1)
