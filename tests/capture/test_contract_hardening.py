from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from pydantic import ValidationError

from saliencegate.capture import capabilities as capabilities_module
from saliencegate.capture import schema as schema_module
from saliencegate.capture.adapters import (
    CaptureAdapterCapabilities,
    CaptureAdapterContractError,
    validated_capture_adapter,
)
from saliencegate.capture.capabilities import (
    CapabilitySupport,
    CaptureCapabilityError,
    CaptureCapabilityManifest,
    CaptureCapabilityRegistry,
    CaptureDetectorCapability,
    CaptureEventCapability,
    CaptureFixtureBinding,
    CaptureProfile,
    CompatibilityStatus,
    capture_capability_digest,
    capture_profile,
    classify_capture_compatibility,
    load_capture_capability_registry,
)
from saliencegate.capture.identities import CaptureDigestContext, CaptureIdentityError
from saliencegate.capture.schema import (
    CaptureEvent,
    CaptureJSONLimits,
    CaptureSchemaError,
    canonical_capture_event,
    canonical_capture_intake,
    load_capture_event,
    read_bounded_json,
    validate_capture_event,
    validate_capture_intake,
)
from saliencegate.security import InstallationKey

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64
DIGEST_E = "e" * 64
CAPABILITY_DIGEST = capture_capability_digest(capture_profile(CaptureProfile.CODEX_HOOKS_V1))


def _session_payload() -> dict[str, object]:
    return {
        "schema_version": "capture-intake/v1",
        "kind": "session_started",
        "adapter_profile": "codex-hooks/v1",
        "capability_manifest_digest": CAPABILITY_DIGEST,
        "connection_id": "connection-1",
        "session_id": DIGEST_B,
        "producer_event_digest": DIGEST_C,
        "intake_tag": DIGEST_D,
        "occurred_at": None,
        "timestamp_authority": "unavailable",
        "producer_sequence": None,
        "sequence_authority": "unavailable",
        "capture_disposition": "captured",
    }


def _finished_payload() -> dict[str, object]:
    return {
        **_session_payload(),
        "kind": "action_finished",
        "call_ref": DIGEST_E,
        "outcome_status": "failed",
        "outcome_authority": "producer_claimed_structured",
        "exit_status": 1,
        "error_code": "tool_error",
        "failure_signature": DIGEST_A,
    }


def _first_profile() -> CaptureCapabilityManifest:
    return capture_profile(CaptureProfile.CODEX_HOOKS_V1)


def test_capability_nested_field_sets_and_detector_claims_fail_closed() -> None:
    profile = _first_profile()
    event = profile.events[0]
    detector = profile.detectors[0]

    invalid_events = (
        event.model_copy(update={"critical_fields": tuple(reversed(event.critical_fields))}),
        event.model_copy(update={"optional_fields": (event.critical_fields[0],)}),
    )
    for invalid in invalid_events:
        with pytest.raises(ValidationError):
            CaptureEventCapability.model_validate(invalid)

    invalid_detectors = (
        detector.model_copy(update={"omissions": (*detector.omissions, detector.omissions[0])}),
        detector.model_copy(update={"support": CapabilitySupport.SUPPORTED}),
        detector.model_copy(update={"omissions": ()}),
    )
    for invalid in invalid_detectors:
        with pytest.raises(ValidationError):
            CaptureDetectorCapability.model_validate(invalid)


def test_fixture_paths_and_manifest_closure_invariants_fail_closed() -> None:
    profile = _first_profile()
    fixture = profile.fixtures[0]
    traversal = fixture.model_copy(update={"path": "fixtures/../synthetic.json"})
    with pytest.raises(ValidationError):
        CaptureFixtureBinding.model_validate(traversal)

    invalid_manifests = (
        profile.model_copy(update={"official_sources": (profile.official_sources[0],) * 2}),
        profile.model_copy(update={"events": (*profile.events, profile.events[0])}),
        profile.model_copy(
            update={"tool_coverage": (*profile.tool_coverage, profile.tool_coverage[0])}
        ),
        profile.model_copy(
            update={"coverage_exclusions": tuple(reversed(profile.coverage_exclusions))}
        ),
        profile.model_copy(update={"detectors": tuple(reversed(profile.detectors))}),
        profile.model_copy(
            update={
                "fixtures": (
                    fixture.model_copy(update={"fixture_id": "other-profile-synthetic/v1"}),
                )
            }
        ),
    )
    for invalid in invalid_manifests:
        with pytest.raises(ValidationError):
            CaptureCapabilityManifest.model_validate(invalid)

    registry = load_capture_capability_registry()
    wrong_order = registry.model_copy(update={"profiles": tuple(reversed(registry.profiles))})
    with pytest.raises(ValidationError):
        CaptureCapabilityRegistry.model_validate(wrong_order)

    assert repr(profile) == "CaptureCapabilityManifest(<redacted>)"


class _ResourceBytes:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def joinpath(self, _name: str) -> _ResourceBytes:
        return self

    def read_bytes(self) -> bytes:
        return self._payload


def test_registry_loader_rejects_noncanonical_and_malformed_installed_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = (
        capabilities_module.resources.files("saliencegate.integrations")
        .joinpath("profiles.json")
        .read_bytes()
    )

    for payload in (source + b"\n", b'{"native-secret":'):
        monkeypatch.setattr(
            capabilities_module.resources,
            "files",
            lambda _package, payload=payload: _ResourceBytes(payload),
        )
        with pytest.raises(CaptureCapabilityError) as captured:
            load_capture_capability_registry()
        assert "native-secret" not in str(captured.value)


def test_manifest_entry_points_revalidate_types_and_hide_dependency_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _first_profile()
    invalid = profile.model_copy(update={"official_sources": (profile.official_sources[0],) * 2})

    with pytest.raises(CaptureCapabilityError):
        capture_capability_digest(object())  # type: ignore[arg-type]
    with pytest.raises(CaptureCapabilityError):
        capture_capability_digest(invalid)
    with pytest.raises(CaptureCapabilityError):
        capture_profile(profile.profile_id.value)  # type: ignore[arg-type]

    def fail_canonical_json(_value: object) -> bytes:
        raise RuntimeError("native-manifest-secret")

    monkeypatch.setattr(capabilities_module, "canonical_json", fail_canonical_json)
    with pytest.raises(CaptureCapabilityError) as captured:
        capture_capability_digest(profile)
    assert "native-manifest-secret" not in str(captured.value)


@pytest.mark.parametrize(
    ("host_version", "observed_event", "observed_fields"),
    (
        ("01.2.3", "SessionStart", frozenset()),
        ("0.144.6", 7, frozenset()),
        ("0.144.6", "SessionStart", {"session_id"}),
        ("0.144.6", "SessionStart", frozenset({7})),
    ),
)
def test_compatibility_classifier_rejects_invalid_boundary_types(
    host_version: object,
    observed_event: object,
    observed_fields: object,
) -> None:
    profile = _first_profile()
    assert (
        classify_capture_compatibility(
            profile,
            host_version=host_version,  # type: ignore[arg-type]
            observed_event=observed_event,  # type: ignore[arg-type]
            observed_fields=observed_fields,  # type: ignore[arg-type]
        )
        is CompatibilityStatus.INCOMPATIBLE
    )


def test_compatibility_classifier_rejects_an_invalid_manifest_instance() -> None:
    profile = _first_profile().model_copy(update={"events": ()})
    assert (
        classify_capture_compatibility(
            profile,
            host_version="0.144.6",
            observed_event="SessionStart",
            observed_fields=frozenset({"hook_event_name", "session_id"}),
        )
        is CompatibilityStatus.INCOMPATIBLE
    )


@pytest.mark.parametrize(
    "invalid",
    (
        {"max_bytes": 0, "max_depth": 1, "max_items": 1, "max_string_bytes": 1},
        {"max_bytes": True, "max_depth": 1, "max_items": 1, "max_string_bytes": 1},
    ),
)
def test_json_limits_require_exact_positive_integers(invalid: dict[str, object]) -> None:
    with pytest.raises(CaptureSchemaError):
        CaptureJSONLimits(**invalid)  # type: ignore[arg-type]


def test_schema_identifiers_outcomes_and_event_chain_fail_closed() -> None:
    with pytest.raises(CaptureSchemaError):
        validate_capture_intake({**_session_payload(), "connection_id": 1})

    unavailable_with_evidence = {
        **_finished_payload(),
        "outcome_status": "failed",
        "outcome_authority": "unavailable",
        "exit_status": None,
        "error_code": None,
        "failure_signature": None,
    }
    succeeded_with_failure = {
        **_finished_payload(),
        "outcome_status": "succeeded",
        "outcome_authority": "producer_claimed_structured",
    }
    for invalid in (unavailable_with_evidence, succeeded_with_failure):
        with pytest.raises(CaptureSchemaError):
            validate_capture_intake(invalid)

    intake = validate_capture_intake(_session_payload())
    for ordinal, previous in ((1, DIGEST_A), (2, None)):
        with pytest.raises(ValidationError):
            CaptureEvent(
                receipt_ordinal=ordinal,
                previous_event_tag=previous,
                event_tag=DIGEST_E,
                intake=intake,
            )


def test_bounded_json_rejects_oversized_containers_nonfinite_numbers_and_surrogates() -> None:
    tight_items = CaptureJSONLimits(
        max_bytes=256,
        max_depth=8,
        max_items=1,
        max_string_bytes=256,
    )
    normal = CaptureJSONLimits(
        max_bytes=256,
        max_depth=8,
        max_items=32,
        max_string_bytes=256,
    )

    for document, limits in (
        (b'{"a":1,"b":2}', tight_items),
        (b'{"value":1e400}', normal),
        (b'{"value":"\\ud800"}', normal),
        (b"[]", normal),
    ):
        with pytest.raises(CaptureSchemaError):
            read_bounded_json(document, limits=limits)

    assert read_bounded_json(b'{"value":1.5}', limits=normal)["value"] == 1.5


def test_schema_codec_boundaries_reject_invalid_and_noncanonical_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intake = validate_capture_intake(_session_payload())
    event = CaptureEvent(
        receipt_ordinal=1,
        previous_event_tag=None,
        event_tag=DIGEST_E,
        intake=intake,
    )

    with pytest.raises(CaptureSchemaError):
        canonical_capture_intake(object())  # type: ignore[arg-type]
    with pytest.raises(CaptureSchemaError):
        validate_capture_event({"schema_version": "capture-event/v1"})
    with pytest.raises(CaptureSchemaError):
        canonical_capture_event(object())  # type: ignore[arg-type]

    noncanonical = json.dumps(
        json.loads(canonical_capture_event(event)),
        indent=2,
    ).encode()
    with pytest.raises(CaptureSchemaError):
        load_capture_event(noncanonical)

    def fail_canonical_json(_value: object) -> bytes:
        raise RuntimeError("native-codec-secret")

    monkeypatch.setattr(schema_module, "canonical_json", fail_canonical_json)
    for callback, value in (
        (canonical_capture_intake, intake),
        (canonical_capture_event, event),
    ):
        with pytest.raises(CaptureSchemaError) as captured:
            callback(value)  # type: ignore[arg-type]
        assert "native-codec-secret" not in str(captured.value)


def test_digest_context_covers_deepcopy_and_hides_hmac_backend_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = CaptureDigestContext(InstallationKey(b"k" * 32))
    with pytest.raises(TypeError):
        copy.deepcopy(context)

    def fail_hmac(
        _self: InstallationKey,
        _message: bytes,
        *,
        domain: bytes,
    ) -> str:
        del domain
        raise RuntimeError("native-hmac-secret")

    monkeypatch.setattr(InstallationKey, "_hmac_sha256", fail_hmac)
    with pytest.raises(CaptureIdentityError) as captured:
        context.session_id(b"native-session")
    assert "native-hmac-secret" not in str(captured.value)


class _ConformingAdapter:
    def __init__(self, declaration: CaptureAdapterCapabilities) -> None:
        self._declaration = declaration

    def capabilities(self) -> CaptureAdapterCapabilities:
        return self._declaration

    def adapt_bytes(
        self,
        source: bytes,
        *,
        context: CaptureDigestContext,
    ) -> tuple[object, ...]:
        del source, context
        return ()


class _HostileAttributeAdapter:
    def __getattribute__(self, name: str) -> Any:
        if name in {"capabilities", "adapt_bytes"}:
            raise RuntimeError("native-adapter-secret")
        return super().__getattribute__(name)


def test_adapter_rejects_hostile_attribute_access_and_nonexact_capability_state() -> None:
    with pytest.raises(CaptureAdapterContractError) as captured:
        validated_capture_adapter(_HostileAttributeAdapter())
    assert "native-adapter-secret" not in str(captured.value)

    profile = _first_profile()
    declaration = CaptureAdapterCapabilities(
        profile_id=profile.profile_id,
        capability_digest=capture_capability_digest(profile),
        host_version=profile.host_version,
    ).model_copy(update={"profile_id": profile.profile_id.value})
    with pytest.raises(CaptureAdapterContractError):
        validated_capture_adapter(_ConformingAdapter(declaration))
