"""Audited capability manifests for provider capture integrations."""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Mapping
from enum import StrEnum
from importlib import resources
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from saliencegate.capture.schema import CAPTURE_NATIVE_JSON_LIMITS, read_bounded_json
from saliencegate.domain import SignalType, canonical_json
from saliencegate.domain.records import Sha256Digest

_REGISTRY_RESOURCE = "profiles.json"
_HOST_VERSION = re.compile(r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CaptureCapabilityError(ValueError):
    """A content-free failure at the capture capability boundary."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("capture capability is invalid")


class CaptureProfile(StrEnum):
    """The complete set of audited Universal Shadow Capture v1 profiles."""

    CODEX_HOOKS_V1 = "codex-hooks/v1"
    CLAUDE_CODE_HOOKS_V1 = "claude-code-hooks/v1"
    OPENCODE_PLUGIN_V1 = "opencode-plugin/v1"
    PI_EXTENSION_V1 = "pi-extension/v1"


class CapabilitySupport(StrEnum):
    """How much evidence a profile can provide to a detector."""

    SUPPORTED = "supported"
    CONDITIONAL = "conditional"
    UNSUPPORTED = "unsupported"


class CompatibilityStatus(StrEnum):
    """Compatibility of one observed native event with an audited profile."""

    VERIFIED = "verified"
    SCHEMA_COMPATIBLE_UNVERIFIED_VERSION = "schema_compatible_unverified_version"
    INCOMPATIBLE = "incompatible"


def _require_exact_string(value: str) -> str:
    if type(value) is not str:
        raise ValueError("capture capability string is invalid")
    return value


_ManifestText: TypeAlias = Annotated[
    str,
    StringConstraints(min_length=1, max_length=2_048),
    AfterValidator(_require_exact_string),
]
_ManifestToken: TypeAlias = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9_@][A-Za-z0-9._@:/+\-\[\]]*$",
    ),
    AfterValidator(_require_exact_string),
]
_HostVersion: TypeAlias = Annotated[
    str,
    StringConstraints(max_length=64, pattern=_HOST_VERSION.pattern),
    AfterValidator(_require_exact_string),
]
_UpstreamRevision: TypeAlias = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{40}$"),
    AfterValidator(_require_exact_string),
]
_OfficialSource: TypeAlias = Annotated[
    str,
    StringConstraints(min_length=9, max_length=2_048, pattern=r"^https://[^\s]+$"),
    AfterValidator(_require_exact_string),
]
_FixturePath: TypeAlias = Annotated[
    str,
    StringConstraints(
        min_length=15,
        max_length=256,
        pattern=r"^fixtures/[A-Za-z0-9_][A-Za-z0-9._/\-]*\.json$",
    ),
    AfterValidator(_require_exact_string),
]

_OutcomeAuthority: TypeAlias = Literal[
    "action_closed_outcome_unavailable",
    "action_observed",
    "confirmed_success_or_ambiguous_error",
    "controller_failure_when_session_correlated",
    "correlation_only",
    "coverage_boundary",
    "coverage_only",
    "flush_only",
    "lineage_hint_only",
    "no_semantic_intake",
    "pre_hook_proposal",
    "provider_claimed_controller_failure",
    "provider_claimed_denial",
    "provider_claimed_failure",
    "provider_claimed_success",
    "provider_claimed_tool_outcome",
    "reconciliation_only",
    "stable_unit_closed",
    "tool_state_discriminator",
    "turn_closed",
    "turn_open",
    "window_closed",
    "window_open",
]


class _CaptureCapabilityModel(BaseModel):
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


class CaptureEventCapability(_CaptureCapabilityModel):
    """The consumed-field and outcome boundary for one provider event."""

    event_name: _ManifestToken
    critical_fields: Annotated[tuple[_ManifestToken, ...], Field(max_length=64)]
    optional_fields: Annotated[tuple[_ManifestToken, ...], Field(max_length=64)]
    ignored_fields: Annotated[tuple[_ManifestToken, ...], Field(max_length=128)]
    outcome_authority: _OutcomeAuthority

    @model_validator(mode="after")
    def field_sets_are_canonical_and_disjoint(self) -> Self:
        groups = (self.critical_fields, self.optional_fields, self.ignored_fields)
        if any(group != tuple(sorted(set(group))) for group in groups):
            raise ValueError("capture event fields are not canonical")
        critical, optional, ignored = (set(group) for group in groups)
        if critical & optional or critical & ignored or optional & ignored:
            raise ValueError("capture event field authorities overlap")
        return self


class CaptureDetectorCapability(_CaptureCapabilityModel):
    """Evidence support and explicit omissions for one detector."""

    signal_type: SignalType
    support: CapabilitySupport
    omissions: Annotated[tuple[_ManifestToken, ...], Field(max_length=32)]

    @model_validator(mode="after")
    def omissions_match_support(self) -> Self:
        if self.omissions != tuple(sorted(set(self.omissions))):
            raise ValueError("capture detector omissions are not canonical")
        if bool(self.omissions) is (self.support is CapabilitySupport.SUPPORTED):
            raise ValueError("capture detector support and omissions disagree")
        return self


class CaptureFixtureBinding(_CaptureCapabilityModel):
    """A canonical digest binding to a fully synthetic native fixture."""

    fixture_id: _ManifestToken
    path: _FixturePath
    sha256: Sha256Digest
    kind: Literal["fully_synthetic_generated"]
    transform_id: Literal["hand_authored_from_audited_shape/v1"]
    source_payload_retained: Literal[False]

    @model_validator(mode="after")
    def fixture_path_is_relative(self) -> Self:
        if ".." in self.path.split("/"):
            raise ValueError("capture fixture path is invalid")
        return self


class CaptureCapabilityManifest(_CaptureCapabilityModel):
    """The immutable, audited capability boundary for one integration profile."""

    schema_version: Literal["capture-capability-manifest/v1"]
    profile_id: CaptureProfile
    host_name: _ManifestText
    host_version: _HostVersion
    audit_date: Literal["2026-07-19"]
    upstream_revision: _UpstreamRevision | None
    official_sources: Annotated[tuple[_OfficialSource, ...], Field(min_length=1, max_length=16)]
    source_authentication: Literal["none_same_user_untrusted"]
    raw_content_persisted: Literal[False]
    transcript_read: Literal[False]
    complete_execution_session_coverage: Literal[False]
    decision_authority: Literal[False]
    model_calls: Literal[0]
    timestamp_authority: Literal["local_observation"]
    sequence_authority: Literal["local_receipt_order"]
    rollback_detection: Literal["none"]
    at_rest_integrity: Literal["hmac_sha256_local_mutation_detection"]
    events: Annotated[tuple[CaptureEventCapability, ...], Field(min_length=1, max_length=64)]
    tool_coverage: Annotated[tuple[_ManifestToken, ...], Field(min_length=1, max_length=32)]
    coverage_exclusions: Annotated[tuple[_ManifestToken, ...], Field(min_length=1, max_length=64)]
    detectors: Annotated[
        tuple[CaptureDetectorCapability, ...],
        Field(min_length=len(SignalType), max_length=len(SignalType)),
    ]
    fixtures: Annotated[tuple[CaptureFixtureBinding, ...], Field(min_length=1, max_length=1)]

    @model_validator(mode="after")
    def manifest_is_closed_and_canonical(self) -> Self:
        if len(set(self.official_sources)) != len(self.official_sources):
            raise ValueError("capture official sources are duplicated")
        if tuple(event.event_name for event in self.events) != tuple(
            dict.fromkeys(event.event_name for event in self.events)
        ):
            raise ValueError("capture events are duplicated")
        if self.tool_coverage != tuple(sorted(set(self.tool_coverage))):
            raise ValueError("capture tool coverage is not canonical")
        if self.coverage_exclusions != tuple(sorted(set(self.coverage_exclusions))):
            raise ValueError("capture coverage exclusions are not canonical")
        if tuple(item.signal_type for item in self.detectors) != tuple(SignalType):
            raise ValueError("capture detector matrix is not closed")
        expected_fixture_id = f"{self.profile_id.value}-synthetic/v1"
        if self.fixtures[0].fixture_id != expected_fixture_id:
            raise ValueError("capture fixture binding is inconsistent")
        return self


class CaptureCapabilityRegistry(_CaptureCapabilityModel):
    """The immutable registry containing exactly the four v1 profiles."""

    schema_version: Literal["capture-capability-registry/v1"]
    profiles: Annotated[
        tuple[CaptureCapabilityManifest, ...],
        Field(min_length=len(CaptureProfile), max_length=len(CaptureProfile)),
    ]

    @model_validator(mode="after")
    def registry_contains_exact_profile_order(self) -> Self:
        if tuple(profile.profile_id for profile in self.profiles) != tuple(CaptureProfile):
            raise ValueError("capture capability profile registry is not closed")
        return self


def _validated_manifest(value: object) -> CaptureCapabilityManifest:
    if type(value) is not CaptureCapabilityManifest:
        raise CaptureCapabilityError()
    try:
        return CaptureCapabilityManifest.model_validate(value)
    except (TypeError, ValidationError, ValueError):
        raise CaptureCapabilityError() from None


def _fixture_path_present(value: object, path: str) -> bool:
    head, *tail = path.split(".", maxsplit=1)
    remainder = tail[0] if tail else None
    if head.endswith("[]"):
        if not isinstance(value, Mapping):
            return False
        nested = value.get(head[:-2])
        return (
            type(nested) is tuple
            and bool(nested)
            and all(remainder is None or _fixture_path_present(item, remainder) for item in nested)
        )
    if not isinstance(value, Mapping) or head not in value:
        return False
    return remainder is None or _fixture_path_present(value[head], remainder)


def _validate_fixture_resources(registry: CaptureCapabilityRegistry) -> None:
    package = resources.files("saliencegate.integrations")
    for profile in registry.profiles:
        declared_events = {event.event_name: event for event in profile.events}
        for fixture in profile.fixtures:
            parts = PurePosixPath(fixture.path).parts
            source = package.joinpath(*parts).read_bytes()
            if not hmac.compare_digest(hashlib.sha256(source).hexdigest(), fixture.sha256):
                raise CaptureCapabilityError()
            body = read_bounded_json(source, limits=CAPTURE_NATIVE_JSON_LIMITS)
            if canonical_json(body) != source or set(body) != {
                "events",
                "profile_id",
                "provenance",
                "schema_version",
            }:
                raise CaptureCapabilityError()
            if (
                body["schema_version"] != "capture-native-fixture/v1"
                or body["profile_id"] != profile.profile_id.value
                or body["provenance"] != "fully_synthetic_no_provider_or_model_call"
                or type(body["events"]) is not tuple
                or not body["events"]
            ):
                raise CaptureCapabilityError()
            observed_events: set[str] = set()
            for native_event in body["events"]:
                if not isinstance(native_event, Mapping) or set(native_event) != {
                    "event_name",
                    "payload",
                }:
                    raise CaptureCapabilityError()
                event_name = native_event["event_name"]
                payload = native_event["payload"]
                if type(event_name) is not str or not isinstance(payload, Mapping):
                    raise CaptureCapabilityError()
                capability = declared_events.get(event_name)
                if capability is None or not all(
                    _fixture_path_present(payload, field) for field in capability.critical_fields
                ):
                    raise CaptureCapabilityError()
                observed_events.add(event_name)
            if observed_events != set(declared_events):
                raise CaptureCapabilityError()


def load_capture_capability_registry() -> CaptureCapabilityRegistry:
    """Load and strictly validate the canonical installed capability registry."""

    try:
        source = (
            resources.files("saliencegate.integrations").joinpath(_REGISTRY_RESOURCE).read_bytes()
        )
        registry = CaptureCapabilityRegistry.model_validate_json(source)
        if canonical_json(registry) != source:
            raise CaptureCapabilityError()
        _validate_fixture_resources(registry)
        return registry
    except CaptureCapabilityError:
        raise
    except Exception:
        raise CaptureCapabilityError() from None


def capture_profile(profile_id: CaptureProfile) -> CaptureCapabilityManifest:
    """Return the installed audited manifest for an exact profile enum member."""

    if type(profile_id) is not CaptureProfile:
        raise CaptureCapabilityError()
    registry = load_capture_capability_registry()
    for profile in registry.profiles:
        if profile.profile_id is profile_id:
            return profile
    raise CaptureCapabilityError()


def capture_capability_digest(profile: CaptureCapabilityManifest) -> str:
    """Return the canonical SHA-256 digest of a validated capability manifest."""

    validated = _validated_manifest(profile)
    try:
        return hashlib.sha256(canonical_json(validated)).hexdigest()
    except Exception:
        raise CaptureCapabilityError() from None


def validate_capture_capability_binding(
    profile_id: CaptureProfile,
    declared_digest: str,
) -> CaptureCapabilityManifest:
    """Bind a connector declaration to its exact installed capability manifest."""

    if (
        type(profile_id) is not CaptureProfile
        or type(declared_digest) is not str
        or _SHA256.fullmatch(declared_digest) is None
    ):
        raise CaptureCapabilityError()
    profile = capture_profile(profile_id)
    expected_digest = capture_capability_digest(profile)
    if not hmac.compare_digest(declared_digest, expected_digest):
        raise CaptureCapabilityError()
    return profile


def classify_capture_compatibility(
    profile: CaptureCapabilityManifest,
    *,
    host_version: str,
    observed_event: str,
    observed_fields: frozenset[str],
) -> CompatibilityStatus:
    """Classify one observed shape without granting authority to additive fields."""

    try:
        validated = _validated_manifest(profile)
    except CaptureCapabilityError:
        return CompatibilityStatus.INCOMPATIBLE
    if (
        type(host_version) is not str
        or not 1 <= len(host_version) <= 64
        or _HOST_VERSION.fullmatch(host_version) is None
        or type(observed_event) is not str
        or len(observed_event) > 256
        or type(observed_fields) is not frozenset
        or len(observed_fields) > 256
        or any(type(field) is not str or not 1 <= len(field) <= 256 for field in observed_fields)
    ):
        return CompatibilityStatus.INCOMPATIBLE

    event = next(
        (item for item in validated.events if item.event_name == observed_event),
        None,
    )
    if event is None or not set(event.critical_fields).issubset(observed_fields):
        return CompatibilityStatus.INCOMPATIBLE
    if host_version == validated.host_version:
        return CompatibilityStatus.VERIFIED
    return CompatibilityStatus.SCHEMA_COMPATIBLE_UNVERIFIED_VERSION


__all__ = [
    "CapabilitySupport",
    "CaptureCapabilityError",
    "CaptureCapabilityManifest",
    "CaptureCapabilityRegistry",
    "CaptureDetectorCapability",
    "CaptureEventCapability",
    "CaptureFixtureBinding",
    "CaptureProfile",
    "CompatibilityStatus",
    "capture_capability_digest",
    "capture_profile",
    "classify_capture_compatibility",
    "load_capture_capability_registry",
    "validate_capture_capability_binding",
]
