"""Closed project/global capture scope and identity contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saliencegate.capture.capabilities import (
    CaptureProfile,
    CompatibilityStatus,
)
from saliencegate.capture.identities import CaptureDigestContext
from saliencegate.domain import canonical_json
from saliencegate.domain.primitives import (
    ComponentIdentifier,
    Sha256Digest,
    UtcDatetime,
)
from saliencegate.security import InstallationKey

MAX_GLOBAL_CHILDREN_PER_PARENT = 1_000
MAX_GLOBAL_EXCLUSIONS_PER_PARENT = 1_000
MAX_GLOBAL_HEALTH_COUNT = 1_000_000
MAX_GLOBAL_INSTALLATION_GENERATION = 1_000_000


class CaptureGlobalScopeError(ValueError):
    """A content-free failure at the global capture scope boundary."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("capture global scope is invalid")


class CaptureConnectionScope(StrEnum):
    """The complete set of supported connection installation scopes."""

    PROJECT = "project"
    USER_GLOBAL = "user_global"


class CaptureGlobalProvider(StrEnum):
    """Providers with a designed user-global capture surface."""

    CODEX = "codex"
    CLAUDE_CODE = "claude-code"
    OPENCODE = "opencode"
    PI = "pi"


class CaptureGlobalParentState(StrEnum):
    """Lifecycle of one authenticated provider-global installation."""

    PENDING = "pending"
    ENABLED = "enabled"
    DRAINING = "draining"
    DISABLED = "disabled"
    DELETING = "deleting"


class CaptureGlobalHealthCode(StrEnum):
    """Bounded health reasons that can exist before a child is enrolled."""

    UNKNOWN_CHILD_EVENT = "unknown_child_event"
    PROJECT_IDENTITY_UNAVAILABLE = "project_identity_unavailable"
    ENROLLMENT_REJECTED = "enrollment_rejected"
    CHILD_LIMIT_REACHED = "child_limit_reached"
    PROJECT_EXCLUDED = "project_excluded"


_PROVIDER_PROFILES = {
    CaptureGlobalProvider.CODEX: CaptureProfile.CODEX_HOOKS_V1,
    CaptureGlobalProvider.CLAUDE_CODE: CaptureProfile.CLAUDE_CODE_HOOKS_V1,
    CaptureGlobalProvider.OPENCODE: CaptureProfile.OPENCODE_PLUGIN_V1,
    CaptureGlobalProvider.PI: CaptureProfile.PI_EXTENSION_V1,
}


class _CaptureGlobalModel(BaseModel):
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


class CaptureGlobalParentRegistration(_CaptureGlobalModel):
    """One deterministic provider-global parent registered in the capture store."""

    schema_version: Literal["capture-global-parent-registration/v1"] = (
        "capture-global-parent-registration/v1"
    )
    global_parent_id: Annotated[ComponentIdentifier, Field(repr=False)]
    provider_id: CaptureGlobalProvider
    config_root_digest: Annotated[Sha256Digest, Field(repr=False)]
    profile_id: CaptureProfile
    capability_manifest_digest: Annotated[Sha256Digest, Field(repr=False)]
    host_version: Annotated[ComponentIdentifier, Field(repr=False)]
    generation: Annotated[
        int,
        Field(ge=1, le=MAX_GLOBAL_INSTALLATION_GENERATION),
    ]
    state: CaptureGlobalParentState = CaptureGlobalParentState.PENDING

    @model_validator(mode="after")
    def provider_profile_is_closed(self) -> Self:
        if _PROVIDER_PROFILES[self.provider_id] is not self.profile_id:
            raise ValueError("capture global provider profile is inconsistent")
        return self


class CaptureGlobalParentTransition(_CaptureGlobalModel):
    global_parent_id: Annotated[ComponentIdentifier, Field(repr=False)]
    previous_state: CaptureGlobalParentState
    state: CaptureGlobalParentState


class CaptureGlobalParentSummary(_CaptureGlobalModel):
    """Authenticated parent state without provider-native content."""

    schema_version: Literal["capture-global-parent-summary/v1"] = "capture-global-parent-summary/v1"
    global_parent_id: Annotated[ComponentIdentifier, Field(repr=False)]
    provider_id: CaptureGlobalProvider
    config_root_digest: Annotated[Sha256Digest, Field(repr=False)]
    profile_id: CaptureProfile
    capability_manifest_digest: Annotated[Sha256Digest, Field(repr=False)]
    host_version: Annotated[ComponentIdentifier, Field(repr=False)]
    generation: Annotated[
        int,
        Field(ge=1, le=MAX_GLOBAL_INSTALLATION_GENERATION),
    ]
    state: CaptureGlobalParentState
    compatibility_status: CompatibilityStatus
    health_marker_count: Annotated[
        int,
        Field(ge=0, le=len(CaptureGlobalHealthCode)),
    ]
    health_set_digest: Annotated[Sha256Digest, Field(repr=False)]
    exclusion_count: Annotated[
        int,
        Field(ge=0, le=MAX_GLOBAL_EXCLUSIONS_PER_PARENT),
    ]
    exclusion_set_digest: Annotated[Sha256Digest, Field(repr=False)]
    created_at: Annotated[UtcDatetime, Field(repr=False)]
    updated_at: Annotated[UtcDatetime, Field(repr=False)]

    @model_validator(mode="after")
    def commitments_are_consistent(self) -> Self:
        if _PROVIDER_PROFILES[self.provider_id] is not self.profile_id:
            raise ValueError("capture global provider profile is inconsistent")
        if self.updated_at < self.created_at:
            raise ValueError("capture global parent timestamps are inconsistent")
        return self


class CaptureGlobalChildIdentity(_CaptureGlobalModel):
    """Deterministic child identity derived from a canonical project identity."""

    schema_version: Literal["capture-global-child-identity/v1"] = "capture-global-child-identity/v1"
    global_parent_id: Annotated[ComponentIdentifier, Field(repr=False)]
    connection_id: Annotated[ComponentIdentifier, Field(repr=False)]
    project_digest: Annotated[Sha256Digest, Field(repr=False)]


class CaptureGlobalChildBinding(_CaptureGlobalModel):
    """Authenticated persisted relationship between a parent and child."""

    schema_version: Literal["capture-global-child-binding/v1"] = "capture-global-child-binding/v1"
    global_parent_id: Annotated[ComponentIdentifier, Field(repr=False)]
    connection_id: Annotated[ComponentIdentifier, Field(repr=False)]
    project_digest: Annotated[Sha256Digest, Field(repr=False)]
    created_at: Annotated[UtcDatetime, Field(repr=False)]


class CaptureGlobalExclusionBinding(_CaptureGlobalModel):
    """One path-free project exclusion below a global parent."""

    schema_version: Literal["capture-global-exclusion-binding/v1"] = (
        "capture-global-exclusion-binding/v1"
    )
    global_parent_id: Annotated[ComponentIdentifier, Field(repr=False)]
    project_digest: Annotated[Sha256Digest, Field(repr=False)]
    created_at: Annotated[UtcDatetime, Field(repr=False)]


class CaptureGlobalHealthCounter(_CaptureGlobalModel):
    """One saturating parent-global health counter."""

    schema_version: Literal["capture-global-health-counter/v1"] = "capture-global-health-counter/v1"
    global_parent_id: Annotated[ComponentIdentifier, Field(repr=False)]
    code: CaptureGlobalHealthCode
    count: Annotated[int, Field(ge=1, le=MAX_GLOBAL_HEALTH_COUNT)]
    saturated: bool
    created_at: Annotated[UtcDatetime, Field(repr=False)]
    updated_at: Annotated[UtcDatetime, Field(repr=False)]

    @model_validator(mode="after")
    def saturation_and_timestamps_are_consistent(self) -> Self:
        if self.saturated is not (self.count == MAX_GLOBAL_HEALTH_COUNT):
            raise ValueError("capture global health saturation is inconsistent")
        if self.updated_at < self.created_at:
            raise ValueError("capture global health timestamps are inconsistent")
        return self


def capture_global_provider_profile(provider_id: CaptureGlobalProvider) -> CaptureProfile:
    """Return the single evidence profile valid for a global provider."""

    if type(provider_id) is not CaptureGlobalProvider:
        raise CaptureGlobalScopeError()
    return _PROVIDER_PROFILES[provider_id]


def _derive_global_parent_id(
    *,
    context: CaptureDigestContext,
    provider_id: CaptureGlobalProvider,
    config_root_digest: str,
    generation: int,
) -> str:
    try:
        if (
            type(context) is not CaptureDigestContext
            or type(provider_id) is not CaptureGlobalProvider
            or type(config_root_digest) is not str
            or len(config_root_digest) != 64
            or any(character not in "0123456789abcdef" for character in config_root_digest)
            or type(generation) is not int
            or not 1 <= generation <= MAX_GLOBAL_INSTALLATION_GENERATION
        ):
            raise CaptureGlobalScopeError()
        material = canonical_json(
            {
                "schema_version": "capture-global-parent-id/v1",
                "provider_id": provider_id.value,
                "config_root_digest": config_root_digest,
                "generation": generation,
            }
        )
        return f"sgg-{context.global_parent_id(material)[:48]}"
    except CaptureGlobalScopeError:
        raise
    except Exception:
        raise CaptureGlobalScopeError() from None


def _derive_global_child_from_project_digest(
    *,
    context: CaptureDigestContext,
    global_parent_id: str,
    provider_id: CaptureGlobalProvider,
    generation: int,
    project_digest: str,
) -> CaptureGlobalChildIdentity:
    try:
        if (
            type(context) is not CaptureDigestContext
            or type(global_parent_id) is not str
            or not global_parent_id.startswith("sgg-")
            or len(global_parent_id) != 52
            or type(provider_id) is not CaptureGlobalProvider
            or type(generation) is not int
            or not 1 <= generation <= MAX_GLOBAL_INSTALLATION_GENERATION
            or type(project_digest) is not str
            or len(project_digest) != 64
            or any(character not in "0123456789abcdef" for character in project_digest)
        ):
            raise CaptureGlobalScopeError()
        material = canonical_json(
            {
                "schema_version": "capture-global-child-id/v1",
                "global_parent_id": global_parent_id,
                "provider_id": provider_id.value,
                "generation": generation,
                "project_digest": project_digest,
            }
        )
        return CaptureGlobalChildIdentity(
            global_parent_id=global_parent_id,
            connection_id=f"sgc-{context.global_child_id(material)[:48]}",
            project_digest=project_digest,
        )
    except CaptureGlobalScopeError:
        raise
    except Exception:
        raise CaptureGlobalScopeError() from None


def derive_global_config_root_digest(
    canonical_config_root: bytes,
    installation_key: InstallationKey,
) -> str:
    """Pseudonymize one canonical provider configuration root."""

    try:
        if (
            type(canonical_config_root) is not bytes
            or type(installation_key) is not InstallationKey
        ):
            raise CaptureGlobalScopeError()
        return CaptureDigestContext(installation_key).global_config_root(canonical_config_root)
    except CaptureGlobalScopeError:
        raise
    except Exception:
        raise CaptureGlobalScopeError() from None


def derive_global_parent_id(
    *,
    provider_id: CaptureGlobalProvider,
    config_root_digest: str,
    generation: int,
    installation_key: InstallationKey,
) -> str:
    """Derive one opaque provider-global parent identifier."""

    try:
        if type(installation_key) is not InstallationKey:
            raise CaptureGlobalScopeError()
        return _derive_global_parent_id(
            context=CaptureDigestContext(installation_key),
            provider_id=provider_id,
            config_root_digest=config_root_digest,
            generation=generation,
        )
    except CaptureGlobalScopeError:
        raise
    except Exception:
        raise CaptureGlobalScopeError() from None


def derive_global_child_identity(
    *,
    global_parent_id: str,
    provider_id: CaptureGlobalProvider,
    generation: int,
    canonical_project_identity: bytes,
    installation_key: InstallationKey,
) -> CaptureGlobalChildIdentity:
    """Derive one normal project connection below a global parent."""

    try:
        if (
            type(canonical_project_identity) is not bytes
            or type(installation_key) is not InstallationKey
        ):
            raise CaptureGlobalScopeError()
        context = CaptureDigestContext(installation_key)
        project_digest = context.workspace_identity(canonical_project_identity)
        return _derive_global_child_from_project_digest(
            context=context,
            global_parent_id=global_parent_id,
            provider_id=provider_id,
            generation=generation,
            project_digest=project_digest,
        )
    except CaptureGlobalScopeError:
        raise
    except Exception:
        raise CaptureGlobalScopeError() from None


__all__ = [
    "MAX_GLOBAL_CHILDREN_PER_PARENT",
    "MAX_GLOBAL_EXCLUSIONS_PER_PARENT",
    "MAX_GLOBAL_HEALTH_COUNT",
    "MAX_GLOBAL_INSTALLATION_GENERATION",
    "CaptureConnectionScope",
    "CaptureGlobalChildBinding",
    "CaptureGlobalChildIdentity",
    "CaptureGlobalExclusionBinding",
    "CaptureGlobalHealthCode",
    "CaptureGlobalHealthCounter",
    "CaptureGlobalParentRegistration",
    "CaptureGlobalParentState",
    "CaptureGlobalParentSummary",
    "CaptureGlobalParentTransition",
    "CaptureGlobalProvider",
    "CaptureGlobalScopeError",
    "capture_global_provider_profile",
    "derive_global_child_identity",
    "derive_global_config_root_digest",
    "derive_global_parent_id",
]
