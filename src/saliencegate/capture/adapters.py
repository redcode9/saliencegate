"""Strict pre-admission contracts for provider capture adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, StringConstraints

from saliencegate.capture.capabilities import (
    CaptureProfile,
    validate_capture_capability_binding,
)
from saliencegate.capture.identities import CaptureDigestContext
from saliencegate.capture.schema import CaptureIntake
from saliencegate.domain.records import Sha256Digest

CAPTURE_ADAPTER_PROTOCOL_VERSION: Final = "capture-adapter/v1"

_CAPABILITY_FIELDS = frozenset(
    {
        "protocol_version",
        "profile_id",
        "capability_digest",
        "host_version",
    }
)
_CaptureHostVersion = Annotated[
    str,
    StringConstraints(
        min_length=5,
        max_length=64,
        pattern=r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$",
    ),
]


class CaptureAdapterCapabilities(BaseModel):
    """Content-free declaration presented before native capture dispatch."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    protocol_version: Literal["capture-adapter/v1"] = CAPTURE_ADAPTER_PROTOCOL_VERSION
    profile_id: CaptureProfile
    capability_digest: Sha256Digest
    host_version: _CaptureHostVersion

    def __repr__(self) -> str:
        return "CaptureAdapterCapabilities(<redacted>)"

    __str__ = __repr__


class CaptureAdapterContractError(ValueError):
    """An adapter failed admission without exposing provider-owned content."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("capture adapter contract is invalid")


@runtime_checkable
class CaptureAdapter(Protocol):
    """Synchronous provider adapter admitted before native bytes are exposed."""

    def capabilities(self) -> CaptureAdapterCapabilities: ...

    def adapt_bytes(
        self,
        source: bytes,
        *,
        context: CaptureDigestContext,
    ) -> tuple[CaptureIntake, ...]: ...


def _adapter_callbacks(
    adapter: object,
) -> tuple[Callable[[], object], Callable[..., object]] | None:
    callbacks: tuple[Callable[[], object], Callable[..., object]] | None = None
    try:
        capabilities = getattr(adapter, "capabilities", None)
        adapt_bytes = getattr(adapter, "adapt_bytes", None)
        if callable(capabilities) and callable(adapt_bytes):
            callbacks = capabilities, adapt_bytes
    except Exception:
        callbacks = None
    return callbacks


def _declared_capabilities(callback: Callable[[], object]) -> object | None:
    declaration: object | None = None
    try:
        declaration = callback()
    except Exception:
        declaration = None
    return declaration


def _revalidate_capabilities(value: object) -> CaptureAdapterCapabilities | None:
    """Rebuild from field storage without calling methods on an untrusted instance."""

    validated: CaptureAdapterCapabilities | None = None
    try:
        if type(value) is not CaptureAdapterCapabilities:
            return None
        state = object.__getattribute__(value, "__dict__")
        if type(state) is not dict or frozenset(state) != _CAPABILITY_FIELDS:
            return None
        snapshot = {
            "protocol_version": state["protocol_version"],
            "profile_id": state["profile_id"],
            "capability_digest": state["capability_digest"],
            "host_version": state["host_version"],
        }
        if (
            type(snapshot["protocol_version"]) is not str
            or type(snapshot["profile_id"]) is not CaptureProfile
            or type(snapshot["capability_digest"]) is not str
            or type(snapshot["host_version"]) is not str
        ):
            return None
        validated = CaptureAdapterCapabilities.model_validate(snapshot)
    except Exception:
        validated = None
    return validated


def _capability_binding_is_valid(declaration: CaptureAdapterCapabilities) -> bool:
    valid = False
    try:
        profile = validate_capture_capability_binding(
            declaration.profile_id,
            declaration.capability_digest,
        )
        valid = profile.profile_id is declaration.profile_id
    except Exception:
        valid = False
    return valid


def validated_capture_adapter(adapter: object) -> CaptureAdapterCapabilities:
    """Validate and bind an adapter declaration before native-byte dispatch."""

    callbacks = _adapter_callbacks(adapter)
    if callbacks is None:
        raise CaptureAdapterContractError()
    capabilities_callback, _adapt_bytes_callback = callbacks

    declaration = _revalidate_capabilities(_declared_capabilities(capabilities_callback))
    if declaration is None or not _capability_binding_is_valid(declaration):
        raise CaptureAdapterContractError()
    return declaration


__all__ = [
    "CAPTURE_ADAPTER_PROTOCOL_VERSION",
    "CaptureAdapter",
    "CaptureAdapterCapabilities",
    "CaptureAdapterContractError",
    "validated_capture_adapter",
]
