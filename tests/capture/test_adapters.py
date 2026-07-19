from __future__ import annotations

import inspect
from typing import get_args, get_origin, get_type_hints

import pytest
from pydantic import ValidationError
from saliencegate.capture.adapters import (
    CAPTURE_ADAPTER_PROTOCOL_VERSION,
    CaptureAdapter,
    CaptureAdapterCapabilities,
    CaptureAdapterContractError,
    validated_capture_adapter,
)
from saliencegate.capture.capabilities import (
    CaptureProfile,
    capture_capability_digest,
    capture_profile,
)

from saliencegate.capture.identities import CaptureDigestContext
from saliencegate.capture.schema import CaptureIntake


def adapter_capabilities(
    *,
    profile_id: CaptureProfile = CaptureProfile.CODEX_HOOKS_V1,
    capability_digest: str | None = None,
    host_version: str | None = None,
) -> CaptureAdapterCapabilities:
    profile = capture_profile(profile_id)
    return CaptureAdapterCapabilities(
        protocol_version=CAPTURE_ADAPTER_PROTOCOL_VERSION,
        profile_id=profile_id,
        capability_digest=(
            capture_capability_digest(profile) if capability_digest is None else capability_digest
        ),
        host_version=profile.host_version if host_version is None else host_version,
    )


class _ConformingAdapter:
    def __init__(self, declaration: CaptureAdapterCapabilities | None = None) -> None:
        self._declaration = adapter_capabilities() if declaration is None else declaration

    def capabilities(self) -> CaptureAdapterCapabilities:
        return self._declaration

    def adapt_bytes(
        self,
        source: bytes,
        *,
        context: CaptureDigestContext,
    ) -> tuple[CaptureIntake, ...]:
        del source, context
        return ()


class _MissingAdaptBytes:
    def capabilities(self) -> CaptureAdapterCapabilities:
        return adapter_capabilities()


class _HostileCapabilitiesAdapter:
    def capabilities(self) -> CaptureAdapterCapabilities:
        raise RuntimeError("raw-provider-capabilities-secret")

    def adapt_bytes(
        self,
        source: bytes,
        *,
        context: CaptureDigestContext,
    ) -> tuple[CaptureIntake, ...]:
        del source, context
        return ()


def test_capture_adapter_protocol_freezes_the_pre_admission_shape() -> None:
    adapter = _ConformingAdapter()

    assert CAPTURE_ADAPTER_PROTOCOL_VERSION == "capture-adapter/v1"
    assert isinstance(adapter, CaptureAdapter)
    assert not isinstance(_MissingAdaptBytes(), CaptureAdapter)

    capabilities_signature = inspect.signature(CaptureAdapter.capabilities)
    assert tuple(capabilities_signature.parameters) == ("self",)
    assert get_type_hints(CaptureAdapter.capabilities)["return"] is CaptureAdapterCapabilities

    adapt_signature = inspect.signature(CaptureAdapter.adapt_bytes)
    assert tuple(adapt_signature.parameters) == ("self", "source", "context")
    assert tuple(parameter.kind for parameter in adapt_signature.parameters.values()) == (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )
    annotations = get_type_hints(CaptureAdapter.adapt_bytes)
    assert annotations["source"] is bytes
    assert annotations["context"] is CaptureDigestContext
    assert get_origin(annotations["return"]) is tuple
    assert get_args(annotations["return"]) == (CaptureIntake, Ellipsis)


def test_adapter_capabilities_are_strict_frozen_versioned_and_content_free() -> None:
    declaration = adapter_capabilities()

    assert declaration.model_dump(mode="python") == {
        "protocol_version": "capture-adapter/v1",
        "profile_id": CaptureProfile.CODEX_HOOKS_V1,
        "capability_digest": capture_capability_digest(
            capture_profile(CaptureProfile.CODEX_HOOKS_V1)
        ),
        "host_version": "0.144.6",
    }
    assert (
        CaptureAdapterCapabilities.model_validate_json(declaration.model_dump_json()) == declaration
    )
    assert "raw" not in repr(declaration).casefold()
    assert "source" not in repr(declaration).casefold()

    with pytest.raises(ValidationError, match="frozen"):
        declaration.__setattr__("host_version", "replacement")
    with pytest.raises(ValidationError):
        CaptureAdapterCapabilities.model_validate(
            {**declaration.model_dump(mode="python"), "unexpected": True}
        )
    with pytest.raises(ValidationError):
        CaptureAdapterCapabilities.model_validate(
            {**declaration.model_dump(mode="python"), "profile_id": declaration.profile_id.value}
        )
    with pytest.raises(ValidationError):
        CaptureAdapterCapabilities.model_validate(
            {**declaration.model_dump(mode="python"), "host_version": ""}
        )


def test_validated_adapter_accepts_exact_and_shape_compatible_unverified_host_versions() -> None:
    exact_adapter = _ConformingAdapter()
    future_adapter = _ConformingAdapter(adapter_capabilities(host_version="0.144.7"))

    assert validated_capture_adapter(exact_adapter) == exact_adapter.capabilities()
    assert validated_capture_adapter(future_adapter) == future_adapter.capabilities()


def test_adapter_profile_and_capability_digest_must_bind_before_native_dispatch() -> None:
    codex = capture_profile(CaptureProfile.CODEX_HOOKS_V1)
    pi = capture_profile(CaptureProfile.PI_EXTENSION_V1)

    wrong_profile = _ConformingAdapter(
        adapter_capabilities(
            profile_id=CaptureProfile.PI_EXTENSION_V1,
            capability_digest=capture_capability_digest(codex),
        )
    )
    wrong_digest = _ConformingAdapter(
        adapter_capabilities(
            profile_id=CaptureProfile.CODEX_HOOKS_V1,
            capability_digest=capture_capability_digest(pi),
        )
    )
    wrong_protocol = _ConformingAdapter(
        adapter_capabilities().model_copy(update={"protocol_version": "capture-adapter/v2"})
    )

    for adapter in (wrong_profile, wrong_digest, wrong_protocol):
        with pytest.raises(CaptureAdapterContractError):
            validated_capture_adapter(adapter)


@pytest.mark.parametrize(
    "adapter",
    (
        object(),
        _MissingAdaptBytes(),
        _HostileCapabilitiesAdapter(),
    ),
)
def test_adapter_shape_and_callback_failures_are_content_free(adapter: object) -> None:
    with pytest.raises(CaptureAdapterContractError) as captured:
        validated_capture_adapter(adapter)

    rendered = f"{captured.value!s}\n{captured.value!r}"
    assert rendered == (
        "capture adapter contract is invalid\n"
        "CaptureAdapterContractError('capture adapter contract is invalid')"
    )
    assert "raw-provider-capabilities-secret" not in rendered


def test_adapter_rejects_forged_capability_objects_without_serializing_them() -> None:
    secret = "forged-capability-secret"
    declaration = adapter_capabilities().model_copy(update={"host_version": secret})
    object.__setattr__(
        declaration,
        "model_dump_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(secret)),
    )
    adapter = _ConformingAdapter(declaration)

    with pytest.raises(CaptureAdapterContractError) as captured:
        validated_capture_adapter(adapter)

    assert secret not in str(captured.value)
    assert secret not in repr(captured.value)
