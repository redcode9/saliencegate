"""Opaque, domain-separated identities for provider capture data."""

from __future__ import annotations

from typing import Never, SupportsIndex, cast

from saliencegate.security import InstallationKey

MAX_CAPTURE_IDENTITY_INPUT_BYTES = 2 * 1_024 * 1_024


class CaptureIdentityError(ValueError):
    """A content-free identity derivation failure."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("capture identity is invalid")


_DOMAINS = {
    "session_id": b"saliencegate:capture:session-id:v1",
    "call_ref": b"saliencegate:capture:call-ref:v1",
    "producer_event": b"saliencegate:capture:producer-event:v1",
    "action_identity": b"saliencegate:capture:action-identity:v1",
    "unavailable_action_identity": (b"saliencegate:capture:unavailable-action-identity:v1"),
    "workspace_identity": b"saliencegate:capture:workspace-identity:v1",
    "environment_identity": b"saliencegate:capture:environment-identity:v1",
    "failure_signature": b"saliencegate:capture:failure-signature:v1",
    "subagent_id": b"saliencegate:capture:subagent-id:v1",
    "turn_id": b"saliencegate:capture:turn-id:v1",
    "transport_batch_ref": b"saliencegate:capture:transport-batch-ref:v1",
    "transport_chunk_digest": b"saliencegate:capture:transport-chunk-digest:v1",
    "integrity_tag": b"saliencegate:capture:integrity-tag:v1",
}


def _exact_identity_input(value: object) -> bytes:
    if type(value) is not bytes or not 1 <= len(value) <= MAX_CAPTURE_IDENTITY_INPUT_BYTES:
        raise CaptureIdentityError()
    return value


class CaptureDigestContext:
    """Derive only named capture identities without exposing key material."""

    __slots__ = ("__key",)

    def __init__(self, key: InstallationKey) -> None:
        if type(key) is not InstallationKey:
            raise CaptureIdentityError()
        object.__setattr__(self, "_CaptureDigestContext__key", key._copy())

    def __setattr__(self, _name: str, _value: object) -> Never:
        raise AttributeError("CaptureDigestContext is immutable")

    def __repr__(self) -> str:
        return "CaptureDigestContext(<redacted>)"

    __str__ = __repr__

    def __copy__(self) -> Never:
        raise TypeError("capture digest contexts cannot be copied")

    def __deepcopy__(self, _memo: object) -> Never:
        raise TypeError("capture digest contexts cannot be copied")

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        del protocol
        raise TypeError("capture digest contexts are transient")

    def _derive(self, purpose: str, value: object) -> str:
        exact = _exact_identity_input(value)
        key = cast(
            InstallationKey,
            object.__getattribute__(self, "_CaptureDigestContext__key"),
        )
        try:
            return key._hmac_sha256(exact, domain=_DOMAINS[purpose])
        except Exception:
            raise CaptureIdentityError() from None

    def session_id(self, value: bytes) -> str:
        return self._derive("session_id", value)

    def call_ref(self, value: bytes) -> str:
        return self._derive("call_ref", value)

    def producer_event(self, value: bytes) -> str:
        return self._derive("producer_event", value)

    def action_identity(self, value: bytes) -> str:
        return self._derive("action_identity", value)

    def unavailable_action_identity(self, value: bytes) -> str:
        return self._derive("unavailable_action_identity", value)

    def workspace_identity(self, value: bytes) -> str:
        return self._derive("workspace_identity", value)

    def environment_identity(self, value: bytes) -> str:
        return self._derive("environment_identity", value)

    def failure_signature(self, value: bytes) -> str:
        return self._derive("failure_signature", value)

    def subagent_id(self, value: bytes) -> str:
        return self._derive("subagent_id", value)

    def turn_id(self, value: bytes) -> str:
        return self._derive("turn_id", value)

    def transport_batch_ref(self, value: bytes) -> str:
        """Pseudonymize one bridge-local batch identifier."""

        return self._derive("transport_batch_ref", value)

    def transport_chunk_digest(self, value: bytes) -> str:
        """Key the exact canonical bytes of one bounded bridge chunk."""

        return self._derive("transport_chunk_digest", value)

    def integrity_tag(self, value: bytes) -> str:
        return self._derive("integrity_tag", value)


__all__ = [
    "MAX_CAPTURE_IDENTITY_INPUT_BYTES",
    "CaptureDigestContext",
    "CaptureIdentityError",
]
