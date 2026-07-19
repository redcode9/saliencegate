from __future__ import annotations

import hmac

from pydantic import ConfigDict, Field, RootModel

from saliencegate.domain import (
    JsonObject,
    PayloadDigest,
    PayloadDigestAlgorithm,
    canonical_json,
    content_digest,
)
from saliencegate.security.keys import InstallationKey


class SyntheticDigestDisabledError(ValueError):
    pass


class MissingInstallationKeyError(ValueError):
    pass


class AmbiguousDigestModeError(ValueError):
    pass


class DigestModeMismatchError(ValueError):
    pass


class RedactedPayload(RootModel[JsonObject]):
    model_config = ConfigDict(frozen=True, strict=True, hide_input_in_errors=True)

    root: JsonObject = Field(repr=False)

    def __repr__(self) -> str:
        return "RedactedPayload(<redacted>)"

    def __str__(self) -> str:
        return "RedactedPayload(<redacted>)"


_PAYLOAD_DIGEST_DOMAIN = b"saliencegate:payload:v1"


def hmac_payload_digest(
    payload: RedactedPayload,
    key: InstallationKey,
) -> PayloadDigest:
    return PayloadDigest(
        algorithm=PayloadDigestAlgorithm.HMAC_SHA256,
        value=key._hmac_sha256(
            canonical_json(payload.root),
            domain=_PAYLOAD_DIGEST_DOMAIN,
        ),
    )


def synthetic_payload_digest(
    payload: RedactedPayload,
    *,
    synthetic_benchmark: bool,
) -> PayloadDigest:
    if not synthetic_benchmark:
        raise SyntheticDigestDisabledError(
            "synthetic digest requires synthetic_benchmark=True on this call"
        )
    return PayloadDigest(
        algorithm=PayloadDigestAlgorithm.SYNTHETIC_SHA256,
        value=content_digest(canonical_json(payload.root)),
    )


def create_payload_digest(
    payload: RedactedPayload,
    *,
    key: InstallationKey | None = None,
    synthetic_benchmark: bool = False,
) -> PayloadDigest:
    if key is not None and synthetic_benchmark:
        raise AmbiguousDigestModeError("installation key and synthetic mode are mutually exclusive")
    if key is not None:
        return hmac_payload_digest(payload, key)
    return synthetic_payload_digest(payload, synthetic_benchmark=synthetic_benchmark)


def verify_payload_digest(
    payload: RedactedPayload,
    digest: PayloadDigest,
    *,
    key: InstallationKey | None = None,
    synthetic_benchmark: bool = False,
) -> bool:
    if key is not None and synthetic_benchmark:
        raise AmbiguousDigestModeError("installation key and synthetic mode are mutually exclusive")
    if digest.algorithm is PayloadDigestAlgorithm.HMAC_SHA256:
        if synthetic_benchmark:
            raise DigestModeMismatchError("HMAC digest cannot be verified in synthetic mode")
        if key is None:
            raise MissingInstallationKeyError(
                "HMAC digest verification requires an installation key"
            )
        expected = hmac_payload_digest(payload, key)
    else:
        if key is not None:
            raise DigestModeMismatchError(
                "synthetic digest cannot be verified with an installation key"
            )
        expected = synthetic_payload_digest(
            payload,
            synthetic_benchmark=synthetic_benchmark,
        )
    return hmac.compare_digest(digest.value, expected.value)
