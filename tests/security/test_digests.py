from __future__ import annotations

import pytest

import saliencegate.security as public_security
from saliencegate.domain import PayloadDigest, PayloadDigestAlgorithm, canonical_json
from saliencegate.security import (
    DigestModeMismatchError,
    InstallationKey,
    MissingInstallationKeyError,
    SyntheticDigestDisabledError,
)
from saliencegate.security.digests import (
    RedactedPayload,
    hmac_payload_digest,
    synthetic_payload_digest,
    verify_payload_digest,
)


def test_hmac_digest_has_a_stable_vector() -> None:
    payload = RedactedPayload(root={"secret": "[REDACTED]"})
    key = InstallationKey(b"k" * 32)

    digest = hmac_payload_digest(payload, key)

    assert digest.algorithm is PayloadDigestAlgorithm.HMAC_SHA256
    assert digest.value == "ff38e9e7fb2e6a97dcefed6bac49c1b67bc66482b366ebcfcc4002f233797f64"


def test_hmac_digest_changes_with_payload_or_key() -> None:
    first = RedactedPayload(root={"value": "one"})
    second = RedactedPayload(root={"value": "two"})
    key = InstallationKey(b"k" * 32)

    assert hmac_payload_digest(first, key) != hmac_payload_digest(second, key)
    assert hmac_payload_digest(first, key) != hmac_payload_digest(first, InstallationKey(b"z" * 32))


def test_hmac_digest_verification_requires_the_installation_key() -> None:
    payload = RedactedPayload(root={"value": "safe"})
    key = InstallationKey(b"k" * 32)
    digest = hmac_payload_digest(payload, key)

    assert verify_payload_digest(payload, digest, key=key)
    assert not verify_payload_digest(payload, digest, key=InstallationKey(b"z" * 32))
    with pytest.raises(MissingInstallationKeyError):
        verify_payload_digest(payload, digest)


def test_synthetic_digest_requires_an_explicit_per_call_flag() -> None:
    payload = RedactedPayload(root={"value": "fixture"})

    with pytest.raises(SyntheticDigestDisabledError):
        synthetic_payload_digest(payload, synthetic_benchmark=False)

    digest = synthetic_payload_digest(payload, synthetic_benchmark=True)
    assert digest.algorithm is PayloadDigestAlgorithm.SYNTHETIC_SHA256
    assert verify_payload_digest(payload, digest, synthetic_benchmark=True)
    with pytest.raises(SyntheticDigestDisabledError):
        verify_payload_digest(payload, digest)


def test_verification_rejects_a_validly_shaped_but_wrong_digest() -> None:
    payload = RedactedPayload(root={"value": "fixture"})
    wrong = PayloadDigest(
        algorithm=PayloadDigestAlgorithm.SYNTHETIC_SHA256,
        value="0" * 64,
    )

    assert not verify_payload_digest(payload, wrong, synthetic_benchmark=True)


def test_verification_rejects_a_digest_from_the_other_mode() -> None:
    payload = RedactedPayload(root={"value": "fixture"})
    key = InstallationKey(b"k" * 32)
    hmac_digest = hmac_payload_digest(payload, key)
    synthetic_digest = synthetic_payload_digest(payload, synthetic_benchmark=True)

    with pytest.raises(DigestModeMismatchError):
        verify_payload_digest(payload, hmac_digest, synthetic_benchmark=True)
    with pytest.raises(DigestModeMismatchError):
        verify_payload_digest(payload, synthetic_digest, key=key)
    assert hmac_digest.value != synthetic_digest.value


def test_redacted_payload_is_deeply_immutable_and_canonical() -> None:
    source = {"nested": [{"value": 1}]}
    payload = RedactedPayload(root=source)
    source["nested"][0]["value"] = 2

    assert canonical_json(payload.root) == b'{"nested":[{"value":1}]}'
    with pytest.raises(TypeError):
        payload.root["nested"][0]["value"] = 3


def test_redacted_payload_never_echoes_contents() -> None:
    secret = "fixture-secret-that-must-not-echo"
    payload = RedactedPayload(root={"value": secret})

    assert secret not in repr(payload)


@pytest.mark.parametrize(
    "name",
    [
        "RedactedPayload",
        "create_payload_digest",
        "hmac_payload_digest",
        "synthetic_payload_digest",
        "verify_payload_digest",
    ],
)
def test_low_level_digest_primitives_are_not_in_the_public_facade(name: str) -> None:
    assert name not in public_security.__all__
    assert not hasattr(public_security, name)
