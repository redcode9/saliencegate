from __future__ import annotations

import hmac

from saliencegate.domain import (
    PayloadDigest,
    PayloadDigestAlgorithm,
    canonical_json,
    length_prefixed_sha256,
)
from saliencegate.security.digests import (
    AmbiguousDigestModeError,
    MissingInstallationKeyError,
)
from saliencegate.security.keys import InstallationKey


class IntegrityContext:
    __slots__ = ("_key", "_synthetic_benchmark")

    def __init__(
        self,
        *,
        key: InstallationKey | None,
        synthetic_benchmark: bool,
    ) -> None:
        if key is not None and synthetic_benchmark:
            raise AmbiguousDigestModeError(
                "installation key and synthetic mode are mutually exclusive"
            )
        if key is None and not synthetic_benchmark:
            raise MissingInstallationKeyError(
                "repository integrity requires an installation key outside synthetic mode"
            )
        if key is not None and type(key) is not InstallationKey:
            raise TypeError("repository integrity requires exactly InstallationKey")
        self._key = None if key is None else key._copy()
        self._synthetic_benchmark = synthetic_benchmark

    def __repr__(self) -> str:
        mode = "synthetic" if self._synthetic_benchmark else "hmac"
        return f"IntegrityContext(mode={mode!r})"

    @property
    def synthetic_benchmark(self) -> bool:
        return self._synthetic_benchmark

    @property
    def key(self) -> InstallationKey | None:
        return None if self._key is None else self._key._copy()

    def tag(self, value: object, *, domain: str) -> PayloadDigest:
        encoded = canonical_json(value)
        if self._synthetic_benchmark:
            return PayloadDigest(
                algorithm=PayloadDigestAlgorithm.SYNTHETIC_SHA256,
                value=length_prefixed_sha256(encoded, domain=domain),
            )
        if self._key is None:  # pragma: no cover - constructor invariant
            raise MissingInstallationKeyError("repository integrity key is unavailable")
        return PayloadDigest(
            algorithm=PayloadDigestAlgorithm.HMAC_SHA256,
            value=self._key._hmac_sha256(encoded, domain=domain.encode("utf-8")),
        )

    def verify(self, value: object, tag: PayloadDigest, *, domain: str) -> bool:
        expected = self.tag(value, domain=domain)
        return tag.algorithm is expected.algorithm and hmac.compare_digest(
            tag.value,
            expected.value,
        )
