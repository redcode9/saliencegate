from __future__ import annotations

import weakref
from typing import Never, SupportsIndex


class PublicGenerationAuthorityError(ValueError):
    """A value-free failure at the public generation authority boundary."""

    def __init__(self) -> None:
        super().__init__("public generation authority failed validation")


class PublicGenerationAuthority:
    """An opaque capability for the future post-review public generation path."""

    __slots__ = ("__weakref__", "_identity")

    def __new__(
        cls,
        *args: object,
        **kwargs: object,
    ) -> Never:
        del cls, args, kwargs
        raise PublicGenerationAuthorityError()

    def __init__(
        self,
        *args: object,
        **kwargs: object,
    ) -> None:
        del self, args, kwargs

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise PublicGenerationAuthorityError()

    def __setattr__(self, name: str, value: object) -> None:
        del self, name, value
        raise PublicGenerationAuthorityError()

    def __delattr__(self, name: str) -> None:
        del self, name
        raise PublicGenerationAuthorityError()

    def __copy__(self) -> PublicGenerationAuthority:
        del self
        raise PublicGenerationAuthorityError()

    def __deepcopy__(self, memo: dict[int, object]) -> PublicGenerationAuthority:
        del self, memo
        raise PublicGenerationAuthorityError()

    def __reduce__(self) -> Never:
        del self
        raise PublicGenerationAuthorityError()

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        del self, protocol
        raise PublicGenerationAuthorityError()

    def __getstate__(self) -> Never:
        del self
        raise PublicGenerationAuthorityError()

    def __repr__(self) -> str:
        return "PublicGenerationAuthority(<opaque>)"

    __str__ = __repr__


class _PublicGenerationAuthorityState:
    __slots__ = ("capability_kind", "registry", "review_subreport")
    capability_kind: str
    registry: object
    review_subreport: object

    def __init__(
        self,
        *,
        registry: object,
        review_subreport: object,
        capability_kind: str,
    ) -> None:
        object.__setattr__(self, "registry", registry)
        object.__setattr__(self, "review_subreport", review_subreport)
        object.__setattr__(self, "capability_kind", capability_kind)

    def __setattr__(self, name: str, value: object) -> None:
        del self, name, value
        raise PublicGenerationAuthorityError()

    def __delattr__(self, name: str) -> None:
        del self, name
        raise PublicGenerationAuthorityError()


_ISSUED_PUBLIC_GENERATION_AUTHORITIES: dict[
    int,
    tuple[
        weakref.ReferenceType[PublicGenerationAuthority],
        _PublicGenerationAuthorityState,
    ],
] = {}


def require_public_generation_authority(value: object) -> _PublicGenerationAuthorityState:
    """Resolve the bound state only when the registered capability identity is live."""

    if type(value) is not PublicGenerationAuthority:
        raise PublicGenerationAuthorityError()
    registered = _ISSUED_PUBLIC_GENERATION_AUTHORITIES.get(id(value))
    if registered is None or registered[0]() is not value:
        raise PublicGenerationAuthorityError()
    return registered[1]


__all__ = [
    "PublicGenerationAuthority",
    "PublicGenerationAuthorityError",
    "require_public_generation_authority",
]
