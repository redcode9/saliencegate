from __future__ import annotations

import copy
import inspect
import json
import pickle
import weakref
from collections.abc import Callable

import pytest

import saliencegate.benchmarks.state_decay_v2.generation_authority as authority_module
from saliencegate.benchmarks.state_decay_v2.generation_authority import (
    PublicGenerationAuthority,
    PublicGenerationAuthorityError,
    require_public_generation_authority,
)


class _DuckAuthority:
    _identity = object()


class _ConstructorPickle:
    def __reduce__(self) -> tuple[type[PublicGenerationAuthority], tuple[()]]:
        return PublicGenerationAuthority, ()


def _forged_uninitialized_authority() -> PublicGenerationAuthority:
    return object.__new__(PublicGenerationAuthority)


def test_exports_only_a_closed_unmintable_authority_boundary() -> None:
    assert authority_module.__all__ == [
        "PublicGenerationAuthority",
        "PublicGenerationAuthorityError",
        "require_public_generation_authority",
    ]
    assert {
        name
        for name, value in vars(authority_module).items()
        if type(value) is PublicGenerationAuthority
    } == set()
    assert {
        name
        for name, value in vars(authority_module).items()
        if inspect.isfunction(value) and not name.startswith("__")
    } == {"require_public_generation_authority"}
    assert authority_module._ISSUED_PUBLIC_GENERATION_AUTHORITIES == {}
    assert not any("sentinel" in name.casefold() for name in vars(authority_module))


@pytest.mark.parametrize(
    "construct",
    (
        lambda: PublicGenerationAuthority(),
        lambda: PublicGenerationAuthority(_identity=None),
        lambda: PublicGenerationAuthority(_identity=object()),
        lambda: PublicGenerationAuthority(identity=object()),
        lambda: PublicGenerationAuthority(sentinel=object()),
        lambda: PublicGenerationAuthority(token=object()),
    ),
)
def test_direct_construction_is_value_free_and_unavailable(
    construct: Callable[[], object],
) -> None:
    with pytest.raises(PublicGenerationAuthorityError) as raised:
        construct()

    assert str(raised.value) == "public generation authority failed validation"
    assert repr(raised.value).count("object at") == 0


def test_subclassing_is_rejected_at_class_definition_time() -> None:
    with pytest.raises(PublicGenerationAuthorityError):

        class _Subclass(PublicGenerationAuthority):
            pass


@pytest.mark.parametrize(
    "candidate",
    (
        None,
        object(),
        _DuckAuthority(),
        {},
        (),
        "authority",
    ),
)
def test_exact_type_and_private_identity_are_both_required(candidate: object) -> None:
    with pytest.raises(PublicGenerationAuthorityError):
        require_public_generation_authority(candidate)


def test_object_new_and_guessed_identity_cannot_forge_authority() -> None:
    forged = _forged_uninitialized_authority()

    with pytest.raises(PublicGenerationAuthorityError):
        require_public_generation_authority(forged)

    object.__setattr__(forged, "_identity", object())
    with pytest.raises(PublicGenerationAuthorityError):
        require_public_generation_authority(forged)
    with pytest.raises(PublicGenerationAuthorityError):
        forged._identity = object()  # type: ignore[attr-defined]


@pytest.mark.parametrize("copier", (copy.copy, copy.deepcopy))
def test_authority_copying_is_prohibited(copier: Callable[[object], object]) -> None:
    forged = _forged_uninitialized_authority()

    with pytest.raises(PublicGenerationAuthorityError):
        copier(forged)


def test_pickle_emission_and_constructor_deserialization_are_prohibited() -> None:
    forged = _forged_uninitialized_authority()

    with pytest.raises(PublicGenerationAuthorityError):
        pickle.dumps(forged)
    payload = pickle.dumps(_ConstructorPickle())
    with pytest.raises(PublicGenerationAuthorityError):
        pickle.loads(payload)


def test_json_emission_and_object_deserialization_cannot_create_authority() -> None:
    forged = _forged_uninitialized_authority()

    with pytest.raises(TypeError):
        json.dumps(forged)
    decoded = json.loads("{}")
    with pytest.raises(PublicGenerationAuthorityError):
        PublicGenerationAuthority(**decoded)


def test_opaque_representation_contains_no_identity_material() -> None:
    forged = _forged_uninitialized_authority()
    object.__setattr__(forged, "_identity", object())

    assert repr(forged) == "PublicGenerationAuthority(<opaque>)"
    assert str(forged) == "PublicGenerationAuthority(<opaque>)"
    assert not hasattr(forged, "__dict__")


def test_uninitialized_authority_protocol_methods_remain_closed() -> None:
    forged = _forged_uninitialized_authority()

    PublicGenerationAuthority.__init__(forged, "ignored", identity=object())
    with pytest.raises(PublicGenerationAuthorityError):
        del forged._identity  # type: ignore[attr-defined]
    with pytest.raises(PublicGenerationAuthorityError):
        forged.__reduce__()
    with pytest.raises(PublicGenerationAuthorityError):
        forged.__getstate__()


def test_registered_live_identity_resolves_only_its_immutable_bound_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _forged_uninitialized_authority()
    state = authority_module._PublicGenerationAuthorityState(
        registry=object(),
        review_subreport=object(),
        capability_kind="coverage-probe",
    )
    monkeypatch.setitem(
        authority_module._ISSUED_PUBLIC_GENERATION_AUTHORITIES,
        id(authority),
        (weakref.ref(authority), state),
    )

    assert require_public_generation_authority(authority) is state
    with pytest.raises(PublicGenerationAuthorityError):
        state.registry = object()
    with pytest.raises(PublicGenerationAuthorityError):
        del state.review_subreport
