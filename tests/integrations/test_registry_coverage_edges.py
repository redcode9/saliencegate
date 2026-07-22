"""Coverage-oriented regressions for otherwise rare contract edges."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from tests.cli.test_capture_connect import _spec

from saliencegate.capture import CaptureProfile
from saliencegate.integrations.registry import (
    BUILTIN_PROVIDER_REGISTRY,
    ProviderAlias,
    ProviderInstallationKind,
    ProviderRegistration,
    ProviderRegistry,
    ProviderRegistryError,
    _exact_absolute_path,
)


def test_registry_rejects_noncanonical_aliases_and_duplicate_profiles() -> None:
    assert repr(BUILTIN_PROVIDER_REGISTRY) == "ProviderRegistry(<redacted>)"
    providers = BUILTIN_PROVIDER_REGISTRY.providers

    duplicated_alias = BUILTIN_PROVIDER_REGISTRY.model_copy(
        update={"providers": (providers[0], providers[0], *providers[2:])}
    )
    with pytest.raises(ValueError, match="canonical"):
        duplicated_alias.providers_are_closed_and_canonical()

    duplicate_profile = providers[1].model_copy(update={"profile": providers[0].profile})
    ambiguous = BUILTIN_PROVIDER_REGISTRY.model_copy(
        update={"providers": (providers[0], duplicate_profile, *providers[2:])}
    )
    with pytest.raises(ValueError, match="ambiguous"):
        ambiguous.providers_are_closed_and_canonical()


def test_registry_resolution_maps_invalid_flags_aliases_and_availability() -> None:
    with pytest.raises(ProviderRegistryError):
        BUILTIN_PROVIDER_REGISTRY.resolve(ProviderAlias.CODEX, require_available=1)  # type: ignore[arg-type]
    with pytest.raises(ProviderRegistryError):
        BUILTIN_PROVIDER_REGISTRY.resolve(object())  # type: ignore[arg-type]
    with pytest.raises(ProviderRegistryError):
        BUILTIN_PROVIDER_REGISTRY.resolve("unknown")

    unavailable = BUILTIN_PROVIDER_REGISTRY.providers[0].model_copy(update={"available": False})
    registry = BUILTIN_PROVIDER_REGISTRY.model_copy(
        update={
            "providers": (unavailable, *BUILTIN_PROVIDER_REGISTRY.providers[1:]),
        }
    )
    with pytest.raises(ProviderRegistryError):
        registry.resolve(ProviderAlias.CODEX)
    assert registry.resolve(ProviderAlias.CODEX, require_available=False) is unavailable


@pytest.mark.parametrize("value", ["relative", Path("relative"), Path("/")])
def test_exact_absolute_path_rejects_invalid_boundary_values(value: object) -> None:
    with pytest.raises(ValueError, match="invalid"):
        _exact_absolute_path(value)  # type: ignore[arg-type]


def test_provider_registration_strictly_rejects_invalid_runtime_shapes() -> None:
    with pytest.raises(ValidationError):
        ProviderRegistration(
            alias=ProviderAlias.CODEX,
            profile=CaptureProfile.CODEX_HOOKS_V1,
            host_name="Codex",
            host_version="1.0.0",
            available=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError):
        ProviderRegistry(providers=BUILTIN_PROVIDER_REGISTRY.providers[:-1])


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"bundle_path": Path("/outside/bundle.js")}, "escape"),
        ({"receipt_path": Path("/tmp/operational/launcher")}, "alias"),
        ({"journal_path": Path("/tmp/elsewhere/journal")}, "share"),
        ({"launcher_path": Path("/tmp/elsewhere/launcher")}, "outside"),
        ({"launcher_bytes": b"bad\x00launcher"}, "launcher"),
    ],
)
def test_installation_spec_defensive_path_and_launcher_invariants(
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
) -> None:
    spec = _spec(tmp_path)
    if "receipt_path" in updates:
        updates["receipt_path"] = spec.launcher_path
    elif "journal_path" in updates:
        updates["journal_path"] = tmp_path / "elsewhere" / "journal"
    elif "launcher_path" in updates:
        updates["launcher_path"] = tmp_path / "elsewhere" / "launcher"
    forged = spec.model_copy(update=updates)
    with pytest.raises(ValueError, match=message):
        forged.paths_and_bundle_contract_are_closed()


def test_receipt_cannot_be_nested_in_project_and_defensive_properties_fail_closed(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    nested_receipt = spec.model_copy(
        update={
            "receipt_path": spec.project_root / "receipt.json",
            "journal_path": spec.project_root / "journal.json",
            "lock_path": spec.project_root / "lock",
            "launcher_path": spec.project_root / "launcher",
        }
    )
    with pytest.raises(ValueError, match="outside the project"):
        nested_receipt.paths_and_bundle_contract_are_closed()

    command_hook = spec.model_copy(
        update={
            "installation_kind": ProviderInstallationKind.COMMAND_HOOK,
            "config_path": None,
            "config": None,
            "bundle_path": None,
            "bootstrap_path": None,
            "bundle_bytes": None,
            "bootstrap_relative_reference": None,
        }
    )
    with pytest.raises(ValueError, match="requires configuration"):
        _ = command_hook.project_local_paths

    broken_bridge = spec.model_copy(update={"bundle_path": None})
    with pytest.raises(ValueError, match="asset binding"):
        _ = broken_bridge.project_local_paths
