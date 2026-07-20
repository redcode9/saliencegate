"""Lazy public exports for bundled capture integration contracts.

The provider-facing hook lives below this package. Keeping package import free of
Pydantic and installer initialization is part of its measured fail-open latency
contract; public symbols retain their normal import surface through PEP 562.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from saliencegate.integrations.bootstrap import (  # noqa: F401
        MAX_INTEGRATION_BOOTSTRAP_BYTES,
        IntegrationBootstrap,
        IntegrationBootstrapError,
        decode_integration_bootstrap,
        encode_integration_bootstrap,
        inspect_integration_bootstrap,
        publish_integration_bootstrap,
    )
    from saliencegate.integrations.config_files import (  # noqa: F401
        MAX_PROVIDER_CONFIG_BYTES,
        ConfigFileError,
        ConfigSyntax,
        OwnedConfigPlan,
        OwnedConfigReverseEdit,
        OwnedConfigSpec,
        plan_owned_config_install,
        remove_owned_config_edit,
    )
    from saliencegate.integrations.installation import (  # noqa: F401
        InstallationDisposition,
        InstallationError,
        InstallationIdentity,
        InstallationJournal,
        InstallationReceipt,
        InstallationState,
        InstallationStatus,
        derive_installation_identity,
        ensure_private_installation_directory,
        git_tracked_project_files,
        inspect_installation_receipt,
        inspect_provider_installation,
        install_provider,
        recover_provider_installation,
        uninstall_provider,
    )
    from saliencegate.integrations.launcher_renderer import (  # noqa: F401
        CaptureLauncherPlatform,
        LauncherRenderError,
        render_capture_launcher,
    )
    from saliencegate.integrations.registry import (  # noqa: F401
        BUILTIN_PROVIDER_REGISTRY,
        MAX_INTEGRATION_BUNDLE_BYTES,
        MAX_INTEGRATION_LAUNCHER_BYTES,
        ProviderAlias,
        ProviderInstallationKind,
        ProviderInstallationSpec,
        ProviderRegistration,
        ProviderRegistry,
        ProviderRegistryError,
    )

_EXPORT_MODULES = {
    "BUILTIN_PROVIDER_REGISTRY": "saliencegate.integrations.registry",
    "MAX_INTEGRATION_BOOTSTRAP_BYTES": "saliencegate.integrations.bootstrap",
    "MAX_INTEGRATION_BUNDLE_BYTES": "saliencegate.integrations.registry",
    "MAX_INTEGRATION_LAUNCHER_BYTES": "saliencegate.integrations.registry",
    "MAX_PROVIDER_CONFIG_BYTES": "saliencegate.integrations.config_files",
    "CaptureLauncherPlatform": "saliencegate.integrations.launcher_renderer",
    "ConfigFileError": "saliencegate.integrations.config_files",
    "ConfigSyntax": "saliencegate.integrations.config_files",
    "InstallationDisposition": "saliencegate.integrations.installation",
    "InstallationError": "saliencegate.integrations.installation",
    "InstallationIdentity": "saliencegate.integrations.installation",
    "InstallationJournal": "saliencegate.integrations.installation",
    "InstallationReceipt": "saliencegate.integrations.installation",
    "InstallationState": "saliencegate.integrations.installation",
    "InstallationStatus": "saliencegate.integrations.installation",
    "IntegrationBootstrap": "saliencegate.integrations.bootstrap",
    "IntegrationBootstrapError": "saliencegate.integrations.bootstrap",
    "LauncherRenderError": "saliencegate.integrations.launcher_renderer",
    "OwnedConfigPlan": "saliencegate.integrations.config_files",
    "OwnedConfigReverseEdit": "saliencegate.integrations.config_files",
    "OwnedConfigSpec": "saliencegate.integrations.config_files",
    "ProviderAlias": "saliencegate.integrations.registry",
    "ProviderInstallationKind": "saliencegate.integrations.registry",
    "ProviderInstallationSpec": "saliencegate.integrations.registry",
    "ProviderRegistration": "saliencegate.integrations.registry",
    "ProviderRegistry": "saliencegate.integrations.registry",
    "ProviderRegistryError": "saliencegate.integrations.registry",
    "decode_integration_bootstrap": "saliencegate.integrations.bootstrap",
    "derive_installation_identity": "saliencegate.integrations.installation",
    "ensure_private_installation_directory": "saliencegate.integrations.installation",
    "encode_integration_bootstrap": "saliencegate.integrations.bootstrap",
    "git_tracked_project_files": "saliencegate.integrations.installation",
    "inspect_integration_bootstrap": "saliencegate.integrations.bootstrap",
    "inspect_installation_receipt": "saliencegate.integrations.installation",
    "inspect_provider_installation": "saliencegate.integrations.installation",
    "install_provider": "saliencegate.integrations.installation",
    "plan_owned_config_install": "saliencegate.integrations.config_files",
    "publish_integration_bootstrap": "saliencegate.integrations.bootstrap",
    "recover_provider_installation": "saliencegate.integrations.installation",
    "remove_owned_config_edit": "saliencegate.integrations.config_files",
    "render_capture_launcher": "saliencegate.integrations.launcher_renderer",
    "uninstall_provider": "saliencegate.integrations.installation",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    if name == "launcher_renderer":
        return import_module("saliencegate.integrations.launcher_renderer")
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORT_MODULES, "launcher_renderer"})
