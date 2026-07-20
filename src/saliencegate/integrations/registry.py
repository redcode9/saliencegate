"""Closed provider metadata and strict provider-neutral installation specifications."""

from __future__ import annotations

import hashlib
import os
import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from saliencegate.capture.capabilities import CaptureProfile
from saliencegate.integrations.config_files import OwnedConfigSpec

MAX_INTEGRATION_BUNDLE_BYTES = 2 * 1_024 * 1_024
MAX_INTEGRATION_LAUNCHER_BYTES = 256 * 1_024
_PROVIDER_ID = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
_HOST_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+_-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RELATIVE_BOOTSTRAP = re.compile(r"^\./[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.json$")


class ProviderRegistryError(ValueError):
    """A content-free provider registry failure."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("capture provider is unavailable")


class ProviderAlias(StrEnum):
    CODEX = "codex"
    CLAUDE_CODE = "claude-code"
    OPENCODE = "opencode"
    PI = "pi"


class _RegistryModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(<redacted>)"

    __str__ = __repr__


class ProviderRegistration(_RegistryModel):
    alias: ProviderAlias
    profile: CaptureProfile
    host_name: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    host_version: Annotated[
        str,
        StringConstraints(min_length=1, max_length=64, pattern=_HOST_VERSION.pattern),
    ]
    available: bool = False


class ProviderRegistry(_RegistryModel):
    schema_version: Literal["provider-installation-registry/v1"] = (
        "provider-installation-registry/v1"
    )
    providers: Annotated[tuple[ProviderRegistration, ...], Field(min_length=1, max_length=16)]

    @model_validator(mode="after")
    def providers_are_closed_and_canonical(self) -> Self:
        aliases = tuple(item.alias for item in self.providers)
        if aliases != tuple(ProviderAlias) or len(set(aliases)) != len(aliases):
            raise ValueError("provider registry is not canonical")
        if len({item.profile for item in self.providers}) != len(self.providers):
            raise ValueError("provider registry profile mapping is ambiguous")
        return self

    def resolve(
        self,
        alias: ProviderAlias | str,
        *,
        require_available: bool = True,
    ) -> ProviderRegistration:
        try:
            if type(require_available) is not bool:
                raise ProviderRegistryError()
            selected = alias if type(alias) is ProviderAlias else ProviderAlias(alias)
            if type(alias) not in (str, ProviderAlias):
                raise ProviderRegistryError()
            result = next(item for item in self.providers if item.alias is selected)
            if require_available and not result.available:
                raise ProviderRegistryError()
            return result
        except ProviderRegistryError:
            raise
        except Exception:
            raise ProviderRegistryError() from None


def _exact_absolute_path(value: Path) -> Path:
    if not isinstance(value, Path) or not value.is_absolute() or ".." in value.parts:
        raise ValueError("integration path is invalid")
    raw = os.fspath(value)
    if not raw or "\x00" in raw or value.name in ("", ".", ".."):
        raise ValueError("integration path is invalid")
    return value


class ProviderInstallationSpec(_RegistryModel):
    """All provider-neutral inputs for one project-local installation generation."""

    provider_id: Annotated[
        str,
        StringConstraints(min_length=1, max_length=64, pattern=_PROVIDER_ID.pattern),
    ]
    profile: CaptureProfile
    host_version: Annotated[
        str,
        StringConstraints(min_length=1, max_length=64, pattern=_HOST_VERSION.pattern),
    ]
    project_root: Path
    config_path: Path
    bundle_path: Path
    bootstrap_path: Path
    receipt_path: Path
    journal_path: Path
    lock_path: Path
    launcher_path: Path
    capability_digest: Annotated[str, StringConstraints(pattern=_SHA256.pattern)]
    bundle_bytes: Annotated[
        bytes,
        Field(min_length=1, max_length=MAX_INTEGRATION_BUNDLE_BYTES, repr=False),
    ]
    launcher_bytes: Annotated[
        bytes,
        Field(min_length=1, max_length=MAX_INTEGRATION_LAUNCHER_BYTES, repr=False),
    ]
    bootstrap_relative_reference: Annotated[
        str,
        StringConstraints(
            min_length=8,
            max_length=132,
            pattern=_RELATIVE_BOOTSTRAP.pattern,
        ),
    ]
    config: OwnedConfigSpec
    generation: Annotated[int, Field(ge=1, le=1_000_000)] = 1

    @field_validator(
        "project_root",
        "config_path",
        "bundle_path",
        "bootstrap_path",
        "receipt_path",
        "journal_path",
        "lock_path",
        "launcher_path",
    )
    @classmethod
    def paths_are_exact_and_absolute(cls, value: Path) -> Path:
        return _exact_absolute_path(value)

    @model_validator(mode="after")
    def paths_and_bundle_contract_are_closed(self) -> Self:
        project_files = (self.config_path, self.bundle_path, self.bootstrap_path)
        try:
            for path in project_files:
                relative = path.relative_to(self.project_root)
                if not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
                    raise ValueError
        except ValueError:
            raise ValueError("integration project paths escape the project root") from None
        all_paths = (
            *project_files,
            self.receipt_path,
            self.journal_path,
            self.lock_path,
            self.launcher_path,
        )
        if len(set(all_paths)) != len(all_paths):
            raise ValueError("integration paths alias")
        if self.bundle_path.parent != self.bootstrap_path.parent:
            raise ValueError("integration bundle and bootstrap do not share a boundary")
        if not (self.receipt_path.parent == self.journal_path.parent == self.lock_path.parent):
            raise ValueError("integration operational paths do not share a boundary")
        if self.launcher_path.parent != self.receipt_path.parent:
            raise ValueError("integration launcher is outside its operational boundary")
        try:
            self.receipt_path.relative_to(self.project_root)
        except ValueError:
            pass
        else:
            raise ValueError("integration receipt must be outside the project")
        expected_reference = f"./{self.bootstrap_path.name}"
        reference_bytes = self.bootstrap_relative_reference.encode("utf-8")
        expected_binding = (
            "export const saliencegateBootstrap = "
            f'new URL("{self.bootstrap_relative_reference}", import.meta.url);\n'
        ).encode()
        if (
            self.bootstrap_relative_reference != expected_reference
            or not self.bundle_bytes.startswith(expected_binding)
            or self.bundle_bytes.count(expected_binding) != 1
            or self.bundle_bytes.count(b"new URL(") != 1
            or self.bundle_bytes.count(b"import.meta.url") != 1
            or self.bundle_bytes.count(reference_bytes) != 1
            or b"\x00" in self.bundle_bytes
            or b"\x00" in self.launcher_bytes
        ):
            raise ValueError("integration bundle bootstrap lookup is invalid")
        return self

    @property
    def bundle_digest(self) -> str:
        return hashlib.sha256(self.bundle_bytes).hexdigest()

    @property
    def launcher_digest(self) -> str:
        return hashlib.sha256(self.launcher_bytes).hexdigest()


BUILTIN_PROVIDER_REGISTRY = ProviderRegistry(
    providers=(
        ProviderRegistration(
            alias=ProviderAlias.CODEX,
            profile=CaptureProfile.CODEX_HOOKS_V1,
            host_name="Codex CLI",
            host_version="0.144.6",
        ),
        ProviderRegistration(
            alias=ProviderAlias.CLAUDE_CODE,
            profile=CaptureProfile.CLAUDE_CODE_HOOKS_V1,
            host_name="Claude Code",
            host_version="2.1.204",
        ),
        ProviderRegistration(
            alias=ProviderAlias.OPENCODE,
            profile=CaptureProfile.OPENCODE_PLUGIN_V1,
            host_name="OpenCode",
            host_version="1.18.3",
        ),
        ProviderRegistration(
            alias=ProviderAlias.PI,
            profile=CaptureProfile.PI_EXTENSION_V1,
            host_name="@earendil-works/pi-coding-agent",
            host_version="0.80.10",
        ),
    )
)


__all__ = [
    "BUILTIN_PROVIDER_REGISTRY",
    "MAX_INTEGRATION_BUNDLE_BYTES",
    "MAX_INTEGRATION_LAUNCHER_BYTES",
    "ProviderAlias",
    "ProviderInstallationSpec",
    "ProviderRegistration",
    "ProviderRegistry",
    "ProviderRegistryError",
]
