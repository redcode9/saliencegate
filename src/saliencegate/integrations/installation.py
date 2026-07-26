"""Crash-recoverable, provider-neutral project integration installation."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import stat
import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PureWindowsPath
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

import saliencegate.security.files as security_files
from saliencegate.capture.capabilities import CaptureProfile, validate_capture_capability_binding
from saliencegate.capture.identities import CaptureDigestContext
from saliencegate.domain import canonical_json
from saliencegate.integrations.bootstrap import (
    IntegrationBootstrap,
    encode_integration_bootstrap,
    inspect_integration_bootstrap,
    publish_integration_bootstrap,
)
from saliencegate.integrations.config_files import (
    ConfigFileError,
    OwnedConfigPlan,
    OwnedConfigReverseEdit,
    _owned_config_edit_matches_spec,
    delete_config_bytes,
    plan_owned_config_install,
    publish_config_bytes,
    read_config_bytes,
    remove_owned_config_edit,
)
from saliencegate.integrations.registry import (
    MAX_INTEGRATION_LAUNCHER_BYTES,
    ProviderInstallationKind,
    ProviderInstallationSpec,
)
from saliencegate.security import InstallationKey
from saliencegate.security.files import (
    StableReadPolicy,
    authorize_atomic_file_publication,
    delete_authorized_private_file,
    ensure_private_directory,
    inspect_private_directory,
    read_stable_file,
)
from saliencegate.security.windows import (
    NativeWindowsSecurityOperations,
    WindowsPathKind,
    WindowsPathSecurity,
    WindowsSecurityOperations,
    authorize_windows_managed_path,
    authorize_windows_private_path,
    ensure_windows_private_directory,
)

MAX_INSTALLATION_RECEIPT_BYTES = 512 * 1_024
MAX_INSTALLATION_JOURNAL_BYTES = 768 * 1_024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONNECTION_ID = re.compile(r"^sg-[0-9a-f]{48}$")
_RECEIPT_DOMAIN = b"saliencegate:integration-installation-receipt:v1"
_JOURNAL_DOMAIN = b"saliencegate:integration-installation-journal:v1"
_CONNECTION_DOMAIN = b"saliencegate:integration-connection:v1"
_DRIFT_ORDER = (
    "receipt",
    "lock",
    "config",
    "bundle",
    "bootstrap",
    "launcher",
    "host_version",
)


class InstallationError(RuntimeError):
    """A content-free installer failure."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("capture integration installation failed")


class InstallationState(StrEnum):
    PENDING = "pending"
    ENABLED = "enabled"
    DRAINING = "draining"
    DISABLED = "disabled"


class InstallationDisposition(StrEnum):
    PLANNED = "planned"
    INSTALLED = "installed"
    NOOP = "noop"
    UPGRADED = "upgraded"
    UNINSTALLED = "uninstalled"
    RECOVERED = "recovered"


class GitProjectFileDisposition(StrEnum):
    """Read-only Git visibility of the provider's managed project files."""

    NOT_REPOSITORY = "not_repository"
    UNAVAILABLE = "unavailable"
    ALL_IGNORED = "all_ignored"
    UNIGNORED = "unignored"


@dataclass(frozen=True, slots=True)
class GitProjectFileReview:
    """Bounded Git review result without exposing project paths in CLI output."""

    disposition: GitProjectFileDisposition
    project_local_files: tuple[Path, ...]
    unignored_files: tuple[Path, ...]
    tracked_files: tuple[Path, ...]


class _JournalOperation(StrEnum):
    INSTALL = "install"
    UNINSTALL = "uninstall"


class _InstallationModel(BaseModel):
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


def _absolute_path(value: Path) -> Path:
    if (
        not isinstance(value, Path)
        or not value.is_absolute()
        or ".." in value.parts
        or "\x00" in os.fspath(value)
        or not value.name
    ):
        raise ValueError("installation path is invalid")
    return value


class InstallationReceipt(_InstallationModel):
    schema_version: Literal["integration-installation-receipt/v1"] = (
        "integration-installation-receipt/v1"
    )
    state: InstallationState
    provider_id: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9-]{0,63}$")]
    profile: CaptureProfile
    host_version: Annotated[
        str,
        StringConstraints(
            min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9.+_-]{0,63}$"
        ),
    ]
    generation: Annotated[int, Field(ge=1, le=1_000_000)]
    connection_id: Annotated[str, StringConstraints(pattern=_CONNECTION_ID.pattern)] = Field(
        repr=False
    )
    project_digest: Annotated[str, StringConstraints(pattern=_SHA256.pattern)] = Field(repr=False)
    capability_digest: Annotated[str, StringConstraints(pattern=_SHA256.pattern)] = Field(
        repr=False
    )
    config_path: Path | None = Field(default=None, repr=False)
    bundle_path: Path | None = Field(default=None, repr=False)
    bootstrap_path: Path | None = Field(default=None, repr=False)
    launcher_path: Path = Field(repr=False)
    receipt_path: Path = Field(repr=False)
    journal_path: Path = Field(repr=False)
    lock_path: Path = Field(repr=False)
    bundle_digest: Annotated[str | None, StringConstraints(pattern=_SHA256.pattern)] = Field(
        default=None,
        repr=False,
    )
    launcher_digest: Annotated[str, StringConstraints(pattern=_SHA256.pattern)] = Field(repr=False)
    config_edit: OwnedConfigReverseEdit | None = Field(default=None, repr=False)
    receipt_mac: Annotated[str, StringConstraints(pattern=_SHA256.pattern)] = Field(repr=False)

    @property
    def installation_kind(self) -> ProviderInstallationKind:
        bridge_values = (self.bundle_path, self.bootstrap_path, self.bundle_digest)
        if all(value is None for value in bridge_values):
            return ProviderInstallationKind.COMMAND_HOOK
        return ProviderInstallationKind.BRIDGE

    @field_validator(
        "launcher_path",
        "receipt_path",
        "journal_path",
        "lock_path",
    )
    @classmethod
    def paths_are_absolute(cls, value: Path) -> Path:
        return _absolute_path(value)

    @field_validator("config_path", "bundle_path", "bootstrap_path")
    @classmethod
    def optional_paths_are_absolute(cls, value: Path | None) -> Path | None:
        return None if value is None else _absolute_path(value)

    @model_validator(mode="after")
    def paths_and_digests_are_unambiguous(self) -> Self:
        has_config = self.config_path is not None
        if has_config != (self.config_edit is not None):
            raise ValueError("installation receipt configuration binding is incomplete")
        bridge_values = (self.bundle_path, self.bootstrap_path, self.bundle_digest)
        if self.installation_kind is ProviderInstallationKind.BRIDGE:
            if any(value is None for value in bridge_values):
                raise ValueError("installation receipt bridge binding is incomplete")
        else:
            if any(value is not None for value in bridge_values):
                raise ValueError("command-hook receipt declares bridge assets")
            if not has_config:
                raise ValueError("command-hook receipt requires configuration")
        paths = {
            self.launcher_path,
            self.receipt_path,
            self.journal_path,
            self.lock_path,
        }
        if self.config_path is not None:
            paths.add(self.config_path)
        if self.bundle_path is not None:
            paths.add(self.bundle_path)
        if self.bootstrap_path is not None:
            paths.add(self.bootstrap_path)
        expected_path_count = (
            4
            + (1 if self.config_path is not None else 0)
            + (2 if self.installation_kind is ProviderInstallationKind.BRIDGE else 0)
        )
        if len(paths) != expected_path_count or not (
            self.receipt_path.parent == self.journal_path.parent == self.lock_path.parent
        ):
            raise ValueError("installation receipt paths alias")
        if (
            self.installation_kind is ProviderInstallationKind.BRIDGE
            and self.bundle_path is not None
            and self.bootstrap_path is not None
            and self.bundle_path.parent != self.bootstrap_path.parent
        ):
            raise ValueError("installation receipt bridge boundary is invalid")
        return self


class InstallationJournal(_InstallationModel):
    schema_version: Literal["integration-installation-journal/v1"] = (
        "integration-installation-journal/v1"
    )
    operation: _JournalOperation
    target_receipt: InstallationReceipt = Field(repr=False)
    target_bootstrap_digest: Annotated[
        str | None,
        StringConstraints(pattern=_SHA256.pattern),
    ] = Field(default=None, repr=False)
    prior_receipt_mac: Annotated[str | None, StringConstraints(pattern=_SHA256.pattern)] = Field(
        default=None,
        repr=False,
    )
    prior_bundle_path: Path | None = Field(default=None, repr=False)
    prior_bundle_digest: Annotated[str | None, StringConstraints(pattern=_SHA256.pattern)] = Field(
        default=None,
        repr=False,
    )
    prior_launcher_path: Path | None = Field(default=None, repr=False)
    prior_launcher_digest: Annotated[
        str | None,
        StringConstraints(pattern=_SHA256.pattern),
    ] = Field(default=None, repr=False)
    prior_bootstrap_digest: Annotated[
        str | None,
        StringConstraints(pattern=_SHA256.pattern),
    ] = Field(default=None, repr=False)
    journal_mac: Annotated[str, StringConstraints(pattern=_SHA256.pattern)] = Field(repr=False)

    @field_validator("prior_bundle_path", "prior_launcher_path")
    @classmethod
    def optional_path_is_absolute(cls, value: Path | None) -> Path | None:
        return None if value is None else _absolute_path(value)

    @model_validator(mode="after")
    def prior_bundle_fields_are_paired(self) -> Self:
        if (self.prior_bundle_path is None) != (self.prior_bundle_digest is None):
            raise ValueError("installation journal prior bundle binding is incomplete")
        if (self.prior_launcher_path is None) != (self.prior_launcher_digest is None):
            raise ValueError("installation journal prior launcher binding is incomplete")
        has_bridge = self.target_receipt.installation_kind is ProviderInstallationKind.BRIDGE
        if has_bridge != (self.target_bootstrap_digest is not None):
            raise ValueError("installation journal target bridge binding is incomplete")
        if (self.prior_bundle_path is None) != (self.prior_bootstrap_digest is None):
            raise ValueError("installation journal prior bridge binding is incomplete")
        if not has_bridge and (
            self.prior_bundle_path is not None
            or self.prior_bundle_digest is not None
            or self.prior_bootstrap_digest is not None
        ):
            raise ValueError("command-hook journal declares bridge assets")
        if (
            self.operation is _JournalOperation.INSTALL
            and self.target_receipt.state is not InstallationState.ENABLED
        ) or (
            self.operation is _JournalOperation.UNINSTALL
            and (
                self.target_receipt.state is not InstallationState.DISABLED
                or self.prior_receipt_mac is None
                or self.prior_bundle_path != self.target_receipt.bundle_path
                or self.prior_bundle_digest != self.target_receipt.bundle_digest
                or self.prior_launcher_path != self.target_receipt.launcher_path
                or self.prior_launcher_digest != self.target_receipt.launcher_digest
            )
        ):
            raise ValueError("installation journal lifecycle binding is invalid")
        return self


class InstallationStatus(_InstallationModel):
    schema_version: Literal["integration-installation-status/v1"] = (
        "integration-installation-status/v1"
    )
    disposition: InstallationDisposition
    state: InstallationState
    provider_id: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9-]{0,63}$")]
    profile: CaptureProfile
    generation: Annotated[int, Field(ge=1, le=1_000_000)]
    connection_id: Annotated[str, StringConstraints(pattern=_CONNECTION_ID.pattern)] = Field(
        repr=False
    )
    project_digest: Annotated[str, StringConstraints(pattern=_SHA256.pattern)] = Field(repr=False)
    installed: bool
    drift: Annotated[tuple[str, ...], Field(max_length=len(_DRIFT_ORDER))]
    would_write: Annotated[tuple[Path, ...], Field(max_length=8)] = Field(default=(), repr=False)
    git_tracked_files: Annotated[tuple[Path, ...], Field(max_length=8)] = Field(
        default=(),
        repr=False,
    )

    @model_validator(mode="after")
    def status_is_canonical(self) -> Self:
        if self.drift != tuple(item for item in _DRIFT_ORDER if item in set(self.drift)):
            raise ValueError("installation drift is not canonical")
        if self.installed != (self.state is InstallationState.ENABLED and not self.drift):
            raise ValueError("installation status summary is inconsistent")
        return self


class InstallationIdentity(_InstallationModel):
    """Shared project and installation-generation binding for CLI/store orchestration."""

    schema_version: Literal["integration-installation-identity/v1"] = (
        "integration-installation-identity/v1"
    )
    project_digest: Annotated[str, StringConstraints(pattern=_SHA256.pattern)] = Field(repr=False)
    connection_id: Annotated[str, StringConstraints(pattern=_CONNECTION_ID.pattern)] = Field(
        repr=False
    )


def _reject_project_symlink_traversal(project_root: Path, path: Path) -> None:
    try:
        relative = path.relative_to(project_root)
    except ValueError:
        raise InstallationError() from None
    if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
        operations = NativeWindowsSecurityOperations()
        windows_current = PureWindowsPath(os.fspath(project_root))
        for index, component in enumerate(relative.parts):
            windows_current /= component
            security = operations.inspect_path(windows_current)
            if security is None:
                return
            authorization = authorize_windows_managed_path(
                windows_current,
                kind=security.kind,
                operations=operations,
            )
            authorization.revalidate()
            if index < len(relative.parts) - 1 and security.kind is not WindowsPathKind.DIRECTORY:
                raise InstallationError()
        return
    current = project_root
    for index, component in enumerate(relative.parts):
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(metadata.st_mode) or (
            index < len(relative.parts) - 1 and not stat.S_ISDIR(metadata.st_mode)
        ):
            raise InstallationError()


def _validated_spec(value: object) -> ProviderInstallationSpec:
    try:
        payload = (
            value.model_dump(mode="python", warnings="error")
            if type(value) is ProviderInstallationSpec
            else value
        )
        spec = ProviderInstallationSpec.model_validate(payload)
        resolved_project = spec.project_root.resolve(strict=True)
        if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
            operations = NativeWindowsSecurityOperations()
            project_security = operations.inspect_path(
                PureWindowsPath(os.fspath(spec.project_root))
            )
            if (
                resolved_project != spec.project_root
                or project_security is None
                or project_security.kind is not WindowsPathKind.DIRECTORY
                or project_security.reparse_tag is not None
                or project_security.hardlink_count != 1
            ):
                raise InstallationError()
            for project_path in spec.project_local_paths:
                _reject_project_symlink_traversal(spec.project_root, project_path)
            validate_capture_capability_binding(spec.profile, spec.capability_digest)
            return spec
        project_metadata = spec.project_root.lstat()
        if (
            resolved_project != spec.project_root
            or not stat.S_ISDIR(project_metadata.st_mode)
            or stat.S_ISLNK(project_metadata.st_mode)
        ):
            raise InstallationError()
        for project_path in spec.project_local_paths:
            _reject_project_symlink_traversal(spec.project_root, project_path)
        validate_capture_capability_binding(spec.profile, spec.capability_digest)
        return spec
    except Exception:
        raise InstallationError() from None


def _keyed_digest(key: InstallationKey, payload: bytes, domain: bytes) -> str:
    if type(key) is not InstallationKey or type(payload) is not bytes:
        raise InstallationError()
    try:
        return key._hmac_sha256(payload, domain=domain)
    except Exception:
        raise InstallationError() from None


def _project_digest(spec: ProviderInstallationSpec, key: InstallationKey) -> str:
    try:
        return CaptureDigestContext(key).workspace_identity(os.fsencode(spec.project_root))
    except Exception:
        raise InstallationError() from None


def _connection_id(spec: ProviderInstallationSpec, key: InstallationKey) -> str:
    material = canonical_json(
        {
            "schema_version": "integration-connection-id/v1",
            "provider_id": spec.provider_id,
            "project_digest": _project_digest(spec, key),
            "generation": spec.generation,
        }
    )
    return f"sg-{_keyed_digest(key, material, _CONNECTION_DOMAIN)[:48]}"


def derive_installation_identity(
    spec: ProviderInstallationSpec,
    installation_key: InstallationKey,
) -> InstallationIdentity:
    """Derive the exact IDs to register in the authenticated capture store."""

    checked = _validated_spec(spec)
    if type(installation_key) is not InstallationKey:
        raise InstallationError()
    return InstallationIdentity(
        project_digest=_project_digest(checked, installation_key),
        connection_id=_connection_id(checked, installation_key),
    )


def _model_body(value: BaseModel, mac_field: str) -> bytes:
    return canonical_json(value.model_dump(mode="json", exclude={mac_field}, warnings="error"))


def _seal_receipt(receipt: InstallationReceipt, key: InstallationKey) -> InstallationReceipt:
    try:
        payload = receipt.model_dump(mode="python", warnings="error")
        payload["receipt_mac"] = "0" * 64
        unsigned = InstallationReceipt.model_validate(payload)
        mac = _keyed_digest(key, _model_body(unsigned, "receipt_mac"), _RECEIPT_DOMAIN)
        payload["receipt_mac"] = mac
        return InstallationReceipt.model_validate(payload)
    except InstallationError:
        raise
    except Exception:
        raise InstallationError() from None


def _verify_receipt(receipt: InstallationReceipt, key: InstallationKey) -> InstallationReceipt:
    try:
        payload = receipt.model_dump(mode="python", warnings="error")
        checked = InstallationReceipt.model_validate(payload)
        expected = _keyed_digest(key, _model_body(checked, "receipt_mac"), _RECEIPT_DOMAIN)
        if not hmac.compare_digest(checked.receipt_mac, expected):
            raise InstallationError()
        return checked
    except InstallationError:
        raise
    except Exception:
        raise InstallationError() from None


def _seal_journal(journal: InstallationJournal, key: InstallationKey) -> InstallationJournal:
    try:
        payload = journal.model_dump(mode="python", warnings="error")
        payload["journal_mac"] = "0" * 64
        unsigned = InstallationJournal.model_validate(payload)
        mac = _keyed_digest(key, _model_body(unsigned, "journal_mac"), _JOURNAL_DOMAIN)
        payload["journal_mac"] = mac
        return InstallationJournal.model_validate(payload)
    except InstallationError:
        raise
    except Exception:
        raise InstallationError() from None


def _verify_journal(journal: InstallationJournal, key: InstallationKey) -> InstallationJournal:
    try:
        payload = journal.model_dump(mode="python", warnings="error")
        checked = InstallationJournal.model_validate(payload)
        expected = _keyed_digest(key, _model_body(checked, "journal_mac"), _JOURNAL_DOMAIN)
        if not hmac.compare_digest(checked.journal_mac, expected):
            raise InstallationError()
        _verify_receipt(checked.target_receipt, key)
        return checked
    except InstallationError:
        raise
    except Exception:
        raise InstallationError() from None


def _encode_model(value: BaseModel, maximum: int) -> bytes:
    try:
        data = canonical_json(value.model_dump(mode="json", warnings="error"))
        if not 2 <= len(data) <= maximum:
            raise InstallationError()
        return data
    except InstallationError:
        raise
    except Exception:
        raise InstallationError() from None


def _decode_receipt(data: bytes, key: InstallationKey) -> InstallationReceipt:
    try:
        if type(data) is not bytes or not 2 <= len(data) <= MAX_INSTALLATION_RECEIPT_BYTES:
            raise InstallationError()
        value = InstallationReceipt.model_validate_json(data)
        if _encode_model(value, MAX_INSTALLATION_RECEIPT_BYTES) != data:
            raise InstallationError()
        return _verify_receipt(value, key)
    except InstallationError:
        raise
    except Exception:
        raise InstallationError() from None


def _decode_journal(data: bytes, key: InstallationKey) -> InstallationJournal:
    try:
        if type(data) is not bytes or not 2 <= len(data) <= MAX_INSTALLATION_JOURNAL_BYTES:
            raise InstallationError()
        value = InstallationJournal.model_validate_json(data)
        if _encode_model(value, MAX_INSTALLATION_JOURNAL_BYTES) != data:
            raise InstallationError()
        return _verify_journal(value, key)
    except InstallationError:
        raise
    except Exception:
        raise InstallationError() from None


def _windows_path_is_stably_absent(
    path: PureWindowsPath,
    operations: WindowsSecurityOperations,
) -> bool:
    """Authenticate an existing prefix and every absent component twice."""

    if operations.inspect_path(path) is not None:
        return False
    missing = [path]
    existing_parent = path.parent
    while operations.inspect_path(existing_parent) is None:
        missing.append(existing_parent)
        next_parent = existing_parent.parent
        if next_parent == existing_parent:
            raise InstallationError()
        existing_parent = next_parent
    boundary = authorize_windows_managed_path(
        existing_parent,
        kind=WindowsPathKind.DIRECTORY,
        operations=operations,
    )
    for _attempt in range(2):
        boundary.revalidate()
        if any(operations.inspect_path(item) is not None for item in missing):
            raise InstallationError()
    boundary.revalidate()
    return True


def _read_private_optional(path: Path, *, maximum: int) -> bytes | None:
    if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
        try:
            operations = NativeWindowsSecurityOperations()
            windows_path = PureWindowsPath(os.fspath(path))
            if _windows_path_is_stably_absent(windows_path, operations):
                return None
            parent = authorize_windows_managed_path(
                windows_path.parent,
                kind=WindowsPathKind.DIRECTORY,
                operations=operations,
            )
            stable = operations.read_private_file(
                windows_path,
                maximum_bytes=maximum,
            )
            parent.revalidate()
            stable.authorization.revalidate()
            return stable.data
        except InstallationError:
            raise
        except Exception:
            raise InstallationError() from None
    try:
        return read_stable_file(
            path,
            maximum_bytes=maximum,
            policy=StableReadPolicy.PRIVATE_EXACT,
        ).data
    except Exception:
        try:
            path.lstat()
        except FileNotFoundError:
            return None
        except Exception:
            pass
        raise InstallationError() from None


def _load_receipt_optional(
    spec: ProviderInstallationSpec, key: InstallationKey
) -> InstallationReceipt | None:
    if not spec.receipt_path.parent.exists():
        return None
    data = _read_private_optional(spec.receipt_path, maximum=MAX_INSTALLATION_RECEIPT_BYTES)
    return None if data is None else _decode_receipt(data, key)


def inspect_installation_receipt(
    path: Path,
    installation_key: InstallationKey,
) -> InstallationReceipt:
    """Read one exact authenticated receipt without searching provider state."""

    try:
        checked_path = _absolute_path(path)
        if type(installation_key) is not InstallationKey:
            raise InstallationError()
        data = _read_private_optional(
            checked_path,
            maximum=MAX_INSTALLATION_RECEIPT_BYTES,
        )
        if data is None:
            raise InstallationError()
        receipt = _decode_receipt(data, installation_key)
        if receipt.receipt_path != checked_path:
            raise InstallationError()
        return receipt
    except InstallationError:
        raise
    except Exception:
        raise InstallationError() from None


def _load_journal_optional(
    spec: ProviderInstallationSpec, key: InstallationKey
) -> InstallationJournal | None:
    if not spec.journal_path.parent.exists():
        return None
    data = _read_private_optional(spec.journal_path, maximum=MAX_INSTALLATION_JOURNAL_BYTES)
    return None if data is None else _decode_journal(data, key)


def _safe_windows_directory_security(
    security: WindowsPathSecurity | None,
    *,
    owner_sid: str,
    private: bool,
) -> bool:
    return (
        type(security) is WindowsPathSecurity
        and security.kind is WindowsPathKind.DIRECTORY
        and security.owner_sid == owner_sid
        and security.hardlink_count == 1
        and security.reparse_tag is None
        and (security.owner_private_dacl if private else security.owner_write_protected_dacl)
    )


def _safe_directory(path: Path, *, private: bool) -> bool:
    if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
        try:
            operations = NativeWindowsSecurityOperations()
            return _safe_windows_directory_security(
                operations.inspect_path(PureWindowsPath(os.fspath(path))),
                owner_sid=operations.current_user_sid(),
                private=private,
            )
        except Exception:
            return False
    try:
        metadata = path.lstat()
    except OSError:
        return False
    mode = stat.S_IMODE(metadata.st_mode)
    return (
        stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and (os.name != "posix" or not hasattr(os, "getuid") or metadata.st_uid == os.getuid())
        and not mode & 0o022
        and (not private or mode == 0o700)
    )


def _ensure_directory(path: Path, *, private: bool) -> None:
    if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
        operations = NativeWindowsSecurityOperations()
        windows_path = PureWindowsPath(os.fspath(path))
        existing = operations.inspect_path(windows_path)
        if existing is not None:
            authorization = (
                authorize_windows_private_path(
                    windows_path,
                    kind=WindowsPathKind.DIRECTORY,
                    operations=operations,
                    create=False,
                )
                if private
                else authorize_windows_managed_path(
                    windows_path,
                    kind=WindowsPathKind.DIRECTORY,
                    operations=operations,
                )
            )
            authorization.revalidate()
            return
        created = ensure_windows_private_directory(
            windows_path,
            operations=operations,
        )
        created.revalidate()
        return
    if path.exists():
        if not _safe_directory(path, private=private):
            raise InstallationError()
        return
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        parent = current.parent
        if parent == current:
            raise InstallationError()
        current = parent
    if not _safe_directory(current, private=False):
        raise InstallationError()
    for item in reversed(missing):
        with suppress(FileExistsError):
            item.mkdir(mode=0o700)
        if not _safe_directory(item, private=True):
            raise InstallationError()


def ensure_private_installation_directory(path: Path) -> None:
    """Create or authorize one exact owner-private integration state directory."""

    try:
        if (
            not isinstance(path, Path)
            or not path.is_absolute()
            or ".." in path.parts
            or not path.name
        ):
            raise InstallationError()
        if os.name == "posix":
            ensure_private_directory(path)
        else:
            _ensure_directory(path, private=True)
    except InstallationError:
        raise
    except Exception:
        raise InstallationError() from None


def _read_launcher_digest(path: Path) -> str:
    if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
        data = _read_private_optional(
            path,
            maximum=MAX_INTEGRATION_LAUNCHER_BYTES,
        )
        if data is None:
            raise InstallationError()
        return hashlib.sha256(data).hexdigest()
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > MAX_INTEGRATION_LAUNCHER_BYTES
            or (
                os.name == "posix"
                and (
                    stat.S_IMODE(metadata.st_mode) != 0o700
                    or (hasattr(os, "getuid") and metadata.st_uid != os.getuid())
                )
            )
            or not os.access(path, os.X_OK)
        ):
            raise InstallationError()
        stable = read_stable_file(
            path,
            maximum_bytes=MAX_INTEGRATION_LAUNCHER_BYTES,
            policy=StableReadPolicy.PRIVATE_EXECUTABLE,
        )
        return hashlib.sha256(stable.data).hexdigest()
    except InstallationError:
        raise
    except Exception:
        raise InstallationError() from None


def _launcher_digest_optional(path: Path) -> str | None:
    if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
        data = _read_private_optional(
            path,
            maximum=MAX_INTEGRATION_LAUNCHER_BYTES,
        )
        return None if data is None else hashlib.sha256(data).hexdigest()
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    except Exception:
        raise InstallationError() from None
    return _read_launcher_digest(path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_launcher(
    path: Path,
    data: bytes,
    *,
    replace_digest: str | None = None,
) -> None:
    """Publish one owner-private executable, replacing only a receipt-bound launcher."""

    temporary: Path | None = None
    try:
        if (
            type(data) is not bytes
            or not 1 <= len(data) <= MAX_INTEGRATION_LAUNCHER_BYTES
            or not _safe_directory(path.parent, private=True)
        ):
            raise InstallationError()
        if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
            operations = NativeWindowsSecurityOperations()
            published = operations.publish_private_file(
                PureWindowsPath(os.fspath(path)),
                data,
                maximum_bytes=MAX_INTEGRATION_LAUNCHER_BYTES,
                validate_replacement=(
                    None
                    if replace_digest is None
                    else lambda current: hmac.compare_digest(
                        hashlib.sha256(current).hexdigest(), replace_digest
                    )
                ),
                validate_published=lambda current: hmac.compare_digest(current, data),
            )
            if not hmac.compare_digest(published.data, data):
                raise InstallationError()
            return
        existing: str | None
        try:
            existing = _read_launcher_digest(path)
        except InstallationError:
            try:
                path.lstat()
            except FileNotFoundError:
                existing = None
            else:
                raise
        if existing is None:
            if replace_digest is not None:
                raise InstallationError()
        elif replace_digest is None or not hmac.compare_digest(existing, replace_digest):
            raise InstallationError()
        temporary = path.with_name(f".{path.name}.{secrets.token_hex(16)}.tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o700)
        try:
            view = memoryview(data)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise InstallationError()
                view = view[written:]
            os.fchmod(descriptor, 0o700)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if existing is None:
            os.link(temporary, path, follow_symlinks=False)
            temporary.unlink()
            temporary = None
        else:
            if _read_launcher_digest(path) != existing:
                raise InstallationError()
            os.replace(temporary, path)
            temporary = None
        _fsync_directory(path.parent)
        if _read_launcher_digest(path) != hashlib.sha256(data).hexdigest():
            raise InstallationError()
    except InstallationError:
        raise
    except Exception:
        raise InstallationError() from None
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink()


def _publish_private(
    path: Path,
    data: bytes,
    *,
    maximum: int,
    replace_digest: str | None = None,
    managed_parent: bool = False,
) -> None:
    try:
        if type(managed_parent) is not bool:
            raise InstallationError()
        if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
            operations = NativeWindowsSecurityOperations()
            publisher = (
                operations.publish_private_file_in_managed_directory
                if managed_parent
                else operations.publish_private_file
            )
            published = publisher(
                PureWindowsPath(os.fspath(path)),
                data,
                maximum_bytes=maximum,
                validate_replacement=(
                    None
                    if replace_digest is None
                    else lambda current: hmac.compare_digest(
                        hashlib.sha256(current).hexdigest(), replace_digest
                    )
                ),
                validate_published=lambda current: hmac.compare_digest(current, data),
            )
            if not hmac.compare_digest(published.data, data):
                raise InstallationError()
            return
        publication = authorize_atomic_file_publication(
            path,
            maximum_bytes=maximum,
            validate_replacement=(
                None
                if replace_digest is None
                else lambda current: hmac.compare_digest(
                    hashlib.sha256(current).hexdigest(), replace_digest
                )
            ),
        )
        stable = publication.publish(data, validate_published=lambda current: current == data)
        if stable.data != data:
            raise InstallationError()
    except InstallationError:
        raise
    except Exception:
        raise InstallationError() from None


def _delete_private_exact(
    path: Path,
    expected_digest: str,
    *,
    maximum: int,
    policy: StableReadPolicy = StableReadPolicy.PRIVATE_EXACT,
) -> None:
    try:
        if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
            operations = NativeWindowsSecurityOperations()
            windows_path = PureWindowsPath(os.fspath(path))
            parent = authorize_windows_managed_path(
                windows_path.parent,
                kind=WindowsPathKind.DIRECTORY,
                operations=operations,
            )
            windows_stable = operations.read_private_file(
                windows_path,
                maximum_bytes=maximum,
            )
            if not hmac.compare_digest(
                hashlib.sha256(windows_stable.data).hexdigest(),
                expected_digest,
            ):
                raise InstallationError()
            parent.revalidate()
            operations.delete_authorized_file(windows_stable.authorization)
            parent.revalidate()
            if operations.inspect_path(windows_path) is not None:
                raise InstallationError()
            return
        stable = read_stable_file(
            path,
            maximum_bytes=maximum,
            policy=policy,
        )
        if not hmac.compare_digest(hashlib.sha256(stable.data).hexdigest(), expected_digest):
            raise InstallationError()
        delete_authorized_private_file(stable.authorization)
    except InstallationError:
        raise
    except Exception:
        raise InstallationError() from None


def _delete_private_if_present(
    path: Path,
    expected_digest: str,
    *,
    maximum: int,
    policy: StableReadPolicy = StableReadPolicy.PRIVATE_EXACT,
) -> None:
    if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
        try:
            operations = NativeWindowsSecurityOperations()
            windows_path = PureWindowsPath(os.fspath(path))
            parent = authorize_windows_managed_path(
                windows_path.parent,
                kind=WindowsPathKind.DIRECTORY,
                operations=operations,
            )
            if operations.inspect_path(windows_path) is None:
                parent.revalidate()
                if operations.inspect_path(windows_path) is not None:
                    raise InstallationError()
                return
        except InstallationError:
            raise
        except Exception:
            raise InstallationError() from None
        _delete_private_exact(path, expected_digest, maximum=maximum, policy=policy)
        return
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except Exception:
        raise InstallationError() from None
    _delete_private_exact(path, expected_digest, maximum=maximum, policy=policy)


def _publish_receipt(
    spec: ProviderInstallationSpec,
    receipt: InstallationReceipt,
    *,
    replace_data: bytes | None,
) -> bytes:
    data = _encode_model(receipt, MAX_INSTALLATION_RECEIPT_BYTES)
    _publish_private(
        spec.receipt_path,
        data,
        maximum=MAX_INSTALLATION_RECEIPT_BYTES,
        replace_digest=None if replace_data is None else hashlib.sha256(replace_data).hexdigest(),
    )
    return data


def _publish_journal(spec: ProviderInstallationSpec, journal: InstallationJournal) -> bytes:
    data = _encode_model(journal, MAX_INSTALLATION_JOURNAL_BYTES)
    _publish_private(spec.journal_path, data, maximum=MAX_INSTALLATION_JOURNAL_BYTES)
    return data


def _delete_journal(spec: ProviderInstallationSpec, data: bytes) -> None:
    _delete_private_exact(
        spec.journal_path,
        hashlib.sha256(data).hexdigest(),
        maximum=MAX_INSTALLATION_JOURNAL_BYTES,
    )


def _safe_lock_metadata(metadata: os.stat_result) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_nlink == 1
        and stat.S_IMODE(metadata.st_mode) == 0o600
        and (not hasattr(os, "getuid") or metadata.st_uid == os.getuid())
    )


@contextmanager
def _installation_lock(spec: ProviderInstallationSpec) -> Iterator[None]:
    if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
        try:
            operations = NativeWindowsSecurityOperations()
            windows_boundary = authorize_windows_private_path(
                PureWindowsPath(os.fspath(spec.lock_path.parent)),
                kind=WindowsPathKind.DIRECTORY,
                operations=operations,
                create=False,
            )
            windows_boundary.revalidate()
            with operations.private_file_lock(
                PureWindowsPath(os.fspath(spec.lock_path)),
            ):
                windows_boundary.revalidate()
                yield
                windows_boundary.revalidate()
            return
        except InstallationError:
            raise
        except BaseException:
            raise
    try:
        posix_boundary = security_files._authorize_private_directory(
            spec.lock_path.parent,
            create=False,
        )
        directory_fd = security_files._open_authorized_private_directory(posix_boundary)
    except Exception:
        raise InstallationError() from None
    descriptor: int | None = None
    operation_failure: BaseException | None = None
    try:
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(spec.lock_path.name, flags, 0o600, dir_fd=directory_fd)
        opened = os.fstat(descriptor)
        named = os.stat(spec.lock_path.name, dir_fd=directory_fd, follow_symlinks=False)
        security_files._require_safe_acl(descriptor)
        if (
            not _safe_lock_metadata(opened)
            or not _safe_lock_metadata(named)
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise InstallationError()
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_EX)
        opened_locked = os.fstat(descriptor)
        named_locked = os.stat(spec.lock_path.name, dir_fd=directory_fd, follow_symlinks=False)
        security_files._require_safe_acl(descriptor)
        if (
            not _safe_lock_metadata(opened_locked)
            or not _safe_lock_metadata(named_locked)
            or (opened_locked.st_dev, opened_locked.st_ino)
            != (named_locked.st_dev, named_locked.st_ino)
        ):
            raise InstallationError()
        security_files._require_authorized_private_directory_descriptor(
            posix_boundary,
            directory_fd,
        )
        posix_boundary.revalidate()
        yield
        opened_after = os.fstat(descriptor)
        named_after = os.stat(spec.lock_path.name, dir_fd=directory_fd, follow_symlinks=False)
        security_files._require_safe_acl(descriptor)
        if (
            not _safe_lock_metadata(opened_after)
            or not _safe_lock_metadata(named_after)
            or (opened_after.st_dev, opened_after.st_ino)
            != (named_after.st_dev, named_after.st_ino)
        ):
            raise InstallationError()
        security_files._require_authorized_private_directory_descriptor(
            posix_boundary,
            directory_fd,
        )
        posix_boundary.revalidate()
    except BaseException as error:
        operation_failure = error
    finally:
        cleanup_failure: BaseException | None = None
        if descriptor is not None:
            try:
                with suppress(OSError):
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            except BaseException as error:
                cleanup_failure = error
        close_failure = security_files._close_independent_descriptors(
            descriptor,
            directory_fd,
        )
        cleanup_failure = security_files._preferred_failure(cleanup_failure, close_failure)
        operation_failure = security_files._preferred_failure(
            operation_failure,
            cleanup_failure,
        )
    if operation_failure is not None:
        raise operation_failure


def _fault(callback: Callable[[str], None] | None, stage: str) -> None:
    if callback is not None:
        callback(stage)


def _spec_bridge_assets(
    spec: ProviderInstallationSpec,
) -> tuple[Path, Path, bytes, str] | None:
    if spec.installation_kind is ProviderInstallationKind.COMMAND_HOOK:
        if any(
            value is not None
            for value in (
                spec.bundle_path,
                spec.bootstrap_path,
                spec.bundle_bytes,
                spec.bundle_digest,
            )
        ):
            raise InstallationError()
        return None
    if (
        spec.bundle_path is None
        or spec.bootstrap_path is None
        or spec.bundle_bytes is None
        or spec.bundle_digest is None
    ):
        raise InstallationError()
    return (
        spec.bundle_path,
        spec.bootstrap_path,
        spec.bundle_bytes,
        spec.bundle_digest,
    )


def _receipt_bridge_assets(
    receipt: InstallationReceipt,
) -> tuple[Path, Path, str] | None:
    if receipt.installation_kind is ProviderInstallationKind.COMMAND_HOOK:
        if any(
            value is not None
            for value in (
                receipt.bundle_path,
                receipt.bootstrap_path,
                receipt.bundle_digest,
            )
        ):
            raise InstallationError()
        return None
    if (
        receipt.bundle_path is None
        or receipt.bootstrap_path is None
        or receipt.bundle_digest is None
    ):
        raise InstallationError()
    return receipt.bundle_path, receipt.bootstrap_path, receipt.bundle_digest


def _make_receipt(
    spec: ProviderInstallationSpec,
    key: InstallationKey,
    *,
    state: InstallationState,
    launcher_digest: str,
    config_edit: OwnedConfigReverseEdit | None,
) -> InstallationReceipt:
    return _seal_receipt(
        InstallationReceipt(
            state=state,
            provider_id=spec.provider_id,
            profile=spec.profile,
            host_version=spec.host_version,
            generation=spec.generation,
            connection_id=_connection_id(spec, key),
            project_digest=_project_digest(spec, key),
            capability_digest=spec.capability_digest,
            config_path=spec.config_path,
            bundle_path=spec.bundle_path,
            bootstrap_path=spec.bootstrap_path,
            launcher_path=spec.launcher_path,
            receipt_path=spec.receipt_path,
            journal_path=spec.journal_path,
            lock_path=spec.lock_path,
            bundle_digest=spec.bundle_digest,
            launcher_digest=launcher_digest,
            config_edit=config_edit,
            receipt_mac="0" * 64,
        ),
        key,
    )


def _receipt_with_state(
    receipt: InstallationReceipt,
    state: InstallationState,
    key: InstallationKey,
) -> InstallationReceipt:
    payload = receipt.model_dump(mode="python", warnings="error")
    payload["state"] = state
    payload["receipt_mac"] = "0" * 64
    return _seal_receipt(InstallationReceipt.model_validate(payload), key)


def _bootstrap_for(receipt: InstallationReceipt) -> IntegrationBootstrap:
    bridge = _receipt_bridge_assets(receipt)
    if bridge is None:
        raise InstallationError()
    _bundle_path, _bootstrap_path, bundle_digest = bridge
    return IntegrationBootstrap(
        profile=receipt.profile,
        connection_id=receipt.connection_id,
        launcher_path=receipt.launcher_path,
        capability_digest=receipt.capability_digest,
        bundle_digest=bundle_digest,
        receipt_mac=receipt.receipt_mac,
    )


def _receipt_matches_spec(
    receipt: InstallationReceipt,
    spec: ProviderInstallationSpec,
    key: InstallationKey,
    *,
    generation: bool = True,
) -> bool:
    receipt_bridge = _receipt_bridge_assets(receipt)
    spec_bridge = _spec_bridge_assets(spec)
    bridge_matches = (
        receipt_bridge is None
        if spec_bridge is None
        else receipt_bridge is not None
        and receipt_bridge[0].parent == spec_bridge[0].parent
        and receipt_bridge[1] == spec_bridge[1]
    )
    return (
        receipt.installation_kind is spec.installation_kind
        and receipt.provider_id == spec.provider_id
        and receipt.profile is spec.profile
        and (receipt.host_version == spec.host_version or not generation)
        and (not generation or receipt.generation == spec.generation)
        and receipt.project_digest == _project_digest(spec, key)
        and receipt.capability_digest == spec.capability_digest
        and receipt.config_path == spec.config_path
        and bridge_matches
        and receipt.launcher_path == spec.launcher_path
        and receipt.receipt_path == spec.receipt_path
        and receipt.journal_path == spec.journal_path
        and receipt.lock_path == spec.lock_path
    )


def _path_digest(path: Path, *, maximum: int) -> str | None:
    try:
        data = _read_private_optional(path, maximum=maximum)
    except InstallationError:
        return "invalid"
    return None if data is None else hashlib.sha256(data).hexdigest()


def _require_private_operational_directory(spec: ProviderInstallationSpec) -> bool:
    """Return whether the operational directory exists, rejecting unsafe boundaries."""

    if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
        try:
            operations = NativeWindowsSecurityOperations()
            windows_path = PureWindowsPath(os.fspath(spec.receipt_path.parent))
            if _windows_path_is_stably_absent(windows_path, operations):
                return False
            authorization = authorize_windows_private_path(
                windows_path,
                kind=WindowsPathKind.DIRECTORY,
                operations=operations,
                create=False,
            )
            authorization.revalidate()
            return True
        except InstallationError:
            raise
        except Exception:
            raise InstallationError() from None
    try:
        return inspect_private_directory(spec.receipt_path.parent)
    except Exception:
        raise InstallationError() from None


def _lock_digest(spec: ProviderInstallationSpec) -> str | None:
    return _path_digest(spec.lock_path, maximum=1)


def _status(
    spec: ProviderInstallationSpec,
    key: InstallationKey,
    *,
    disposition: InstallationDisposition,
    state: InstallationState,
    drift: tuple[str, ...] = (),
    would_write: tuple[Path, ...] = (),
    tracked: tuple[Path, ...] = (),
    connection_id: str | None = None,
) -> InstallationStatus:
    canonical_drift = tuple(item for item in _DRIFT_ORDER if item in set(drift))
    return InstallationStatus(
        disposition=disposition,
        state=state,
        provider_id=spec.provider_id,
        profile=spec.profile,
        generation=spec.generation,
        connection_id=_connection_id(spec, key) if connection_id is None else connection_id,
        project_digest=_project_digest(spec, key),
        installed=state is InstallationState.ENABLED and not canonical_drift,
        drift=canonical_drift,
        would_write=would_write,
        git_tracked_files=tracked,
    )


def _config_edit_matches_spec(
    edit: OwnedConfigReverseEdit | None,
    spec: ProviderInstallationSpec,
) -> bool:
    if edit is None or spec.config is None:
        return edit is None and spec.config is None and spec.config_path is None
    return spec.config_path is not None and _owned_config_edit_matches_spec(edit, spec.config)


def inspect_provider_installation(
    spec: ProviderInstallationSpec,
    installation_key: InstallationKey,
) -> InstallationStatus:
    """Inspect journal, receipt, config, assets, and launcher without repairing them."""

    checked = _validated_spec(spec)
    key = installation_key
    if type(key) is not InstallationKey:
        raise InstallationError()
    try:
        operational_exists = _require_private_operational_directory(checked)
    except InstallationError:
        return _status(
            checked,
            key,
            disposition=InstallationDisposition.NOOP,
            state=InstallationState.DISABLED,
            drift=("receipt",),
        )
    journal = _load_journal_optional(checked, key)
    observed_lock_digest = _lock_digest(checked) if operational_exists else None
    lifecycle_state = (
        None
        if journal is None
        else (
            InstallationState.PENDING
            if journal.operation is _JournalOperation.INSTALL
            else InstallationState.DRAINING
        )
    )
    try:
        receipt = _load_receipt_optional(checked, key)
    except InstallationError:
        receipt_drift = ["receipt"]
        if observed_lock_digest in (None, "invalid"):
            receipt_drift.append("lock")
        return _status(
            checked,
            key,
            disposition=InstallationDisposition.NOOP,
            state=(InstallationState.DISABLED if lifecycle_state is None else lifecycle_state),
            drift=tuple(receipt_drift),
        )
    if receipt is None:
        orphan_drift: list[str] = []
        if observed_lock_digest == "invalid" or (
            journal is not None and observed_lock_digest is None
        ):
            orphan_drift.append("lock")
        if checked.config is not None:
            try:
                orphan_config = _current_config_for_plan(checked)
                if (
                    orphan_config is not None
                    and checked.config.marker.encode("ascii") in orphan_config
                ):
                    orphan_drift.append("config")
            except (ConfigFileError, InstallationError):
                orphan_drift.append("config")
        checked_bridge = _spec_bridge_assets(checked)
        if checked_bridge is not None:
            bundle_path, bootstrap_path, _bundle_bytes, _bundle_digest = checked_bridge
            if _path_digest(bundle_path, maximum=2 * 1_024 * 1_024) is not None:
                orphan_drift.append("bundle")
            if _path_digest(bootstrap_path, maximum=16 * 1_024) is not None:
                orphan_drift.append("bootstrap")
        try:
            orphan_launcher = _launcher_digest_optional(checked.launcher_path)
        except InstallationError:
            orphan_drift.append("launcher")
        else:
            if orphan_launcher is not None:
                orphan_drift.append("launcher")
        if journal is not None:
            orphan_drift.append("receipt")
        return _status(
            checked,
            key,
            disposition=InstallationDisposition.NOOP,
            state=(InstallationState.DISABLED if lifecycle_state is None else lifecycle_state),
            drift=tuple(orphan_drift),
        )
    drift: list[str] = []
    if journal is not None:
        drift.append("receipt")
    if observed_lock_digest in (None, "invalid"):
        drift.append("lock")
    if not _receipt_matches_spec(receipt, checked, key):
        drift.append("receipt")
    config_path = receipt.config_path
    config_edit = receipt.config_edit
    if (config_path is None) != (config_edit is None):
        raise InstallationError()
    current_config: bytes | None = None
    marker: bytes | None = None
    if config_path is not None and config_edit is not None:
        try:
            current_config = read_config_bytes(config_path)
        except ConfigFileError:
            drift.append("config")
        marker = config_edit.marker.encode("ascii")
    checked_bridge = _spec_bridge_assets(checked)
    receipt_bridge = _receipt_bridge_assets(receipt)
    if receipt.state is InstallationState.DISABLED:
        if current_config is not None and marker is not None and marker in current_config:
            drift.append("config")
        if checked_bridge is not None and receipt_bridge is not None:
            bundle_path, bootstrap_path, bundle_bytes, _bundle_digest = checked_bridge
            if _path_digest(bundle_path, maximum=len(bundle_bytes) + 1) is not None:
                drift.append("bundle")
            if _path_digest(bootstrap_path, maximum=16 * 1_024) is not None:
                drift.append("bootstrap")
        try:
            receipt.launcher_path.lstat()
        except FileNotFoundError:
            pass
        except Exception:
            drift.append("launcher")
        else:
            drift.append("launcher")
    else:
        if not _config_edit_matches_spec(receipt.config_edit, checked):
            drift.append("config")
        if receipt.launcher_digest != checked.launcher_digest:
            drift.append("launcher")
        if checked_bridge is not None and receipt_bridge is not None:
            checked_bundle, _checked_bootstrap, _bundle_bytes, checked_digest = checked_bridge
            receipt_bundle, receipt_bootstrap, receipt_digest = receipt_bridge
            if receipt_bundle != checked_bundle or receipt_digest != checked_digest:
                drift.append("bundle")
            if _path_digest(receipt_bundle, maximum=2 * 1_024 * 1_024) != receipt_digest:
                drift.append("bundle")
            try:
                bootstrap = inspect_integration_bootstrap(receipt_bootstrap)
                expected_bootstrap = _bootstrap_for(receipt)
                if bootstrap != expected_bootstrap:
                    drift.append("bootstrap")
            except Exception:
                drift.append("bootstrap")
        if config_edit is not None and (
            current_config is None
            or hashlib.sha256(current_config).hexdigest() != config_edit.installed_digest
        ):
            drift.append("config")
        try:
            if _read_launcher_digest(receipt.launcher_path) != receipt.launcher_digest:
                drift.append("launcher")
        except InstallationError:
            drift.append("launcher")
    return _status(
        checked,
        key,
        disposition=InstallationDisposition.NOOP,
        state=receipt.state if lifecycle_state is None else lifecycle_state,
        drift=tuple(drift),
        connection_id=receipt.connection_id if journal is None else None,
    )


def _current_config_for_plan(spec: ProviderInstallationSpec) -> bytes | None:
    if spec.config_path is None:
        if spec.config is not None:
            raise InstallationError()
        return None
    try:
        parent = spec.config_path.parent.lstat()
    except FileNotFoundError:
        return None
    except OSError:
        raise InstallationError() from None
    if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
        raise InstallationError()
    return read_config_bytes(spec.config_path)


def _journal_for_install(
    target: InstallationReceipt,
    key: InstallationKey,
    *,
    prior: InstallationReceipt | None,
) -> InstallationJournal:
    target_bridge = _receipt_bridge_assets(target)
    target_bootstrap_digest = (
        None
        if target_bridge is None
        else hashlib.sha256(encode_integration_bootstrap(_bootstrap_for(target))).hexdigest()
    )
    prior_enabled = prior is not None and prior.state is InstallationState.ENABLED
    prior_bridge = None if prior is None else _receipt_bridge_assets(prior)
    prior_bootstrap = (
        None
        if not prior_enabled or prior is None or prior_bridge is None
        else hashlib.sha256(encode_integration_bootstrap(_bootstrap_for(prior))).hexdigest()
    )
    return _seal_journal(
        InstallationJournal(
            operation=_JournalOperation.INSTALL,
            target_receipt=target,
            target_bootstrap_digest=target_bootstrap_digest,
            prior_receipt_mac=None if prior is None else prior.receipt_mac,
            prior_bundle_path=(
                None if not prior_enabled or prior_bridge is None else prior_bridge[0]
            ),
            prior_bundle_digest=(
                None if not prior_enabled or prior_bridge is None else prior_bridge[2]
            ),
            prior_launcher_path=None if not prior_enabled or prior is None else prior.launcher_path,
            prior_launcher_digest=(
                None if not prior_enabled or prior is None else prior.launcher_digest
            ),
            prior_bootstrap_digest=prior_bootstrap,
            journal_mac="0" * 64,
        ),
        key,
    )


def _managed_directory_identity(path: Path) -> object:
    if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
        operations = NativeWindowsSecurityOperations()
        authorization = authorize_windows_managed_path(
            PureWindowsPath(os.fspath(path)),
            kind=WindowsPathKind.DIRECTORY,
            operations=operations,
        )
        authorization.revalidate()
        return authorization.component_security
    metadata = path.lstat()
    mode = stat.S_IMODE(metadata.st_mode)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or (hasattr(os, "getuid") and metadata.st_uid != os.getuid())
        or mode & 0o022
    ):
        raise InstallationError()
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        mode,
    )


@dataclass(frozen=True, slots=True, repr=False)
class _ManagedProjectBoundary:
    identities: tuple[tuple[Path, object], ...]

    @classmethod
    def capture(
        cls,
        spec: ProviderInstallationSpec,
        *,
        expected_project_identity: object | None = None,
    ) -> _ManagedProjectBoundary:
        paths = tuple(
            dict.fromkeys((spec.project_root, *(path.parent for path in spec.project_local_paths)))
        )
        result = cls(identities=tuple((path, _managed_directory_identity(path)) for path in paths))
        if (
            expected_project_identity is not None
            and result.identities[0][1] != expected_project_identity
        ):
            raise InstallationError()
        result.revalidate()
        return result

    def revalidate(self) -> None:
        if not self.identities:
            raise InstallationError()
        for path, expected in (*self.identities, self.identities[0]):
            if _managed_directory_identity(path) != expected:
                raise InstallationError()

    def __repr__(self) -> str:
        return "_ManagedProjectBoundary(<redacted>)"


def _safe_posix_project_directory(metadata: os.stat_result, *, private: bool) -> bool:
    mode = stat.S_IMODE(metadata.st_mode)
    return (
        stat.S_ISDIR(metadata.st_mode)
        and metadata.st_uid == os.getuid()
        and not mode & 0o022
        and (not private or mode == 0o700)
    )


def _posix_project_directory_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        stat.S_IMODE(metadata.st_mode),
    )


def _ensure_posix_project_directory(
    project_root: Path,
    path: Path,
    *,
    expected_project_identity: object,
) -> None:
    try:
        relative = path.relative_to(project_root)
        if any(component in ("", ".", "..") for component in relative.parts):
            raise InstallationError()
        if type(expected_project_identity) is not tuple or len(expected_project_identity) != 4:
            raise InstallationError()
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
        descriptor = os.open(project_root, flags)
        try:
            root_metadata = os.fstat(descriptor)
            if _posix_project_directory_identity(
                root_metadata
            ) != expected_project_identity or not _safe_posix_project_directory(
                root_metadata, private=False
            ):
                raise InstallationError()
            final_identity = _posix_project_directory_identity(root_metadata)
            for component in relative.parts:
                created = False
                try:
                    child = os.open(component, flags, dir_fd=descriptor)
                except FileNotFoundError:
                    try:
                        os.mkdir(component, 0o700, dir_fd=descriptor)
                        created = True
                    except FileExistsError:
                        pass
                    child = os.open(component, flags, dir_fd=descriptor)
                try:
                    metadata = os.fstat(child)
                    if not _safe_posix_project_directory(metadata, private=created):
                        raise InstallationError()
                    final_identity = _posix_project_directory_identity(metadata)
                except BaseException:
                    os.close(child)
                    raise
                os.close(descriptor)
                descriptor = child
            if (
                _managed_directory_identity(project_root) != expected_project_identity
                or _managed_directory_identity(path) != final_identity
            ):
                raise InstallationError()
        finally:
            os.close(descriptor)
    except InstallationError:
        raise
    except Exception:
        raise InstallationError() from None


def _ensure_project_directory(
    spec: ProviderInstallationSpec,
    path: Path,
    *,
    expected_project_identity: object,
) -> None:
    if os.name == "posix":
        _ensure_posix_project_directory(
            spec.project_root,
            path,
            expected_project_identity=expected_project_identity,
        )
        return
    if _managed_directory_identity(spec.project_root) != expected_project_identity:
        raise InstallationError()
    _ensure_directory(path, private=False)
    if _managed_directory_identity(spec.project_root) != expected_project_identity:
        raise InstallationError()


def _ensure_install_directories(
    spec: ProviderInstallationSpec,
    *,
    expected_project_identity: object,
) -> None:
    for path in tuple(
        dict.fromkeys(project_path.parent for project_path in spec.project_local_paths)
    ):
        _ensure_project_directory(
            spec,
            path,
            expected_project_identity=expected_project_identity,
        )
    if _managed_directory_identity(spec.project_root) != expected_project_identity:
        raise InstallationError()


def _install_locked(
    spec: ProviderInstallationSpec,
    key: InstallationKey,
    boundary: _ManagedProjectBoundary,
    *,
    fault_injector: Callable[[str], None] | None,
) -> InstallationStatus:
    boundary.revalidate()
    spec_bridge = _spec_bridge_assets(spec)
    journal = _load_journal_optional(spec, key)
    if journal is not None:
        return _recover_locked(
            spec,
            key,
            boundary,
            journal=journal,
            fault_injector=fault_injector,
        )
    prior = _load_receipt_optional(spec, key)
    current = _current_config_for_plan(spec)
    launcher_digest = spec.launcher_digest
    current_launcher_digest = _launcher_digest_optional(spec.launcher_path)
    upgrade = False
    has_config = spec.config_path is not None
    if has_config != (spec.config is not None):
        raise InstallationError()
    config_write = has_config
    config_edit: OwnedConfigReverseEdit | None = None
    config_plan: OwnedConfigPlan | None = None
    if prior is not None:
        if not _receipt_matches_spec(prior, spec, key, generation=False):
            raise InstallationError()
        prior_bridge = _receipt_bridge_assets(prior)
        if prior.state is InstallationState.ENABLED:
            if current_launcher_digest != prior.launcher_digest:
                raise InstallationError()
            if prior.generation == spec.generation:
                if prior.launcher_digest != spec.launcher_digest or not _config_edit_matches_spec(
                    prior.config_edit, spec
                ):
                    raise InstallationError()
                status = inspect_provider_installation(spec, key)
                if status.drift:
                    raise InstallationError()
                if spec_bridge is not None and (
                    prior_bridge is None
                    or prior_bridge[0] != spec_bridge[0]
                    or prior_bridge[2] != spec_bridge[3]
                ):
                    raise InstallationError()
                boundary.revalidate()
                return status.model_copy(update={"disposition": InstallationDisposition.NOOP})
            if prior.generation >= spec.generation:
                raise InstallationError()
            bridge_invalid = False
            if spec_bridge is not None:
                if prior_bridge is None:
                    raise InstallationError()
                spec_bundle_path, _spec_bootstrap_path, _spec_bundle_bytes, spec_bundle_digest = (
                    spec_bridge
                )
                prior_bundle_path, prior_bootstrap_path, prior_bundle_digest = prior_bridge
                if (
                    prior_bundle_path == spec_bundle_path
                    and prior_bundle_digest != spec_bundle_digest
                ):
                    raise InstallationError()
                bridge_invalid = (
                    _path_digest(prior_bundle_path, maximum=2 * 1_024 * 1_024)
                    != prior_bundle_digest
                    or _path_digest(prior_bootstrap_path, maximum=16 * 1_024)
                    != hashlib.sha256(
                        encode_integration_bootstrap(_bootstrap_for(prior))
                    ).hexdigest()
                )
            config_invalid = not _config_edit_matches_spec(prior.config_edit, spec)
            if prior.config_edit is not None:
                config_invalid = (
                    config_invalid
                    or current is None
                    or (hashlib.sha256(current).hexdigest() != prior.config_edit.installed_digest)
                )
            elif spec.config is not None:
                config_invalid = True
            if config_invalid or bridge_invalid:
                raise InstallationError()
            upgrade = True
            config_write = False
            config_edit = prior.config_edit
        elif prior.state is InstallationState.DISABLED:
            if current_launcher_digest is not None:
                raise InstallationError()
        else:
            raise InstallationError()
    else:
        if current_launcher_digest is not None:
            raise InstallationError()
    if config_write:
        if spec.config is None:
            raise InstallationError()
        config_plan = plan_owned_config_install(current, spec.config)
        config_edit = config_plan.reverse_edit
    target_bundle_digest: str | None = None
    target_bootstrap_digest: str | None = None
    if spec_bridge is not None:
        bundle_path, bootstrap_path, _bundle_bytes, _bundle_digest = spec_bridge
        target_bundle_digest = _path_digest(bundle_path, maximum=2 * 1_024 * 1_024)
        target_bootstrap_digest = _path_digest(bootstrap_path, maximum=16 * 1_024)
        if upgrade:
            if prior is None:
                raise InstallationError()
            prior_bridge = _receipt_bridge_assets(prior)
            if prior_bridge is None:
                raise InstallationError()
            if bundle_path != prior_bridge[0] and target_bundle_digest is not None:
                raise InstallationError()
        elif target_bundle_digest is not None or target_bootstrap_digest is not None:
            raise InstallationError()
    target = _make_receipt(
        spec,
        key,
        state=InstallationState.ENABLED,
        launcher_digest=launcher_digest,
        config_edit=config_edit,
    )
    pending = _receipt_with_state(target, InstallationState.PENDING, key)
    transaction = _journal_for_install(target, key, prior=prior)
    journal_data = _publish_journal(spec, transaction)
    _fault(fault_injector, "after_journal_publish")
    prior_data = None if prior is None else _encode_model(prior, MAX_INSTALLATION_RECEIPT_BYTES)
    pending_data = _publish_receipt(spec, pending, replace_data=prior_data)
    _fault(fault_injector, "after_pending_receipt_publish")
    _publish_launcher(
        spec.launcher_path,
        spec.launcher_bytes,
        replace_digest=(prior.launcher_digest if upgrade and prior is not None else None),
    )
    _fault(fault_injector, "after_launcher_publish")
    if spec_bridge is not None:
        bundle_path, bootstrap_path, bundle_bytes, bundle_digest = spec_bridge
        if target_bundle_digest is None:
            boundary.revalidate()
            _publish_private(
                bundle_path,
                bundle_bytes,
                maximum=2 * 1_024 * 1_024,
                managed_parent=True,
            )
            boundary.revalidate()
        elif target_bundle_digest != bundle_digest:
            raise InstallationError()
        _fault(fault_injector, "after_bundle_publish")
        bootstrap = _bootstrap_for(target)
        boundary.revalidate()
        publish_integration_bootstrap(
            bootstrap_path,
            bootstrap,
            replace_digest=transaction.prior_bootstrap_digest,
        )
        boundary.revalidate()
        _fault(fault_injector, "after_bootstrap_publish")
    if config_write:
        if spec.config_path is None or config_plan is None:
            raise InstallationError()
        boundary.revalidate()
        publish_config_bytes(
            spec.config_path,
            expected=current,
            data=config_plan.installed_bytes,
        )
        boundary.revalidate()
    _fault(fault_injector, "after_config_publish")
    boundary.revalidate()
    _publish_receipt(spec, target, replace_data=pending_data)
    _fault(fault_injector, "after_enabled_receipt_publish")
    if (
        transaction.prior_bundle_path is not None
        and transaction.prior_bundle_digest is not None
        and (spec_bridge is None or transaction.prior_bundle_path != spec_bridge[0])
    ):
        boundary.revalidate()
        _delete_private_if_present(
            transaction.prior_bundle_path,
            transaction.prior_bundle_digest,
            maximum=2 * 1_024 * 1_024,
        )
        boundary.revalidate()
    boundary.revalidate()
    _delete_journal(spec, journal_data)
    return _status(
        spec,
        key,
        disposition=(
            InstallationDisposition.UPGRADED if upgrade else InstallationDisposition.INSTALLED
        ),
        state=InstallationState.ENABLED,
        connection_id=target.connection_id,
    )


def install_provider(
    spec: ProviderInstallationSpec,
    installation_key: InstallationKey,
    *,
    dry_run: bool = False,
    _fault_injector: Callable[[str], None] | None = None,
) -> InstallationStatus:
    """Plan or install one provider integration without touching provider trust settings."""

    checked = _validated_spec(spec)
    if type(installation_key) is not InstallationKey or type(dry_run) is not bool:
        raise InstallationError()
    try:
        operational_exists = _require_private_operational_directory(checked)
        initial_project_identity = _managed_directory_identity(checked.project_root)
        current = _current_config_for_plan(checked)
        prior = _load_receipt_optional(checked, installation_key)
        journal = _load_journal_optional(checked, installation_key)
        observed_lock_digest = _lock_digest(checked) if operational_exists else None
        if observed_lock_digest == "invalid" or (
            (prior is not None or journal is not None) and observed_lock_digest is None
        ):
            raise InstallationError()
        if prior is not None and not _receipt_matches_spec(
            prior,
            checked,
            installation_key,
            generation=False,
        ):
            raise InstallationError()
        if (
            prior is None or prior.state is InstallationState.DISABLED
        ) and checked.config is not None:
            plan_owned_config_install(current, checked.config)
        if dry_run:
            return _status(
                checked,
                installation_key,
                disposition=InstallationDisposition.PLANNED,
                state=InstallationState.PENDING,
                would_write=(
                    checked.launcher_path,
                    *checked.project_local_paths,
                    checked.receipt_path,
                    checked.journal_path,
                ),
                tracked=git_tracked_project_files(checked),
            )
        ensure_private_installation_directory(checked.receipt_path.parent)
        with _installation_lock(checked):
            _ensure_install_directories(
                checked,
                expected_project_identity=initial_project_identity,
            )
            boundary = _ManagedProjectBoundary.capture(
                checked,
                expected_project_identity=initial_project_identity,
            )
            return _install_locked(
                checked,
                installation_key,
                boundary,
                fault_injector=_fault_injector,
            )
    except RuntimeError as error:
        if _fault_injector is not None and not isinstance(error, InstallationError):
            raise
        if isinstance(error, InstallationError):
            raise
        raise InstallationError() from None
    except (ConfigFileError, ValueError, OSError):
        raise InstallationError() from None


def _recover_install(
    spec: ProviderInstallationSpec,
    key: InstallationKey,
    boundary: _ManagedProjectBoundary,
    journal: InstallationJournal,
    journal_data: bytes,
    *,
    fault_injector: Callable[[str], None] | None,
) -> InstallationStatus:
    boundary.revalidate()
    target = journal.target_receipt
    spec_bridge = _spec_bridge_assets(spec)
    target_bridge = _receipt_bridge_assets(target)
    if not _receipt_matches_spec(target, spec, key):
        raise InstallationError()
    if (
        target.launcher_digest != spec.launcher_digest
        or target.launcher_path != spec.launcher_path
        or (
            journal.prior_bundle_path is not None
            and (spec_bridge is None or journal.prior_bundle_path.parent != spec_bridge[0].parent)
        )
    ):
        raise InstallationError()
    if spec_bridge is not None and (
        target_bridge is None
        or target_bridge[0] != spec_bridge[0]
        or target_bridge[2] != spec_bridge[3]
    ):
        raise InstallationError()
    current_receipt = _load_receipt_optional(spec, key)
    current_data = (
        None
        if current_receipt is None
        else _encode_model(current_receipt, MAX_INSTALLATION_RECEIPT_BYTES)
    )
    pending = _receipt_with_state(target, InstallationState.PENDING, key)
    enabled_receipt = False
    if current_receipt is None:
        if journal.prior_receipt_mac is not None:
            raise InstallationError()
        current_data = _publish_receipt(spec, pending, replace_data=None)
    elif current_receipt.receipt_mac == target.receipt_mac:
        enabled_receipt = True
    elif current_receipt.receipt_mac == pending.receipt_mac:
        if current_receipt.state is not InstallationState.PENDING:
            raise InstallationError()
    elif (
        journal.prior_receipt_mac is not None
        and current_receipt.receipt_mac == journal.prior_receipt_mac
        and current_receipt.state in (InstallationState.ENABLED, InstallationState.DISABLED)
        and _receipt_matches_spec(current_receipt, spec, key, generation=False)
        and (
            current_receipt.state is InstallationState.DISABLED
            or current_receipt.generation < target.generation
        )
    ):
        current_data = _publish_receipt(spec, pending, replace_data=current_data)
    else:
        raise InstallationError()
    launcher_digest = _launcher_digest_optional(spec.launcher_path)
    if launcher_digest is None:
        _publish_launcher(spec.launcher_path, spec.launcher_bytes)
    elif launcher_digest == journal.prior_launcher_digest:
        _publish_launcher(
            spec.launcher_path,
            spec.launcher_bytes,
            replace_digest=launcher_digest,
        )
    elif launcher_digest != target.launcher_digest:
        raise InstallationError()
    if spec_bridge is not None and target_bridge is not None:
        bundle_path, bootstrap_path, bundle_bytes, _bundle_digest = spec_bridge
        bundle_digest = _path_digest(bundle_path, maximum=2 * 1_024 * 1_024)
        if bundle_digest is None:
            boundary.revalidate()
            _publish_private(
                bundle_path,
                bundle_bytes,
                maximum=2 * 1_024 * 1_024,
                managed_parent=True,
            )
            boundary.revalidate()
        elif bundle_digest != target_bridge[2]:
            raise InstallationError()
        expected_bootstrap = encode_integration_bootstrap(_bootstrap_for(target))
        bootstrap_digest = _path_digest(bootstrap_path, maximum=16 * 1_024)
        if bootstrap_digest is None or bootstrap_digest == journal.prior_bootstrap_digest:
            boundary.revalidate()
            publish_integration_bootstrap(
                bootstrap_path,
                _bootstrap_for(target),
                replace_digest=bootstrap_digest,
            )
            boundary.revalidate()
        elif bootstrap_digest != hashlib.sha256(expected_bootstrap).hexdigest():
            raise InstallationError()
    if not _config_edit_matches_spec(target.config_edit, spec):
        raise InstallationError()
    if target.config_edit is not None:
        if spec.config is None or spec.config_path is None:
            raise InstallationError()
        current = _current_config_for_plan(spec)
        current_digest = None if current is None else hashlib.sha256(current).hexdigest()
        if current_digest != target.config_edit.installed_digest:
            plan = plan_owned_config_install(current, spec.config)
            if plan.reverse_edit != target.config_edit:
                raise InstallationError()
            boundary.revalidate()
            publish_config_bytes(spec.config_path, expected=current, data=plan.installed_bytes)
            boundary.revalidate()
    _fault(fault_injector, "recovery_before_enabled_receipt")
    boundary.revalidate()
    if not enabled_receipt:
        _publish_receipt(spec, target, replace_data=current_data)
    if (
        journal.prior_bundle_path is not None
        and journal.prior_bundle_digest is not None
        and (spec_bridge is None or journal.prior_bundle_path != spec_bridge[0])
    ):
        boundary.revalidate()
        _delete_private_if_present(
            journal.prior_bundle_path,
            journal.prior_bundle_digest,
            maximum=2 * 1_024 * 1_024,
        )
        boundary.revalidate()
    boundary.revalidate()
    _delete_journal(spec, journal_data)
    return _status(
        spec,
        key,
        disposition=InstallationDisposition.RECOVERED,
        state=InstallationState.ENABLED,
        connection_id=target.connection_id,
    )


def _journal_for_uninstall(
    receipt: InstallationReceipt,
    key: InstallationKey,
) -> InstallationJournal:
    disabled = _receipt_with_state(receipt, InstallationState.DISABLED, key)
    receipt_bridge = _receipt_bridge_assets(receipt)
    bootstrap_digest = (
        None
        if receipt_bridge is None
        else hashlib.sha256(encode_integration_bootstrap(_bootstrap_for(receipt))).hexdigest()
    )
    return _seal_journal(
        InstallationJournal(
            operation=_JournalOperation.UNINSTALL,
            target_receipt=disabled,
            target_bootstrap_digest=bootstrap_digest,
            prior_receipt_mac=receipt.receipt_mac,
            prior_bundle_path=None if receipt_bridge is None else receipt_bridge[0],
            prior_bundle_digest=None if receipt_bridge is None else receipt_bridge[2],
            prior_launcher_path=receipt.launcher_path,
            prior_launcher_digest=receipt.launcher_digest,
            prior_bootstrap_digest=bootstrap_digest,
            journal_mac="0" * 64,
        ),
        key,
    )


_ConfigRemovalPlan = tuple[bytes | None, bytes | None, bool]


def _plan_config_removal(receipt: InstallationReceipt) -> _ConfigRemovalPlan | None:
    config_path = receipt.config_path
    config_edit = receipt.config_edit
    if config_path is None or config_edit is None:
        if config_path is None and config_edit is None:
            return None
        raise InstallationError()
    current = read_config_bytes(config_path)
    if current is None:
        if config_edit.target_existed:
            raise InstallationError()
        return None, None, False
    marker = config_edit.marker.encode("ascii")
    if marker not in current:
        if (
            config_edit.target_existed
            and hashlib.sha256(current).hexdigest() == config_edit.preimage_digest
        ):
            return current, current, False
        if not config_edit.target_existed:
            return current, current, False
        raise InstallationError()
    restored = remove_owned_config_edit(current, config_edit)
    return current, restored, True


def _apply_config_removal(
    receipt: InstallationReceipt,
    plan: _ConfigRemovalPlan | None,
) -> None:
    config_path = receipt.config_path
    config_edit = receipt.config_edit
    if plan is None:
        if config_path is None and config_edit is None:
            return
        raise InstallationError()
    if config_path is None or config_edit is None:
        raise InstallationError()
    current, restored, needs_write = plan
    if not needs_write:
        observed = read_config_bytes(config_path)
        if (current is None and observed is not None) or (
            current is not None and (observed is None or not hmac.compare_digest(observed, current))
        ):
            raise InstallationError()
        return
    if current is None:
        raise InstallationError()
    if restored is None:
        delete_config_bytes(config_path, expected=current)
    else:
        publish_config_bytes(config_path, expected=current, data=restored)


def _remove_config_for_receipt(receipt: InstallationReceipt) -> None:
    _apply_config_removal(receipt, _plan_config_removal(receipt))


def _recover_uninstall(
    spec: ProviderInstallationSpec,
    key: InstallationKey,
    boundary: _ManagedProjectBoundary,
    journal: InstallationJournal,
    journal_data: bytes,
    *,
    fault_injector: Callable[[str], None] | None,
) -> InstallationStatus:
    boundary.revalidate()
    target = journal.target_receipt
    spec_bridge = _spec_bridge_assets(spec)
    target_bridge = _receipt_bridge_assets(target)
    if (
        not _receipt_matches_spec(target, spec, key)
        or target.launcher_path != spec.launcher_path
        or target.launcher_digest != spec.launcher_digest
    ):
        raise InstallationError()
    if spec_bridge is not None and (
        target_bridge is None
        or target_bridge[0] != spec_bridge[0]
        or target_bridge[2] != spec_bridge[3]
    ):
        raise InstallationError()
    current = _load_receipt_optional(spec, key)
    if current is None:
        raise InstallationError()
    current_data = _encode_model(current, MAX_INSTALLATION_RECEIPT_BYTES)
    draining = _receipt_with_state(target, InstallationState.DRAINING, key)
    publish_draining = False
    if (
        current.state is InstallationState.ENABLED
        and journal.prior_receipt_mac is not None
        and current.receipt_mac == journal.prior_receipt_mac
    ):
        publish_draining = True
    elif current.receipt_mac == draining.receipt_mac:
        if current.state is not InstallationState.DRAINING:
            raise InstallationError()
    elif current.receipt_mac == target.receipt_mac:
        if current.state is not InstallationState.DISABLED:
            raise InstallationError()
    else:
        raise InstallationError()
    config_plan = _plan_config_removal(target)
    expected_assets: tuple[tuple[Path, str, int], ...] = ()
    if target_bridge is not None:
        if journal.target_bootstrap_digest is None:
            raise InstallationError()
        expected_assets = (
            (target_bridge[1], journal.target_bootstrap_digest, 16 * 1_024),
            (target_bridge[0], target_bridge[2], 2 * 1_024 * 1_024),
        )
    for path, digest, maximum in expected_assets:
        observed = _path_digest(path, maximum=maximum)
        if observed is not None and observed != digest:
            raise InstallationError()
    launcher_digest = _launcher_digest_optional(target.launcher_path)
    if launcher_digest is not None and launcher_digest != target.launcher_digest:
        raise InstallationError()
    if publish_draining:
        current_data = _publish_receipt(spec, draining, replace_data=current_data)
        current = draining
    boundary.revalidate()
    _apply_config_removal(target, config_plan)
    if target_bridge is not None:
        if journal.target_bootstrap_digest is None:
            raise InstallationError()
        boundary.revalidate()
        _delete_private_if_present(
            target_bridge[1],
            journal.target_bootstrap_digest,
            maximum=16 * 1_024,
        )
        boundary.revalidate()
        _delete_private_if_present(
            target_bridge[0],
            target_bridge[2],
            maximum=2 * 1_024 * 1_024,
        )
    boundary.revalidate()
    _delete_private_if_present(
        target.launcher_path,
        target.launcher_digest,
        maximum=MAX_INTEGRATION_LAUNCHER_BYTES,
        policy=StableReadPolicy.PRIVATE_EXECUTABLE,
    )
    _fault(fault_injector, "recovery_before_disabled_receipt")
    boundary.revalidate()
    if current.state is not InstallationState.DISABLED:
        _publish_receipt(spec, target, replace_data=current_data)
    boundary.revalidate()
    _delete_journal(spec, journal_data)
    return _status(
        spec,
        key,
        disposition=InstallationDisposition.RECOVERED,
        state=InstallationState.DISABLED,
        connection_id=target.connection_id,
    )


def _recover_locked(
    spec: ProviderInstallationSpec,
    key: InstallationKey,
    boundary: _ManagedProjectBoundary,
    *,
    journal: InstallationJournal,
    fault_injector: Callable[[str], None] | None,
) -> InstallationStatus:
    boundary.revalidate()
    checked = _verify_journal(journal, key)
    journal_data = _encode_model(checked, MAX_INSTALLATION_JOURNAL_BYTES)
    if checked.operation is _JournalOperation.INSTALL:
        return _recover_install(
            spec,
            key,
            boundary,
            checked,
            journal_data,
            fault_injector=fault_injector,
        )
    return _recover_uninstall(
        spec,
        key,
        boundary,
        checked,
        journal_data,
        fault_injector=fault_injector,
    )


def recover_provider_installation(
    spec: ProviderInstallationSpec,
    installation_key: InstallationKey,
    *,
    _fault_injector: Callable[[str], None] | None = None,
) -> InstallationStatus:
    """Roll one authenticated pending or draining transaction forward."""

    checked = _validated_spec(spec)
    if type(installation_key) is not InstallationKey:
        raise InstallationError()
    try:
        if not _require_private_operational_directory(checked):
            raise InstallationError()
        if _lock_digest(checked) in (None, "invalid"):
            raise InstallationError()
        boundary = _ManagedProjectBoundary.capture(checked)
        with _installation_lock(checked):
            boundary.revalidate()
            journal = _load_journal_optional(checked, installation_key)
            if journal is None:
                raise InstallationError()
            return _recover_locked(
                checked,
                installation_key,
                boundary,
                journal=journal,
                fault_injector=_fault_injector,
            )
    except RuntimeError as error:
        if _fault_injector is not None and not isinstance(error, InstallationError):
            raise
        if isinstance(error, InstallationError):
            raise
        raise InstallationError() from None
    except Exception:
        raise InstallationError() from None


def uninstall_provider(
    spec: ProviderInstallationSpec,
    installation_key: InstallationKey,
    *,
    _fault_injector: Callable[[str], None] | None = None,
) -> InstallationStatus:
    """Disable one receipt-bound integration and reverse only its owned config span."""

    checked = _validated_spec(spec)
    if type(installation_key) is not InstallationKey:
        raise InstallationError()
    try:
        if not _require_private_operational_directory(checked):
            raise InstallationError()
        if _lock_digest(checked) in (None, "invalid"):
            raise InstallationError()
        boundary = _ManagedProjectBoundary.capture(checked)
        with _installation_lock(checked):
            boundary.revalidate()
            journal = _load_journal_optional(checked, installation_key)
            if journal is not None:
                return _recover_locked(
                    checked,
                    installation_key,
                    boundary,
                    journal=journal,
                    fault_injector=_fault_injector,
                )
            receipt = _load_receipt_optional(checked, installation_key)
            if (
                receipt is None
                or receipt.state is not InstallationState.ENABLED
                or not _receipt_matches_spec(receipt, checked, installation_key)
                or receipt.launcher_digest != checked.launcher_digest
                or receipt.launcher_path != checked.launcher_path
            ):
                raise InstallationError()
            checked_bridge = _spec_bridge_assets(checked)
            receipt_bridge = _receipt_bridge_assets(receipt)
            if checked_bridge is not None and (
                receipt_bridge is None
                or receipt_bridge[0] != checked_bridge[0]
                or receipt_bridge[2] != checked_bridge[3]
            ):
                raise InstallationError()
            # Plan the reversal and authenticate every owned asset before any write.
            config_plan = _plan_config_removal(receipt)
            bootstrap_digest: str | None = None
            if receipt_bridge is not None:
                bootstrap_data = encode_integration_bootstrap(_bootstrap_for(receipt))
                if _path_digest(receipt_bridge[0], maximum=2 * 1_024 * 1_024) != receipt_bridge[2]:
                    raise InstallationError()
                bootstrap_digest = hashlib.sha256(bootstrap_data).hexdigest()
                if _path_digest(receipt_bridge[1], maximum=16 * 1_024) != bootstrap_digest:
                    raise InstallationError()
            if _read_launcher_digest(receipt.launcher_path) != receipt.launcher_digest:
                raise InstallationError()
            transaction = _journal_for_uninstall(receipt, installation_key)
            boundary.revalidate()
            journal_data = _publish_journal(checked, transaction)
            _fault(_fault_injector, "after_journal_publish")
            boundary.revalidate()
            draining = _receipt_with_state(receipt, InstallationState.DRAINING, installation_key)
            receipt_data = _encode_model(receipt, MAX_INSTALLATION_RECEIPT_BYTES)
            _publish_receipt(checked, draining, replace_data=receipt_data)
            _fault(_fault_injector, "after_draining_receipt_publish")
            boundary.revalidate()
            _apply_config_removal(receipt, config_plan)
            boundary.revalidate()
            _fault(_fault_injector, "after_config_remove")
            if receipt_bridge is not None:
                if bootstrap_digest is None:
                    raise InstallationError()
                _delete_private_exact(
                    receipt_bridge[1],
                    bootstrap_digest,
                    maximum=16 * 1_024,
                )
                boundary.revalidate()
                _delete_private_exact(
                    receipt_bridge[0],
                    receipt_bridge[2],
                    maximum=2 * 1_024 * 1_024,
                )
                boundary.revalidate()
            _delete_private_exact(
                receipt.launcher_path,
                receipt.launcher_digest,
                maximum=MAX_INTEGRATION_LAUNCHER_BYTES,
                policy=StableReadPolicy.PRIVATE_EXECUTABLE,
            )
            _fault(_fault_injector, "after_asset_remove")
            boundary.revalidate()
            disabled = transaction.target_receipt
            draining_data = _encode_model(draining, MAX_INSTALLATION_RECEIPT_BYTES)
            _publish_receipt(checked, disabled, replace_data=draining_data)
            boundary.revalidate()
            _delete_journal(checked, journal_data)
            return _status(
                checked,
                installation_key,
                disposition=InstallationDisposition.UNINSTALLED,
                state=InstallationState.DISABLED,
                connection_id=receipt.connection_id,
            )
    except RuntimeError as error:
        if _fault_injector is not None and not isinstance(error, InstallationError):
            raise
        if isinstance(error, InstallationError):
            raise
        raise InstallationError() from None
    except Exception:
        raise InstallationError() from None


def _git_environment() -> dict[str, str]:
    sterile_path = os.pathsep.join(
        item
        for item in os.defpath.split(os.pathsep)
        if item and "\x00" not in item and Path(item).is_absolute()
    )
    environment = {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "NO_COLOR": "1",
        "PATH": sterile_path,
    }
    for name in ("SYSTEMROOT", "WINDIR"):
        value = os.environ.get(name)
        if value is not None:
            environment[name] = value
    return environment


def _git_executable_is_usable(path: Path, *, native_windows: bool) -> bool:
    """Validate an already-resolved Git executable without trusting child PATH lookup."""

    try:
        metadata = path.stat()
    except OSError:
        return False
    if not path.is_absolute() or (
        native_windows and path.suffix.casefold() not in {".com", ".exe"}
    ):
        return False
    return stat.S_ISREG(metadata.st_mode) and (native_windows or os.access(path, os.X_OK))


def _path_may_be_within(path: Path, boundary: Path) -> bool:
    """Conservatively detect containment by existing-object identity."""

    try:
        boundary_metadata = boundary.stat()
        current = path
        while True:
            metadata = current.stat()
            if (
                metadata.st_dev == boundary_metadata.st_dev
                and metadata.st_ino == boundary_metadata.st_ino
            ):
                return True
            parent = current.parent
            if parent == current:
                return False
            current = parent
    except (AttributeError, OSError, ValueError):
        return True


def _posix_executable_boundary_is_trusted(path: Path) -> bool:
    """Require a root/current-user-owned path with no group or world-writable component."""

    if os.name != "posix" or not _git_executable_is_usable(path, native_windows=False):
        return False
    try:
        effective_user_id = os.geteuid()
        trusted_owners = {0, effective_user_id}
        current = path
        while True:
            metadata = current.stat()
            if (
                metadata.st_uid not in trusted_owners
                or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
                or (current == path and metadata.st_uid != 0 and metadata.st_nlink != 1)
                or (current == path and not stat.S_ISREG(metadata.st_mode))
                or (current != path and not stat.S_ISDIR(metadata.st_mode))
            ):
                return False
            parent = current.parent
            if parent == current:
                return True
            current = parent
    except (AttributeError, OSError):
        return False


def _windows_executable_boundary_is_trusted(path: Path) -> bool:
    """Authorize a current-user-owned executable and its complete native Windows ancestry."""

    try:
        operations = NativeWindowsSecurityOperations()
        authorization = authorize_windows_managed_path(
            PureWindowsPath(os.fspath(path)),
            kind=WindowsPathKind.FILE,
            operations=operations,
        )
        authorization.revalidate()
    except Exception:
        return False
    return True


def _windows_git_names(windows_pathext: str | None) -> tuple[str, ...]:
    extensions = ".COM;.EXE" if windows_pathext is None else windows_pathext
    names: list[str] = []
    for item in extensions.split(";"):
        normalized = item.strip().casefold()
        if normalized not in {".com", ".exe"}:
            continue
        name = f"git{normalized}"
        if name not in names:
            names.append(name)
    return tuple(names)


def _find_git_executable(
    search_path: str | None,
    *,
    project_root: Path,
    current_directory: Path,
    native_windows: bool,
    windows_pathext: str | None,
    candidate_is_trusted: Callable[[Path], bool],
) -> str | None:
    """Manually enumerate bounded PATH entries without implicit current-directory lookup."""

    try:
        if (
            search_path is None
            or not isinstance(search_path, str)
            or "\x00" in search_path
            or not callable(candidate_is_trusted)
        ):
            return None
        checked_project = project_root.resolve(strict=True)
        checked_current = current_directory.resolve(strict=True)
        if not checked_project.is_dir() or not checked_current.is_dir():
            return None
    except (OSError, RuntimeError, TypeError):
        return None

    names = _windows_git_names(windows_pathext) if native_windows else ("git",)
    separator = ";" if native_windows else os.pathsep
    seen_directories: set[Path] = set()
    seen_candidates: set[Path] = set()
    for item in search_path.split(separator):
        if not item or "\x00" in item:
            continue
        directory = Path(item)
        if not directory.is_absolute():
            continue
        try:
            resolved_directory = directory.resolve(strict=True)
            if (
                resolved_directory in seen_directories
                or not resolved_directory.is_dir()
                or _path_may_be_within(resolved_directory, checked_project)
                or _path_may_be_within(resolved_directory, checked_current)
            ):
                continue
            seen_directories.add(resolved_directory)
        except (OSError, RuntimeError):
            continue

        for name in names:
            try:
                candidate = (resolved_directory / name).resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if (
                candidate in seen_candidates
                or _path_may_be_within(candidate, checked_project)
                or _path_may_be_within(candidate, checked_current)
            ):
                continue
            seen_candidates.add(candidate)
            if not _git_executable_is_usable(candidate, native_windows=native_windows):
                continue
            try:
                if not candidate_is_trusted(candidate):
                    continue
            except Exception:
                continue
            return str(candidate)
    return None


def _resolved_git_executable(project_root: Path) -> str | None:
    """Resolve Git only through an absolute, non-project, validated trust boundary."""

    native_windows = os.name == "nt"
    try:
        current_directory = Path.cwd()
    except OSError:
        return None
    return _find_git_executable(
        os.environ.get("PATH"),
        project_root=project_root,
        current_directory=current_directory,
        native_windows=native_windows,
        windows_pathext=os.environ.get("PATHEXT") if native_windows else None,
        candidate_is_trusted=(
            _windows_executable_boundary_is_trusted
            if native_windows
            else _posix_executable_boundary_is_trusted
        ),
    )


def git_project_file_review(spec: ProviderInstallationSpec) -> GitProjectFileReview:
    """Classify managed-file Git visibility without mutating repository state."""

    checked = _validated_spec(spec)
    candidates = checked.project_local_paths
    relative = tuple(path.relative_to(checked.project_root).as_posix() for path in candidates)
    git_executable = _resolved_git_executable(checked.project_root)
    if git_executable is None:
        return GitProjectFileReview(
            GitProjectFileDisposition.UNAVAILABLE,
            candidates,
            (),
            (),
        )
    command_prefix = (
        git_executable,
        "-c",
        "color.ui=false",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "maintenance.auto=false",
    )
    try:
        repository = subprocess.run(
            (
                *command_prefix,
                "rev-parse",
                "--is-inside-work-tree",
            ),
            cwd=checked.project_root,
            env=_git_environment(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=3,
            check=False,
        )
    except FileNotFoundError:
        return GitProjectFileReview(
            GitProjectFileDisposition.UNAVAILABLE,
            candidates,
            (),
            (),
        )
    except (OSError, subprocess.SubprocessError):
        return GitProjectFileReview(
            GitProjectFileDisposition.UNAVAILABLE,
            candidates,
            (),
            (),
        )
    repository_stderr = repository.stderr
    if (
        repository.returncode == 128
        and isinstance(repository_stderr, bytes)
        and b"not a git repository" in repository_stderr
        and len(repository_stderr) <= 64 * 1_024
    ):
        return GitProjectFileReview(
            GitProjectFileDisposition.NOT_REPOSITORY,
            candidates,
            (),
            (),
        )
    if repository.returncode != 0 or repository.stdout != b"true\n":
        return GitProjectFileReview(
            GitProjectFileDisposition.UNAVAILABLE,
            candidates,
            (),
            (),
        )

    try:
        tracked = subprocess.run(
            (
                *command_prefix,
                "ls-files",
                "-z",
                "--",
                *relative,
            ),
            cwd=checked.project_root,
            env=_git_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )
        ignored = subprocess.run(
            (
                *command_prefix,
                "check-ignore",
                "--no-index",
                "-z",
                "--stdin",
            ),
            cwd=checked.project_root,
            env=_git_environment(),
            input=b"".join(name.encode("utf-8") + b"\0" for name in relative),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return GitProjectFileReview(
            GitProjectFileDisposition.UNAVAILABLE,
            candidates,
            (),
            (),
        )
    if (
        tracked.returncode != 0
        or ignored.returncode not in (0, 1)
        or len(tracked.stdout) > 64 * 1_024
        or len(ignored.stdout) > 64 * 1_024
    ):
        return GitProjectFileReview(
            GitProjectFileDisposition.UNAVAILABLE,
            candidates,
            (),
            (),
        )
    try:
        tracked_names = frozenset(
            item.decode("utf-8", errors="strict") for item in tracked.stdout.split(b"\0") if item
        )
        ignored_names = frozenset(
            item.decode("utf-8", errors="strict") for item in ignored.stdout.split(b"\0") if item
        )
    except UnicodeDecodeError:
        return GitProjectFileReview(
            GitProjectFileDisposition.UNAVAILABLE,
            candidates,
            (),
            (),
        )
    expected_names = frozenset(relative)
    if not tracked_names <= expected_names or not ignored_names <= expected_names:
        return GitProjectFileReview(
            GitProjectFileDisposition.UNAVAILABLE,
            candidates,
            (),
            (),
        )
    tracked_files = tuple(
        path for path, name in zip(candidates, relative, strict=True) if name in tracked_names
    )
    unignored_files = tuple(
        path
        for path, name in zip(candidates, relative, strict=True)
        if name in tracked_names or name not in ignored_names
    )
    disposition = (
        GitProjectFileDisposition.UNIGNORED
        if unignored_files
        else GitProjectFileDisposition.ALL_IGNORED
    )
    return GitProjectFileReview(
        disposition,
        candidates,
        unignored_files,
        tracked_files,
    )


def git_tracked_project_files(spec: ProviderInstallationSpec) -> tuple[Path, ...]:
    """Return managed files already tracked by Git; never mutate ignore configuration."""

    return git_project_file_review(spec).tracked_files


__all__ = [
    "MAX_INSTALLATION_JOURNAL_BYTES",
    "MAX_INSTALLATION_RECEIPT_BYTES",
    "GitProjectFileDisposition",
    "GitProjectFileReview",
    "InstallationDisposition",
    "InstallationError",
    "InstallationIdentity",
    "InstallationJournal",
    "InstallationReceipt",
    "InstallationState",
    "InstallationStatus",
    "derive_installation_identity",
    "ensure_private_installation_directory",
    "git_project_file_review",
    "git_tracked_project_files",
    "inspect_installation_receipt",
    "inspect_provider_installation",
    "install_provider",
    "recover_provider_installation",
    "uninstall_provider",
]
