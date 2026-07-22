"""Stable security public API without eager redaction/model imports."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from saliencegate.security.digests import (
        AmbiguousDigestModeError,
        DigestModeMismatchError,
        MissingInstallationKeyError,
        SyntheticDigestDisabledError,
    )
    from saliencegate.security.files import (
        AtomicFilePublication,
        SecureFileBoundError,
        SecureFileError,
        SecureFileUnsupportedError,
        StableFileAuthorization,
        StableFileRead,
        StableReadPolicy,
        authorize_atomic_file_publication,
        authorize_private_sqlite_path,
        claim_private_sqlite_location,
        ensure_private_directory,
        inspect_private_directory,
        inspect_private_file_location,
        read_stable_file,
    )
    from saliencegate.security.keys import (
        InsecureKeyFileError,
        InsecureKeyPathError,
        InstallationKey,
        InvalidInstallationKeyError,
        default_installation_key_path,
        generate_installation_key,
        load_installation_key,
        load_or_create_installation_key,
    )
    from saliencegate.security.redaction import (
        REDACTED,
        REDACTED_PRIVATE_KEY,
        AmbiguousFieldNameError,
        EventRedactionResult,
        RedactionFinding,
        RedactionPolicy,
        RedactionResult,
        Redactor,
        SecretInFieldNameError,
        verify_redacted_event,
    )

_EXPORT_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "saliencegate.security.digests": (
        "AmbiguousDigestModeError",
        "DigestModeMismatchError",
        "MissingInstallationKeyError",
        "SyntheticDigestDisabledError",
    ),
    "saliencegate.security.files": (
        "AtomicFilePublication",
        "SecureFileBoundError",
        "SecureFileError",
        "SecureFileUnsupportedError",
        "StableFileAuthorization",
        "StableFileRead",
        "StableReadPolicy",
        "authorize_atomic_file_publication",
        "authorize_private_sqlite_path",
        "claim_private_sqlite_location",
        "ensure_private_directory",
        "inspect_private_directory",
        "inspect_private_file_location",
        "read_stable_file",
    ),
    "saliencegate.security.keys": (
        "InsecureKeyFileError",
        "InsecureKeyPathError",
        "InstallationKey",
        "InvalidInstallationKeyError",
        "default_installation_key_path",
        "generate_installation_key",
        "load_installation_key",
        "load_or_create_installation_key",
    ),
    "saliencegate.security.redaction": (
        "REDACTED",
        "REDACTED_PRIVATE_KEY",
        "AmbiguousFieldNameError",
        "EventRedactionResult",
        "RedactionFinding",
        "RedactionPolicy",
        "RedactionResult",
        "Redactor",
        "SecretInFieldNameError",
        "verify_redacted_event",
    ),
}

_EXPORTS: Final[dict[str, str]] = {
    name: module_name for module_name, names in _EXPORT_GROUPS.items() for name in names
}

__all__ = [
    "REDACTED",
    "REDACTED_PRIVATE_KEY",
    "AmbiguousDigestModeError",
    "AmbiguousFieldNameError",
    "AtomicFilePublication",
    "DigestModeMismatchError",
    "EventRedactionResult",
    "InsecureKeyFileError",
    "InsecureKeyPathError",
    "InstallationKey",
    "InvalidInstallationKeyError",
    "MissingInstallationKeyError",
    "RedactionFinding",
    "RedactionPolicy",
    "RedactionResult",
    "Redactor",
    "SecretInFieldNameError",
    "SecureFileBoundError",
    "SecureFileError",
    "SecureFileUnsupportedError",
    "StableFileAuthorization",
    "StableFileRead",
    "StableReadPolicy",
    "SyntheticDigestDisabledError",
    "authorize_atomic_file_publication",
    "authorize_private_sqlite_path",
    "claim_private_sqlite_location",
    "default_installation_key_path",
    "ensure_private_directory",
    "generate_installation_key",
    "inspect_private_directory",
    "inspect_private_file_location",
    "load_installation_key",
    "load_or_create_installation_key",
    "read_stable_file",
    "verify_redacted_event",
]

if len(_EXPORTS) != len(__all__) or frozenset(_EXPORTS) != frozenset(__all__):
    raise RuntimeError("security public API export map is invalid")


def __getattr__(name: str) -> object:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(frozenset(globals()).union(__all__))
