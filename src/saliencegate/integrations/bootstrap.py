"""Content-free bootstrap sidecars for project-local connector bundles."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from pathlib import Path, PureWindowsPath
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)

from saliencegate.capture.capabilities import CaptureProfile
from saliencegate.domain import canonical_json
from saliencegate.security.files import (
    StableReadPolicy,
    authorize_atomic_file_publication,
    read_stable_file,
)
from saliencegate.security.windows import (
    NativeWindowsSecurityOperations,
    WindowsPathKind,
    authorize_windows_managed_path,
)

MAX_INTEGRATION_BOOTSTRAP_BYTES = 16 * 1_024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONNECTION_ID = re.compile(r"^[a-z0-9][a-z0-9._:-]{11,127}$")


class IntegrationBootstrapError(ValueError):
    """A content-free bootstrap validation or publication failure."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("integration bootstrap is invalid")


class IntegrationBootstrap(BaseModel):
    """Only operational routing and authenticated digest commitments."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    schema_version: Literal["integration-bootstrap/v1"] = "integration-bootstrap/v1"
    profile: CaptureProfile
    connection_id: Annotated[
        str,
        StringConstraints(min_length=12, max_length=128, pattern=_CONNECTION_ID.pattern),
    ] = Field(repr=False)
    launcher_path: Path = Field(repr=False)
    capability_digest: Annotated[str, StringConstraints(pattern=_SHA256.pattern)] = Field(
        repr=False
    )
    bundle_digest: Annotated[str, StringConstraints(pattern=_SHA256.pattern)] = Field(repr=False)
    receipt_mac: Annotated[str, StringConstraints(pattern=_SHA256.pattern)] = Field(repr=False)

    @field_validator("launcher_path")
    @classmethod
    def launcher_is_exact_absolute_path(cls, value: Path) -> Path:
        if (
            not isinstance(value, Path)
            or not value.is_absolute()
            or ".." in value.parts
            or "\x00" in os.fspath(value)
            or not value.name
        ):
            raise ValueError("bootstrap launcher path is invalid")
        return value

    def __repr__(self) -> str:
        return "IntegrationBootstrap(<redacted>)"

    __str__ = __repr__


def _validated_bootstrap(value: object) -> IntegrationBootstrap:
    try:
        payload = (
            value.model_dump(mode="python", warnings="error")
            if type(value) is IntegrationBootstrap
            else value
        )
        return IntegrationBootstrap.model_validate(payload)
    except Exception:
        raise IntegrationBootstrapError() from None


def encode_integration_bootstrap(value: IntegrationBootstrap) -> bytes:
    try:
        checked = _validated_bootstrap(value)
        encoded = canonical_json(checked.model_dump(mode="json", warnings="error"))
        if not 2 <= len(encoded) <= MAX_INTEGRATION_BOOTSTRAP_BYTES:
            raise IntegrationBootstrapError()
        return encoded
    except IntegrationBootstrapError:
        raise
    except Exception:
        raise IntegrationBootstrapError() from None


def decode_integration_bootstrap(data: bytes) -> IntegrationBootstrap:
    try:
        if type(data) is not bytes or not 2 <= len(data) <= MAX_INTEGRATION_BOOTSTRAP_BYTES:
            raise IntegrationBootstrapError()
        checked = IntegrationBootstrap.model_validate_json(data)
        if not hmac.compare_digest(encode_integration_bootstrap(checked), data):
            raise IntegrationBootstrapError()
        return checked
    except IntegrationBootstrapError:
        raise
    except Exception:
        raise IntegrationBootstrapError() from None


def publish_integration_bootstrap(
    path: Path,
    value: IntegrationBootstrap,
    *,
    replace_digest: str | None = None,
) -> bytes:
    """Publish canonical owner-private bytes, replacing only one expected digest."""

    try:
        if not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts:
            raise IntegrationBootstrapError()
        if replace_digest is not None and (
            type(replace_digest) is not str or _SHA256.fullmatch(replace_digest) is None
        ):
            raise IntegrationBootstrapError()
        encoded = encode_integration_bootstrap(value)
        if os.name == "nt":  # pragma: no cover - exercised by native Windows R01
            operations = NativeWindowsSecurityOperations()
            windows_published = operations.publish_private_file_in_managed_directory(
                PureWindowsPath(os.fspath(path)),
                encoded,
                maximum_bytes=MAX_INTEGRATION_BOOTSTRAP_BYTES,
                validate_replacement=(
                    None
                    if replace_digest is None
                    else lambda existing: hmac.compare_digest(
                        hashlib.sha256(existing).hexdigest(), replace_digest
                    )
                ),
                validate_published=lambda observed: hmac.compare_digest(observed, encoded),
            )
            if not hmac.compare_digest(windows_published.data, encoded):
                raise IntegrationBootstrapError()
            return encoded
        publication = authorize_atomic_file_publication(
            path,
            maximum_bytes=MAX_INTEGRATION_BOOTSTRAP_BYTES,
            validate_replacement=(
                None
                if replace_digest is None
                else lambda existing: hmac.compare_digest(
                    hashlib.sha256(existing).hexdigest(), replace_digest
                )
            ),
        )
        published = publication.publish(
            encoded,
            validate_published=lambda observed: observed == encoded,
        )
        if published.data != encoded:
            raise IntegrationBootstrapError()
        return encoded
    except IntegrationBootstrapError:
        raise
    except Exception:
        raise IntegrationBootstrapError() from None


def inspect_integration_bootstrap(path: Path) -> IntegrationBootstrap:
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
                maximum_bytes=MAX_INTEGRATION_BOOTSTRAP_BYTES,
            )
            parent.revalidate()
            windows_stable.authorization.revalidate()
            return decode_integration_bootstrap(windows_stable.data)
        stable = read_stable_file(
            path,
            maximum_bytes=MAX_INTEGRATION_BOOTSTRAP_BYTES,
            policy=StableReadPolicy.PRIVATE_EXACT,
        )
        return decode_integration_bootstrap(stable.data)
    except IntegrationBootstrapError:
        raise
    except Exception:
        raise IntegrationBootstrapError() from None


__all__ = [
    "MAX_INTEGRATION_BOOTSTRAP_BYTES",
    "IntegrationBootstrap",
    "IntegrationBootstrapError",
    "decode_integration_bootstrap",
    "encode_integration_bootstrap",
    "inspect_integration_bootstrap",
    "publish_integration_bootstrap",
]
