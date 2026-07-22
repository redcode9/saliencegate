"""Provider-credential-free environment projection for capture subprocesses."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Final

PROVIDER_CREDENTIAL_ENVIRONMENT_KEYS: Final = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_ORGANIZATION",
        "OPENAI_ORG_ID",
        "OPENAI_PROJECT",
        "OPENAI_PROJECT_ID",
    }
)


class CaptureEnvironmentError(ValueError):
    """An environment boundary was invalid without exposing a value."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("capture environment is invalid")


def environment_without_provider_credentials(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Copy an environment without ever resolving provider-credential values."""

    source = os.environ if environ is None else environ
    if not isinstance(source, Mapping):
        raise CaptureEnvironmentError()
    result: dict[str, str] = {}
    try:
        for key in source:
            if type(key) is not str:
                raise CaptureEnvironmentError()
            if key.upper() in PROVIDER_CREDENTIAL_ENVIRONMENT_KEYS:
                continue
            value = source[key]
            if type(value) is not str:
                raise CaptureEnvironmentError()
            result[key] = value
    except CaptureEnvironmentError:
        raise
    except Exception:
        raise CaptureEnvironmentError() from None
    return result
