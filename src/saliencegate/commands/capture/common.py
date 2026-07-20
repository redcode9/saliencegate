"""Shared value-free failures and project identity for capture commands."""

from __future__ import annotations

import os
from pathlib import Path

from saliencegate.capture.identities import CaptureDigestContext
from saliencegate.security import InstallationKey


class CaptureCommandError(RuntimeError):
    """Base class for stable, content-free capture command failures."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("capture command failed")


class CaptureCommandInputError(CaptureCommandError):
    __slots__ = ()

    def __init__(self) -> None:
        RuntimeError.__init__(self, "capture command input is invalid")


class CaptureCommandConfigurationError(CaptureCommandError):
    __slots__ = ()

    def __init__(self) -> None:
        RuntimeError.__init__(self, "capture configuration is invalid")


class CaptureCommandRequiresDisconnectError(CaptureCommandConfigurationError):
    __slots__ = ()

    def __init__(self) -> None:
        RuntimeError.__init__(self, "capture must be disconnected before project deletion")


class CaptureCommandUnavailableError(CaptureCommandError):
    __slots__ = ()

    def __init__(self) -> None:
        RuntimeError.__init__(self, "capture integration is unavailable")


class CaptureCommandIntegrityError(CaptureCommandError):
    __slots__ = ()

    def __init__(self) -> None:
        RuntimeError.__init__(self, "capture integrity check failed")


def resolve_capture_project(
    value: str | os.PathLike[str] | None,
    *,
    cwd: Path | None = None,
) -> Path:
    """Resolve one existing project directory without mutating it."""

    result: Path | None = None
    try:
        if value is not None and not isinstance(value, (str, os.PathLike)):
            raise TypeError
        if isinstance(value, str) and not value:
            raise ValueError
        base = Path.cwd() if cwd is None else cwd
        if not isinstance(base, Path):
            raise TypeError
        base = base.resolve(strict=True)
        candidate = base if value is None else Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        candidate = candidate.resolve(strict=True)
        if not candidate.is_dir():
            raise ValueError
        result = candidate
    except (OSError, RuntimeError, TypeError, ValueError):
        result = None
    if result is None:
        raise CaptureCommandInputError()
    return result


def capture_project_digest(
    project: str | os.PathLike[str] | Path,
    *,
    installation_key: InstallationKey,
) -> str:
    """Derive the installation-key-bound project identity stored by capture."""

    result: str | None = None
    try:
        if type(installation_key) is not InstallationKey:
            raise TypeError
        resolved = resolve_capture_project(project)
        result = CaptureDigestContext(installation_key).workspace_identity(os.fsencode(resolved))
    except CaptureCommandInputError:
        raise
    except Exception:
        result = None
    if result is None:
        raise CaptureCommandConfigurationError()
    return result


__all__ = [
    "CaptureCommandConfigurationError",
    "CaptureCommandError",
    "CaptureCommandInputError",
    "CaptureCommandIntegrityError",
    "CaptureCommandRequiresDisconnectError",
    "CaptureCommandUnavailableError",
    "capture_project_digest",
    "resolve_capture_project",
]
