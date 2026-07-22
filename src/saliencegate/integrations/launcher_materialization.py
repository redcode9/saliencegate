"""Bind a packaged launcher to one authenticated provider installation."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

from saliencegate.commands.capture.common import CaptureCommandUnavailableError
from saliencegate.integrations.installation import derive_installation_identity
from saliencegate.integrations.launcher_renderer import (
    CaptureLauncherPlatform,
    LauncherRenderError,
    render_capture_launcher,
)
from saliencegate.integrations.registry import ProviderInstallationSpec
from saliencegate.security import InstallationKey


def _trusted_launcher_watchdog(platform: CaptureLauncherPlatform) -> Path:
    try:
        candidates: tuple[Path, ...]
        if platform is CaptureLauncherPlatform.POSIX:
            candidates = (Path("/bin/sleep"), Path("/usr/bin/sleep"))
        else:  # pragma: no cover - exercised by native Windows R01
            import ctypes

            buffer = ctypes.create_unicode_buffer(32_768)
            length = ctypes.windll.kernel32.GetSystemDirectoryW(  # type: ignore[attr-defined]
                buffer,
                len(buffer),
            )
            if not 0 < length < len(buffer):
                raise OSError
            candidates = (Path(buffer.value) / "WindowsPowerShell" / "v1.0" / "powershell.exe",)
        for candidate in candidates:
            try:
                resolved = candidate.resolve(strict=True)
                metadata = resolved.lstat()
            except OSError:
                continue
            if (
                stat.S_ISREG(metadata.st_mode)
                and not resolved.is_symlink()
                and (os.name != "posix" or os.access(resolved, os.X_OK))
            ):
                return resolved
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    raise CaptureCommandUnavailableError()


def materialize_provider_launcher(
    spec: ProviderInstallationSpec,
    key: InstallationKey,
    *,
    capture_executable: str | os.PathLike[str] | Path | None = None,
) -> ProviderInstallationSpec:
    """Bind the packaged launcher to this install's absolute executable and ID."""

    try:
        selected = (
            shutil.which("saliencegate-capture-hook")
            if capture_executable is None
            else capture_executable
        )
        if selected is None or not isinstance(selected, (str, os.PathLike)):
            raise CaptureCommandUnavailableError()
        executable = Path(selected).expanduser()
        if not executable.is_absolute():
            executable = Path.cwd() / executable
        executable = executable.resolve(strict=True)
        metadata = executable.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or executable.is_symlink()
            or (os.name == "posix" and not os.access(executable, os.X_OK))
        ):
            raise CaptureCommandUnavailableError()
        identity = derive_installation_identity(spec, key)
        platform = (
            CaptureLauncherPlatform.WINDOWS if os.name == "nt" else CaptureLauncherPlatform.POSIX
        )
        launcher = render_capture_launcher(
            executable=executable,
            profile=spec.profile,
            connection_id=identity.connection_id,
            platform=platform,
            watchdog_executable=_trusted_launcher_watchdog(platform),
        )
        payload = spec.model_dump(mode="python", warnings="error")
        payload["launcher_bytes"] = launcher
        return ProviderInstallationSpec.model_validate(payload)
    except CaptureCommandUnavailableError:
        raise
    except (FileNotFoundError, LauncherRenderError, OSError, TypeError, ValueError):
        raise CaptureCommandUnavailableError() from None


__all__ = ["materialize_provider_launcher"]
