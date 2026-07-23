"""Bind a packaged launcher to one authenticated provider installation."""

from __future__ import annotations

import os
import stat
import sysconfig
from collections.abc import Callable
from pathlib import Path

from saliencegate.commands.capture.common import CaptureCommandUnavailableError
from saliencegate.integrations.installation import (
    _path_may_be_within,
    _posix_executable_boundary_is_trusted,
    _windows_executable_boundary_is_trusted,
    derive_installation_identity,
)
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


def _capture_executable_is_usable(
    path: Path,
    *,
    native_windows: bool,
    windows_suffixes: tuple[str, ...] = (".exe",),
) -> bool:
    try:
        metadata = path.stat()
    except OSError:
        return False
    return (
        path.is_absolute()
        and stat.S_ISREG(metadata.st_mode)
        and (not native_windows or path.suffix.casefold() in windows_suffixes)
        and (native_windows or os.access(path, os.X_OK))
    )


def _interpreter_capture_executable(
    *,
    project_root: Path,
    scripts_directory: str | os.PathLike[str] | Path | None = None,
    native_windows: bool | None = None,
    candidate_is_trusted: Callable[[Path], bool] | None = None,
) -> Path:
    """Resolve the installed hook through a non-project interpreter scripts boundary."""

    try:
        selected_platform = os.name == "nt" if native_windows is None else native_windows
        if type(selected_platform) is not bool:
            raise CaptureCommandUnavailableError()
        checked_project = project_root.resolve(strict=True)
        if not checked_project.is_dir():
            raise CaptureCommandUnavailableError()
        selected_directory = (
            sysconfig.get_path("scripts") if scripts_directory is None else scripts_directory
        )
        if not isinstance(selected_directory, (str, os.PathLike)):
            raise CaptureCommandUnavailableError()
        directory = Path(selected_directory)
        if not directory.is_absolute():
            raise CaptureCommandUnavailableError()
        directory = directory.resolve(strict=True)
        if not directory.is_dir() or _path_may_be_within(directory, checked_project):
            raise CaptureCommandUnavailableError()
        executable_name = (
            "saliencegate-capture-hook.exe" if selected_platform else "saliencegate-capture-hook"
        )
        unresolved = directory / executable_name
        if unresolved.is_symlink():
            raise CaptureCommandUnavailableError()
        executable = unresolved.resolve(strict=True)
        if (
            executable.parent != directory
            or _path_may_be_within(executable, checked_project)
            or not _capture_executable_is_usable(
                executable,
                native_windows=selected_platform,
            )
        ):
            raise CaptureCommandUnavailableError()
        trust_check = candidate_is_trusted
        if trust_check is None:
            trust_check = (
                _windows_executable_boundary_is_trusted
                if selected_platform
                else _posix_executable_boundary_is_trusted
            )
        if not callable(trust_check):
            raise CaptureCommandUnavailableError()
        try:
            trusted = trust_check(executable)
        except Exception:
            raise CaptureCommandUnavailableError() from None
        if type(trusted) is not bool or not trusted:
            raise CaptureCommandUnavailableError()
        return executable
    except CaptureCommandUnavailableError:
        raise
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        raise CaptureCommandUnavailableError() from None


def _explicit_capture_executable(
    selected: str | os.PathLike[str] | Path,
    *,
    native_windows: bool | None = None,
) -> Path:
    """Validate one caller-authorized absolute executable without ambient path lookup."""

    try:
        selected_platform = os.name == "nt" if native_windows is None else native_windows
        if type(selected_platform) is not bool:
            raise CaptureCommandUnavailableError()
        if not isinstance(selected, (str, os.PathLike)):
            raise CaptureCommandUnavailableError()
        unresolved = Path(selected)
        if not unresolved.is_absolute():
            raise CaptureCommandUnavailableError()
        executable = unresolved.resolve(strict=True)
        if not _capture_executable_is_usable(
            executable,
            native_windows=selected_platform,
            windows_suffixes=(".com", ".exe"),
        ):
            raise CaptureCommandUnavailableError()
        return executable
    except CaptureCommandUnavailableError:
        raise
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        raise CaptureCommandUnavailableError() from None


def materialize_provider_launcher(
    spec: ProviderInstallationSpec,
    key: InstallationKey,
    *,
    capture_executable: str | os.PathLike[str] | Path | None = None,
) -> ProviderInstallationSpec:
    """Bind the packaged launcher to this install's absolute executable and ID."""

    try:
        executable = (
            _interpreter_capture_executable(project_root=spec.project_root)
            if capture_executable is None
            else _explicit_capture_executable(capture_executable)
        )
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
