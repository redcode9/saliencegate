"""Strict rendering of the packaged provider-neutral capture launchers."""

from __future__ import annotations

import re
import shlex
from enum import StrEnum
from importlib import resources
from pathlib import PurePath, PurePosixPath, PureWindowsPath
from typing import Final

from saliencegate.capture.capabilities import CaptureProfile
from saliencegate.integrations.registry import MAX_INTEGRATION_LAUNCHER_BYTES

_CONNECTION_ID: Final = re.compile(r"^[a-z0-9][a-z0-9._:-]{11,127}$")
_LEFTOVER_TOKEN: Final = re.compile(r"__SALIENCEGATE_[A-Z0-9_]+__")
_WINDOWS_INVALID_COMPONENT: Final = frozenset('<>:"|?*')

_POSIX_TOKENS: Final = (
    "__SALIENCEGATE_EXECUTABLE_SHELL__",
    "__SALIENCEGATE_WATCHDOG_SHELL__",
    "__SALIENCEGATE_PROFILE_SHELL__",
    "__SALIENCEGATE_CONNECTION_SHELL__",
)
_WINDOWS_TOKENS: Final = (
    "__SALIENCEGATE_EXECUTABLE_BATCH__",
    "__SALIENCEGATE_WATCHDOG_BATCH__",
    "__SALIENCEGATE_PROFILE_BATCH__",
    "__SALIENCEGATE_CONNECTION_BATCH__",
)
_DEFAULT_POSIX_WATCHDOG: Final = PurePosixPath("/bin/sleep")
_DEFAULT_WINDOWS_WATCHDOG: Final = PureWindowsPath(
    r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
)


class CaptureLauncherPlatform(StrEnum):
    """The supported native launcher formats."""

    POSIX = "posix"
    WINDOWS = "windows"


class LauncherRenderError(ValueError):
    """A content-free launcher rendering failure."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("capture launcher is invalid")


def _read_launcher_template(platform: CaptureLauncherPlatform) -> str:
    resource_name = "posix.sh" if platform is CaptureLauncherPlatform.POSIX else "windows.cmd"
    try:
        payload = (
            resources.files("saliencegate.integrations")
            .joinpath("launchers")
            .joinpath(resource_name)
            .read_bytes()
        )
        if not payload or len(payload) > MAX_INTEGRATION_LAUNCHER_BYTES or b"\x00" in payload:
            raise LauncherRenderError()
        return payload.decode("utf-8", errors="strict")
    except LauncherRenderError:
        raise
    except Exception:
        raise LauncherRenderError() from None


def _has_control_character(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _raw_executable(executable: str | PurePath) -> str:
    if type(executable) is str:
        return executable
    if isinstance(executable, PurePath):
        return str(executable)
    raise LauncherRenderError()


def _validated_posix_executable(executable: str | PurePath) -> str:
    raw = _raw_executable(executable)
    if not raw or len(raw) > 32_767 or _has_control_character(raw):
        raise LauncherRenderError()
    path = PurePosixPath(raw)
    if not path.is_absolute() or not path.name or ".." in path.parts:
        raise LauncherRenderError()
    return str(path)


def _validated_windows_program(
    executable: str | PurePath,
    *,
    suffixes: tuple[str, ...],
) -> str:
    raw = _raw_executable(executable)
    if not raw or len(raw) > 32_767 or _has_control_character(raw):
        raise LauncherRenderError()
    path = PureWindowsPath(raw)
    if (
        not path.is_absolute()
        or re.fullmatch(r"[A-Za-z]:\\", path.anchor) is None
        or not path.name
        or ".." in path.parts
        or path.suffix.lower() not in suffixes
        or path.is_reserved()
    ):
        raise LauncherRenderError()
    for component in path.parts[1:]:
        if (
            not component
            or component.endswith((" ", "."))
            or any(character in _WINDOWS_INVALID_COMPONENT for character in component)
        ):
            raise LauncherRenderError()
    return str(path)


def _validated_windows_executable(executable: str | PurePath) -> str:
    return _validated_windows_program(executable, suffixes=(".exe", ".com"))


def _validated_windows_watchdog(executable: str | PurePath) -> str:
    return _validated_windows_program(executable, suffixes=(".exe",))


def _render_values(
    *,
    executable: str | PurePath,
    profile: CaptureProfile,
    connection_id: str,
    platform: CaptureLauncherPlatform,
    watchdog_executable: str | PurePath | None,
) -> tuple[tuple[str, str], ...]:
    if type(profile) is not CaptureProfile:
        raise LauncherRenderError()
    if type(connection_id) is not str or _CONNECTION_ID.fullmatch(connection_id) is None:
        raise LauncherRenderError()
    if platform is CaptureLauncherPlatform.POSIX:
        watchdog = _DEFAULT_POSIX_WATCHDOG if watchdog_executable is None else watchdog_executable
        values = (
            _validated_posix_executable(executable),
            _validated_posix_executable(watchdog),
            profile.value,
            connection_id,
        )
        if any(_LEFTOVER_TOKEN.search(value) is not None for value in values):
            raise LauncherRenderError()
        return tuple(zip(_POSIX_TOKENS, (shlex.quote(value) for value in values), strict=True))

    watchdog = _DEFAULT_WINDOWS_WATCHDOG if watchdog_executable is None else watchdog_executable
    values = (
        _validated_windows_executable(executable),
        _validated_windows_watchdog(watchdog),
        profile.value,
        connection_id,
    )
    if any(
        '"' in value or _has_control_character(value) or _LEFTOVER_TOKEN.search(value) is not None
        for value in values
    ):
        raise LauncherRenderError()
    return tuple(zip(_WINDOWS_TOKENS, (value.replace("%", "%%") for value in values), strict=True))


def render_capture_launcher(
    *,
    executable: str | PurePath,
    profile: CaptureProfile,
    connection_id: str,
    platform: CaptureLauncherPlatform,
    watchdog_executable: str | PurePath | None = None,
) -> bytes:
    """Render one attested launcher without interpreting provider-controlled data."""

    try:
        if type(platform) is not CaptureLauncherPlatform:
            raise LauncherRenderError()
        replacements = _render_values(
            executable=executable,
            profile=profile,
            connection_id=connection_id,
            platform=platform,
            watchdog_executable=watchdog_executable,
        )
        template = _read_launcher_template(platform)
        if any(template.count(token) != 1 for token, _value in replacements):
            raise LauncherRenderError()
        rendered = template
        for token, value in replacements:
            rendered = rendered.replace(token, value)
        if _LEFTOVER_TOKEN.search(rendered) is not None:
            raise LauncherRenderError()
        encoded = rendered.encode("utf-8", errors="strict")
        if not encoded or len(encoded) > MAX_INTEGRATION_LAUNCHER_BYTES or b"\x00" in encoded:
            raise LauncherRenderError()
        return encoded
    except LauncherRenderError:
        raise
    except Exception:
        raise LauncherRenderError() from None


__all__ = [
    "CaptureLauncherPlatform",
    "LauncherRenderError",
    "render_capture_launcher",
]
