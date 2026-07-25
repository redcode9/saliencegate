from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import PurePosixPath, PureWindowsPath

import pytest

from saliencegate.capture import CaptureProfile
from saliencegate.integrations import launcher_renderer
from saliencegate.integrations.launcher_renderer import (
    CaptureLauncherPlatform,
    LauncherRenderError,
    render_capture_launcher,
)

PROFILE = CaptureProfile.CODEX_HOOKS_V1
CONNECTION_ID = "sg-0123456789abcdef0123456789abcdef0123456789abcdef"


def test_posix_renderer_shell_quotes_every_operational_value() -> None:
    executable = PurePosixPath("/tmp/capture ' $HOME $(touch injected); & target")

    rendered = render_capture_launcher(
        executable=executable,
        profile=PROFILE,
        connection_id=CONNECTION_ID,
        platform=CaptureLauncherPlatform.POSIX,
    ).decode("utf-8")

    assert f"capture_executable={shlex.quote(str(executable))}\n" in rendered
    assert "capture_sleep=/bin/sleep\n" in rendered
    assert f"capture_profile={shlex.quote(PROFILE.value)}\n" in rendered
    assert f"capture_connection={shlex.quote(CONNECTION_ID)}\n" in rendered
    assert "__SALIENCEGATE_" not in rendered


@pytest.mark.skipif(os.name == "nt", reason="POSIX launcher execution contract")
def test_posix_rendered_launcher_treats_executable_metacharacters_as_data(tmp_path) -> None:
    captured = tmp_path / "captured"
    injected = tmp_path / "injected"
    executable = tmp_path / "capture ' $(touch injected) ; & target"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys\n"
        f"pathlib.Path({str(captured)!r}).write_text("
        "repr((sys.argv[1:], sys.stdin.buffer.read())), encoding='utf-8')\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    launcher = tmp_path / "launcher"
    launcher.write_bytes(
        render_capture_launcher(
            executable=executable,
            profile=PROFILE,
            connection_id=CONNECTION_ID,
            platform=CaptureLauncherPlatform.POSIX,
        )
    )
    launcher.chmod(0o700)

    completed = subprocess.run(
        (str(launcher),),
        input=b'{"provider":"payload"}',
        capture_output=True,
        check=False,
        cwd=tmp_path,
        timeout=5,
    )

    assert (completed.returncode, completed.stdout, completed.stderr) == (0, b"", b"")
    assert captured.read_text(encoding="utf-8") == repr(
        (
            ["--profile", PROFILE.value, "--connection", CONNECTION_ID],
            b'{"provider":"payload"}',
        )
    )
    assert not injected.exists()


def test_windows_renderer_preserves_safe_metacharacters_and_escapes_percent() -> None:
    executable = PureWindowsPath(r"C:\Program Files\Salience & Gate^(x)! 100%\capture-hook.exe")

    rendered = render_capture_launcher(
        executable=executable,
        profile=PROFILE,
        connection_id=CONNECTION_ID,
        platform=CaptureLauncherPlatform.WINDOWS,
    ).decode("utf-8")

    assert (
        'set "capture_executable='
        r"C:\Program Files\Salience & Gate^(x)! 100%%\capture-hook.exe"
        '"\n'
    ) in rendered
    assert (
        'set "capture_powershell='
        r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
        '"\n'
    ) in rendered
    assert f'set "capture_profile={PROFILE.value}"\n' in rendered
    assert f'set "capture_connection={CONNECTION_ID}"\n' in rendered
    assert "setlocal DisableDelayedExpansion" in rendered
    assert "__SALIENCEGATE_" not in rendered


@pytest.mark.parametrize(
    ("platform", "executable"),
    (
        (CaptureLauncherPlatform.POSIX, "relative/capture-hook"),
        (CaptureLauncherPlatform.POSIX, "/tmp/capture-hook\nmalicious"),
        (CaptureLauncherPlatform.WINDOWS, r"relative\capture-hook.exe"),
        (CaptureLauncherPlatform.WINDOWS, r"\rooted\capture-hook.exe"),
        (CaptureLauncherPlatform.WINDOWS, r"C:drive-relative.exe"),
        (CaptureLauncherPlatform.WINDOWS, 'C:\\capture"hook.exe'),
        (CaptureLauncherPlatform.WINDOWS, r"C:\capture|hook.exe"),
        (CaptureLauncherPlatform.WINDOWS, r"\\server\share\capture-hook.exe"),
        (CaptureLauncherPlatform.WINDOWS, r"\\?\C:\capture-hook.exe"),
        (
            CaptureLauncherPlatform.WINDOWS,
            r"\\?\GLOBALROOT\Device\HarddiskVolumeShadowCopy1\capture-hook.exe",
        ),
        (CaptureLauncherPlatform.WINDOWS, r"\\.\C:\capture-hook.exe"),
    ),
)
def test_renderer_rejects_nonabsolute_or_unsafe_executables(
    platform: CaptureLauncherPlatform,
    executable: str,
) -> None:
    with pytest.raises(LauncherRenderError, match=r"^capture launcher is invalid$"):
        render_capture_launcher(
            executable=executable,
            profile=PROFILE,
            connection_id=CONNECTION_ID,
            platform=platform,
        )


@pytest.mark.parametrize(
    "watchdog",
    (
        PureWindowsPath(r"C:\Windows\System32\watchdog.com"),
        PureWindowsPath(r"\\server\share\powershell.exe"),
        PureWindowsPath(r"\\?\C:\Windows\System32\powershell.exe"),
    ),
)
def test_windows_renderer_requires_a_drive_rooted_exe_watchdog(
    watchdog: PureWindowsPath,
) -> None:
    with pytest.raises(LauncherRenderError, match=r"^capture launcher is invalid$"):
        render_capture_launcher(
            executable=PureWindowsPath(r"C:\SalienceGate\capture-hook.exe"),
            profile=PROFILE,
            connection_id=CONNECTION_ID,
            platform=CaptureLauncherPlatform.WINDOWS,
            watchdog_executable=watchdog,
        )


@pytest.mark.parametrize(
    ("profile", "connection_id"),
    (
        (PROFILE.value, CONNECTION_ID),
        (PROFILE, "too-short"),
        (PROFILE, "sg-0123456789&command"),
        (PROFILE, "sg-0123456789\ncommand"),
    ),
)
def test_renderer_rejects_untyped_profile_and_invalid_connection_id(
    profile: CaptureProfile | str,
    connection_id: str,
) -> None:
    with pytest.raises(LauncherRenderError, match=r"^capture launcher is invalid$"):
        render_capture_launcher(
            executable="/usr/bin/saliencegate-capture-hook",
            profile=profile,  # type: ignore[arg-type]
            connection_id=connection_id,
            platform=CaptureLauncherPlatform.POSIX,
        )


@pytest.mark.parametrize("platform", ("posix", "windows", object()))
def test_renderer_requires_the_exact_platform_enum(platform: object) -> None:
    with pytest.raises(LauncherRenderError, match=r"^capture launcher is invalid$"):
        render_capture_launcher(
            executable="/usr/bin/saliencegate-capture-hook",
            profile=PROFILE,
            connection_id=CONNECTION_ID,
            platform=platform,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "leftover"))
def test_renderer_rejects_tampered_template_tokens(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    original = launcher_renderer._read_launcher_template(CaptureLauncherPlatform.POSIX)
    token = "__SALIENCEGATE_EXECUTABLE_SHELL__"
    if mutation == "missing":
        tampered = original.replace(token, "")
    elif mutation == "duplicate":
        tampered = f"{original}\n{token}\n"
    else:
        tampered = f"{original}\n__SALIENCEGATE_UNRECOGNIZED_TOKEN__\n"
    monkeypatch.setattr(
        launcher_renderer,
        "_read_launcher_template",
        lambda _platform: tampered,
    )

    with pytest.raises(LauncherRenderError, match=r"^capture launcher is invalid$"):
        render_capture_launcher(
            executable="/usr/bin/saliencegate-capture-hook",
            profile=PROFILE,
            connection_id=CONNECTION_ID,
            platform=CaptureLauncherPlatform.POSIX,
        )


@pytest.mark.parametrize(
    "token",
    (
        "__SALIENCEGATE_UNRECOGNIZED_TOKEN__",
        "__SALIENCEGATE_PROFILE_SHELL__",
    ),
)
def test_renderer_rejects_a_token_smuggled_through_an_operational_value(token: str) -> None:
    with pytest.raises(LauncherRenderError, match=r"^capture launcher is invalid$"):
        render_capture_launcher(
            executable=f"/tmp/{token}/capture-hook",
            profile=PROFILE,
            connection_id=CONNECTION_ID,
            platform=CaptureLauncherPlatform.POSIX,
        )
