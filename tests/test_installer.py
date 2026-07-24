from __future__ import annotations

import errno
import os
import re
import select
import signal
import stat
import subprocess
import time
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POSIX_INSTALLER = ROOT / "scripts" / "install.sh"
POWERSHELL_INSTALLER = ROOT / "scripts" / "install.ps1"
UV_VERSION = "0.11.28"


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)


def _fake_uv(path: Path) -> None:
    _write_executable(
        path,
        f"""#!/bin/sh
set -eu
if [ "$1" = "--version" ]; then
    printf '%s\\n' 'uv {UV_VERSION}'
    exit 0
fi
if [ "$1" = "tool" ] && [ "$2" = "update-shell" ]; then
    printf '%s\\n' "$@" > "$SALIENCEGATE_FAKE_UV_UPDATE_ARGUMENTS"
    exit 0
fi
printf '%s\\n' "$UV_TOOL_DIR" > "$SALIENCEGATE_FAKE_UV_TOOL_DIR"
printf '%s\\n' "$UV_TOOL_BIN_DIR" > "$SALIENCEGATE_FAKE_UV_BIN_DIR"
printf '%s\\n' "$UV_PYTHON_INSTALL_DIR" > "$SALIENCEGATE_FAKE_UV_PYTHON_DIR"
printf '%s\\n' "$@" > "$SALIENCEGATE_FAKE_UV_ARGUMENTS"
install -d -m 700 "$UV_TOOL_BIN_DIR"
cp "$SALIENCEGATE_FAKE_SETUP_SOURCE" "$UV_TOOL_BIN_DIR/saliencegate"
chmod 700 "$UV_TOOL_BIN_DIR/saliencegate"
""",
    )


def _fake_setup(path: Path) -> None:
    _write_executable(
        path,
        """#!/bin/sh
set -eu
printf '%s\\n' "$0" > "$SALIENCEGATE_FAKE_SETUP_PATH"
printf '%s\\n' "$@" > "$SALIENCEGATE_FAKE_SETUP_ARGUMENTS"
if [ "${SALIENCEGATE_FAKE_SETUP_READ_STDIN:-0}" = "1" ]; then
    IFS= read -r setup_input
    printf '%s\\n' "$setup_input" > "$SALIENCEGATE_FAKE_SETUP_INPUT"
fi
""",
    )


def _installer_environment(tmp_path: Path) -> tuple[dict[str, str], dict[str, Path]]:
    home = tmp_path / "home"
    fake_bin = tmp_path / "fake-bin"
    install_root = tmp_path / "runtime"
    command_bin = tmp_path / "commands"
    for directory in (home, fake_bin):
        directory.mkdir(mode=0o700)
    setup_source = tmp_path / "setup-source"
    _fake_setup(setup_source)
    paths = {
        "uv_arguments": tmp_path / "uv-arguments",
        "uv_update_arguments": tmp_path / "uv-update-arguments",
        "uv_tool_dir": tmp_path / "uv-tool-dir",
        "uv_bin_dir": tmp_path / "uv-bin-dir",
        "uv_python_dir": tmp_path / "uv-python-dir",
        "setup_path": tmp_path / "setup-path",
        "setup_arguments": tmp_path / "setup-arguments",
        "setup_input": tmp_path / "setup-input",
        "setup_source": setup_source,
        "install_root": install_root,
        "command_bin": command_bin,
        "fake_bin": fake_bin,
    }
    environment = {
        "HOME": str(home),
        "PATH": os.pathsep.join((str(fake_bin), "/usr/bin", "/bin")),
        "SALIENCEGATE_INSTALL_ROOT": str(install_root),
        "SALIENCEGATE_INSTALL_BIN_DIR": str(command_bin),
        "SALIENCEGATE_FAKE_UV_ARGUMENTS": str(paths["uv_arguments"]),
        "SALIENCEGATE_FAKE_UV_UPDATE_ARGUMENTS": str(paths["uv_update_arguments"]),
        "SALIENCEGATE_FAKE_UV_TOOL_DIR": str(paths["uv_tool_dir"]),
        "SALIENCEGATE_FAKE_UV_BIN_DIR": str(paths["uv_bin_dir"]),
        "SALIENCEGATE_FAKE_UV_PYTHON_DIR": str(paths["uv_python_dir"]),
        "SALIENCEGATE_FAKE_SETUP_PATH": str(paths["setup_path"]),
        "SALIENCEGATE_FAKE_SETUP_ARGUMENTS": str(paths["setup_arguments"]),
        "SALIENCEGATE_FAKE_SETUP_INPUT": str(paths["setup_input"]),
        "SALIENCEGATE_FAKE_SETUP_SOURCE": str(setup_source),
    }
    return environment, paths


def _run_piped_installer_with_tty(
    *,
    environment: dict[str, str],
    cwd: Path,
    terminal_input: bytes,
) -> tuple[int, str]:
    pty = pytest.importorskip("pty", reason="POSIX pseudo-terminal contract")
    pid, master_fd = pty.fork()
    if pid == 0:
        try:
            os.chdir(cwd)
            os.execve(
                "/bin/sh",
                (
                    "/bin/sh",
                    "-c",
                    '/bin/cat "$SALIENCEGATE_FAKE_INSTALLER" | /bin/sh',
                ),
                environment,
            )
        except BaseException:
            os._exit(127)

    output = bytearray()
    child_status: int | None = None
    deadline = time.monotonic() + 20
    try:
        os.write(master_fd, terminal_input)
        while time.monotonic() < deadline:
            readable, _, _ = select.select((master_fd,), (), (), 0.1)
            if readable:
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError as error:
                    if error.errno != errno.EIO:
                        raise
                else:
                    output.extend(chunk)
            waited_pid, status = os.waitpid(pid, os.WNOHANG)
            if waited_pid == pid:
                child_status = status
                break
        if child_status is None:
            os.kill(pid, signal.SIGKILL)
            _, child_status = os.waitpid(pid, 0)
            pytest.fail("piped installer did not finish")
    finally:
        os.close(master_fd)

    return os.waitstatus_to_exitcode(child_status), output.decode(
        "utf-8",
        errors="replace",
    )


@pytest.mark.skipif(os.name != "posix", reason="POSIX installer contract")
def test_posix_installer_uses_exact_persistent_release_and_absolute_setup(
    tmp_path: Path,
) -> None:
    environment, paths = _installer_environment(tmp_path)
    _fake_uv(paths["fake_bin"] / "uv")
    network_canary = paths["fake_bin"] / "curl"
    _write_executable(network_canary, "#!/bin/sh\nexit 91\n")

    result = subprocess.run(
        (
            "sh",
            str(POSIX_INSTALLER),
            "--provider",
            "all",
            "--scope",
            "global",
            "--yes",
        ),
        env=environment,
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    assert paths["uv_arguments"].read_text(encoding="utf-8").splitlines() == [
        "tool",
        "install",
        "--force",
        "--python",
        "3.12",
        "--managed-python",
        "--no-config",
        "--no-build",
        "--no-sources",
        (
            "https://github.com/redcode9/saliencegate/releases/download/"
            f"v{version}/saliencegate-{version}-py3-none-any.whl"
        ),
    ]
    assert paths["uv_tool_dir"].read_text(encoding="utf-8").strip() == str(
        paths["install_root"] / "tools"
    )
    assert paths["uv_bin_dir"].read_text(encoding="utf-8").strip() == str(paths["command_bin"])
    assert paths["uv_python_dir"].read_text(encoding="utf-8").strip() == str(
        paths["install_root"] / "python"
    )
    assert paths["setup_path"].read_text(encoding="utf-8").strip() == str(
        paths["command_bin"] / "saliencegate"
    )
    assert paths["uv_update_arguments"].read_text(encoding="utf-8").splitlines() == [
        "tool",
        "update-shell",
        "--no-config",
    ]
    assert paths["setup_arguments"].read_text(encoding="utf-8").splitlines() == [
        "setup",
        "--provider",
        "all",
        "--scope",
        "global",
        "--yes",
    ]


@pytest.mark.skipif(os.name != "posix", reason="POSIX installer contract")
def test_posix_one_line_install_reads_interactive_setup_from_tty(
    tmp_path: Path,
) -> None:
    environment, paths = _installer_environment(tmp_path)
    _fake_uv(paths["fake_bin"] / "uv")
    _write_executable(paths["fake_bin"] / "curl", "#!/bin/sh\nexit 91\n")
    environment.update(
        {
            "SALIENCEGATE_FAKE_INSTALLER": str(POSIX_INSTALLER),
            "SALIENCEGATE_FAKE_SETUP_READ_STDIN": "1",
        }
    )

    returncode, output = _run_piped_installer_with_tty(
        environment=environment,
        cwd=tmp_path,
        terminal_input=b"interactive-selection\n",
    )

    assert returncode == 0, output
    assert paths["setup_arguments"].read_text(encoding="utf-8").splitlines() == ["setup"]
    assert paths["setup_input"].read_text(encoding="utf-8").strip() == ("interactive-selection")


@pytest.mark.skipif(os.name != "posix", reason="POSIX installer contract")
def test_posix_installer_accepts_only_an_explicit_test_local_artifact(
    tmp_path: Path,
) -> None:
    environment, paths = _installer_environment(tmp_path)
    fake_uv = paths["fake_bin"] / "private-uv"
    _fake_uv(fake_uv)
    wheel = tmp_path / "saliencegate-test.whl"
    wheel.write_bytes(b"offline-test-artifact")
    environment.update(
        {
            "SALIENCEGATE_INSTALL_TESTING": "1",
            "SALIENCEGATE_INSTALL_TEST_PACKAGE": str(wheel),
            "SALIENCEGATE_INSTALL_TEST_UV": str(fake_uv),
        }
    )

    result = subprocess.run(
        ("sh", str(POSIX_INSTALLER), "--install-only", "--yes"),
        env=environment,
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert paths["uv_arguments"].read_text(encoding="utf-8").splitlines()[-1] == str(wheel)
    rejected_environment = dict(environment)
    rejected_environment["SALIENCEGATE_INSTALL_TESTING"] = "0"
    paths["uv_arguments"].unlink()
    rejected = subprocess.run(
        ("sh", str(POSIX_INSTALLER), "--install-only", "--yes"),
        env=rejected_environment,
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert rejected.returncode != 0
    assert "test overrides require explicit test mode" in rejected.stderr
    assert not paths["uv_arguments"].exists()

    linked_uv = paths["fake_bin"] / "linked-uv"
    linked_uv.symlink_to(fake_uv)
    linked_environment = dict(environment)
    linked_environment["SALIENCEGATE_INSTALL_TEST_UV"] = str(linked_uv)
    linked = subprocess.run(
        ("sh", str(POSIX_INSTALLER), "--install-only", "--yes"),
        env=linked_environment,
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert linked.returncode != 0
    assert "test uv executable is invalid" in linked.stderr
    assert not paths["uv_arguments"].exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX installer contract")
def test_posix_installer_bootstraps_the_pinned_uv_without_network_in_test(
    tmp_path: Path,
) -> None:
    environment, paths = _installer_environment(tmp_path)
    uv_payload = tmp_path / "uv-payload"
    _fake_uv(uv_payload)
    uv_bootstrap = tmp_path / "uv-bootstrap.sh"
    _write_executable(
        uv_bootstrap,
        """#!/bin/sh
set -eu
printf '%s\\n' 'uv bootstrap output'
install -d -m 700 "$UV_UNMANAGED_INSTALL"
cp "$SALIENCEGATE_FAKE_UV_PAYLOAD" "$UV_UNMANAGED_INSTALL/uv"
chmod 700 "$UV_UNMANAGED_INSTALL/uv"
""",
    )
    curl_log = tmp_path / "curl-log"
    _write_executable(
        paths["fake_bin"] / "curl",
        """#!/bin/sh
set -eu
target=
url=
while [ "$#" -gt 0 ]; do
    case "$1" in
        --output)
            shift
            target=$1
            ;;
        https://*)
            url=$1
            ;;
    esac
    shift
done
printf '%s\\n' "$url" > "$SALIENCEGATE_FAKE_CURL_LOG"
cp "$SALIENCEGATE_FAKE_UV_BOOTSTRAP" "$target"
""",
    )
    environment.update(
        {
            "SALIENCEGATE_FAKE_UV_PAYLOAD": str(uv_payload),
            "SALIENCEGATE_FAKE_UV_BOOTSTRAP": str(uv_bootstrap),
            "SALIENCEGATE_FAKE_CURL_LOG": str(curl_log),
        }
    )

    result = subprocess.run(
        ("sh", str(POSIX_INSTALLER), "--install-only", "--yes"),
        env=environment,
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert "uv bootstrap output" in result.stderr
    assert curl_log.read_text(encoding="utf-8").strip() == (
        f"https://releases.astral.sh/github/uv/releases/download/{UV_VERSION}/uv-installer.sh"
    )
    assert (paths["install_root"] / "bootstrap" / "uv").is_file()


def test_installers_freeze_versions_scope_and_non_privileged_contract() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = project["project"]["version"]
    posix = POSIX_INSTALLER.read_text(encoding="utf-8")
    powershell = POWERSHELL_INSTALLER.read_text(encoding="utf-8")

    assert POSIX_INSTALLER.stat().st_mode & stat.S_IXUSR
    assert re.search(
        rf'^SALIENCEGATE_RELEASE_VERSION="{re.escape(version)}"$',
        posix,
        re.MULTILINE,
    )
    assert f'$SalienceGateReleaseVersion = "{version}"' in powershell
    assert "IsPathFullyQualified" not in powershell
    assert (
        '"$SalienceGateUvVersion/uv-installer.ps1"' in powershell
        and "https://releases.astral.sh/github/uv/releases/download/" in powershell
    )
    assert "$installerOutput = @(" in powershell
    assert "$installerExitCode = $LASTEXITCODE" in powershell
    assert "[Console]::Error.WriteLine([string] $line)" in powershell
    for source in (posix, powershell):
        assert UV_VERSION in source
        assert "https://github.com/redcode9/saliencegate/releases/download/" in source
        assert "py3-none-any.whl" in source
        assert "managed-python" in source
        assert "no-config" in source
        assert "no-build" in source
        assert "UV_TOOL_DIR" in source
        assert "UV_TOOL_BIN_DIR" in source
        assert "tool update-shell" in source
        assert "SALIENCEGATE_INSTALL_TESTING" in source
        assert "SALIENCEGATE_INSTALL_TEST_PACKAGE" in source
        assert "SALIENCEGATE_INSTALL_TEST_UV" in source
        assert "sudo" not in source.casefold()
        assert "runas" not in source.casefold()
    assert '"$saliencegate_executable" setup "$@"' in posix
    assert "& $saliencegateExecutable setup @SetupArguments" in powershell
