#!/bin/sh

set -eu

SALIENCEGATE_RELEASE_VERSION="0.2.0"
SALIENCEGATE_RELEASE_BASE_URL="https://github.com/redcode9/saliencegate/releases/download/v$SALIENCEGATE_RELEASE_VERSION"
SALIENCEGATE_UV_VERSION="0.11.28"
SALIENCEGATE_PYTHON_VERSION="3.12"

installer_error() {
    printf '%s\n' "saliencegate installer: $1" >&2
    exit 1
}

require_absolute_path() {
    case "$1" in
        /*) ;;
        *) installer_error "$2 must be an absolute path" ;;
    esac
    case "$1" in
        */../*|*/..|../*|..)
            installer_error "$2 must not contain a parent traversal"
            ;;
    esac
}

ensure_directory() {
    if [ -L "$1" ]; then
        installer_error "$2 must not be a symbolic link"
    fi
    if [ -e "$1" ] && [ ! -d "$1" ]; then
        installer_error "$2 must be a directory"
    fi
    mkdir -p "$1" || installer_error "could not create $2"
    if [ -L "$1" ] || [ ! -d "$1" ]; then
        installer_error "$2 is unavailable"
    fi
}

exact_uv() {
    [ -n "$1" ] || return 1
    require_absolute_path "$1" "uv executable"
    [ -f "$1" ] && [ -x "$1" ] && [ ! -L "$1" ] || return 1
    [ "$("$1" --version 2>/dev/null || true)" = "uv $SALIENCEGATE_UV_VERSION" ]
}

download_file() {
    download_target=$1
    download_url=$2
    curl_path=$(command -v curl 2>/dev/null || true)
    if [ -n "$curl_path" ]; then
        "$curl_path" \
            --proto '=https' \
            --tlsv1.2 \
            --location \
            --silent \
            --show-error \
            --fail \
            --output "$download_target" \
            "$download_url"
        return
    fi
    wget_path=$(command -v wget 2>/dev/null || true)
    if [ -n "$wget_path" ]; then
        "$wget_path" --quiet --output-document="$download_target" "$download_url"
        return
    fi
    installer_error "curl or wget is required to download uv"
}

bootstrap_uv() {
    bootstrap_directory=$1
    temporary_base=${TMPDIR:-/tmp}
    require_absolute_path "$temporary_base" "temporary directory"
    temporary_directory=$(
        mktemp -d "$temporary_base/saliencegate-installer.XXXXXXXX"
    ) || installer_error "could not create a temporary directory"
    uv_installer="$temporary_directory/uv-install.sh"
    cleanup_installer() {
        rm -f "$uv_installer"
        rmdir "$temporary_directory" 2>/dev/null || true
    }
    trap cleanup_installer EXIT HUP INT TERM

    uv_installer_url="https://releases.astral.sh/github/uv/releases/download/$SALIENCEGATE_UV_VERSION/uv-installer.sh"
    download_file "$uv_installer" "$uv_installer_url"
    [ -f "$uv_installer" ] && [ -s "$uv_installer" ] ||
        installer_error "the downloaded uv installer is invalid"
    UV_UNMANAGED_INSTALL="$bootstrap_directory" \
        UV_NO_MODIFY_PATH=1 \
        sh "$uv_installer" >&2 ||
        installer_error "uv installation failed"

    uv_executable="$bootstrap_directory/uv"
    exact_uv "$uv_executable" ||
        installer_error "the installed uv version is invalid"
    cleanup_installer
    trap - EXIT HUP INT TERM
    printf '%s\n' "$uv_executable"
}

umask 077

installer_home=${HOME:-}
[ -n "$installer_home" ] || installer_error "HOME is required"
require_absolute_path "$installer_home" "HOME"

installer_data_home=${XDG_DATA_HOME:-"$installer_home/.local/share"}
installer_bin_directory=${SALIENCEGATE_INSTALL_BIN_DIR:-${XDG_BIN_HOME:-"$installer_home/.local/bin"}}
installer_root=${SALIENCEGATE_INSTALL_ROOT:-"$installer_data_home/saliencegate/runtime"}
installer_tool_directory="$installer_root/tools"
installer_python_directory="$installer_root/python"
installer_bootstrap_directory="$installer_root/bootstrap"

require_absolute_path "$installer_data_home" "data directory"
require_absolute_path "$installer_bin_directory" "executable directory"
require_absolute_path "$installer_root" "installation root"
for installer_directory in \
    "$installer_root" \
    "$installer_tool_directory" \
    "$installer_python_directory" \
    "$installer_bootstrap_directory" \
    "$installer_bin_directory"
do
    ensure_directory "$installer_directory" "installation directory"
done

installer_testing=${SALIENCEGATE_INSTALL_TESTING:-0}
installer_test_package=${SALIENCEGATE_INSTALL_TEST_PACKAGE:-}
installer_test_uv=${SALIENCEGATE_INSTALL_TEST_UV:-}
if [ "$installer_testing" != "0" ] && [ "$installer_testing" != "1" ]; then
    installer_error "the test mode value is invalid"
fi
if [ -n "$installer_test_package" ] || [ -n "$installer_test_uv" ]; then
    [ "$installer_testing" = "1" ] ||
        installer_error "test overrides require explicit test mode"
fi

if [ -n "$installer_test_package" ]; then
    require_absolute_path "$installer_test_package" "test package"
    [ -f "$installer_test_package" ] && [ ! -L "$installer_test_package" ] ||
        installer_error "the test package must be a regular local file"
    installer_package=$installer_test_package
else
    installer_package="$SALIENCEGATE_RELEASE_BASE_URL/saliencegate-$SALIENCEGATE_RELEASE_VERSION-py3-none-any.whl"
fi

installer_uv=
if [ -n "$installer_test_uv" ]; then
    exact_uv "$installer_test_uv" ||
        installer_error "the test uv executable is invalid"
    installer_uv=$installer_test_uv
else
    discovered_uv=$(command -v uv 2>/dev/null || true)
    case "$discovered_uv" in
        /*)
            if exact_uv "$discovered_uv"; then
                installer_uv=$discovered_uv
            fi
            ;;
    esac
fi
if [ -z "$installer_uv" ]; then
    installer_uv=$(bootstrap_uv "$installer_bootstrap_directory")
fi

UV_TOOL_DIR="$installer_tool_directory" \
    UV_TOOL_BIN_DIR="$installer_bin_directory" \
    UV_PYTHON_INSTALL_DIR="$installer_python_directory" \
    "$installer_uv" tool install \
        --force \
        --python "$SALIENCEGATE_PYTHON_VERSION" \
        --managed-python \
        --no-config \
        --no-build \
        --no-sources \
        "$installer_package" ||
    installer_error "SalienceGate installation failed"

saliencegate_executable="$installer_bin_directory/saliencegate"
require_absolute_path "$saliencegate_executable" "SalienceGate executable"
[ -f "$saliencegate_executable" ] && [ -x "$saliencegate_executable" ] ||
    installer_error "the installed SalienceGate executable is unavailable"

case ":${PATH:-}:" in
    *":$installer_bin_directory:"*) ;;
    *)
        UV_TOOL_DIR="$installer_tool_directory" \
            UV_TOOL_BIN_DIR="$installer_bin_directory" \
            "$installer_uv" tool update-shell --no-config >&2 ||
            printf '%s\n' \
                "Add $installer_bin_directory to PATH to use saliencegate in a new terminal." >&2
        ;;
esac

if [ "$#" -eq 0 ]; then
    if ! (: </dev/tty) 2>/dev/null; then
        installer_error "interactive setup requires a terminal"
    fi
    "$saliencegate_executable" setup </dev/tty
else
    "$saliencegate_executable" setup "$@"
fi
