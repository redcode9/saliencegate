[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $SetupArguments
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$SalienceGateReleaseVersion = "0.2.0"
$SalienceGateReleaseBaseUrl = "https://github.com/redcode9/saliencegate/releases/download/v$SalienceGateReleaseVersion"
$SalienceGateUvVersion = "0.11.28"
$SalienceGatePythonVersion = "3.12"

function Stop-Installer {
    param([Parameter(Mandatory = $true)][string] $Message)
    [Console]::Error.WriteLine("saliencegate installer: {0}", $Message)
    exit 1
}

function Assert-AbsolutePath {
    param(
        [Parameter(Mandatory = $true)][string] $Path,
        [Parameter(Mandatory = $true)][string] $Label
    )
    $isDriveQualified = $Path -match "^[A-Za-z]:[\\/]"
    $isUncQualified = $Path -match "^[\\/]{2}[^\\/]+[\\/][^\\/]+(?:[\\/]|$)"
    if (-not ($isDriveQualified -or $isUncQualified)) {
        Stop-Installer "$Label must be an absolute path"
    }
    if ($Path -split "[\\/]" -contains "..") {
        Stop-Installer "$Label must not contain a parent traversal"
    }
}

function Ensure-InstallerDirectory {
    param(
        [Parameter(Mandatory = $true)][string] $Path,
        [Parameter(Mandatory = $true)][string] $Label
    )
    Assert-AbsolutePath $Path $Label
    if (Test-Path -LiteralPath $Path) {
        $item = Get-Item -LiteralPath $Path -Force
        if (-not $item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            Stop-Installer "$Label must be a non-linked directory"
        }
        return
    }
    $created = New-Item -ItemType Directory -Path $Path -Force
    if (-not $created.PSIsContainer) {
        Stop-Installer "could not create $Label"
    }
}

function Test-ExactUv {
    param([Parameter(Mandatory = $true)][string] $Path)
    try {
        Assert-AbsolutePath $Path "uv executable"
        $item = Get-Item -LiteralPath $Path -Force
        if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            return $false
        }
        $versionOutput = @(& $Path --version 2>$null)
        return (
            $LASTEXITCODE -eq 0 -and
            $versionOutput.Count -eq 1 -and
            $versionOutput[0] -ceq "uv $SalienceGateUvVersion"
        )
    }
    catch {
        return $false
    }
}

function Install-PrivateUv {
    param([Parameter(Mandatory = $true)][string] $BootstrapDirectory)
    $installerUri = (
        "https://releases.astral.sh/github/uv/releases/download/" +
        "$SalienceGateUvVersion/uv-installer.ps1"
    )
    $temporaryInstaller = Join-Path `
        ([IO.Path]::GetTempPath()) `
        ("saliencegate-uv-{0}.ps1" -f [Guid]::NewGuid().ToString("N"))
    $priorUnmanagedInstall = $env:UV_UNMANAGED_INSTALL
    $priorNoModifyPath = $env:UV_NO_MODIFY_PATH
    try {
        $null = Invoke-WebRequest `
            -UseBasicParsing `
            -Uri $installerUri `
            -OutFile $temporaryInstaller
        $downloaded = Get-Item -LiteralPath $temporaryInstaller -Force
        if ($downloaded.PSIsContainer -or $downloaded.Length -le 0) {
            Stop-Installer "the downloaded uv installer is invalid"
        }
        $env:UV_UNMANAGED_INSTALL = $BootstrapDirectory
        $env:UV_NO_MODIFY_PATH = "1"
        $powerShellExecutable = (Get-Process -Id $PID).Path
        $installerOutput = @(
            & $powerShellExecutable `
                -NoLogo `
                -NoProfile `
                -NonInteractive `
                -ExecutionPolicy Bypass `
                -File $temporaryInstaller 2>&1
        )
        $installerExitCode = $LASTEXITCODE
        foreach ($line in $installerOutput) {
            [Console]::Error.WriteLine([string] $line)
        }
        if ($installerExitCode -ne 0) {
            Stop-Installer "uv installation failed"
        }
    }
    finally {
        $env:UV_UNMANAGED_INSTALL = $priorUnmanagedInstall
        $env:UV_NO_MODIFY_PATH = $priorNoModifyPath
        Remove-Item -LiteralPath $temporaryInstaller -Force -ErrorAction SilentlyContinue
    }

    $uvExecutable = Join-Path $BootstrapDirectory "uv.exe"
    if (-not (Test-ExactUv $uvExecutable)) {
        Stop-Installer "the installed uv version is invalid"
    }
    return $uvExecutable
}

$installerHome = [Environment]::GetFolderPath([Environment+SpecialFolder]::UserProfile)
if ([string]::IsNullOrWhiteSpace($installerHome)) {
    Stop-Installer "the user profile directory is required"
}
$installerLocalData = $env:LOCALAPPDATA
if ([string]::IsNullOrWhiteSpace($installerLocalData)) {
    $installerLocalData = Join-Path $installerHome "AppData\Local"
}
$installerRoot = $env:SALIENCEGATE_INSTALL_ROOT
if ([string]::IsNullOrWhiteSpace($installerRoot)) {
    $installerRoot = Join-Path $installerLocalData "SalienceGate\runtime"
}
$installerBinDirectory = $env:SALIENCEGATE_INSTALL_BIN_DIR
if ([string]::IsNullOrWhiteSpace($installerBinDirectory)) {
    $installerBinDirectory = Join-Path $installerRoot "bin"
}
$installerToolDirectory = Join-Path $installerRoot "tools"
$installerPythonDirectory = Join-Path $installerRoot "python"
$installerBootstrapDirectory = Join-Path $installerRoot "bootstrap"

Ensure-InstallerDirectory $installerRoot "installation root"
Ensure-InstallerDirectory $installerBinDirectory "executable directory"
Ensure-InstallerDirectory $installerToolDirectory "tool directory"
Ensure-InstallerDirectory $installerPythonDirectory "Python directory"
Ensure-InstallerDirectory $installerBootstrapDirectory "bootstrap directory"

$installerTesting = $env:SALIENCEGATE_INSTALL_TESTING
if ([string]::IsNullOrEmpty($installerTesting)) {
    $installerTesting = "0"
}
if ($installerTesting -notin @("0", "1")) {
    Stop-Installer "the test mode value is invalid"
}
$installerTestPackage = $env:SALIENCEGATE_INSTALL_TEST_PACKAGE
$installerTestUv = $env:SALIENCEGATE_INSTALL_TEST_UV
if (
    (
        -not [string]::IsNullOrWhiteSpace($installerTestPackage) -or
        -not [string]::IsNullOrWhiteSpace($installerTestUv)
    ) -and
    $installerTesting -ne "1"
) {
    Stop-Installer "test overrides require explicit test mode"
}

if (-not [string]::IsNullOrWhiteSpace($installerTestPackage)) {
    Assert-AbsolutePath $installerTestPackage "test package"
    $testPackageItem = Get-Item -LiteralPath $installerTestPackage -Force
    if (
        $testPackageItem.PSIsContainer -or
        ($testPackageItem.Attributes -band [IO.FileAttributes]::ReparsePoint)
    ) {
        Stop-Installer "the test package must be a regular local file"
    }
    $installerPackage = $installerTestPackage
}
else {
    $installerPackage = "$SalienceGateReleaseBaseUrl/saliencegate-$SalienceGateReleaseVersion-py3-none-any.whl"
}

$installerUv = $null
if (-not [string]::IsNullOrWhiteSpace($installerTestUv)) {
    if (-not (Test-ExactUv $installerTestUv)) {
        Stop-Installer "the test uv executable is invalid"
    }
    $installerUv = $installerTestUv
}
else {
    $discoveredUv = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -ne $discoveredUv -and (Test-ExactUv $discoveredUv.Source)) {
        $installerUv = $discoveredUv.Source
    }
}
if ([string]::IsNullOrWhiteSpace($installerUv)) {
    $installerUv = Install-PrivateUv $installerBootstrapDirectory
}

$priorToolDirectory = $env:UV_TOOL_DIR
$priorToolBinDirectory = $env:UV_TOOL_BIN_DIR
$priorPythonInstallDirectory = $env:UV_PYTHON_INSTALL_DIR
try {
    $env:UV_TOOL_DIR = $installerToolDirectory
    $env:UV_TOOL_BIN_DIR = $installerBinDirectory
    $env:UV_PYTHON_INSTALL_DIR = $installerPythonDirectory
    & $installerUv tool install `
        --force `
        --python $SalienceGatePythonVersion `
        --managed-python `
        --no-config `
        --no-build `
        --no-sources `
        $installerPackage
    if ($LASTEXITCODE -ne 0) {
        Stop-Installer "SalienceGate installation failed"
    }
}
finally {
    $env:UV_TOOL_DIR = $priorToolDirectory
    $env:UV_TOOL_BIN_DIR = $priorToolBinDirectory
    $env:UV_PYTHON_INSTALL_DIR = $priorPythonInstallDirectory
}

$saliencegateExecutable = Join-Path $installerBinDirectory "saliencegate.exe"
Assert-AbsolutePath $saliencegateExecutable "SalienceGate executable"
if (-not (Test-Path -LiteralPath $saliencegateExecutable -PathType Leaf)) {
    Stop-Installer "the installed SalienceGate executable is unavailable"
}

$pathEntries = @($env:PATH -split [IO.Path]::PathSeparator)
if ($installerBinDirectory -notin $pathEntries) {
    try {
        $env:UV_TOOL_DIR = $installerToolDirectory
        $env:UV_TOOL_BIN_DIR = $installerBinDirectory
        & $installerUv tool update-shell --no-config
        if ($LASTEXITCODE -ne 0) {
            throw "PATH update failed"
        }
    }
    catch {
        [Console]::Error.WriteLine(
            "Add {0} to PATH to use saliencegate in a new terminal.",
            $installerBinDirectory
        )
    }
    finally {
        $env:UV_TOOL_DIR = $priorToolDirectory
        $env:UV_TOOL_BIN_DIR = $priorToolBinDirectory
    }
}

& $saliencegateExecutable setup @SetupArguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
