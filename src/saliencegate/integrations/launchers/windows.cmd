@echo off
setlocal DisableDelayedExpansion
call :capture_main >nul 2>nul
exit /b 0

:capture_main
rem Render percent signs as %% in the three prevalidated operational values.
rem Provider arguments are intentionally ignored.
set "capture_executable=__SALIENCEGATE_EXECUTABLE_BATCH__"
set "capture_powershell=__SALIENCEGATE_WATCHDOG_BATCH__"
set "capture_profile=__SALIENCEGATE_PROFILE_BATCH__"
set "capture_connection=__SALIENCEGATE_CONNECTION_BATCH__"

if not defined capture_executable exit /b 0
if not "%capture_executable:~1,2%"==":\" exit /b 0
if not exist "%capture_executable%" exit /b 0

for %%I in ("%capture_executable%") do set "capture_full=%%~fI"
for %%I in ("%capture_executable%") do set "capture_attributes=%%~aI"
for %%I in ("%capture_executable%") do set "capture_extension=%%~xI"
if /I not "%capture_full%"=="%capture_executable%" exit /b 0
if /I "%capture_attributes:~0,1%"=="d" exit /b 0
if /I not "%capture_extension%"==".exe" if /I not "%capture_extension%"==".com" exit /b 0

if not defined capture_powershell exit /b 0
if not "%capture_powershell:~1,2%"==":\" exit /b 0
if not exist "%capture_powershell%" exit /b 0
for %%I in ("%capture_powershell%") do set "capture_powershell_full=%%~fI"
for %%I in ("%capture_powershell%") do set "capture_powershell_attributes=%%~aI"
for %%I in ("%capture_powershell%") do set "capture_powershell_extension=%%~xI"
if /I not "%capture_powershell_full%"=="%capture_powershell%" exit /b 0
if /I "%capture_powershell_attributes:~0,1%"=="d" exit /b 0
if /I not "%capture_powershell_extension%"==".exe" exit /b 0

if "%capture_profile%"=="codex-hooks/v1" goto capture_profile_ok
if "%capture_profile%"=="claude-code-hooks/v1" goto capture_profile_ok
if "%capture_profile%"=="opencode-plugin/v1" goto capture_profile_ok
if "%capture_profile%"=="pi-extension/v1" goto capture_profile_ok
exit /b 0

:capture_profile_ok
rem Provider credentials are outside the capture protocol. Remove them before
rem PowerShell creates the capture process so the child cannot read them.
set "ANTHROPIC_API_KEY="
set "AZURE_OPENAI_API_KEY="
set "OPENAI_API_KEY="
set "OPENAI_ORGANIZATION="
set "OPENAI_ORG_ID="
set "OPENAI_PROJECT="
set "OPENAI_PROJECT_ID="
set "SALIENCEGATE_CAPTURE_EXECUTABLE=%capture_executable%"
set "SALIENCEGATE_CAPTURE_PROFILE=%capture_profile%"
set "SALIENCEGATE_CAPTURE_CONNECTION=%capture_connection%"

"%capture_powershell%" -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "$ErrorActionPreference='SilentlyContinue'; $exe=[Environment]::GetEnvironmentVariable('SALIENCEGATE_CAPTURE_EXECUTABLE','Process'); $profile=[Environment]::GetEnvironmentVariable('SALIENCEGATE_CAPTURE_PROFILE','Process'); $connection=[Environment]::GetEnvironmentVariable('SALIENCEGATE_CAPTURE_CONNECTION','Process'); if([String]::IsNullOrEmpty($exe) -or [String]::IsNullOrEmpty($profile) -or [String]::IsNullOrEmpty($connection)){exit 0}; $info=New-Object System.Diagnostics.ProcessStartInfo; $info.FileName=$exe; $info.Arguments='--profile "'+$profile+'" --connection "'+$connection+'"'; $info.UseShellExecute=$false; $info.CreateNoWindow=$true; $info.RedirectStandardInput=$true; $process=New-Object System.Diagnostics.Process; $process.StartInfo=$info; [Console]::InputEncoding=[System.Text.UTF8Encoding]::new($false); if(-not $process.Start()){exit 0}; $clock=[Diagnostics.Stopwatch]::StartNew(); $copy=[Console]::OpenStandardInput().CopyToAsync($process.StandardInput.BaseStream); while((-not $copy.IsCompleted) -and (-not $process.HasExited) -and $clock.ElapsedMilliseconds -lt 2000){Start-Sleep -Milliseconds 5}; try{$process.StandardInput.Close()}catch{}; $remaining=[Math]::Max(0,2000-$clock.ElapsedMilliseconds); if((-not $process.HasExited) -and (-not $process.WaitForExit([int]$remaining))){try{$taskkill=[IO.Path]::Combine([Environment]::SystemDirectory,'taskkill.exe'); & $taskkill /PID $process.Id /T /F}catch{try{$process.Kill()}catch{}}; try{$process.WaitForExit()}catch{}}; exit 0"
exit /b 0
