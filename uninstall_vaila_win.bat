@echo off
REM Script: uninstall_vaila_win.bat
REM Description: Uninstalls the vailá - Multimodal Toolbox on Windows,
REM              removing the Conda environment, program files from
REM              C:\vaila_programs\vaila, and removing the vailá profile from
REM              Windows Terminal or desktop shortcut if it exists.
REM 
REM Usage:
REM   1. Right-click the script and select "Run as Administrator."
REM   2. Follow the on-screen instructions.
REM
REM Notes:
REM   - Ensure Conda is installed before running this script.
REM   - Administrative privileges are required for certain operations like
REM     removing files from system directories.
REM
REM Author: Prof. Dr. Paulo R. P. Santiago
REM Date: September 22, 2024
REM Version: 1.0
REM OS: Windows 11

@echo off
REM This script uninstalls the vailá environment on Windows

echo Starting vailá uninstallation on Windows...

REM Ensure the script is running as administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo This script requires administrative privileges. Please run as administrator.
    pause
    exit /b
)

REM Initialize Conda
call "%ProgramData%\Anaconda3\Scripts\activate.bat" base

REM Check if conda is available
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo Conda not found. Please make sure Anaconda or Miniconda is installed.
    pause
    exit /b
)

REM Deactivate any active environment
call conda deactivate

REM Remove the 'vaila' Conda environment if it exists
echo Checking for 'vaila' Conda environment...
conda env list | findstr /i "^vaila" >nul 2>&1
if %errorlevel% equ 0 (
    echo Removing 'vaila' Conda environment...
    conda remove --name vaila --all -y
    if %errorlevel% neq 0 (
        echo Failed to remove 'vaila' environment.
    ) else (
        echo 'vaila' environment removed successfully.
    )
) else (
    echo 'vaila' environment does not exist. Skipping environment removal.
)

REM Define installation path
set "VAILA_PROGRAM_PATH=C:\vaila_programs\vaila"

REM Remove the vailá directory
if exist "%VAILA_PROGRAM_PATH%" (
    echo Removing vailá directory at %VAILA_PROGRAM_PATH%...
    rmdir /S /Q "%VAILA_PROGRAM_PATH%"
    if %errorlevel% neq 0 (
        echo Failed to remove vailá directory.
    ) else (
        echo vailá directory removed successfully.
    )
) else (
    echo vailá directory not found at %VAILA_PROGRAM_PATH%. Skipping removal.
)

REM Remove the Windows Terminal profile if Windows Terminal is installed
if exist "%LocalAppData%\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe" (
    echo Removing vailá profile from Windows Terminal...
    set "WT_CONFIG_PATH=%LocalAppData%\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState\settings.json"
    set "WT_BACKUP_PATH=%LocalAppData%\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState\settings_backup_uninstall.json"
    
    REM Backup the current settings.json
    copy "%WT_CONFIG_PATH%" "%WT_BACKUP_PATH%" >nul
    
    REM Use PowerShell to remove the profile from the JSON
    powershell -Command ^
    $settings = Get-Content -Path '%WT_CONFIG_PATH%' -Raw | ConvertFrom-Json; ^
    $profileIndex = $settings.profiles.list.FindIndex({ $_.name -eq 'vailá' -or $_.name -eq 'vaila' }); ^
    if ($profileIndex -ge 0) { ^
        $settings.profiles.list.RemoveAt($profileIndex); ^
        $settings | ConvertTo-Json -Depth 100 | Set-Content -Path '%WT_CONFIG_PATH%'; ^
        Write-Host 'vailá profile removed from Windows Terminal.'
    } else { ^
        Write-Host 'vailá profile not found in Windows Terminal settings.'
    }
) else (
    echo Windows Terminal is not installed. Skipping profile removal.
)

REM Remove desktop shortcut if it exists
set "SHORTCUT_PATH=%USERPROFILE%\Desktop\vailá.lnk"
if exist "%SHORTCUT_PATH%" (
    echo Removing desktop shortcut...
    del "%SHORTCUT_PATH%" /F /Q
    if %errorlevel% neq 0 (
        echo Failed to remove desktop shortcut.
    ) else (
        echo Desktop shortcut removed successfully.
    )
) else (
    echo Desktop shortcut not found. Skipping removal.
)

echo Uninstallation completed. vailá has been removed from your system.
pause
