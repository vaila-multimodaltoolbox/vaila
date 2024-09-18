@echo off
REM This script installs or updates the vailá environment on Windows

echo Starting vailá installation on Windows...

REM Ensure the script is running as administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo This script requires administrative privileges. Please run as administrator.
    pause
    exit /b
)

REM Check if winget is available in the system path
where winget >nul 2>&1
if %errorlevel% neq 0 (
    echo Winget not found. Please install Winget manually.
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

REM Check if the "vaila" environment already exists
echo Checking if vailá Conda environment exists...
conda env list | findstr /i "^vaila" >nul 2>&1
if %errorlevel% equ 0 (
    echo Conda environment 'vaila' already exists. Updating it...
    REM Update the existing environment
    conda env update -n vaila -f yaml_for_conda_env\\vaila_win.yaml --prune
    if %errorlevel% neq 0 (
        echo Failed to update 'vaila' environment.
        pause
        exit /b
    ) else (
        echo 'vaila' environment updated successfully.
    )
) else (
    REM Create the environment if it does not exist
    echo Creating Conda environment from vaila_win.yaml...
    conda env create -f yaml_for_conda_env\\vaila_win.yaml
    if %errorlevel% neq 0 (
        echo Failed to create 'vaila' environment.
        pause
        exit /b
    ) else (
        echo 'vaila' environment created successfully.
    )
)

REM Install FFmpeg using winget
echo Installing FFmpeg...
winget install --id Gyan.FFmpeg -e --silent

REM Check if Windows Terminal is installed
if not exist "%LocalAppData%\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe" (
    echo Windows Terminal is not installed. Installing via Microsoft Store...
    winget install --id Microsoft.WindowsTerminal -e
    if %errorlevel% neq 0 (
        echo Failed to install Windows Terminal. Please install it manually from the Microsoft Store.
        pause
        exit /b
    )
)

REM Configure vailá in the Windows Terminal JSON
echo Configuring Windows Terminal...

REM Path to the Windows Terminal settings JSON
set "WT_CONFIG_PATH=%LocalAppData%\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState\settings.json"

REM Define the vailá profile JSON entry
set "VAILA_PROFILE={
    \"colorScheme\": \"Vintage\",
    \"commandline\": \"pwsh.exe -ExecutionPolicy ByPass -NoExit -Command '& \"^%ProgramData^%\\Anaconda3\\shell\\condabin\\conda-hook.ps1\" ; conda activate \"vaila\" ; python \"^%USERPROFILE^%\\vaila\\vaila.py\"'\",
    \"guid\": \"{17ce5bfe-17ed-5f3a-ab15-5cd5baafed5b}\",
    \"hidden\": false,
    \"icon\": \"^%USERPROFILE^%\\vaila\\docs\\images\\vaila_ico.png\",
    \"name\": \"vailá\",
    \"startingDirectory\": \"^%USERPROFILE^%\\vaila\"
}"

REM Backup the current settings.json
copy "%WT_CONFIG_PATH%" "%WT_CONFIG_PATH%.bak" >nul

REM Use PowerShell to insert the profile into the JSON
powershell -Command "
$settings = Get-Content -Path '%WT_CONFIG_PATH%' -Raw | ConvertFrom-Json;
$profile = %VAILA_PROFILE%;
$profileJson = $profile | ConvertFrom-Json;
if (-not $settings.profiles.list) {
    $settings.profiles.list = @();
}
$settings.profiles.list += $profileJson;
$settings | ConvertTo-Json -Depth 100 | Set-Content -Path '%WT_CONFIG_PATH%'
"

REM Open settings.json in Notepad for verification
echo Opening Windows Terminal settings.json in Notepad for verification...
notepad "%WT_CONFIG_PATH%"

echo Installation and configuration completed successfully!
pause

