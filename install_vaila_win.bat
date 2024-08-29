@echo off
REM This script installs the vailá environment on Windows

echo "Starting vailá installation on Windows..."

REM Check if winget is available
if not exist "%SystemRoot%\System32\winget.exe" (
    echo "Winget not found. Please install Winget manually."
    exit /b
)

REM Install FFmpeg using winget
echo "Installing FFmpeg..."
winget install ffmpeg

REM Create the Conda environment using the YAML file
echo "Creating Conda environment..."
conda env create -f yaml_for_conda_env\vaila_win.yaml

REM Configure vailá in the Windows Terminal JSON
echo "Configuring Windows Terminal..."
REM Path to the Windows Terminal settings JSON (adjust as needed)
set WT_CONFIG_PATH=%LocalAppData%\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState\settings.json

REM Define the vailá profile JSON entry
set VAILA_PROFILE="{
    \"colorScheme\": \"Vintage\",
    \"commandline\": \"pwsh.exe -ExecutionPolicy ByPass -NoExit -Command \\\"& 'C:\\ProgramData\\anaconda3\\shell\\condabin\\conda-hook.ps1' ; conda activate 'vaila' ; python 'vaila.py' \\\"\",
    \"guid\": \"{17ce5bfe-17ed-5f3a-ab15-5cd5baafed5b}\",
    \"hidden\": false,
    \"icon\": \"C:\\vaila_programs\\vaila\\vaila\\images\\vaila_ico.png\",
    \"name\": \"vailá\",
    \"startingDirectory\": \"C:\\vaila_programs\\vaila\"
}"

REM Backup the current settings.json
copy %WT_CONFIG_PATH% %WT_CONFIG_PATH%.bak

REM Append the vailá profile to the profiles section in the settings.json
powershell -Command "(Get-Content -Raw %WT_CONFIG_PATH%).Replace('\"profiles\": {', '\"profiles\": {\n    \"list\": [\n        %VAILA_PROFILE%,') | Set-Content %WT_CONFIG_PATH%"

echo "Installation and configuration completed successfully!"
pause
