<#
    Script: win_launch_vaila.ps1
    Description: Launches the vailá - Multimodal Toolbox using the Anaconda PowerShell Prompt on Windows 11.
                 The script activates the 'vaila' Conda environment and runs the vailá Python script.

    Usage:
      1. Open "Anaconda PowerShell Prompt" with administrator privileges.
      2. Run the script using: .\win_launch_vaila.ps1

    Features:
      - Activates the 'vaila' Conda environment.
      - Runs the vailá Python script from the new installation directory (AppData\Local\vaila).
      - Provides feedback to the user on progress and any issues encountered.

    Notes:
      - Ensure that Conda is installed and that the 'vaila' environment has been created during installation.
      - This script should be run from the Anaconda PowerShell Prompt.

    Author: Prof. Dr. Paulo R. P. Santiago
    Date: September 23, 2024
    Version: 1.1
    OS: Windows 11
#>

# Print message to user
Write-Host "Starting vailá toolbox launch..." -ForegroundColor Cyan
Write-Host "Checking if 'vaila' Conda environment exists..." -ForegroundColor Cyan

# Check if the vaila environment exists and activate it
conda activate vaila

# Check if activation succeeded
if ($?) {
    Write-Host "'vaila' Conda environment activated successfully." -ForegroundColor Green

    # Run the vailá Python script
    $scriptPath = "$env:LOCALAPPDATA\vaila\vaila.py"
    if (Test-Path $scriptPath) {
        Write-Host "Found vaila.py at $scriptPath" -ForegroundColor Green
        Write-Host "Navigating to the vaila directory and starting the vailá toolbox..." -ForegroundColor Cyan
        cd "$env:LOCALAPPDATA\vaila"
        python vaila.py
    } else {
        Write-Host "Error: vaila.py not found at $scriptPath" -ForegroundColor Red
        Read-Host "Press Enter to exit..."
        exit
    }
} else {
    Write-Host "Error: Failed to activate the 'vaila' environment." -ForegroundColor Red
    Read-Host "Press Enter to exit..."
    exit
}

# Pause to keep the terminal open after execution
Write-Host "vailá toolbox execution complete." -ForegroundColor Green
Read-Host "Press Enter to exit..."

