<#
    Script: win_launch_vaila.ps1
    Author: Prof. Dr. Paulo R. P. Santiago
    Date: November 27, 2025
    Update: November 27, 2025
    Description: Launches the vailá - Multimodal Toolbox using the Anaconda PowerShell Prompt on Windows 11.
                 The script activates the 'vaila' Conda environment and runs the vailá Python script.

    Usage:
      1. Open "Anaconda PowerShell Prompt" or PowerShell with Anaconda initialized.
      2. Run the script using: .\win_launch_vaila.ps1

    Features:
      - Activates the 'vaila' Conda environment.
      - Runs the vailá Python script from the new installation directory (AppData\Local\vaila).
      - Provides feedback to the user on progress and any issues encountered.

    Notes:
      - Ensure that Conda is installed and that the 'vaila' environment has been created during installation.
      - This script can be run from any PowerShell instance where Conda is initialized.

    Author: Prof. Dr. Paulo R. P. Santiago
    Date: November 25, 2024
    Version: 1.2
    OS: Windows 11
#>

# Inform the user about the process
Write-Host "Launching vailá toolbox..." -ForegroundColor Cyan
Write-Host "Checking if the 'vaila' Conda environment exists..." -ForegroundColor Cyan

# Attempt to activate the vaila Conda environment
try {
    conda activate vaila
    if (-not $?) {
        throw "Failed to activate the 'vaila' environment. Ensure it was installed correctly."
    }
    Write-Host "'vaila' Conda environment activated successfully." -ForegroundColor Green
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit..."
    exit
}

# Define the path to the vailá Python script
$vailaPath = "$env:LOCALAPPDATA\vaila"
$scriptPath = "$vailaPath\vaila.py"

# Check if vaila.py exists and run it
if (Test-Path $scriptPath) {
    Write-Host "Found vaila.py at $scriptPath" -ForegroundColor Green
    Write-Host "Navigating to the vailá directory and starting the toolbox..." -ForegroundColor Cyan
    Set-Location -Path $vailaPath
    try {
        python vaila.py
    } catch {
        Write-Host "Error running vaila.py: $_" -ForegroundColor Red
    }
} else {
    Write-Host "Error: vaila.py not found at $scriptPath" -ForegroundColor Red
    Write-Host "Ensure the installation was completed successfully." -ForegroundColor Yellow
    Read-Host "Press Enter to exit..."
    exit
}

# Inform the user that execution has completed
Write-Host "vailá toolbox execution complete." -ForegroundColor Green
Read-Host "Press Enter to exit..."

