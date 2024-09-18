# Initialize Conda in PowerShell
$condaInstallPath = "C:\ProgramData\Anaconda3"  # Adjust if Conda is installed elsewhere
$condaHookScript = Join-Path $condaInstallPath "shell\condabin\conda-hook.ps1"

if (Test-Path $condaHookScript) {
    & $condaHookScript
    conda activate vaila
} else {
    Write-Host "Conda initialization script not found at $condaHookScript"
    Read-Host "Press Enter to exit..."
    exit
}

# Run the vailá Python script
$scriptPath = "C:\vaila_programs\vaila\vaila.py"
if (Test-Path $scriptPath) {
    cd "C:\vaila_programs\vaila"
    python vaila.py
} else {
    Write-Host "vaila.py not found at $scriptPath"
}

# Pause to keep the terminal open
Read-Host "Press Enter to exit..."

