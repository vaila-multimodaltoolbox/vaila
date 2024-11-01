<#
    Script: install_vaila_win.ps1
    Description: Installs or updates the vaila - Multimodal Toolbox on Windows 11,
                 setting up the Conda environment, copying program files to
                 AppData\Local\vaila, installing FFmpeg, configuring Windows
                 Terminal, and setting up oh-my-posh for PowerShell.
#>

# Define installation path in AppData\Local
$vailaProgramPath = "$env:LOCALAPPDATA\vaila"
$sourcePath = (Get-Location)

# Ensure the directory exists
If (-Not (Test-Path $vailaProgramPath)) {
    Write-Output "Creating directory $vailaProgramPath..."
    New-Item -ItemType Directory -Force -Path $vailaProgramPath
}

# Copy the vaila program files to AppData\Local\vaila
Write-Output "Copying vaila program files from $sourcePath to $vailaProgramPath..."
Copy-Item -Path "$sourcePath\*" -Destination "$vailaProgramPath" -Recurse -Force

# Check if Conda is installed
If (-Not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Warning "Conda is not installed or not in the PATH. Please install Conda first."
    Exit
}

# Get Conda installation path
$condaPath = (conda info --base).Trim()

# Initialize Conda for PowerShell if it is not already initialized
Write-Output "Initializing Conda for PowerShell..."
Try {
    & "$condaPath\Scripts\conda.exe" init powershell
    Write-Output "Conda initialized successfully in PowerShell."
} Catch {
    Write-Error "Failed to initialize Conda in PowerShell. Error: $_"
    Exit
}

# Add Conda initialization to Windows Terminal and PowerShell profiles
$profilePath = "$PROFILE"
If (-Not (Test-Path $profilePath)) {
    Write-Output "PowerShell profile not found. Creating it..."
    New-Item -ItemType File -Path $profilePath -Force
}

# Add conda initialization to PowerShell profile
Add-Content -Path $profilePath -Value "& '$condaPath\shell\condabin\conda-hook.ps1'"

# Reload PowerShell profile to reflect changes
. $profilePath

# Install Windows Terminal, PowerShell 7, and oh-my-posh via winget
Write-Output "Installing Windows Terminal, PowerShell 7, and oh-my-posh..."
Try {
    winget install --id Microsoft.WindowsTerminal -e --source winget
    winget install --id Microsoft.Powershell -e --source winget
    winget install --id JanDeDobbeleer.OhMyPosh -e --source winget
    Write-Output "Windows Terminal, PowerShell 7, and oh-my-posh installed successfully."
} Catch {
    Write-Warning "Failed to install one or more applications via winget."
}

# Configure oh-my-posh for PowerShell
Write-Output "Configuring oh-my-posh for PowerShell..."

# Confirm the POSH_THEMES_PATH environment variable exists after installation
$ohMyPoshConfig = "$env:POSH_THEMES_PATH\jandedobbeleer.omp.json"
If (-Not (Test-Path $ohMyPoshConfig)) {
    Write-Output "Error locating oh-my-posh theme path. Please verify installation."
    Exit
}

# Add oh-my-posh initialization to PowerShell profile
Add-Content -Path $profilePath -Value "`noh-my-posh init pwsh --config '$ohMyPoshConfig' | Invoke-Expression`"

# Reload PowerShell profile
. $profilePath

# Install FFmpeg if not already installed
If (-Not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Output "Installing FFmpeg via winget..."
    Try {
        winget install --id Gyan.FFmpeg -e --source winget
        Write-Output "FFmpeg installed successfully."
    } Catch {
        Write-Warning "Failed to install FFmpeg via winget."
    }
}

# Add vaila profile to Windows Terminal
Write-Output "Adding vaila profile to Windows Terminal..."
$wtPath = "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState\settings.json"
If (Test-Path $wtPath) {
    $settingsJson = Get-Content -Path $wtPath -Raw | ConvertFrom-Json
    $vailaProfile = @{
        name = "vaila"
        commandline = "pwsh.exe -ExecutionPolicy Bypass -NoExit -Command `"& '$condaPath\shell\condabin\conda-hook.ps1' ; conda activate 'vaila' ; cd '$vailaProgramPath' ; python 'vaila.py'`""
        startingDirectory = "$vailaProgramPath"
        icon = "$vailaProgramPath\docs\images\vaila_ico.png"
        guid = "{17ce5bfe-17ed-5f3a-ab15-5cd5baafed5b}"
        hidden = $false
    }
    $settingsJson.profiles.list += $vailaProfile
    $settingsJson | ConvertTo-Json -Depth 100 | Out-File -FilePath $wtPath -Encoding UTF8
    Write-Output "vaila profile added to Windows Terminal."
}

# Final message
Write-Output "Installation and configuration completed successfully!"
Pause
