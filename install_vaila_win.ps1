<#
    Script: install_vaila_win.ps1
    Description: Installs or updates the vaila - Multimodal Toolbox on Windows 11,
                 setting up the Conda environment, copying program files to
                 AppData\Local\vaila, installing FFmpeg, configuring Windows
                 Terminal, and adding a profile for easy access with shortcuts.
#>

# Define installation path in AppData\Local dynamically for the current user
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
$condaPath = (& conda info --base).Trim()

# Initialize Conda for PowerShell if it is not already initialized
Write-Output "Initializing Conda for PowerShell..."
Try {
    & "$condaPath\Scripts\conda.exe" init powershell
    Write-Output "Conda initialized successfully in PowerShell."
} Catch {
    Write-Error "Failed to initialize Conda in PowerShell. Error: $_"
    Exit
}

# Add Conda initialization to PowerShell profile
$profilePath = $PROFILE
If (-Not (Test-Path $profilePath)) {
    Write-Output "PowerShell profile not found. Creating it..."
    New-Item -ItemType File -Path $profilePath -Force
}

Write-Output "Ensuring Conda initialization is added to PowerShell profile..."
Add-Content -Path $profilePath -Value "`n& '$condaPath\shell\condabin\conda-hook.ps1'"

# Reload PowerShell profile to reflect changes
Write-Output "Reloading PowerShell profile to apply changes..."
. $profilePath

# Check if the 'vaila' Conda environment exists and create/update it
Write-Output "Checking if the 'vaila' Conda environment exists..."
$envExists = conda env list | Select-String -Pattern "^vaila"
If ($envExists) {
    Write-Output "Conda environment 'vaila' already exists. Updating..."
    Try {
        & conda env update -n vaila -f "$vailaProgramPath\yaml_for_conda_env\vaila_win.yaml" --prune
        Write-Output "'vaila' environment updated successfully."
    } Catch {
        Write-Error "Failed to update 'vaila' environment. Error: $_"
        Exit
    }
} Else {
    Write-Output "Creating Conda environment from vaila_win.yaml..."
    Try {
        & conda env create -f "$vailaProgramPath\yaml_for_conda_env\vaila_win.yaml"
        Write-Output "'vaila' environment created successfully."
    } Catch {
        Write-Error "Failed to create 'vaila' environment. Error: $_"
        Exit
    }
}

# Install moviepy using pip in the 'vaila' environment
Write-Output "Installing moviepy in the 'vaila' environment..."
conda activate vaila
Try {
    pip install moviepy
    Write-Output "moviepy installed successfully."
} Catch {
    Write-Error "Failed to install moviepy. Error: $_"
}

# Remove ffmpeg installed via Conda, if any
Write-Output "Removing ffmpeg installed via Conda, if any..."
conda remove -n vaila ffmpeg -y

# Install FFmpeg via winget if not already installed
If (-Not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Output "Installing FFmpeg via winget..."
    Try {
        winget install --id Gyan.FFmpeg -e --source winget
        Write-Output "FFmpeg installed successfully."
    } Catch {
        Write-Warning "Failed to install FFmpeg via winget."
    }
}

# Install Windows Terminal and PowerShell 7 via winget
Write-Output "Installing Windows Terminal and PowerShell 7..."
Try {
    winget install --id Microsoft.WindowsTerminal -e --source winget
    winget install --id Microsoft.Powershell -e --source winget
    Write-Output "Windows Terminal and PowerShell 7 installed successfully."
} Catch {
    Write-Warning "Failed to install Windows Terminal or PowerShell 7 via winget."
}

# Configure the vaila profile in Windows Terminal
$wtPath = "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe"
If (Test-Path $wtPath) {
    Write-Output "Configuring the vaila profile in Windows Terminal..."
    $settingsPath = "$wtPath\LocalState\settings.json"
    $settingsBackupPath = "$wtPath\LocalState\settings_backup.json"

    # Backup the current settings.json
    Copy-Item -Path $settingsPath -Destination $settingsBackupPath -Force

    # Load settings.json
    $settingsJson = Get-Content -Path $settingsPath -Raw | ConvertFrom-Json

    # Remove existing 'vaila' profiles if they exist
    $settingsJson.profiles.list = $settingsJson.profiles.list | Where-Object { $_.name -ne "vaila" }

    # Add the new vaila profile
    $vailaProfile = @{
        name = "vaila"
        commandline = "pwsh.exe -ExecutionPolicy Bypass -NoExit -Command `"& `'$condaPath\shell\condabin\conda-hook.ps1`'; conda activate `'vaila`'; cd `'$vailaProgramPath`'; python `'vaila.py`'"
        startingDirectory = "$vailaProgramPath"
        icon = "$vailaProgramPath\docs\images\vaila_ico.png"
        guid = "{17ce5bfe-17ed-5f3a-ab15-5cd5baafed5b}"
        hidden = $false
    }

    $settingsJson.profiles.list += $vailaProfile
    $settingsJson | ConvertTo-Json -Depth 100 | Out-File -FilePath $settingsPath -Encoding UTF8
    Write-Output "vaila profile added to Windows Terminal successfully."
}

# Create a Desktop shortcut for vaila
Write-Output "Creating Desktop shortcut for vaila..."
$desktopShortcutPath = "$env:USERPROFILE\Desktop\vaila.lnk"
$wshell = New-Object -ComObject WScript.Shell
$desktopShortcut = $wshell.CreateShortcut($desktopShortcutPath)
$desktopShortcut.TargetPath = "pwsh.exe"
$desktopShortcut.Arguments = "-ExecutionPolicy Bypass -NoExit -Command `"& `'$condaPath\shell\condabin\conda-hook.ps1`'; conda activate `'vaila`'; cd `'$vailaProgramPath`'; python `'vaila.py`'"
$desktopShortcut.IconLocation = "$vailaProgramPath\docs\images\vaila_ico.ico"
$desktopShortcut.WorkingDirectory = "$vailaProgramPath"
$desktopShortcut.Save()

Write-Output "Desktop shortcut for vaila created at $desktopShortcutPath."

Write-Output "Installation and configuration completed successfully!"
Pause
