<#
    Script: install_vaila_win.ps1
    Description: Installs or updates the vailá - Multimodal Toolbox on Windows 11,
                 setting up the Conda environment, configuring MediaPipe, and
                 ensuring all dependencies are properly installed.
    Creation Date: 2024-12-17
    Author: Paulo R. P. Santiago & David Williams
#>

# Check for administrative privileges
If (-NOT ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator"))
{
    Write-Warning "This script must be run as Administrator."
    Exit
}

# Define installation path in AppData\Local dynamically for the current user
$vailaProgramPath = "$env:LOCALAPPDATA\vaila"
$sourcePath = (Get-Location)

# Ensure the directory exists
If (-Not (Test-Path $vailaProgramPath)) {
    Write-Output "Creating directory $vailaProgramPath..."
    New-Item -ItemType Directory -Force -Path $vailaProgramPath | Out-Null
}

# Copy the vaila program files to AppData\Local\vaila
Write-Output "Copying vaila program files from $sourcePath to $vailaProgramPath..."
Copy-Item -Path "$sourcePath\*" -Destination "$vailaProgramPath" -Recurse -Force

# Check if Conda is installed
If (-Not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Warning "Conda is not installed or not in the PATH. Please install Conda first."
    Exit
}

# 2 - Update conda and all packages before proceeding with the vaila environment setup
Write-Output "Updating conda and all related packages..."
Try {
    conda update conda -y
    conda update --all -y
    conda upgrade --all -y
    Write-Output "Conda and all packages updated successfully."
} Catch {
    Write-Warning "Failed to update conda or packages. Attempting to continue."
}

# Get Conda installation path
$condaPath = (& conda info --base).Trim()

# Initialize Conda for PowerShell if it is not already initialized
Write-Output "Initializing Conda for PowerShell..."
Try {
    & "$condaPath\Scripts\conda.exe" init powershell | Out-Null
    Write-Output "Conda initialized successfully in PowerShell."
} Catch {
    Write-Error "Failed to initialize Conda in PowerShell. Error: $_"
    Exit
}

# Add Conda initialization to PowerShell profile
$profilePath = $PROFILE
If (-Not (Test-Path $profilePath)) {
    Write-Output "PowerShell profile not found. Creating it..."
    New-Item -ItemType File -Path $profilePath -Force | Out-Null
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
    Write-Output "Creating 'vaila' Conda environment from vaila_win.yaml..."
    Try {
        & conda env create -f "$vailaProgramPath\yaml_for_conda_env\vaila_win.yaml"
        Write-Output "'vaila' environment created successfully."
    } Catch {
        Write-Error "Failed to create 'vaila' environment. Error: $_"
        Exit
    }
}

# Install dependencies using pip in the 'vaila' environment
Write-Output "Installing additional dependencies (mediapipe, moviepy) in the 'vaila' environment..."
conda activate vaila
Try {
    pip install mediapipe moviepy
    Write-Output "mediapipe and moviepy installed successfully."
} Catch {
    Write-Error "Failed to install mediapipe or moviepy. Error: $_"
}

# Remove ffmpeg installed via Conda, if any
Write-Output "Removing ffmpeg installed via Conda, if any..."
conda remove -n vaila ffmpeg -y

# Separate Installation and Upgrade Logic for Specific Packages

# 1. PowerShell 7
Write-Output "Checking if PowerShell 7 is installed..."
$psInstalled = Get-Command pwsh -ErrorAction SilentlyContinue
If ($psInstalled) {
    Write-Output "PowerShell 7 is already installed. Upgrading..."
    Try {
        winget upgrade --id Microsoft.Powershell -e --source winget --silent
        Write-Output "PowerShell 7 upgraded successfully."
    } Catch {
        Write-Warning "Failed to upgrade PowerShell 7."
    }
} Else {
    Write-Output "PowerShell 7 is not installed. Installing..."
    Try {
        winget install --id Microsoft.Powershell -e --source winget --silent
        Write-Output "PowerShell 7 installed successfully."
    } Catch {
        Write-Warning "Failed to install PowerShell 7 via winget."
    }
}

# 2. FFmpeg
Write-Output "Checking if FFmpeg is installed..."
$ffmpegInstalled = Get-Command ffmpeg -ErrorAction SilentlyContinue
If ($ffmpegInstalled) {
    Write-Output "FFmpeg is already installed. Upgrading..."
    Try {
        winget upgrade --id Gyan.FFmpeg -e --source winget --silent
        Write-Output "FFmpeg upgraded successfully."
    } Catch {
        Write-Warning "Failed to upgrade FFmpeg."
    }
} Else {
    Write-Output "FFmpeg is not installed. Installing..."
    Try {
        winget install --id Gyan.FFmpeg -e --source winget --silent
        Write-Output "FFmpeg installed successfully."
    } Catch {
        Write-Warning "Failed to install FFmpeg via winget."
    }
}

# 3. Windows Terminal
Write-Output "Checking if Windows Terminal is installed..."
$wtInstalled = Get-Command wt.exe -ErrorAction SilentlyContinue
If ($wtInstalled) {
    Write-Output "Windows Terminal is already installed. Upgrading..."
    Try {
        winget upgrade --id Microsoft.WindowsTerminal -e --source winget --silent
        Write-Output "Windows Terminal upgraded successfully."
    } Catch {
        Write-Warning "Failed to upgrade Windows Terminal."
    }
} Else {
    Write-Output "Windows Terminal is not installed. Installing..."
    Try {
        winget install --id Microsoft.WindowsTerminal -e --source winget --silent
        Write-Output "Windows Terminal installed successfully."
    } Catch {
        Write-Warning "Failed to install Windows Terminal via winget."
    }
}

# Configure the vaila profile in Windows Terminal
$wtPath = "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe"
If (Test-Path $wtPath) {
    Write-Output "Configuring the 'vaila' profile in Windows Terminal..."
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
    Write-Output "'vaila' profile added to Windows Terminal successfully."
}

# Grant full permissions to the MediaPipe model directory in the Conda environment
$mediapipeModelDir = "$condaPath\envs\vaila\Lib\site-packages\mediapipe\modules\pose_landmark"
Write-Output "Granting full permissions to the directory $mediapipeModelDir..."

If (Test-Path $mediapipeModelDir) {
    Try {
        Start-Process "icacls.exe" -ArgumentList "`"$mediapipeModelDir`" /grant Users:(OI)(CI)F /T" -Wait -NoNewWindow
        Write-Output "Permissions successfully granted for $mediapipeModelDir."
    } Catch {
        Write-Warning "Failed to grant permissions for $mediapipeModelDir. Details: $_"
    }
} Else {
    Write-Warning "Directory $mediapipeModelDir not found. No permissions were changed."
}

# Create a Desktop shortcut for vaila
Write-Output "Creating Desktop shortcut for 'vaila'..."
$desktopPath = [Environment]::GetFolderPath("Desktop")
$desktopShortcutPath = Join-Path $desktopPath "vaila.lnk"
$wshell = New-Object -ComObject WScript.Shell
$desktopShortcut = $wshell.CreateShortcut($desktopShortcutPath)
$desktopShortcut.TargetPath = "pwsh.exe"
$desktopShortcut.Arguments = "-ExecutionPolicy Bypass -NoExit -Command `"& `'$condaPath\shell\condabin\conda-hook.ps1`'; conda activate `'vaila`'; cd `'$vailaProgramPath`'; python `'vaila.py`'"
$desktopShortcut.IconLocation = "$vailaProgramPath\docs\images\vaila_ico.ico"
$desktopShortcut.WorkingDirectory = "$vailaProgramPath"
$desktopShortcut.Save()

Write-Output "Desktop shortcut for 'vaila' created at $desktopShortcutPath."

# Create Start Menu shortcut for vaila
Write-Output "Creating Start Menu shortcut for 'vaila'..."

$startMenuPath = "C:\ProgramData\Microsoft\Windows\Start Menu\Programs\vaila"
If (-Not (Test-Path $startMenuPath)) {
    Write-Output "Creating directory $startMenuPath..."
    New-Item -ItemType Directory -Force -Path $startMenuPath | Out-Null
}

$startMenuShortcutPath = "$startMenuPath\vaila.lnk"
$startMenuShortcut = $wshell.CreateShortcut($startMenuShortcutPath)
$startMenuShortcut.TargetPath = "pwsh.exe"
$startMenuShortcut.Arguments = "-ExecutionPolicy Bypass -NoExit -Command `"& `'$condaPath\shell\condabin\conda-hook.ps1`'; conda activate `'vaila`'; cd `'$vailaProgramPath`'; python `'vaila.py`'"
$startMenuShortcut.IconLocation = "$vailaProgramPath\docs\images\vaila_ico.ico"
$startMenuShortcut.WorkingDirectory = "$vailaProgramPath"
$startMenuShortcut.Save()

Write-Output "Start Menu shortcut for 'vaila' created at $startMenuShortcutPath."

Write-Output "Installation and configuration of 'vaila' completed successfully!"
Pause
