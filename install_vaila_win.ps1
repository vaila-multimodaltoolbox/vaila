<#
    Script: install_vaila_win.ps1
    Description: Installs or updates the vailá - Multimodal Toolbox on Windows 11.
                 Ensures a clean Conda environment, sets up all dependencies,
                 configures Desktop/Start Menu shortcuts, Windows Terminal profiles,
                 and robustly avoids copying __pycache__ or .pyc files.
    Creation Date: 17 December 2024
    Updated Date: 31 May 2025
    Authors: Paulo R. P. Santiago (USP) & David Williams (UNF), revised by ChatGPT
#>

# Check for administrative privileges
If (-NOT ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Warning "This script must be run as Administrator."
    Exit
}

# Define project source and installation paths
$vailaProgramPath = "$env:LOCALAPPDATA\vaila"
$sourcePath = (Get-Location)

# Ensure the installation directory exists
If (-Not (Test-Path $vailaProgramPath)) {
    Write-Output "Creating directory $vailaProgramPath..."
    New-Item -ItemType Directory -Force -Path $vailaProgramPath | Out-Null
}

# Copy project files, excluding __pycache__ and .pyc files
Write-Output "Copying vaila files (excluding __pycache__ and .pyc)..."
Get-ChildItem -Path $sourcePath -Recurse -Force | Where-Object {
    -not ($_.PSIsContainer -and $_.Name -eq '__pycache__') -and
    -not ($_.Extension -eq '.pyc')
} | ForEach-Object {
    $target = $_.FullName.Replace($sourcePath, $vailaProgramPath)
    if ($_.PSIsContainer) {
        if (-not (Test-Path $target)) {
            New-Item -ItemType Directory -Force -Path $target | Out-Null
        }
    } else {
        Copy-Item -Path $_.FullName -Destination $target -Force
    }
}

# Check if conda is installed
If (-Not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Warning "Conda is not installed or not in the PATH. Please install Conda first."
    Exit
}

# Update conda and all base packages
Write-Output "Updating conda and all base packages..."
Try {
    conda update conda -y
    conda update --all -y
    Write-Output "Conda and base packages updated."
} Catch {
    Write-Warning "Failed to update conda or packages. Continuing anyway."
}

# Remove existing 'vaila' environment for a fresh install
Write-Output "Checking for previous 'vaila' environment for clean install..."
$envExists = conda env list | Select-String -Pattern "^vaila"
If ($envExists) {
    Try {
        Write-Output "Removing old 'vaila' environment..."
        conda deactivate
        conda env remove -n vaila -y
        Write-Output "Old 'vaila' environment removed successfully."
    } Catch {
        Write-Warning "Could not remove old environment. Continuing anyway."
    }
} Else {
    Write-Output "'vaila' environment does not exist. Proceeding with creation."
}

# Clean conda cache
Write-Output "Cleaning conda cache..."
conda clean --all -y

# Create the vaila environment from YAML
Try {
    Write-Output "Creating 'vaila' environment from YAML..."
    & conda env create -f "$vailaProgramPath\yaml_for_conda_env\vaila_win.yaml"
    Write-Output "'vaila' environment created successfully."
} Catch {
    Write-Error "Failed to create the 'vaila' environment."
    Exit
}

# Activate vaila environment
Write-Output "Activating 'vaila' environment for pip upgrades..."
conda activate vaila

# Upgrade pip and install/upgrade pip dependencies
Write-Output "Upgrading pip and required pip packages..."
Try {
    python -m pip install --upgrade pip
    pip install --upgrade mediapipe moviepy
    Write-Output "pip, mediapipe, and moviepy installed/upgraded successfully."
} Catch {
    Write-Warning "Error upgrading pip or pip dependencies."
}

# Remove ffmpeg installed via conda, if present
Write-Output "Removing ffmpeg from conda (if present)..."
conda remove -n vaila ffmpeg -y

# --- System dependency installs/upgrades ---
Write-Output "Checking/installing/upgrading system dependencies (PowerShell 7, Chocolatey, rsync, FFmpeg, Windows Terminal)..."

# PowerShell 7
$psInstalled = Get-Command pwsh -ErrorAction SilentlyContinue
If ($psInstalled) {
    Write-Output "PowerShell 7 is already installed. Upgrading..."
    Try {
        winget upgrade --id Microsoft.PowerShell -e --source winget --silent
        Write-Output "PowerShell 7 upgraded successfully."
    } Catch {
        Write-Warning "Failed to upgrade PowerShell 7."
    }
} Else {
    Write-Output "PowerShell 7 is not installed. Installing..."
    Try {
        winget install --id Microsoft.PowerShell -e --source winget --silent
        Write-Output "PowerShell 7 installed successfully."
    } Catch {
        Write-Warning "Failed to install PowerShell 7 via winget."
    }
}

# Chocolatey
Write-Output "Checking for Chocolatey..."
if (-Not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Output "Installing Chocolatey..."
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    Write-Output "Chocolatey installed successfully."
}

# rsync via Chocolatey
Write-Output "Installing rsync via Chocolatey..."
Try {
    choco install rsync -y
    Write-Output "rsync installed successfully via Chocolatey."
} Catch {
    Write-Warning "Failed to install rsync via Chocolatey."
}

# FFmpeg
Write-Output "Checking for FFmpeg..."
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

# Windows Terminal
Write-Output "Checking for Windows Terminal..."
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

# -------- Windows Terminal Profile Setup --------
$condaPath = (& conda info --base).Trim()
$wtPath = "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe"
If (Test-Path $wtPath) {
    Write-Output "Configuring the 'vaila' profile in Windows Terminal..."
    $settingsPath = "$wtPath\LocalState\settings.json"
    $settingsBackupPath = "$wtPath\LocalState\settings_backup.json"

    # Backup current settings.json
    If (Test-Path $settingsPath) {
        Copy-Item -Path $settingsPath -Destination $settingsBackupPath -Force
        $settingsJson = Get-Content -Path $settingsPath -Raw | ConvertFrom-Json

        # Remove any existing 'vaila' profile
        $settingsJson.profiles.list = $settingsJson.profiles.list | Where-Object { $_.name -ne "vaila" }

        # Add new 'vaila' profile
        $vailaProfile = @{
            name = "vaila"
            commandline = "pwsh.exe -ExecutionPolicy Bypass -NoExit -Command `"& `'$condaPath\shell\condabin\conda-hook.ps1`'; conda activate `'vaila`'; cd `'$vailaProgramPath`'; python `'vaila.py`'`""
            startingDirectory = "$vailaProgramPath"
            icon = "$vailaProgramPath\docs\images\vaila_ico.png"
            guid = "{17ce5bfe-17ed-5f3a-ab15-5cd5baafed5b}"
            hidden = $false
        }

        $settingsJson.profiles.list += $vailaProfile
        $settingsJson | ConvertTo-Json -Depth 100 | Out-File -FilePath $settingsPath -Encoding UTF8
        Write-Output "'vaila' profile added to Windows Terminal successfully."
    } Else {
        Write-Output "Windows Terminal settings.json not found. Skipping profile setup."
    }
}

# --------- Desktop Shortcut ---------
Write-Output "Creating Desktop shortcut for 'vaila'..."
$desktopPath = [Environment]::GetFolderPath("Desktop")
$desktopShortcutPath = Join-Path $desktopPath "vaila.lnk"
$wshell = New-Object -ComObject WScript.Shell
$desktopShortcut = $wshell.CreateShortcut($desktopShortcutPath)
$desktopShortcut.TargetPath = "pwsh.exe"
$desktopShortcut.Arguments = "-ExecutionPolicy Bypass -NoExit -Command `"& `'$condaPath\shell\condabin\conda-hook.ps1`'; conda activate `'vaila`'; cd `'$vailaProgramPath`'; python `'vaila.py`'`""
$desktopShortcut.IconLocation = "$vailaProgramPath\docs\images\vaila_ico.ico"
$desktopShortcut.WorkingDirectory = "$vailaProgramPath"
$desktopShortcut.Save()
Write-Output "Desktop shortcut for 'vaila' created at $desktopShortcutPath."

# --------- Start Menu Shortcut ---------
Write-Output "Creating Start Menu shortcut for 'vaila'..."
$startMenuPath = "C:\ProgramData\Microsoft\Windows\Start Menu\Programs\vaila"
If (-Not (Test-Path $startMenuPath)) {
    Write-Output "Creating directory $startMenuPath..."
    New-Item -ItemType Directory -Force -Path $startMenuPath | Out-Null
}
$startMenuShortcutPath = "$startMenuPath\vaila.lnk"
$startMenuShortcut = $wshell.CreateShortcut($startMenuShortcutPath)
$startMenuShortcut.TargetPath = "pwsh.exe"
$startMenuShortcut.Arguments = "-ExecutionPolicy Bypass -NoExit -Command `"& `'$condaPath\shell\condabin\conda-hook.ps1`'; conda activate `'vaila`'; cd `'$vailaProgramPath`'; python `'vaila.py`'`""
$startMenuShortcut.IconLocation = "$vailaProgramPath\docs\images\vaila_ico.ico"
$startMenuShortcut.WorkingDirectory = "$vailaProgramPath"
$startMenuShortcut.Save()
Write-Output "Start Menu shortcut for 'vaila' created at $startMenuShortcutPath."

# --------- Adjust site-packages Permissions ---------
$vailaSitePackagesDir = Join-Path $condaPath "envs\vaila\Lib\site-packages"
Write-Output "Adjusting permissions on site-packages directory '$vailaSitePackagesDir' to allow read, write, and execute access..."
If (Test-Path $vailaSitePackagesDir) {
    Try {
        Start-Process "icacls.exe" -ArgumentList "`"$vailaSitePackagesDir`" /grant Users:(OI)(CI)F /T" -Wait -NoNewWindow
        Write-Output "Permissions successfully adjusted for '$vailaSitePackagesDir'."
    } Catch {
        Write-Warning "Failed to adjust permissions for '$vailaSitePackagesDir'."
    }
} Else {
    Write-Warning "Directory '$vailaSitePackagesDir' not found. No permissions were changed."
}

Write-Output "Installation and configuration of vaila completed successfully!"
Write-Output "Restart your computer to ensure all changes take effect."
Pause
