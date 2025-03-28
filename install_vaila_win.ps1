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

# Install Chocolatey
Write-Output "Installing Chocolatey package manager..."
Try {
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    Write-Output "Chocolatey installed successfully."
} Catch {
    Write-Warning "Failed to install Chocolatey. Error: $_"
}

# Install rsync via Chocolatey
Write-Output "Installing rsync via Chocolatey..."
Try {
    choco install rsync -y
    Write-Output "rsync installed successfully via Chocolatey."
} Catch {
    Write-Warning "Failed to install rsync via Chocolatey. Error: $_"
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

# Grant full permissions to the vaila directory
Write-Output "Adjusting permissions on vaila directory '$vailaProgramPath' to allow read, write, and execute access..."
Try {
    # Grant full control to the folder and all its files/subfolders for the 'Users' group
    Start-Process "icacls.exe" -ArgumentList "`"$vailaProgramPath`" /grant Users:(OI)(CI)F /T" -Wait -NoNewWindow
    Write-Output "Permissions have been successfully adjusted for '$vailaProgramPath'."
} Catch {
    Write-Warning "Failed to adjust permissions for '$vailaProgramPath'. Details: $_"
}

# Grant full permissions to the vaila Anaconda environment
$vailaEnvDir = "C:\ProgramData\anaconda3\envs\vaila"
Write-Output "Adjusting permissions on Anaconda environment directory '$vailaEnvDir' to allow read, write, and execute access..."
If (Test-Path $vailaEnvDir) {
    Try {
        # Grant full control to the folder and all its files/subfolders for the 'Users' group
        Start-Process "icacls.exe" -ArgumentList "`"$vailaEnvDir`" /grant Users:(OI)(CI)F /T" -Wait -NoNewWindow
        Write-Output "Permissions have been successfully adjusted for '$vailaEnvDir'."
    } Catch {
        Write-Warning "Failed to adjust permissions for '$vailaEnvDir'. Details: $_"
    }
} Else {
    Write-Warning "Directory '$vailaEnvDir' was not found. No permissions were changed."
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

# Install and enable OpenSSH Client and Server
Write-Output "Checking if OpenSSH is installed..."
$sshClientInstalled = Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Client*' | Select-Object -ExpandProperty State
$sshServerInstalled = Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Server*' | Select-Object -ExpandProperty State

# Install OpenSSH Client if not already installed
if ($sshClientInstalled -ne "Installed") {
    Write-Output "Installing OpenSSH Client..."
    Try {
        Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0
        Write-Output "OpenSSH Client installed successfully."
    } Catch {
        Write-Warning "Failed to install OpenSSH Client. Error: $_"
    }
} else {
    Write-Output "OpenSSH Client is already installed."
}

# Install OpenSSH Server if not already installed
if ($sshServerInstalled -ne "Installed") {
    Write-Output "Installing OpenSSH Server..."
    Try {
        Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
        Write-Output "OpenSSH Server installed successfully."
    } Catch {
        Write-Warning "Failed to install OpenSSH Server. Error: $_"
    }
}

# Configure and start OpenSSH Server
if ((Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Server*' | Select-Object -ExpandProperty State) -eq "Installed") {
    Write-Output "Configuring OpenSSH Server..."
    Try {
        # Start the service
        Start-Service sshd
        # Set it to start automatically
        Set-Service -Name sshd -StartupType 'Automatic'
        # Confirm the Firewall rule is configured
        if (!(Get-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -ErrorAction SilentlyContinue | Select-Object Name, Enabled)) {
            New-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -DisplayName 'OpenSSH Server (sshd)' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
        }
        Write-Output "OpenSSH Server configured and started successfully."
    } Catch {
        Write-Warning "Failed to configure OpenSSH Server. Error: $_"
    }
}

Write-Output "Installation and configuration of 'vaila' completed successfully!"

Write-Output "Please restart your computer to apply all changes."

Write-Output "If double-clicking on vaila icon in the desktop or Start Menu is not working, please check the following:   "
Write-Output "1. Open the Start Menu and search for 'Windows PowerShell'."
Write-Output "2. Right-click on 'Windows PowerShell' and select 'Run as administrator'."
Write-Output "3. Type 'conda activate vaila' and press Enter."
Write-Output "4. Type 'cd C:\Users\<username>\AppData\Local\vaila' and press Enter."
Write-Output "5. Type 'python vaila.py' and press Enter."

Write-Output "Thank you for using vailá - Multimodal Toolbox!"
Pause
