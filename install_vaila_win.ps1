<#
    Script: install_vaila_win.ps1
    Description: Installs or updates the vaila - Multimodal Toolbox on Windows 11,
                 setting up the Conda environment, copying program files to the user's
                 home directory, installing FFmpeg, adding a profile to Windows Terminal,
                 and creating a desktop shortcut if Windows Terminal is not installed.

    Usage:
      1. Download the repository from GitHub manually and extract it.
      2. Right-click the script and select "Run with PowerShell" as Administrator.

    Notes:
      - Ensure Conda is installed and accessible from the command line before running.
      - The script checks for and installs Windows Terminal if necessary.
      - If Windows Terminal is not available, a desktop shortcut will be created.

    Author: Prof. Dr. Paulo R. P. Santiago
    Date: September 17, 2024
    Version: 1.1
    OS: Windows 11
#>

# Ensure the script is running as administrator
If (-Not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Warning "This script requires administrative privileges. Please run as administrator."
    Exit
}

# Check if Conda is installed
If (-Not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Warning "Conda is not installed. Please install Conda first."
    Exit
}

# Get Conda installation path
$condaPath = (conda info --base).Trim()

# Initialize Conda in the script
& "$condaPath\Scripts\activate.bat" base

# Define user paths
$userHome = $env:USERPROFILE
$vailaHome = "$userHome\vaila"

# Check if the "vaila" environment already exists
Write-Output "Checking if the 'vaila' Conda environment exists..."
$envExists = conda env list | Select-String -Pattern "^vaila"
If ($envExists) {
    Write-Output "Conda environment 'vaila' already exists. Updating..."
    Try {
        conda env update -n vaila -f "yaml_for_conda_env\vaila_win.yaml" --prune
        Write-Output "'vaila' environment updated successfully."
    } Catch {
        Write-Error "Failed to update 'vaila' environment. Error: $_"
        Exit
    }
} Else {
    Write-Output "Creating Conda environment from vaila_win.yaml..."
    Try {
        conda env create -f "yaml_for_conda_env\vaila_win.yaml"
        Write-Output "'vaila' environment created successfully."
    } Catch {
        Write-Error "Failed to create 'vaila' environment. Error: $_"
        Exit
    }
}

# Activate the 'vaila' environment
& "$condaPath\Scripts\activate.bat" vaila

# Install moviepy using pip
Write-Output "Installing moviepy..."
Try {
    pip install moviepy
    Write-Output "moviepy installed successfully."
} Catch {
    Write-Error "Failed to install moviepy. Error: $_"
}

# Remove ffmpeg installed via Conda, if any
Write-Output "Removing ffmpeg installed via Conda, if any..."
conda remove -n vaila ffmpeg -y

# Check if FFmpeg is installed
If (-Not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Output "Installing FFmpeg via winget..."
    Try {
        winget install --id Gyan.FFmpeg -e --silent
        Write-Output "FFmpeg installed successfully."
    } Catch {
        Write-Warning "Failed to install FFmpeg via winget. Attempting installation via Chocolatey..."
        If (-Not (Get-Command choco -ErrorAction SilentlyContinue)) {
            Write-Warning "Chocolatey is not installed. Installing Chocolatey..."
            Set-ExecutionPolicy Bypass -Scope Process -Force
            [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
            Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
        }
        choco install ffmpeg -y
    }
} Else {
    Write-Output "FFmpeg is already installed."
}

# Copy the vaila program to the user's home directory
Write-Output "Copying vaila program to the user's home directory..."
New-Item -ItemType Directory -Force -Path "$vailaHome"
Copy-Item -Path (Get-Location) -Destination "$vailaHome" -Recurse -Force

# Check if Windows Terminal is installed
$wtPath = "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe"
If (-Not (Test-Path $wtPath)) {
    Write-Warning "Windows Terminal is not installed. Installing via Microsoft Store..."
    Try {
        winget install --id Microsoft.WindowsTerminal -e
        Write-Output "Windows Terminal installed successfully."
        $wtInstalled = $true
    } Catch {
        Write-Warning "Failed to install Windows Terminal. A desktop shortcut will be created instead."
        $wtInstalled = $false
    }
} Else {
    $wtInstalled = $true
}

# Configure the vaila profile in Windows Terminal
If ($wtInstalled) {
    Write-Output "Configuring the vaila profile in Windows Terminal..."
    $settingsPath = "$wtPath\LocalState\settings.json"
    $settingsBackupPath = "$wtPath\LocalState\settings_backup.json"

    # Backup the current settings.json
    Copy-Item -Path $settingsPath -Destination $settingsBackupPath -Force

    # Load settings.json
    $settingsJson = Get-Content -Path $settingsPath -Raw | ConvertFrom-Json

    # Define the new profile
    $vailaProfile = @{
        name = "vaila"
        commandline = "pwsh.exe -ExecutionPolicy Bypass -NoExit -Command `"& '$condaPath\shell\condabin\conda-hook.ps1' ; conda activate 'vaila' ; python '$vailaHome\vaila.py'`""
        startingDirectory = "$vailaHome"
        icon = "$vailaHome\docs\images\vaila_ico.png"
        colorScheme = "Vintage"
        guid = "{17ce5bfe-17ed-5f3a-ab15-5cd5baafed5b}"
        hidden = $false
    }

    # Add the profile to settings.json
    If (-Not $settingsJson.profiles.list) {
        $settingsJson.profiles.list = @()
    }
    $settingsJson.profiles.list += $vailaProfile

    # Save the updated settings.json
    $settingsJson | ConvertTo-Json -Depth 100 | Set-Content -Path $settingsPath -Encoding UTF8

    Write-Output "vaila profile added to Windows Terminal successfully."

    # Open settings.json in Notepad for verification
    Write-Output "Opening settings.json in Notepad for verification..."
    notepad "$settingsPath"
} Else {
    # Create a desktop shortcut
    Write-Output "Creating a desktop shortcut for vaila..."
    $shortcutPath = "$env:USERPROFILE\Desktop\vaila.lnk"
    $wshell = New-Object -ComObject WScript.Shell
    $shortcut = $wshell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = "pwsh.exe"
    $shortcut.Arguments = "-ExecutionPolicy Bypass -NoExit -Command `"& '$condaPath\shell\condabin\conda-hook.ps1' ; conda activate 'vaila' ; python '$vailaHome\vaila.py'`""
    $shortcut.IconLocation = "$vailaHome\docs\images\vaila_ico.ico"
    $shortcut.WorkingDirectory = "$vailaHome"
    $shortcut.Save()

    Write-Output "Desktop shortcut created at $shortcutPath"
}

Write-Output "Installation and configuration completed successfully!"
Pause

