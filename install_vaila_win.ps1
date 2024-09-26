<#
    Script: install_vaila_win.ps1
    Description: Installs or updates the vaila - Multimodal Toolbox on Windows 11,
                 setting up the Conda environment, copying program files to 
                 C:\ProgramData\vaila, installing FFmpeg, configuring Windows 
                 Terminal, adding a profile to it, creating a desktop shortcut 
                 if Windows Terminal is not installed, and adding a Start Menu shortcut.

    Usage:
      1. Download the repository from GitHub manually and extract it.
      2. Right-click the script and select "Run with PowerShell" as Administrator.

    Features:
      - Checks if Conda is installed, activates the 'vaila' environment, 
        installs necessary packages (e.g., moviepy), and updates the environment if it exists.
      - Copies the vaila program to the installation directory (C:\ProgramData\vaila).
      - Installs FFmpeg using winget or Chocolatey if not installed.
      - Adds Conda to the PowerShell and Windows Terminal environment, making 
        Conda commands accessible in future sessions.
      - Configures a vaila profile in Windows Terminal, creates a desktop shortcut if Terminal is not available.
      - Adds a shortcut for vaila to the Start Menu.

    Notes:
      - Make sure Conda is installed and accessible from the command line before running this script.
      - The script checks for and installs Windows Terminal if necessary.
      - Administrator privileges are required to run this script.
      - Conda will be initialized for PowerShell if not already done.

    Author: Prof. Dr. Paulo R. P. Santiago
    Date: September 23, 2024
    Version: 1.8
    OS: Windows 11
#>

# Define installation path
$vailaProgramPath = "C:\ProgramData\vaila"
$sourcePath = (Get-Location)

# Ensure the directory exists
If (-Not (Test-Path $vailaProgramPath)) {
    Write-Output "Creating directory $vailaProgramPath..."
    New-Item -ItemType Directory -Force -Path $vailaProgramPath
}

# Copy the vaila program files to C:\ProgramData\vaila (adjusting to avoid extra directory creation)
Write-Output "Copying vaila program files from $sourcePath to $vailaProgramPath..."
Copy-Item -Path "$sourcePath\*" -Destination "$vailaProgramPath" -Recurse -Force

# Verify that files were copied successfully
Write-Output "Verifying that files were copied correctly..."
$sourceFiles = Get-ChildItem -Recurse "$sourcePath" | Where-Object { -not $_.PSIsContainer }
$destFiles = Get-ChildItem -Recurse "$vailaProgramPath" | Where-Object { -not $_.PSIsContainer }

# Compare the number of files
$sourceCount = $sourceFiles.Count
$destCount = $destFiles.Count

If ($sourceCount -eq $destCount) {
    Write-Output "All files were copied successfully."
} Else {
    Write-Warning "Some files may not have been copied. Source files: $sourceCount, Destination files: $destCount"
}

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

Write-Output "Ensuring Conda initialization is added to PowerShell profile..."
Add-Content -Path $profilePath -Value "& '$condaPath\shell\condabin\conda-hook.ps1'"

# Reload PowerShell profile to reflect changes
Write-Output "Reloading PowerShell profile to apply changes..."
. $profilePath

# Check if the 'vaila' environment already exists
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

# Configure the vaila profile in Windows Terminal using "vaila" for system configurations
If ($wtInstalled) {
    Write-Output "Configuring the vaila profile in Windows Terminal..."
    $settingsPath = "$wtPath\LocalState\settings.json"
    $settingsBackupPath = "$wtPath\LocalState\settings_backup.json"

    # Backup the current settings.json
    Copy-Item -Path $settingsPath -Destination $settingsBackupPath -Force

    # Load settings.json
    $settingsJson = Get-Content -Path $settingsPath -Raw | ConvertFrom-Json

    # Remove existing vaila profile if it exists
    $existingProfileIndex = $settingsJson.profiles.list | Where-Object { $_.name -eq "vaila" }
    If ($existingProfileIndex) {
        Write-Output "Removing existing vaila profile..."
        $settingsJson.profiles.list = $settingsJson.profiles.list | Where-Object { $_.name -ne "vaila" }
    }

    # Define the new profile using "vaila" for system-related configurations
    $vailaProfile = @{
        name = "vaila"
        commandline = "pwsh.exe -ExecutionPolicy Bypass -NoExit -Command `"& '$condaPath\shell\condabin\conda-hook.ps1' ; conda activate 'vaila' ; cd '$vailaProgramPath' ; python 'vaila.py'`""
        startingDirectory = "$vailaProgramPath"
        icon = "$vailaProgramPath\docs\images\vaila_ico.png"
        colorScheme = "Vintage"
        guid = "{17ce5bfe-17ed-5f3a-ab15-5cd5baafed5b}"
        hidden = $false
    }

    # Add the profile to settings.json
    $settingsJson.profiles.list += $vailaProfile

    # Save the updated settings.json with UTF-8 encoding
    $settingsJson | ConvertTo-Json -Depth 100 | Out-File -FilePath $settingsPath -Encoding UTF8

    Write-Output "vaila profile added to Windows Terminal successfully."

    # Open settings.json in Notepad for verification
    Write-Output "Opening settings.json in Notepad for verification..."
    notepad "$settingsPath"
} Else {
    # Create a desktop shortcut
    Write-Output "Creating a desktop shortcut for vailá..."
    $shortcutPath = "$env:USERPROFILE\Desktop\vailá.lnk"
    $wshell = New-Object -ComObject WScript.Shell
    $shortcut = $wshell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = "pwsh.exe"
    $shortcut.Arguments = "-ExecutionPolicy Bypass -NoExit -Command `"& '$condaPath\shell\condabin\conda-hook.ps1' ; conda activate 'vaila' ; cd '$vailaProgramPath' ; python 'vaila.py'`""
    $shortcut.IconLocation = "$vailaProgramPath\docs\images\vaila_ico.ico"
    $shortcut.WorkingDirectory = "$vailaProgramPath"
    $shortcut.Save()

    Write-Output "Desktop shortcut created at $shortcutPath"
}

# Create Start Menu shortcut using "vaila" for system configurations
Write-Output "Creating Start Menu shortcut for vaila..."
$startMenuPath = "C:\ProgramData\Microsoft\Windows\Start Menu\Programs\vaila.lnk"
$wshell = New-Object -ComObject WScript.Shell
$startShortcut = $wshell.CreateShortcut($startMenuPath)
$startShortcut.TargetPath = "pwsh.exe"
$startShortcut.Arguments = "-ExecutionPolicy Bypass -NoExit -Command `"& '$condaPath\shell\condabin\conda-hook.ps1' ; conda activate 'vaila' ; cd '$vailaProgramPath' ; python 'vaila.py'`""
$startShortcut.IconLocation = "$vailaProgramPath\docs\images\vaila_ico.ico"  # Ensure this points to the correct .ico file
$startShortcut.WorkingDirectory = "$vailaProgramPath"
$startShortcut.Save()

Write-Output "Start Menu shortcut for vaila created at $startMenuPath."

Write-Output "Installation and configuration completed successfully!"
Pause