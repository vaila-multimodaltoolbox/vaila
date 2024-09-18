<#
    Script: uninstall_vaila_win.ps1
    Description: Uninstalls the vaila - Multimodal Toolbox from Windows 11,
                 including removing the Conda environment, deleting program files
                 from the user's home directory, removing the Windows Terminal profile,
                 and deleting the desktop shortcut if it was created.

    Usage:
      1. Right-click the script and select "Run with PowerShell" as Administrator.

    Notes:
      - Ensure Conda is installed and accessible from the command line before running.
      - The script will remove the 'vaila' Conda environment and program files.

    Author: Prof. Dr. Paulo R. P. Santiago
    Date: September 17, 2024
    Version: 1.0
    OS: Windows 11
#>

# Ensure the script is running as administrator
If (-Not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Warning "This script requires administrative privileges. Please run as administrator."
    Exit
}

# Initialize Conda
$condaCommand = Get-Command conda -ErrorAction SilentlyContinue
If (-Not $condaCommand) {
    Write-Warning "Conda is not installed or not in PATH. Cannot proceed with uninstallation."
    Exit
}

# Get Conda installation path
$condaBase = (conda info --base).Trim()

# Activate Conda base environment
& "$condaBase\Scripts\activate.bat" base

# Remove the 'vaila' Conda environment if it exists
Write-Output "Checking for 'vaila' Conda environment..."
$envExists = conda env list | Select-String -Pattern "^vaila"
If ($envExists) {
    Write-Output "Removing 'vaila' Conda environment..."
    Try {
        conda remove --name vaila --all -y
        Write-Output "'vaila' environment removed successfully."
    } Catch {
        Write-Error "Failed to remove 'vaila' environment. Error: $_"
    }
} Else {
    Write-Output "'vaila' environment does not exist. Skipping environment removal."
}

# Define user paths
$userHome = $env:USERPROFILE
$vailaHome = "$userHome\vaila"

# Remove the vaila directory from the user's home directory
If (Test-Path $vailaHome) {
    Write-Output "Removing vaila directory from user's home directory..."
    Try {
        Remove-Item -Path $vailaHome -Recurse -Force
        Write-Output "vaila directory removed successfully."
    } Catch {
        Write-Error "Failed to remove vaila directory. Error: $_"
    }
} Else {
    Write-Output "vaila directory not found in user's home directory. Skipping removal."
}

# Remove the Windows Terminal profile if Windows Terminal is installed
$wtPath = "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe"
If (Test-Path $wtPath) {
    Write-Output "Removing vaila profile from Windows Terminal..."
    $settingsPath = "$wtPath\LocalState\settings.json"
    $settingsBackupPath = "$wtPath\LocalState\settings_backup_uninstall.json"

    # Backup the current settings.json
    Copy-Item -Path $settingsPath -Destination $settingsBackupPath -Force

    # Load settings.json
    $settingsJson = Get-Content -Path $settingsPath -Raw | ConvertFrom-Json

    # Remove the vaila profile
    $profileIndex = $null
    For ($i = 0; $i -lt $settingsJson.profiles.list.Count; $i++) {
        If ($settingsJson.profiles.list[$i].name -eq "vaila" -or $settingsJson.profiles.list[$i].name -eq "vailá") {
            $profileIndex = $i
            Break
        }
    }

    If ($null -ne $profileIndex) {
        $settingsJson.profiles.list.RemoveAt($profileIndex)
        # Save the updated settings.json
        $settingsJson | ConvertTo-Json -Depth 100 | Set-Content -Path $settingsPath -Encoding UTF8
        Write-Output "vaila profile removed from Windows Terminal."
    } Else {
        Write-Output "vaila profile not found in Windows Terminal settings."
    }
} Else {
    Write-Output "Windows Terminal is not installed. Skipping profile removal."
}

# Remove desktop shortcut if it exists
$shortcutPath = "$env:USERPROFILE\Desktop\vaila.lnk"
If (Test-Path $shortcutPath) {
    Write-Output "Removing desktop shortcut..."
    Try {
        Remove-Item -Path $shortcutPath -Force
        Write-Output "Desktop shortcut removed successfully."
    } Catch {
        Write-Error "Failed to remove desktop shortcut. Error: $_"
    }
} Else {
    Write-Output "Desktop shortcut not found. Skipping removal."
}

Write-Output "Uninstallation completed. vaila has been removed from your system."
Pause

