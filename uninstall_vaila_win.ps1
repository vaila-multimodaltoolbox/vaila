<# 
    Script: uninstall_vaila_win.ps1
    Description: Uninstalls the vailá - Multimodal Toolbox from Windows,
                 removing the Conda environment, program files from C:\vaila_programs\vaila,
                 and Windows Terminal profile or desktop shortcut.

    Usage:
      1. Right-click the script and select "Run with PowerShell" as Administrator.

    Notes:
      - Ensure Conda is installed and accessible from the command line before running.

    Author: Prof. Dr. Paulo R. P. Santiago
    Date: September 22, 2024
    Version: 1.0
    OS: Windows 11
#>

# Ensure the script is running as administrator
If (-Not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Warning "This script requires administrative privileges. Please run as administrator."
    Exit
}

# Check if Conda is installed
If (-Not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Warning "Conda is not installed. Uninstallation cannot proceed."
    Exit
}

# Get Conda installation path
$condaPath = (conda info --base).Trim()

# Initialize Conda in the script
& "$condaPath\Scripts\activate.bat" base

# Deactivate the current Conda environment if any
Write-Output "Deactivating any active Conda environment..."
conda deactivate

# Check if the 'vaila' environment exists
Write-Output "Checking for 'vaila' Conda environment..."
$envExists = conda env list | Select-String -Pattern "^vaila"
If ($envExists) {
    Write-Output "Removing 'vaila' Conda environment..."
    Try {
        conda remove --name vaila --all -y
        Write-Output "'vaila' environment removed successfully."
    } Catch {
        Write-Error "Failed to remove 'vaila' environment. Error: $_"
        Exit
    }
} Else {
    Write-Output "'vaila' Conda environment does not exist. Skipping environment removal."
}

# Define installation path
$vailaProgramPath = "C:\vaila_programs\vaila"

# Remove the vailá directory
If (Test-Path $vailaProgramPath) {
    Write-Output "Removing vailá program directory at $vailaProgramPath..."
    Try {
        Remove-Item -Recurse -Force -Path $vailaProgramPath
        Write-Output "vailá directory removed successfully."
    } Catch {
        Write-Error "Failed to remove vailá directory. Error: $_"
    }
} Else {
    Write-Output "vailá directory not found. Skipping directory removal."
}

# Remove the vailá profile from Windows Terminal
$wtPath = "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe"
If (Test-Path $wtPath) {
    Write-Output "Removing vailá profile from Windows Terminal..."
    $settingsPath = "$wtPath\LocalState\settings.json"
    $settingsBackupPath = "$wtPath\LocalState\settings_backup_uninstall.json"

    # Backup the current settings.json
    Copy-Item -Path $settingsPath -Destination $settingsBackupPath -Force

    # Load settings.json
    $settingsJson = Get-Content -Path $settingsPath -Raw | ConvertFrom-Json

    # Find and remove vailá profile
    $existingProfileIndex = $settingsJson.profiles.list.FindIndex({ $_.name -eq "vailá" -or $_.name -eq "vaila" })
    If ($existingProfileIndex -ge 0) {
        Write-Output "Removing vailá profile..."
        $settingsJson.profiles.list.RemoveAt($existingProfileIndex)

        # Save the updated settings.json
        $settingsJson | ConvertTo-Json -Depth 100 | Set-Content -Path $settingsPath -Encoding UTF8
        Write-Output "vailá profile removed from Windows Terminal."
    } Else {
        Write-Output "vailá profile not found in Windows Terminal. Skipping."
    }
} Else {
    Write-Output "Windows Terminal is not installed. Skipping profile removal."
}

# Remove desktop shortcut if it exists
$shortcutPath = "$env:USERPROFILE\Desktop\vailá.lnk"
If (Test-Path $shortcutPath) {
    Write-Output "Removing desktop shortcut..."
    Try {
        Remove-Item -Force -Path $shortcutPath
        Write-Output "Desktop shortcut removed successfully."
    } Catch {
        Write-Error "Failed to remove desktop shortcut. Error: $_"
    }
} Else {
    Write-Output "Desktop shortcut not found. Skipping."
}

Write-Output "vailá has been successfully uninstalled."
Pause

