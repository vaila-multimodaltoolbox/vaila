<#
    Script: uninstall_vaila_win.ps1
    Description: Uninstalls the vaila - Multimodal Toolbox from Windows 11,
                 removing the Conda environment, deleting program files from
                 AppData\Local\vaila, removing FFmpeg if installed,
                 removing vaila profiles from Windows Terminal, and deleting
                 Start Menu and Desktop shortcuts.
    Creation Date: 2024-12-10
    Author: Paulo R. P. Santiago
#>

# Check for administrative privileges
If (-NOT ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator"))
{
    Write-Warning "Este script precisa ser executado como Administrador."
    Exit
}

# Define installation path in AppData\Local
$vailaProgramPath = "$env:LOCALAPPDATA\vaila"

# Check if Conda is installed
If (-Not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Warning "Conda is not installed or not in the PATH. Uninstallation cannot proceed."
    Exit
}

# Get Conda installation path
$condaPath = (& conda info --base).Trim()

# Remove the 'vaila' Conda environment
Write-Output "Checking if the 'vaila' Conda environment exists..."
$envExists = conda env list | Select-String -Pattern "^vaila"
If ($envExists) {
    Write-Output "Removing the 'vaila' Conda environment..."
    Try {
        conda env remove -n vaila -y
        Write-Output "'vaila' Conda environment removed successfully."
    } Catch {
        Write-Error "Failed to remove the 'vaila' environment. Error: $_"
    }
} Else {
    Write-Output "'vaila' Conda environment does not exist."
}

# Uninstall FFmpeg if installed by the script
Write-Output "Checking if FFmpeg is installed..."
If (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    Write-Output "Uninstalling FFmpeg..."
    Try {
        winget uninstall --id Gyan.FFmpeg -e --silent
        Write-Output "FFmpeg uninstalled successfully."
    } Catch {
        Write-Warning "Failed to uninstall FFmpeg via winget. Skipping."
    }
} Else {
    Write-Output "FFmpeg is not installed via this script."
}

# Remove program files from AppData\Local\vaila
If (Test-Path $vailaProgramPath) {
    Write-Output "Deleting vaila program files from $vailaProgramPath..."
    Remove-Item -Recurse -Force -Path $vailaProgramPath
    Write-Output "Program files deleted."
} Else {
    Write-Output "vaila program files not found at $vailaProgramPath."
}

# Remove Windows Terminal profile for vaila
$wtPath = "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe"
If (Test-Path $wtPath) {
    Write-Output "Checking for vaila profile in Windows Terminal settings..."
    $settingsPath = "$wtPath\LocalState\settings.json"
    If (Test-Path $settingsPath) {
        $settingsJson = Get-Content -Path $settingsPath -Raw | ConvertFrom-Json

        # Remove all vaila profiles from the list
        $updatedProfiles = $settingsJson.profiles.list | Where-Object { $_.name -ne "vaila" }
        $settingsJson.profiles.list = $updatedProfiles

        # Save updated settings with UTF-8 encoding
        $settingsJson | ConvertTo-Json -Depth 100 | Out-File -FilePath $settingsPath -Encoding UTF8
        Write-Output "vaila profile removed from Windows Terminal."
    } Else {
        Write-Output "Windows Terminal settings.json not found. Skipping profile removal."
    }
} Else {
    Write-Output "Windows Terminal is not installed, skipping profile removal."
}

# Remove Desktop shortcut
$desktopShortcutPath = "$env:USERPROFILE\Desktop\vaila.lnk"
If (Test-Path $desktopShortcutPath) {
    Write-Output "Removing Desktop shortcut..."
    Remove-Item $desktopShortcutPath -Force
    Write-Output "Desktop shortcut removed."
} Else {
    Write-Output "Desktop shortcut not found."
}

# ---------------------------------------------------
# Remove Start Menu shortcuts from both common and user locations
# ---------------------------------------------------

# Common Start Menu
$commonStartMenuPrograms = [System.Environment]::GetFolderPath("CommonPrograms")
$commonStartMenuVailaLnk = Join-Path $commonStartMenuPrograms "vaila\vaila.lnk"
If (Test-Path $commonStartMenuVailaLnk) {
    Write-Output "Removing common Start Menu shortcut..."
    Remove-Item $commonStartMenuVailaLnk -Force
    Write-Output "Common Start Menu shortcut removed."
    
    $commonStartMenuVailaFolder = Join-Path $commonStartMenuPrograms "vaila"
    if (Test-Path $commonStartMenuVailaFolder -and (Get-ChildItem $commonStartMenuVailaFolder | Measure-Object).Count -eq 0) {
        Write-Output "Removing empty common Start Menu folder..."
        Remove-Item $commonStartMenuVailaFolder -Force
        Write-Output "Common Start Menu folder removed."
    }
} Else {
    Write-Output "No common Start Menu shortcut found."
}

# User Start Menu
$userStartMenuPrograms = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
$userStartMenuVailaLnk = Join-Path $userStartMenuPrograms "vaila\vaila.lnk"

If (Test-Path $userStartMenuVailaLnk) {
    Write-Output "Removing user Start Menu shortcut..."
    Remove-Item $userStartMenuVailaLnk -Force
    Write-Output "User Start Menu shortcut removed."
    
    $userStartMenuVailaFolder = Join-Path $userStartMenuPrograms "vaila"
    if (Test-Path $userStartMenuVailaFolder -and (Get-ChildItem $userStartMenuVailaFolder | Measure-Object).Count -eq 0) {
        Write-Output "Removing empty user Start Menu folder..."
        Remove-Item $userStartMenuVailaFolder -Force
        Write-Output "User Start Menu folder removed."
    }
} Else {
    Write-Output "No user Start Menu shortcut found."
}

Write-Output "vaila uninstallation completed successfully!"
Pause
