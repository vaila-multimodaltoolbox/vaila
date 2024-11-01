<#
    Script: uninstall_vaila_win.ps1
    Description: Uninstalls the vaila - Multimodal Toolbox from Windows 11,
                 removing the Conda environment, deleting program files from
                 AppData\Local\vaila, removing FFmpeg if installed, 
                 removing vaila profiles from Windows Terminal, and deleting
                 Start Menu and Desktop shortcuts.
#>

# Define installation path in AppData\Local
$vailaProgramPath = "$env:LOCALAPPDATA\vaila"

# Check if Conda is installed
If (-Not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Warning "Conda is not installed or not in the PATH. Uninstallation cannot proceed."
    Exit
}

# Get Conda installation path
$condaPath = (conda info --base).Trim()

# Remove the 'vaila' Conda environment
Write-Output "Checking if the 'vaila' Conda environment exists..."
$envExists = conda env list | Select-String -Pattern "^vaila"
If ($envExists) {
    Write-Output "Removing the 'vaila' Conda environment..."
    Try {
        conda env remove -n vaila
        Write-Output "'vaila' Conda environment removed successfully."
    } Catch {
        Write-Error "Failed to remove the 'vaila' environment. Error: $_"
    }
} Else {
    Write-Output "'vaila' Conda environment does not exist."
}

# Uninstall oh-my-posh if installed via winget
Write-Output "Checking if oh-my-posh is installed..."
If (Get-Command oh-my-posh -ErrorAction SilentlyContinue) {
    Write-Output "Uninstalling oh-my-posh..."
    Try {
        winget uninstall --id JanDeDobbeleer.OhMyPosh -e
        Write-Output "oh-my-posh uninstalled successfully."
    } Catch {
        Write-Warning "Failed to uninstall oh-my-posh via winget. Skipping."
    }
} Else {
    Write-Output "oh-my-posh is not installed via this script."
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

        # Find and remove vaila profile
        $profileIndex = -1
        For ($i = 0; $i -lt $settingsJson.profiles.list.Count; $i++) {
            if ($settingsJson.profiles.list[$i].name -eq "vaila") {
                $profileIndex = $i
                break
            }
        }

        If ($profileIndex -ge 0) {
            Write-Output "Removing vaila profile from Windows Terminal..."
            $settingsJson.profiles.list.RemoveAt($profileIndex)

            # Save updated settings with UTF-8 encoding
            $settingsJson | ConvertTo-Json -Depth 100 | Out-File -FilePath $settingsPath -Encoding UTF8
            Write-Output "vaila profile removed from Windows Terminal."
        } Else {
            Write-Output "vaila profile not found in Windows Terminal."
        }
    }
} Else {
    Write-Output "Windows Terminal is not installed, skipping profile removal."
}

# Remove Desktop shortcut (uses "vailá")
$desktopShortcutPath = "$env:USERPROFILE\Desktop\vailá.lnk"
If (Test-Path $desktopShortcutPath) {
    Write-Output "Removing Desktop shortcut..."
    Remove-Item $desktopShortcutPath -Force
    Write-Output "Desktop shortcut removed."
} Else {
    Write-Output "Desktop shortcut not found."
}

# Remove Start Menu shortcut (uses "vaila")
$startMenuShortcutPath = "C:\ProgramData\Microsoft\Windows\Start Menu\Programs\vaila.lnk"
If (Test-Path $startMenuShortcutPath) {
    Write-Output "Removing Start Menu shortcut..."
    Remove-Item $startMenuShortcutPath -Force
    Write-Output "Start Menu shortcut removed."
} Else {
    Write-Output "Start Menu shortcut not found."
}

Write-Output "vaila uninstallation completed successfully!"
Pause
