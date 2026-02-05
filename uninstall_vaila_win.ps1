<#
    Script: uninstall_vaila_win.ps1
    Description: Uninstalls the vaila - Multimodal Toolbox from Windows,
                 removing both uv virtual environment (.venv) and Conda environment (legacy),
                 deleting program files from installation locations, removing FFmpeg if installed,
                 removing vaila profiles from Windows Terminal, and deleting
                 Start Menu and Desktop shortcuts.
    Creation Date: 10 Jan 2025
    Last Update: 03 February 2026
    Author: Paulo R. P. Santiago
    Version: 0.3.19
#>

$ErrorActionPreference = "Continue"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "vaila - Multimodal Toolbox Uninstallation" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as administrator (optional, but helpful for system-wide installations)
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
If ($isAdmin) {
    Write-Host "Running with Administrator privileges." -ForegroundColor Green
} Else {
    Write-Host "Running without Administrator privileges (user-level uninstallation)." -ForegroundColor Yellow
}
Write-Host ""

# Define possible installation paths
# Only check 64-bit Program Files (x86 is not used for vaila)
# Always include Program Files so we find the path when uninstaller runs as current user (e.g. Inno [UninstallRun] runascurrentuser)
$possiblePaths = @("${env:ProgramFiles}\vaila", "$env:USERPROFILE\vaila", "$env:LOCALAPPDATA\vaila")
If ($isAdmin) {
    Write-Host "Checking system-wide installation locations (Administrator mode)..." -ForegroundColor Yellow
}

# Find actual installation path
$vailaProgramPath = $null
ForEach ($path in $possiblePaths) {
    # Skip x86 paths - vaila should only be in 64-bit Program Files
    If ($path -like "*Program Files (x86)*") {
        Write-Host "Skipping x86 path (vaila should only be in 64-bit Program Files): $path" -ForegroundColor Yellow
        Continue
    }
    If (Test-Path $path) {
        $vailaProgramPath = $path
        Write-Host "Found vaila installation at: $vailaProgramPath" -ForegroundColor Green
        Break
    }
}

# Remove uv virtual environment (.venv) if it exists
If ($vailaProgramPath -and (Test-Path "$vailaProgramPath\.venv")) {
    Write-Host "Removing uv virtual environment (.venv)..." -ForegroundColor Yellow
    Try {
        Remove-Item -Path "$vailaProgramPath\.venv" -Recurse -Force -ErrorAction Stop
        Write-Host "uv virtual environment removed successfully." -ForegroundColor Green
    } Catch {
        Write-Warning "Failed to remove uv virtual environment: $_"
    }
} Else {
    Write-Host "No uv virtual environment (.venv) found." -ForegroundColor Yellow
}

# Remove Conda environment (legacy) if Conda is installed
If (Get-Command conda -ErrorAction SilentlyContinue) {
    Write-Host "Checking for Conda environment (legacy)..." -ForegroundColor Yellow
    Try {
        $envExists = conda env list 2>$null | Select-String -Pattern "^vaila"
        If ($envExists) {
            Write-Host "Removing the 'vaila' Conda environment (legacy)..." -ForegroundColor Yellow
            conda env remove -n vaila -y 2>$null
            Write-Host "'vaila' Conda environment removed successfully." -ForegroundColor Green
        } Else {
            Write-Host "'vaila' Conda environment does not exist." -ForegroundColor Yellow
        }
    } Catch {
        Write-Warning "Could not check/remove Conda environment: $_"
    }
    
    # Cleanup Conda cache (optional)
    Write-Host "Cleaning Conda cache (if applicable)..." -ForegroundColor Yellow
    Try {
        conda clean --all -y 2>$null | Out-Null
        Write-Host "Conda cache cleaned." -ForegroundColor Green
    } Catch {
        Write-Host "Conda cache cleanup skipped (not critical)." -ForegroundColor Yellow
    }
} Else {
    Write-Host "Conda is not installed. Skipping Conda environment removal." -ForegroundColor Yellow
}

# Uninstall FFmpeg if installed by the script
Write-Host "Checking if FFmpeg is installed..." -ForegroundColor Yellow
If (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    Write-Host "Uninstalling FFmpeg..." -ForegroundColor Yellow
    Try {
        winget uninstall --id Gyan.FFmpeg -e --silent 2>$null | Out-Null
        Write-Host "FFmpeg uninstalled successfully." -ForegroundColor Green
    } Catch {
        Write-Warning "Failed to uninstall FFmpeg via winget. Skipping."
    }
} Else {
    Write-Host "FFmpeg is not installed via this script." -ForegroundColor Yellow
}

# Remove program files from installation location
If ($vailaProgramPath -and (Test-Path $vailaProgramPath)) {
    Write-Host "Deleting vaila program files from $vailaProgramPath..." -ForegroundColor Yellow
    Try {
        Remove-Item -Recurse -Force -Path $vailaProgramPath -ErrorAction Stop
        Write-Host "Program files deleted successfully." -ForegroundColor Green
    } Catch {
        Write-Warning "Failed to delete program files: $_"
        Write-Host "You may need to manually delete: $vailaProgramPath" -ForegroundColor Yellow
    }
} Else {
    Write-Host "vaila program files not found in standard installation locations." -ForegroundColor Yellow
    Write-Host "Checked locations:" -ForegroundColor Yellow
    ForEach ($path in $possiblePaths) {
        Write-Host "  - $path" -ForegroundColor Gray
    }
}

# Remove Windows Terminal profile for vaila
Write-Host "Removing Windows Terminal profile..." -ForegroundColor Yellow
$wtPath = "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe"
If (Test-Path $wtPath) {
    $settingsPath = "$wtPath\LocalState\settings.json"
    If (Test-Path $settingsPath) {
        Try {
            $settingsJson = Get-Content -Path $settingsPath -Raw | ConvertFrom-Json

            # Remove all vaila profiles from the list
            $updatedProfiles = $settingsJson.profiles.list | Where-Object { $_.name -ne "vaila" }
            $settingsJson.profiles.list = $updatedProfiles

            # Save updated settings with UTF-8 encoding
            $settingsJson | ConvertTo-Json -Depth 100 | Out-File -FilePath $settingsPath -Encoding UTF8
            Write-Host "vaila profile removed from Windows Terminal." -ForegroundColor Green
        } Catch {
            Write-Warning "Failed to remove Windows Terminal profile: $_"
        }
    } Else {
        Write-Host "Windows Terminal settings.json not found. Skipping profile removal." -ForegroundColor Yellow
    }
} Else {
    Write-Host "Windows Terminal is not installed, skipping profile removal." -ForegroundColor Yellow
}

# Remove Desktop shortcut
Write-Host "Removing Desktop shortcut..." -ForegroundColor Yellow
$desktopShortcutPath = "$env:USERPROFILE\Desktop\vaila.lnk"
If (Test-Path $desktopShortcutPath) {
    Try {
        Remove-Item $desktopShortcutPath -Force -ErrorAction Stop
        Write-Host "Desktop shortcut removed." -ForegroundColor Green
    } Catch {
        Write-Warning "Failed to remove Desktop shortcut: $_"
    }
} Else {
    Write-Host "Desktop shortcut not found." -ForegroundColor Yellow
}

# Remove Start Menu shortcuts from both common and user locations
Write-Host "Removing Start Menu shortcuts..." -ForegroundColor Yellow

# Common Start Menu
$commonStartMenuPrograms = [System.Environment]::GetFolderPath("CommonPrograms")
$commonStartMenuVailaLnk = Join-Path $commonStartMenuPrograms "vaila\vaila.lnk"
If (Test-Path $commonStartMenuVailaLnk) {
    Try {
        Remove-Item $commonStartMenuVailaLnk -Force -ErrorAction Stop
        Write-Host "Common Start Menu shortcut removed." -ForegroundColor Green
        
        $commonStartMenuVailaFolder = Join-Path $commonStartMenuPrograms "vaila"
        if ((Test-Path $commonStartMenuVailaFolder) -and ((Get-ChildItem $commonStartMenuVailaFolder -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0)) {
            Remove-Item $commonStartMenuVailaFolder -Force -ErrorAction SilentlyContinue
            Write-Host "Common Start Menu folder removed." -ForegroundColor Green
        }
    } Catch {
        Write-Warning "Failed to remove common Start Menu shortcut: $_"
    }
} Else {
    Write-Host "No common Start Menu shortcut found." -ForegroundColor Yellow
}

# User Start Menu
$userStartMenuPrograms = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
$userStartMenuVailaLnk = Join-Path $userStartMenuPrograms "vaila\vaila.lnk"

If (Test-Path $userStartMenuVailaLnk) {
    Try {
        Remove-Item $userStartMenuVailaLnk -Force -ErrorAction Stop
        Write-Host "User Start Menu shortcut removed." -ForegroundColor Green
        
        $userStartMenuVailaFolder = Join-Path $userStartMenuPrograms "vaila"
        if ((Test-Path $userStartMenuVailaFolder) -and ((Get-ChildItem $userStartMenuVailaFolder -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0)) {
            Remove-Item $userStartMenuVailaFolder -Force -ErrorAction SilentlyContinue
            Write-Host "User Start Menu folder removed." -ForegroundColor Green
        }
    } Catch {
        Write-Warning "Failed to remove user Start Menu shortcut: $_"
    }
} Else {
    Write-Host "No user Start Menu shortcut found." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "vaila uninstallation completed successfully!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Pause
