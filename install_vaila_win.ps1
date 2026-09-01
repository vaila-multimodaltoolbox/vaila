<#
    Script: install_vaila_win.ps1
    Description: Installs the vaila - Multimodal Toolbox on Windows using uv (Astral).
                 Official install method: uv (Astral).
    Usage:
        1. Download/clone the repository.
        2. Open PowerShell (Administrator recommended for full installation).
        3. Navigate to the root directory of the repository.
        4. Run: .\install_vaila_win.ps1
    Notes:
        - PowerShell 7 (pwsh), git, and Node.js LTS are auto-installed via winget if
          missing (a "naked" Windows box ships with none of them) before anything
          else runs. If pwsh cannot be installed, generated shortcuts fall back to
          powershell.exe instead of failing silently.
        - uv will be automatically installed if not present.
        - Python 3.12.14 will be installed via `uv python install`.
        - Installation location (prompt; Enter = portable default):
          * [1] Current directory (Local/Portable) - default / recommended
          * [2] User profile or Program Files:
              - With admin: C:\Program Files\vaila
              - Without admin: C:\Users\<user>\vaila
        - Portable/git installs keep the committed uv.lock so `git pull` works.
        - Profile / Program Files copies may regenerate uv.lock (no git tree).
        - Remote one-liner: after IWR download, run `Unblock-File` on the temp script
          (Mark-of-the-Web) before `-File` execution — see README Install Now.
        - Safe with `irm ... | iex`: empty $PSScriptRoot falls back to the current
          directory; if that folder is not a vaila clone, portable installs to .\vaila.
        - Can run without administrator privileges (some features may be skipped).
    Author: Prof. Dr. Paulo R. P. Santiago
    Creation: 17 December 2024
    Updated: 01 September 2026
    Version: 0.3.118
    OS: Windows 11
    Reference: https://docs.astral.sh/uv/
    Parameters:
        -InstallLocation prompt|portable|profile
          prompt   = ask interactively (default for manual runs)
          portable = current script/cwd directory (Local/Portable; used by Inno Setup)
          profile  = Program Files (admin) or %USERPROFILE%\vaila
#>

param(
    [ValidateSet("prompt", "portable", "profile")]
    [string]$InstallLocation = "prompt"
)

$ErrorActionPreference = "Stop"

function Repair-GitLfsCheckoutIfNeeded {
    param([string]$RepoRoot)

    If (-Not (Test-Path (Join-Path $RepoRoot ".git"))) { return }

    $probe = Join-Path $RepoRoot "vaila\models\osnet_x0_25_msmt17.onnx"
    If (-Not (Test-Path $probe)) { return }

    $probeSize = (Get-Item $probe -ErrorAction SilentlyContinue).Length
    If ($probeSize -ge 8192) { return }

    Write-Host "Git LFS checkout incomplete (pointer file ~$probeSize bytes at osnet model)." -ForegroundColor Yellow
    Write-Host "Repairing via HTTPS LFS endpoint (common after SSH clone on Windows)..." -ForegroundColor Yellow

    If (-Not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Warning "git not found — cannot repair LFS checkout automatically."
        return
    }

    Push-Location $RepoRoot
    Try {
        git config lfs.url https://github.com/vaila-multimodaltoolbox/vaila.git/info/lfs 2>$null
        git lfs install 2>$null
        git lfs pull 2>&1 | ForEach-Object { Write-Host $_ }
        git restore --source=HEAD :/ 2>&1 | ForEach-Object { Write-Host $_ }
        $after = (Get-Item $probe -ErrorAction SilentlyContinue).Length
        If ($after -ge 8192) {
            Write-Host "Git LFS repair completed ($after bytes)." -ForegroundColor Green
        } Else {
            Write-Warning "Git LFS repair may have failed — osnet model is still $after bytes."
            Write-Warning "Try: git clone https://github.com/vaila-multimodaltoolbox/vaila.git"
        }
    } Finally {
        Pop-Location
    }
}

function Repair-SshOriginIfNeeded {
    # An SSH origin (git@github.com:...) needs a GitHub SSH key most Windows
    # machines don't have configured, which breaks both a plain `git pull`
    # and the GUI's "Check for Updates" (git fetch) with a raw
    # "Permission denied (publickey)" error. Rewrite it to HTTPS in place.
    param(
        [string]$RepoRoot,
        [string]$RemoteName = "origin"
    )

    If (-Not (Test-Path (Join-Path $RepoRoot ".git"))) { return }
    If (-Not (Get-Command git -ErrorAction SilentlyContinue)) { return }

    Push-Location $RepoRoot
    Try {
        $remoteUrl = git remote get-url $RemoteName 2>$null
        If ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($remoteUrl)) { return }
        If ($remoteUrl -like "git@github.com:*" -or $remoteUrl -like "ssh://git@github.com/*") {
            $httpsUrl = "https://github.com/vaila-multimodaltoolbox/vaila.git"
            Write-Host "Remote '$RemoteName' uses SSH ($remoteUrl)." -ForegroundColor Yellow
            Write-Host "Switching '$RemoteName' to HTTPS ($httpsUrl) so git pull / Check for Updates work without an SSH key..." -ForegroundColor Yellow
            git remote set-url $RemoteName $httpsUrl 2>$null
            If ($LASTEXITCODE -eq 0) {
                Write-Host "Remote '$RemoteName' now uses HTTPS." -ForegroundColor Green
            } Else {
                Write-Warning "Could not rewrite remote '$RemoteName' automatically. Run manually:"
                Write-Warning "  git -C `"$RepoRoot`" remote set-url $RemoteName $httpsUrl"
            }
        }
    } Finally {
        Pop-Location
    }
}

trap {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host "ERROR: An error occurred during installation!" -ForegroundColor Red
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host "Details: $_" -ForegroundColor Red
    Write-Host "Line: $($_.InvocationInfo.ScriptLineNumber)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Press Enter to close this window..." -ForegroundColor Yellow
    Read-Host
    exit 1
}

# Enable TLS 1.2 (and 1.3 if available) for HTTPS
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072

# Mark-of-the-Web: scripts downloaded to TEMP need Unblock-File before execution
If ($PSCommandPath -and ($PSCommandPath -like "$env:TEMP*")) {
    Unblock-File -Path $PSCommandPath -ErrorAction SilentlyContinue
}

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

# irm|iex has empty $PSScriptRoot — fall back to the caller's working directory.
$cwd = (Get-Location).Path
$scriptRoot = $PSScriptRoot
$runningViaIex = $false
If ([string]::IsNullOrWhiteSpace($scriptRoot)) {
    $scriptRoot = $cwd
    $runningViaIex = $true
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "vaila - Multimodal Toolbox Installation/Update (uv)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This script will install or update vaila."
Write-Host "If vaila is already installed, it will be updated with the latest code."
Write-Host ""
If ($runningViaIex) {
    Write-Host "Detected irm|iex (no script path). Using working directory: $scriptRoot" -ForegroundColor Yellow
    Write-Host ""
}

# ============================================================================
# PREREQUISITES: PowerShell 7 (pwsh), Git, Node.js
# ============================================================================
# A freshly-installed ("naked") Windows machine ships with none of these.
# Git is required below to clone the repo (Bootstrap Mode); pwsh is what the
# generated shortcuts/run scripts target; Node.js is not needed by vaila's
# Python toolchain but is a common companion dependency (e.g. Claude Code
# CLI) so we install it too, best-effort. Must run BEFORE any git usage.

function Update-SessionPath {
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
}

function Install-WingetPrerequisite {
    param(
        [string]$Name,
        [string]$CommandName,
        [string]$WingetId,
        [switch]$Optional
    )

    If (Get-Command $CommandName -ErrorAction SilentlyContinue) {
        Write-Host "$Name is already installed." -ForegroundColor Green
        return $true
    }

    Write-Host "$Name not found." -ForegroundColor Yellow
    $wingetAvailable = Get-Command winget -ErrorAction SilentlyContinue
    If (-Not $wingetAvailable) {
        If ($Optional) {
            Write-Warning "$Name is missing and winget is not available to install it. Skipping (optional)."
        } Else {
            Write-Warning "$Name is missing and winget is not available. Install App Installer from the Microsoft Store (https://aka.ms/getwinget), then re-run this script."
        }
        return $false
    }

    Write-Host "Installing $Name via winget ($WingetId)..." -ForegroundColor Cyan
    Try {
        & winget install --id $WingetId -e --silent --accept-package-agreements --accept-source-agreements 2>&1 | ForEach-Object { Write-Host $_ }
    } Catch {
        Write-Warning "winget install of $Name failed: $_"
    }
    Start-Sleep -Seconds 3
    Update-SessionPath
    If (Get-Command $CommandName -ErrorAction SilentlyContinue) {
        Write-Host "$Name installed successfully." -ForegroundColor Green
        return $true
    } Else {
        Write-Warning "$Name install could not be verified (may need admin rights or a new terminal session)."
        return $false
    }
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Checking prerequisites (PowerShell 7, Git, Node.js)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$null = Install-WingetPrerequisite -Name "Git" -CommandName "git" -WingetId "Git.Git"
$script:PwshOk = Install-WingetPrerequisite -Name "PowerShell 7" -CommandName "pwsh" -WingetId "Microsoft.PowerShell"
$null = Install-WingetPrerequisite -Name "Node.js LTS" -CommandName "node" -WingetId "OpenJS.NodeJS.LTS" -Optional

# Shortcuts / run scripts prefer pwsh (PowerShell 7) but fall back to Windows
# PowerShell so a machine where the pwsh install failed (no admin, no winget,
# offline) is still left with working shortcuts instead of dead ones.
If ($script:PwshOk -or (Get-Command pwsh -ErrorAction SilentlyContinue)) {
    $script:PwshExe = "pwsh.exe"
} Else {
    Write-Warning "pwsh.exe is not available — generated shortcuts will use powershell.exe instead."
    $script:PwshExe = "powershell.exe"
}
Write-Host ""

# ============================================================================
# INSTALL LOCATION
# ============================================================================

# Decide portable target early for the prompt label:
# - Already inside a vaila clone → that folder
# - Otherwise → .\vaila under the current directory
$portableHint = $scriptRoot
If (-Not (Test-Path (Join-Path $scriptRoot "pyproject.toml"))) {
    $portableHint = Join-Path $cwd "vaila"
}

# Map CLI/Inno -InstallLocation to the interactive option codes:
#   1 = portable (script/cwd dir), 2 = profile/Program Files
$installLocOption = $null
If ($InstallLocation -eq "portable") {
    $installLocOption = "1"
    Write-Host "Install location preset: portable ($portableHint)" -ForegroundColor Green
} ElseIf ($InstallLocation -eq "profile") {
    $installLocOption = "2"
    Write-Host "Install location preset: profile/Program Files" -ForegroundColor Green
} Else {
    Write-Host "---------------------------------------------" -ForegroundColor Cyan
    Write-Host "Install Location Selection" -ForegroundColor Cyan
    Write-Host "  [1] Current Directory ($portableHint) - Local/Portable - Recommended" -ForegroundColor Yellow
    Write-Host "  [2] User Profile or Program Files" -ForegroundColor Yellow
    Write-Host "---------------------------------------------" -ForegroundColor Cyan
    $installLocOption = Read-Host "Choose an option [1-2] (default: 1)"
    If ([string]::IsNullOrWhiteSpace($installLocOption)) {
        $installLocOption = "1"
    }
}

If ($installLocOption -eq "2") {
    If ($isAdmin) {
        $vailaProgramPath = "${env:ProgramFiles}\vaila"
        Write-Host "Installation location: $vailaProgramPath" -ForegroundColor Green
        Write-Host "(Administrator privileges detected - using Program Files)" -ForegroundColor Green
    } Else {
        $vailaProgramPath = "$env:USERPROFILE\vaila"
        Write-Host "Installation location: $vailaProgramPath" -ForegroundColor Yellow
        Write-Host "(No administrator privileges - using user directory)" -ForegroundColor Yellow
        Write-Host "Note: Some features (FFmpeg, Windows Terminal installation) may be skipped." -ForegroundColor Yellow
        Write-Host "Run as administrator for installation to Program Files." -ForegroundColor Yellow
    }
} Else {
    # Portable: use clone dir if present, else create/use .\vaila under cwd
    If (Test-Path (Join-Path $scriptRoot "pyproject.toml")) {
        $vailaProgramPath = $scriptRoot
    } ElseIf (Test-Path (Join-Path $cwd "pyproject.toml")) {
        $vailaProgramPath = $cwd
    } Else {
        $vailaProgramPath = Join-Path $cwd "vaila"
    }
    Write-Host "Installing in current directory: $vailaProgramPath" -ForegroundColor Green
    Write-Host "(Local/Portable mode - default)" -ForegroundColor Green
}
Write-Host ""

$sourcePath = $cwd
$projectDir = $cwd
# Prefer an existing clone as the source tree when present
If (Test-Path (Join-Path $scriptRoot "pyproject.toml")) {
    $projectDir = $scriptRoot
} ElseIf (Test-Path (Join-Path $cwd "pyproject.toml")) {
    $projectDir = $cwd
} ElseIf (Test-Path (Join-Path $vailaProgramPath "pyproject.toml")) {
    $projectDir = $vailaProgramPath
}

Repair-GitLfsCheckoutIfNeeded -RepoRoot $projectDir

# Bootstrap: clone repo if destination (or cwd) has no pyproject.toml
If (-Not (Test-Path "$projectDir\pyproject.toml")) {
    Write-Host "Bootstrap Mode: vaila source not found locally." -ForegroundColor Cyan
    Write-Host "Cloning vaila repository from GitHub..." -ForegroundColor Cyan

    If (-Not (Get-Command git -ErrorAction SilentlyContinue)) {
         Write-Error "git is not installed and could not be auto-installed (see prerequisites check above). Install it manually: https://git-scm.com/download/win"
         Exit 1
    }

    $cloneTarget = $vailaProgramPath
    # Profile/Program Files: clone to a temp dir, then the nested installer copies
    If ($installLocOption -eq "2") {
        $cloneTarget = Join-Path ([System.IO.Path]::GetTempPath()) "vaila_install_temp"
        If (Test-Path $cloneTarget) {
            Remove-Item -Path $cloneTarget -Recurse -Force -ErrorAction SilentlyContinue
        }
    } Else {
        If (-Not (Test-Path $cloneTarget)) {
            New-Item -ItemType Directory -Force -Path $cloneTarget | Out-Null
        }
        # If target exists but is empty-ish without pyproject, clone into it
        If (Test-Path (Join-Path $cloneTarget "pyproject.toml")) {
            Write-Host "Found existing clone at $cloneTarget" -ForegroundColor Green
        } ElseIf ((Get-ChildItem -Force $cloneTarget -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0) {
            # Non-empty without pyproject — use a subfolder only if this is not already named vaila
            # Keep cloning into cloneTarget; git clone needs empty or new dir
            $marker = Join-Path $cloneTarget ".git"
            If (-Not (Test-Path $marker)) {
                Write-Host "Target $cloneTarget is not empty and is not a git clone." -ForegroundColor Yellow
                Write-Host "Cloning into a fresh temp dir, then installing to the chosen location..." -ForegroundColor Yellow
                $cloneTarget = Join-Path ([System.IO.Path]::GetTempPath()) "vaila_install_temp"
                If (Test-Path $cloneTarget) {
                    Remove-Item -Path $cloneTarget -Recurse -Force -ErrorAction SilentlyContinue
                }
            }
        }
    }

    Write-Host "Downloading to: $cloneTarget" -ForegroundColor Yellow
    If (-Not (Test-Path (Join-Path $cloneTarget "pyproject.toml"))) {
        # git clone requires the destination not to exist, or be empty
        If ((Test-Path $cloneTarget) -and (Get-ChildItem -Force $cloneTarget -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0) {
            Remove-Item -Path $cloneTarget -Force -ErrorAction SilentlyContinue
        }
        If (Test-Path $cloneTarget) {
            Repair-SshOriginIfNeeded -RepoRoot $cloneTarget
            git -C $cloneTarget pull --ff-only 2>$null
            If (-Not (Test-Path (Join-Path $cloneTarget "pyproject.toml"))) {
                Remove-Item -Path $cloneTarget -Recurse -Force -ErrorAction SilentlyContinue
                git clone --depth 1 https://github.com/vaila-multimodaltoolbox/vaila.git $cloneTarget
            }
        } Else {
            git clone --depth 1 https://github.com/vaila-multimodaltoolbox/vaila.git $cloneTarget
        }
    }

    Write-Host "Running installer from downloaded source..." -ForegroundColor Cyan
    Set-Location $cloneTarget
    $nestedLocation = If ($installLocOption -eq "2") { "profile" } Else { "portable" }
    # When portable and we cloned straight into the final dir, nested install stays there.
    # When profile, nested install uses Program Files / user profile and copies from clone.
    & ".\install_vaila_win.ps1" -InstallLocation $nestedLocation
    Exit $LASTEXITCODE
}

# Check Windows version
Write-Host "Checking Windows version..." -ForegroundColor Yellow
$osVersion = [System.Environment]::OSVersion.Version
If ($osVersion.Major -lt 10) {
    Write-Warning "This application is optimized for Windows 10/11. You may experience compatibility issues."
    Write-Host "Current Windows version: $($osVersion.Major).$($osVersion.Minor)" -ForegroundColor Yellow
}

# Check available disk space
Write-Host "Checking available disk space..." -ForegroundColor Yellow
Try {
    $drive = (Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Root -like "$vailaProgramPath*" } | Select-Object -First 1)
    If (-Not $drive) {
        $drive = (Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Root -like "${env:ProgramFiles}*" } | Select-Object -First 1)
    }
    If ($drive) {
        $freeSpace = $drive.Free / 1GB
        If ($freeSpace -lt 2) {
            Write-Warning "Insufficient disk space. At least 2GB required. Available: $([math]::Round($freeSpace, 2))GB"
            Write-Host "Continuing anyway, but installation may fail..." -ForegroundColor Yellow
        }
    }
} Catch {
    Write-Warning "Could not check disk space. Continuing anyway..."
}

# Check internet connectivity
Write-Host "Checking internet connectivity..." -ForegroundColor Yellow
Try {
    $null = Invoke-WebRequest -Uri "https://www.google.com" -TimeoutSec 5 -UseBasicParsing
    Write-Host "Internet connection available." -ForegroundColor Green
} Catch {
    Write-Warning "No internet connection detected. Some features may not work properly."
}

# ============================================================================
# SHORTCUT / TERMINAL HELPERS
# ============================================================================

function New-DesktopShortcut {
    param(
        [string]$TargetPath,
        [string]$Arguments,
        [string]$IconPath,
        [string]$WorkingDirectory
    )

    Write-Host "Creating Desktop shortcut for 'vaila'..." -ForegroundColor Yellow
    $desktopPath = [Environment]::GetFolderPath("Desktop")
    $desktopShortcutPath = Join-Path $desktopPath "vaila.lnk"
    $wshell = New-Object -ComObject WScript.Shell
    $desktopShortcut = $wshell.CreateShortcut($desktopShortcutPath)
    $desktopShortcut.TargetPath = $TargetPath
    $desktopShortcut.Arguments = $Arguments

    $iconFile = $null
    $possibleIconPaths = @(
        "$vailaProgramPath\vaila\images\vaila.ico",
        "$vailaProgramPath\vaila\images\vaila_ico_trans.ico",
        "$vailaProgramPath\vaila\images\vaila_icon_win_original.ico",
        "$vailaProgramPath\docs\images\vaila_ico.ico",
        "$vailaProgramPath\docs\images\vaila_ico_trans.ico"
    )
    ForEach ($path in $possibleIconPaths) {
        If (Test-Path $path) {
            $iconFile = $path
            Break
        }
    }
    If ($iconFile) {
        $desktopShortcut.IconLocation = "$iconFile,0"
        Write-Host "Using icon: $iconFile" -ForegroundColor Green
    } ElseIf ($IconPath) {
        $desktopShortcut.IconLocation = "$IconPath,0"
    } Else {
        Write-Warning "Icon file not found. Shortcut will use default icon."
    }

    $desktopShortcut.WorkingDirectory = $WorkingDirectory
    $desktopShortcut.Save()
    Write-Host "Desktop shortcut for 'vaila' created at $desktopShortcutPath." -ForegroundColor Green
}

function New-StartMenuShortcut {
    param(
        [string]$TargetPath,
        [string]$Arguments,
        [string]$IconPath,
        [string]$WorkingDirectory
    )

    Write-Host "Creating Start Menu shortcut for 'vaila'..." -ForegroundColor Yellow
    If ($isAdmin) {
        $startMenuPath = "C:\ProgramData\Microsoft\Windows\Start Menu\Programs\vaila"
    } Else {
        $startMenuPath = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\vaila"
    }
    If (-Not (Test-Path $startMenuPath)) {
        New-Item -ItemType Directory -Force -Path $startMenuPath | Out-Null
    }
    $startMenuShortcutPath = "$startMenuPath\vaila.lnk"
    $wshell = New-Object -ComObject WScript.Shell
    $startMenuShortcut = $wshell.CreateShortcut($startMenuShortcutPath)
    $startMenuShortcut.TargetPath = $TargetPath
    $startMenuShortcut.Arguments = $Arguments

    $iconFile = $null
    $possibleIconPaths = @(
        "$vailaProgramPath\vaila\images\vaila.ico",
        "$vailaProgramPath\vaila\images\vaila_ico_trans.ico",
        "$vailaProgramPath\vaila\images\vaila_icon_win_original.ico",
        "$vailaProgramPath\docs\images\vaila_ico.ico",
        "$vailaProgramPath\docs\images\vaila_ico_trans.ico"
    )
    ForEach ($path in $possibleIconPaths) {
        If (Test-Path $path) {
            $iconFile = $path
            Break
        }
    }
    If ($iconFile) {
        $startMenuShortcut.IconLocation = "$iconFile,0"
    } ElseIf ($IconPath) {
        $startMenuShortcut.IconLocation = "$IconPath,0"
    }

    $startMenuShortcut.WorkingDirectory = $WorkingDirectory
    $startMenuShortcut.Save()
    Write-Host "Start Menu shortcut for 'vaila' created at $startMenuShortcutPath." -ForegroundColor Green
}

function Set-WindowsTerminalProfile {
    param(
        [string]$CommandLine,
        [string]$IconPath
    )

    $wtPath = "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe"
    If (Test-Path $wtPath) {
        Write-Host "Configuring the 'vaila' profile in Windows Terminal..." -ForegroundColor Yellow
        $settingsPath = "$wtPath\LocalState\settings.json"
        $settingsBackupPath = "$wtPath\LocalState\settings_backup.json"

        If (Test-Path $settingsPath) {
            Copy-Item -Path $settingsPath -Destination $settingsBackupPath -Force -ErrorAction SilentlyContinue
            $settingsContent = Get-Content -Path $settingsPath -Raw
            $settingsJson = $settingsContent | ConvertFrom-Json

            $settingsJson.profiles.list = $settingsJson.profiles.list | Where-Object { $_.name -ne "vaila" }

            $vailaProfile = @{
                name = "vaila"
                commandline = $CommandLine
                startingDirectory = "$vailaProgramPath"
                guid = "{17ce5bfe-17ed-5f3a-ab15-5cd5baafed5b}"
                hidden = $false
            }
            If ($IconPath -and (Test-Path $IconPath)) {
                $vailaProfile.icon = $IconPath
            }

            $settingsJson.profiles.list += $vailaProfile
            $settingsJson | ConvertTo-Json -Depth 100 | Out-File -FilePath $settingsPath -Encoding UTF8
            Write-Host "'vaila' profile added to Windows Terminal successfully." -ForegroundColor Green
        } Else {
            Write-Host "Windows Terminal settings.json not found. Skipping profile setup." -ForegroundColor Yellow
        }
    }
}

# ============================================================================
# UV INSTALLATION
# ============================================================================

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Installing vaila using uv" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Install uv if not present
Write-Host "Checking for uv installation..." -ForegroundColor Yellow
$uvInstalled = Get-Command uv -ErrorAction SilentlyContinue

If (-Not $uvInstalled) {
    Write-Host "uv is not installed. Installing uv..." -ForegroundColor Yellow

    $uvInstalledSuccessfully = $false

    If ($isAdmin) {
        $wingetAvailable = Get-Command winget -ErrorAction SilentlyContinue
        If ($wingetAvailable) {
            Write-Host "Attempting to install uv via winget..." -ForegroundColor Cyan
            Try {
                & winget install --id=astral-sh.uv -e --silent
                Start-Sleep -Seconds 3
                $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
                $uvInstalled = Get-Command uv -ErrorAction SilentlyContinue
                If ($uvInstalled) {
                    $uvInstalledSuccessfully = $true
                    Write-Host "uv installed successfully via winget!" -ForegroundColor Green
                }
            } Catch {
                Write-Host "winget installation failed, trying official installer..." -ForegroundColor Yellow
            }
        }
    }

    If (-Not $uvInstalledSuccessfully) {
        Write-Host "Using official installer..." -ForegroundColor Cyan
        Try {
            powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
            Start-Sleep -Seconds 3
            $uvInstalled = Get-Command uv -ErrorAction SilentlyContinue

            If (-Not $uvInstalled) {
                Write-Error "uv installation failed. Please install manually:"
                Write-Host "  powershell -ExecutionPolicy ByPass -c `"irm https://astral.sh/uv/install.ps1 | iex`"" -ForegroundColor Yellow
                Exit 1
            }
            Write-Host "uv installed successfully!" -ForegroundColor Green
        } Catch {
            Write-Error "Failed to install uv. Please install manually."
            Exit 1
        }
    }
} Else {
    Write-Host "uv is already installed." -ForegroundColor Green
    Write-Host "Updating uv..." -ForegroundColor Yellow
    Try {
        & uv self update
        Write-Host "uv updated successfully." -ForegroundColor Green
    } Catch {
        Write-Warning "Failed to update uv. Continuing with current version."
    }
}

$uvVersion = & uv --version 2>$null
If ($uvVersion) {
    Write-Host "uv version: $uvVersion" -ForegroundColor Green
}
Write-Host ""

# Install Python 3.12.14 via uv if needed
Write-Host "Checking Python version..." -ForegroundColor Yellow
Try {
    $pythonVersion = & uv python list 2>$null | Select-String "3.12.14"
    If (-Not $pythonVersion) {
        Write-Host "Python 3.12.14 not found. Installing via uv..." -ForegroundColor Yellow
        & uv python install 3.12.14
        Write-Host "Python 3.12.14 installed successfully." -ForegroundColor Green
    } Else {
        Write-Host "Python 3.12.14 found." -ForegroundColor Green
    }
} Catch {
    Write-Warning "Could not verify Python 3.12.14 installation. Continuing..."
}

# Check if we're already in the installation directory
$normalizedProjectDir = (Resolve-Path $projectDir -ErrorAction SilentlyContinue).Path
$normalizedVailaPath = (Resolve-Path $vailaProgramPath -ErrorAction SilentlyContinue).Path
If (-Not $normalizedProjectDir) { $normalizedProjectDir = $projectDir }
If (-Not $normalizedVailaPath) { $normalizedVailaPath = $vailaProgramPath }
$isAlreadyInstalled = ($normalizedProjectDir -eq $normalizedVailaPath)

If ($isAlreadyInstalled -or ($installLocOption -ne "2")) {
    Write-Host "Script is running from installation directory (or local install selected). Files are already in place." -ForegroundColor Green
    Write-Host "Skipping file copy step." -ForegroundColor Green
} Else {
    Write-Host ""
    If (Test-Path $vailaProgramPath) {
        Write-Host "Updating existing vaila installation in $vailaProgramPath..." -ForegroundColor Yellow
        Write-Host "Removing old files (keeping .venv to be recreated)..." -ForegroundColor Yellow
        Get-ChildItem -Path $vailaProgramPath -Exclude ".venv" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    } Else {
        Write-Host "Installing vaila to $vailaProgramPath..." -ForegroundColor Yellow
        New-Item -ItemType Directory -Force -Path $vailaProgramPath | Out-Null
    }

    Write-Host "Copying vaila files..." -ForegroundColor Yellow
    $excludeItems = @(".venv", "__pycache__", "*.pyc", ".git", "uv.lock", ".python-version")
    Get-ChildItem -Path $projectDir -Force | Where-Object {
        $item = $_
        $shouldExclude = $false
        ForEach ($exclude in $excludeItems) {
            If ($item.Name -like $exclude -or $item.Name -eq $exclude) {
                $shouldExclude = $true
                Break
            }
        }
        -Not $shouldExclude
    } | ForEach-Object {
        $targetPath = Join-Path $vailaProgramPath $_.Name
        If ($_.PSIsContainer) {
            Copy-Item -Path $_.FullName -Destination $targetPath -Recurse -Force -Exclude $excludeItems
        } Else {
            Copy-Item -Path $_.FullName -Destination $targetPath -Force
        }
    }
}

# Portable install into a git clone: keep committed uv.lock so `git pull` works.
$isGitTree = Test-Path (Join-Path $vailaProgramPath ".git")
$script:LockWasRegenerated = $false

If ($isGitTree) {
    Write-Host "Git working tree detected at $vailaProgramPath — keeping committed uv.lock." -ForegroundColor Green
    Repair-SshOriginIfNeeded -RepoRoot $vailaProgramPath
} ElseIf (Test-Path "$vailaProgramPath\uv.lock") {
    Write-Host "Removing uv.lock in profile/Program Files install (no git tree)..." -ForegroundColor Yellow
    Remove-Item -Path "$vailaProgramPath\uv.lock" -Force -ErrorAction SilentlyContinue
}

Set-Location $vailaProgramPath

# Set permissions for installation directory if needed
If ($isAdmin -and $vailaProgramPath -like "*Program Files*") {
    Write-Host "Setting permissions for installation directory..." -ForegroundColor Yellow
    Try {
        $acl = Get-Acl $vailaProgramPath
        $userGroup = "BUILTIN\Users"
        $permission = $userGroup, "FullControl", "ContainerInherit,ObjectInherit", "None", "Allow"
        $accessRule = New-Object System.Security.AccessControl.FileSystemAccessRule $permission
        $acl.AddAccessRule($accessRule)

        $adminGroup = "BUILTIN\Administrators"
        $adminPermission = $adminGroup, "FullControl", "ContainerInherit,ObjectInherit", "None", "Allow"
        $adminAccessRule = New-Object System.Security.AccessControl.FileSystemAccessRule $adminPermission
        $acl.AddAccessRule($adminAccessRule)

        Set-Acl $vailaProgramPath $acl
        Write-Host "Permissions set successfully (Users group granted FullControl)." -ForegroundColor Green
    } Catch {
        Write-Warning "Could not set permissions: $_"
        Write-Warning "You may need to manually grant 'Full Control' to 'Users' for '$vailaProgramPath'"
    }
}

# Select appropriate pyproject.toml template based on GPU detection and user choice
Write-Host ""
Write-Host "Selecting pyproject.toml configuration..." -ForegroundColor Yellow

$hasNvidiaGPU = Get-Command nvidia-smi -ErrorAction SilentlyContinue
$useGPU = $false

If ($hasNvidiaGPU) {
    Write-Host "NVIDIA GPU detected. Install with GPU support (CUDA 12.1)? [Y/n]" -ForegroundColor Cyan
    $gpuChoice = Read-Host
    $useGPU = ($gpuChoice -ne "n" -and $gpuChoice -ne "N")
} Else {
    Write-Host "No NVIDIA GPU detected. Using CPU-only configuration." -ForegroundColor Yellow
}

$useSamExtra = $false
Write-Host ""
Write-Host "Install optional SAM 3 (Meta) segmentation stack (extra 'sam')? [y/N]" -ForegroundColor Cyan
$samChoice = Read-Host
if ($samChoice -eq "y" -or $samChoice -eq "Y") {
    $useSamExtra = $true
}

$useSapiensExtra = $false
If ($useGPU) {
    Write-Host ""
    Write-Host "Install optional Sapiens2 Pose (Meta 308-keypoint pose, extra 'sapiens', CUDA)? [y/N]" -ForegroundColor Cyan
    $sapiensChoice = Read-Host
    if ($sapiensChoice -eq "y" -or $sapiensChoice -eq "Y") {
        $useSapiensExtra = $true
    }
}

$useFifaExtra = $false
If ($useGPU) {
    Write-Host ""
    Write-Host "Install optional FIFA Skeletal Tracking Light / SAM 3D Body stack (markerless 3D mesh, extra 'fifa', CUDA)? [y/N]" -ForegroundColor Cyan
    $fifaChoice = Read-Host
    if ($fifaChoice -eq "y" -or $fifaChoice -eq "Y") {
        $useFifaExtra = $true
    }
}

# Choose template
If ($useGPU) {
    If (Test-Path "$vailaProgramPath\pyproject_win_cuda12.toml") {
        Copy-Item "$vailaProgramPath\pyproject_win_cuda12.toml" "$vailaProgramPath\pyproject.toml" -Force
        Write-Host "Using Windows CUDA 12.1 configuration." -ForegroundColor Green
    } Else {
        Write-Warning "pyproject_win_cuda12.toml not found. Using CPU-only configuration."
        Copy-Item "$vailaProgramPath\pyproject_universal_cpu.toml" "$vailaProgramPath\pyproject.toml" -Force
        $useGPU = $false
    }
} Else {
    Copy-Item "$vailaProgramPath\pyproject_universal_cpu.toml" "$vailaProgramPath\pyproject.toml" -Force
    Write-Host "Using CPU-only configuration." -ForegroundColor Green
}

# Initialize uv project
Write-Host ""
Write-Host "Initializing uv project..." -ForegroundColor Yellow
If (-Not (Test-Path ".python-version")) {
    & uv python pin 3.12.14
}

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment (.venv)..." -ForegroundColor Yellow
If (Test-Path ".venv") {
    Write-Host "Virtual environment already exists. Removing old one..." -ForegroundColor Yellow
    Try {
        Remove-Item -Path ".venv" -Recurse -Force -ErrorAction Stop
    } Catch {
        Write-Warning "Could not remove existing .venv. Attempting to create new .venv anyway..."
    }
}

Try {
    & uv venv --python 3.12.14
    If ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment."
        Exit 1
    }
    Write-Host "Virtual environment created successfully." -ForegroundColor Green
} Catch {
    Write-Error "Failed to create virtual environment: $_"
    Exit 1
}

# Lock + sync (same policy as Linux/mac installers)
Write-Host ""
If ($isGitTree) {
    If ($useGPU) {
        Write-Host "CUDA template selected in a git clone — regenerating uv.lock for this machine..." -ForegroundColor Yellow
        & uv lock
        $script:LockWasRegenerated = $true
    } Else {
        Write-Host "Using committed uv.lock (no uv lock --upgrade) so git pull is not blocked." -ForegroundColor Green
    }
} Else {
    Write-Host "Generating lock file (uv.lock)..." -ForegroundColor Yellow
    & uv lock --upgrade
    $script:LockWasRegenerated = $true
}

# Sync dependencies
Write-Host ""
Write-Host "Installing vaila dependencies with uv..." -ForegroundColor Yellow
Write-Host "This may take a few minutes on first run..." -ForegroundColor Yellow

function Invoke-VailaUvSync {
    param([switch]$Frozen)
    $syncArgs = @("sync")
    If ($Frozen) { $syncArgs += "--frozen" }
    if ($useGPU) { $syncArgs += @("--extra", "gpu") }
    if ($useSamExtra) { $syncArgs += @("--extra", "sam") }
    if ($useSapiensExtra) { $syncArgs += @("--extra", "sapiens") }
    if ($useFifaExtra) { $syncArgs += @("--extra", "fifa") }
    & uv @syncArgs
    return $LASTEXITCODE
}

Try {
    $wantFrozen = ($isGitTree -and -not $script:LockWasRegenerated)
    $syncExitCode = Invoke-VailaUvSync -Frozen:$wantFrozen

    If ($syncExitCode -ne 0 -and $wantFrozen) {
        Write-Host "Frozen sync failed — retrying without --frozen (may update uv.lock)..." -ForegroundColor Yellow
        $syncExitCode = Invoke-VailaUvSync
        If ($syncExitCode -eq 0) {
            $script:LockWasRegenerated = $true
        }
    }

    If ($syncExitCode -ne 0) {
        throw "uv sync failed with exit code $syncExitCode"
    }
    Write-Host "Dependencies installed successfully." -ForegroundColor Green

    # Verify + repair CUDA wheel integrity (GPU template only).
    # Real bug hit in production (Linux): uv sync can report "nothing to do" for
    # an nvidia-*-cu12 package whose dist-info is present but whose actual .dll/.so
    # payload is missing on disk (broken hardlink / interrupted extraction / disk
    # full) -- import torch then fails, invisible to uv's own bookkeeping.
    if ($useGPU) {
        Write-Host ""
        Write-Host "Verifying NVIDIA/PyTorch CUDA wheel integrity..." -ForegroundColor Yellow
        Try {
            $broken = & uv run python bin\verify_cuda_libs.py --quiet 2>$null
            $broken = $broken | Where-Object { $_ -and $_.Trim() -ne "" }
            If ($broken) {
                Write-Warning "Corrupted CUDA wheels detected (metadata present, files missing): $($broken -join ' ')"
                Write-Host "Reinstalling only the broken packages..." -ForegroundColor Yellow
                $repairArgs = @("sync")
                ForEach ($pkg in $broken) { $repairArgs += @("--reinstall-package", $pkg) }
                if ($useGPU) { $repairArgs += @("--extra", "gpu") }
                if ($useSamExtra) { $repairArgs += @("--extra", "sam") }
                if ($useSapiensExtra) { $repairArgs += @("--extra", "sapiens") }
                if ($useFifaExtra) { $repairArgs += @("--extra", "fifa") }
                & uv @repairArgs
                $stillBroken = & uv run python bin\verify_cuda_libs.py --quiet 2>$null
                $stillBroken = $stillBroken | Where-Object { $_ -and $_.Trim() -ne "" }
                If ($stillBroken) {
                    Write-Warning "Still broken after reinstall: $($stillBroken -join ' ')"
                    Write-Warning "Check disk space and, if the uv cache and .venv are on different drives, try: `$env:UV_LINK_MODE = 'copy'  then re-run this installer."
                } Else {
                    Write-Host "CUDA wheel integrity repaired." -ForegroundColor Green
                }
            } Else {
                Write-Host "CUDA wheel integrity verified." -ForegroundColor Green
            }
        } Catch {
            Write-Warning "Could not run CUDA wheel integrity check: $_"
        }
        & uv run python -c "import torch; print('torch', torch.__version__, '- CUDA available:', torch.cuda.is_available())"
        If ($LASTEXITCODE -ne 0) {
            Write-Warning "torch import still failing after CUDA wheel repair -- see errors above."
        }
    }

    # Verify + repair the Sapiens2 editable install.
    # Real bug hit in production (Linux): `uv sync` (even with --extra sapiens) does
    # not know about the local editable checkout at .local\third_party\sapiens2 --
    # a plain sync can silently drop it. Re-register it if the checkout exists on
    # disk but the package no longer imports (cheap, no network).
    if ($useSapiensExtra) {
        $sapiensImportOk = $false
        Try {
            & uv run python -c "import sapiens" 2>$null | Out-Null
            $sapiensImportOk = ($LASTEXITCODE -eq 0)
        } Catch {
            $sapiensImportOk = $false
        }
        If (-Not $sapiensImportOk) {
            $sapiensCheckout = Join-Path $vailaProgramPath ".local\third_party\sapiens2"
            If (Test-Path $sapiensCheckout) {
                Write-Warning "sapiens checkout exists but is not importable -- re-registering editable install..."
                Try {
                    & uv pip install -e $sapiensCheckout
                    Write-Host "sapiens editable install repaired." -ForegroundColor Green
                } Catch {
                    Write-Warning "Failed to repair sapiens editable install; run: pwsh bin\setup_sapiens2.ps1"
                }
            }
        }
    }

    if ($useSamExtra) {
        Write-Host ""
        Write-Host "------------------------------------------------------------" -ForegroundColor Cyan
        Write-Host "SAM 3: accept https://huggingface.co/facebook/sam3 then run:" -ForegroundColor Cyan
        Write-Host "  cd `"$vailaProgramPath`" ; uv run hf auth login" -ForegroundColor Cyan
        Write-Host "------------------------------------------------------------" -ForegroundColor Cyan
        $hfNow = Read-Host "Run 'uv run hf auth login' now? [y/N]"
        if ($hfNow -eq "y" -or $hfNow -eq "Y") {
            Set-Location $vailaProgramPath
            try {
                & uv run hf auth login
            } catch {
                Write-Warning "hf auth login failed or was cancelled."
            }
        }
    }

    if ($useSapiensExtra) {
        Write-Host ""
        Write-Host "------------------------------------------------------------" -ForegroundColor Cyan
        Write-Host "Sapiens2 Pose (optional): clone + weights via bin/setup_sapiens2.ps1" -ForegroundColor Cyan
        Write-Host "  - Clones facebookresearch/sapiens2 into .local\third_party\sapiens2 (editable install)" -ForegroundColor Cyan
        Write-Host "  - Downloads pose (1B default) + DETR detector to vaila\models\sapiens2\" -ForegroundColor Cyan
        Write-Host "  - GUI: Frame B -> YOLO + FB -> Sapiens2 Pose" -ForegroundColor Cyan
        Write-Host "  - License: Meta Sapiens2 License (not AGPL) — see vaila\help\vaila_sapiens.md" -ForegroundColor Cyan
        Write-Host "------------------------------------------------------------" -ForegroundColor Cyan
        $sapiensSetupNow = Read-Host "Run 'bin/setup_sapiens2.ps1' now from $vailaProgramPath? [y/N]"
        if ($sapiensSetupNow -eq "y" -or $sapiensSetupNow -eq "Y") {
            $sapiensScript = Join-Path $vailaProgramPath "bin\setup_sapiens2.ps1"
            If (Test-Path $sapiensScript) {
                Try {
                    Set-Location $vailaProgramPath
                    & $sapiensScript
                } Catch {
                    Write-Warning "setup_sapiens2.ps1 failed or was cancelled. You can run it later:"
                    Write-Warning "  cd `"$vailaProgramPath`" ; pwsh bin\setup_sapiens2.ps1"
                }
            } Else {
                Write-Warning "bin\setup_sapiens2.ps1 not found. Run manually after updating the repo."
            }
        }
    }

    if ($useFifaExtra) {
        Write-Host ""
        Write-Host "------------------------------------------------------------" -ForegroundColor Cyan
        Write-Host "FIFA Skeletal Tracking Light / SAM 3D Body (optional): clone + weights via bin/setup_fifa_sam3d.ps1" -ForegroundColor Cyan
        Write-Host "  - Clones facebookresearch/sam-3d-body into sam_3d_body/ (NOT pip-installable; runtime deps only)" -ForegroundColor Cyan
        Write-Host "  - Downloads gated facebook/sam-3d-body-dinov3 weights into vaila\models\sam-3d-dinov3\" -ForegroundColor Cyan
        Write-Host "  - GUI: Frame B -> YOLO + FB -> SAM3+DINOv3 3D (markerless 3D mesh + metric joints)" -ForegroundColor Cyan
        Write-Host "  - License: SAM 3D Body keeps its Meta license (not AGPL) — see vaila\help\sam3dinov3.md" -ForegroundColor Cyan
        Write-Host "------------------------------------------------------------" -ForegroundColor Cyan
        $fifaSetupNow = Read-Host "Run 'bin/setup_fifa_sam3d.ps1' now from $vailaProgramPath? [y/N]"
        if ($fifaSetupNow -eq "y" -or $fifaSetupNow -eq "Y") {
            $fifaScript = Join-Path $vailaProgramPath "bin\setup_fifa_sam3d.ps1"
            If (Test-Path $fifaScript) {
                Try {
                    Set-Location $vailaProgramPath
                    & $fifaScript
                } Catch {
                    Write-Warning "setup_fifa_sam3d.ps1 failed or was cancelled. You can run it later:"
                    Write-Warning "  cd `"$vailaProgramPath`" ; pwsh bin\setup_fifa_sam3d.ps1"
                }
            } Else {
                Write-Warning "bin\setup_fifa_sam3d.ps1 not found. Run manually after updating the repo."
            }
        }
    }
} Catch {
    Write-Warning "uv sync failed. Restoring universal CPU configuration..."
    Copy-Item "$vailaProgramPath\pyproject_universal_cpu.toml" "$vailaProgramPath\pyproject.toml" -Force
    Write-Error "Installation failed. Please check the error messages above."
    Exit 1
}

Write-Host ""
Write-Host "PyTorch, torchvision, torchaudio, ultralytics, and boxmot are installed via uv sync from pyproject.toml." -ForegroundColor Green

# Install pycairo
Write-Host ""
Write-Host "Installing pycairo..." -ForegroundColor Yellow
Try {
    & uv pip install pycairo
    Write-Host "pycairo installed successfully." -ForegroundColor Green
} Catch {
    Write-Warning "pycairo installation failed. Trying with force-reinstall..."
    Try {
        & uv pip install --force-reinstall --no-cache-dir pycairo
        Write-Host "pycairo installed successfully." -ForegroundColor Green
    } Catch {
        Write-Warning "pycairo installation failed. This may cause issues with the application."
    }
}

# Verify environment is properly set up by checking for PIL (Pillow)
Write-Host ""
Write-Host "Verifying environment setup..." -ForegroundColor Yellow
Try {
    $testResult = & uv run python -c "import PIL; print('PIL OK')" 2>&1
    If ($testResult -match "PIL OK") {
        Write-Host "Environment verification successful." -ForegroundColor Green
    } Else {
        Write-Warning "Environment verification failed. PIL module not found. Running uv sync again..."
        $retrySyncCode = Invoke-VailaUvSync
        If ($retrySyncCode -ne 0) {
            Write-Error "Failed to sync dependencies during verification."
            Exit 1
        }
    }
} Catch {
    Write-Warning "Could not verify environment. Continuing anyway..."
}

# Create run_vaila.ps1 script
$runScript = Join-Path $vailaProgramPath "run_vaila.ps1"
Write-Host ""
Write-Host "Creating run_vaila.ps1 script..." -ForegroundColor Yellow
@"
# Run vaila using uv
Set-Location "$vailaProgramPath"
& uv run --no-sync "$vailaProgramPath\vaila.py"
# Keep terminal open after execution
Write-Host ""
Write-Host "Program finished. Press Enter to close this window..." -ForegroundColor Yellow
Read-Host
"@ | Out-File -FilePath $runScript -Encoding UTF8

# Create run_vaila.bat script
$runScriptBat = Join-Path $vailaProgramPath "run_vaila.bat"
Write-Host "Creating run_vaila.bat script..." -ForegroundColor Yellow
@"
@echo off
cd /d "$vailaProgramPath"
$script:PwshExe -ExecutionPolicy Bypass -File "run_vaila.ps1"
pause
"@ | Out-File -FilePath $runScriptBat -Encoding ASCII

# Find icon for Windows Terminal
$wtIconPath = $null
$possibleWtIconPaths = @(
    "$vailaProgramPath\vaila\images\vaila_ico.png",
    "$vailaProgramPath\vaila\images\vaila.ico",
    "$vailaProgramPath\vaila\images\vaila_ico_trans.ico",
    "$vailaProgramPath\docs\images\vaila_ico.ico",
    "$vailaProgramPath\docs\images\vaila_ico.png"
)
ForEach ($path in $possibleWtIconPaths) {
    If (Test-Path $path) {
        $wtIconPath = $path
        Break
    }
}

# Setup Windows Terminal profile
$wtCommandLine = "$script:PwshExe -ExecutionPolicy Bypass -NoExit -File `"$runScript`""
Set-WindowsTerminalProfile -CommandLine $wtCommandLine -IconPath $wtIconPath

# Create shortcuts
New-DesktopShortcut -TargetPath $script:PwshExe -Arguments "-ExecutionPolicy Bypass -NoExit -File `"$runScript`"" -WorkingDirectory $vailaProgramPath
New-StartMenuShortcut -TargetPath $script:PwshExe -Arguments "-ExecutionPolicy Bypass -NoExit -File `"$runScript`"" -WorkingDirectory $vailaProgramPath

# ============================================================================
# SYSTEM DEPENDENCIES (FFmpeg, Windows Terminal, rsync/scp)
# ============================================================================

Write-Host ""
Write-Host "Checking/installing system dependencies (FFmpeg, Windows Terminal, rsync)..." -ForegroundColor Yellow

# FFmpeg
$ffmpegInstalled = Get-Command ffmpeg -ErrorAction SilentlyContinue
If ($ffmpegInstalled) {
    Write-Host "FFmpeg is already installed." -ForegroundColor Green
} Else {
    If ($isAdmin) {
        Write-Host "FFmpeg is not installed. Installing via winget..." -ForegroundColor Yellow
        Try {
            & winget install --id Gyan.FFmpeg -e --source winget --silent
            Write-Host "FFmpeg installed successfully." -ForegroundColor Green
        } Catch {
            Write-Warning "Failed to install FFmpeg via winget."
        }
    } Else {
        Write-Host "FFmpeg is not installed. Administrator privileges required for installation." -ForegroundColor Yellow
    }
}

# Windows Terminal
$wtInstalled = Get-Command wt.exe -ErrorAction SilentlyContinue
If ($wtInstalled) {
    Write-Host "Windows Terminal is already installed." -ForegroundColor Green
} Else {
    If ($isAdmin) {
        Write-Host "Windows Terminal is not installed. Installing via winget..." -ForegroundColor Yellow
        Try {
            & winget install --id Microsoft.WindowsTerminal -e --source winget --silent
            Write-Host "Windows Terminal installed successfully." -ForegroundColor Green
        } Catch {
            Write-Warning "Failed to install Windows Terminal via winget."
        }
    } Else {
        Write-Host "Windows Terminal is not installed. Administrator privileges required for installation." -ForegroundColor Yellow
    }
}

# rsync and OpenSSH Client (for transfer scripts)
If ($isAdmin) {
    Write-Host "Checking for file transfer tools (rsync/scp)..." -ForegroundColor Yellow
    $rsyncInstalled = Get-Command rsync -ErrorAction SilentlyContinue
    $scpInstalled = Get-Command scp -ErrorAction SilentlyContinue

    If (-Not $rsyncInstalled) {
        Write-Host "rsync is not installed. Attempting installation..." -ForegroundColor Yellow

        $wingetAvailable = Get-Command winget -ErrorAction SilentlyContinue
        If ($wingetAvailable) {
            Write-Host "Trying to install rsync via winget..." -ForegroundColor Cyan
            Try {
                & winget install --id=Git.Git -e --silent --accept-package-agreements --accept-source-agreements
                Start-Sleep -Seconds 3
                $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
                $rsyncInstalled = Get-Command rsync -ErrorAction SilentlyContinue
                If ($rsyncInstalled) {
                    Write-Host "rsync installed successfully via Git for Windows (winget)." -ForegroundColor Green
                }
            } Catch {
                Write-Host "winget installation failed, trying Chocolatey..." -ForegroundColor Yellow
            }
        }

        If (-Not $rsyncInstalled) {
            $chocoInstalled = Get-Command choco -ErrorAction SilentlyContinue
            If (-Not $chocoInstalled) {
                Write-Host "Installing Chocolatey..." -ForegroundColor Yellow
                Try {
                    Set-ExecutionPolicy Bypass -Scope Process -Force
                    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
                    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
                    Write-Host "Chocolatey installed successfully." -ForegroundColor Green
                } Catch {
                    Write-Warning "Failed to install Chocolatey."
                }
            }

            If (Get-Command choco -ErrorAction SilentlyContinue) {
                Write-Host "Installing rsync via Chocolatey..." -ForegroundColor Yellow
                Try {
                    choco install rsync -y
                    Start-Sleep -Seconds 3
                    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
                    $rsyncInstalled = Get-Command rsync -ErrorAction SilentlyContinue
                    If ($rsyncInstalled) {
                        Write-Host "rsync installed successfully via Chocolatey." -ForegroundColor Green
                    } Else {
                        Write-Warning "Chocolatey installation completed but rsync not found in PATH."
                    }
                } Catch {
                    Write-Warning "Failed to install rsync via Chocolatey."
                }
            }
        }
    } Else {
        Write-Host "rsync is already installed." -ForegroundColor Green
    }

    If (-Not $scpInstalled) {
        Write-Host "Checking for OpenSSH Client (includes scp)..." -ForegroundColor Yellow
        Try {
            $opensshStatus = Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Client*'
            If ($opensshStatus -and $opensshStatus.State -ne 'Installed') {
                Write-Host "Installing OpenSSH Client (includes scp)..." -ForegroundColor Yellow
                Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0
                Start-Sleep -Seconds 2
                $scpInstalled = Get-Command scp -ErrorAction SilentlyContinue
                If ($scpInstalled) {
                    Write-Host "OpenSSH Client installed successfully. scp is now available." -ForegroundColor Green
                }
            } ElseIf ($opensshStatus -and $opensshStatus.State -eq 'Installed') {
                $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
                $scpInstalled = Get-Command scp -ErrorAction SilentlyContinue
                If ($scpInstalled) {
                    Write-Host "OpenSSH Client is installed. scp is available." -ForegroundColor Green
                }
            }
        } Catch {
            Write-Warning "Could not install OpenSSH Client automatically. You can install it manually:"
            Write-Host "  Settings > Apps > Optional Features > Add OpenSSH Client" -ForegroundColor Cyan
            Write-Host "  Or run: dism /online /Add-Capability /CapabilityName:OpenSSH.Client~~~~0.0.1.0" -ForegroundColor Cyan
        }
    } Else {
        Write-Host "scp (OpenSSH Client) is already installed." -ForegroundColor Green
    }

    Write-Host ""
    If ($rsyncInstalled) {
        Write-Host "[OK] rsync is available for file transfers." -ForegroundColor Green
    } ElseIf ($scpInstalled) {
        Write-Host "[OK] scp is available for file transfers (rsync not installed)." -ForegroundColor Yellow
        Write-Host "  Note: The transfer script will use scp as an alternative to rsync." -ForegroundColor Yellow
    } Else {
        Write-Warning "Neither rsync nor scp is available. File transfer feature may not work."
        Write-Host "  To enable file transfers, install one of:" -ForegroundColor Yellow
        Write-Host "    - rsync: via Git for Windows, Chocolatey, or WSL" -ForegroundColor Cyan
        Write-Host "    - scp: Enable OpenSSH Client in Windows Optional Features" -ForegroundColor Cyan
    }
} Else {
    Write-Host "Skipping rsync/scp installation (requires Administrator privileges)." -ForegroundColor Yellow
    Write-Host "  File transfer feature will use scp if OpenSSH Client is already installed." -ForegroundColor Yellow
}

# Ensure correct permissions
Write-Host "Ensuring correct permissions for the application..." -ForegroundColor Yellow
Try {
    $acl = Get-Acl $vailaProgramPath
    $permission = "$env:USERNAME", "FullControl", "ContainerInherit,ObjectInherit", "None", "Allow"
    $accessRule = New-Object System.Security.AccessControl.FileSystemAccessRule $permission
    $acl.SetAccessRule($accessRule)
    Set-Acl $vailaProgramPath $acl
    Write-Host "Permissions set successfully." -ForegroundColor Green
} Catch {
    Write-Warning "Failed to set permissions. Continuing anyway."
}


Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "vaila installation completed successfully!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
If ($isGitTree) {
    $lockDirty = $script:LockWasRegenerated
    If (-not $lockDirty) {
        Push-Location $vailaProgramPath
        Try {
            git diff --quiet -- uv.lock pyproject.toml 2>$null
            If ($LASTEXITCODE -ne 0) { $lockDirty = $true }
        } Catch {
            # ignore git probe failures
        }
        Pop-Location
    }
    If ($lockDirty) {
        Write-Host "NOTE (git pull): local pyproject.toml / uv.lock differ from the repo" -ForegroundColor Yellow
        Write-Host "(common after a CUDA template switch). Before pulling:" -ForegroundColor Yellow
        Write-Host "  git -C `"$vailaProgramPath`" restore uv.lock pyproject.toml" -ForegroundColor Cyan
        Write-Host "  git -C `"$vailaProgramPath`" pull" -ForegroundColor Cyan
        Write-Host "Then re-apply the platform template if needed:" -ForegroundColor Yellow
        Write-Host "  pwsh bin/setup_pyproject.ps1 -Target win-cuda -Yes" -ForegroundColor Cyan
        Write-Host ""
    } Else {
        Write-Host "Git tree kept clean for uv.lock — you can 'git pull' normally." -ForegroundColor Green
        Write-Host ""
    }
}
Write-Host "You can now launch vaila using:" -ForegroundColor Cyan
Write-Host "  - Desktop shortcut" -ForegroundColor Yellow
Write-Host "  - Start Menu shortcut" -ForegroundColor Yellow
Write-Host "  - Windows Terminal profile 'vaila'" -ForegroundColor Yellow
Write-Host "  - Double-click run_vaila.bat" -ForegroundColor Yellow
Write-Host "  - uv run vaila   (from `"$vailaProgramPath`", no activation needed)" -ForegroundColor Yellow
Write-Host ""
Write-Host "------------------------------------------------------------" -ForegroundColor Cyan
Write-Host "Activate the .venv manually (optional — the launchers above" -ForegroundColor Cyan
Write-Host "and 'uv run' do not require this):" -ForegroundColor Cyan
Write-Host "  PowerShell : .venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "               (blocked by policy? Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)" -ForegroundColor DarkYellow
Write-Host "  CMD        : .venv\Scripts\activate.bat" -ForegroundColor Yellow
Write-Host "  Git Bash   : source .venv/Scripts/activate" -ForegroundColor Yellow
Write-Host "------------------------------------------------------------" -ForegroundColor Cyan
If ($isGitTree) {
    Write-Host "Keeping this git clone up to date:" -ForegroundColor Cyan
    Write-Host "  git pull --ff-only origin main   (or use the GUI's Check for Updates)" -ForegroundColor Yellow
    Write-Host "  uv self update                   (updates uv itself)" -ForegroundColor Yellow
    Write-Host "  uv sync                          (re-syncs deps to the committed uv.lock)" -ForegroundColor Yellow
    Write-Host "  uv sync --upgrade                (opt-in: also bumps deps beyond uv.lock)" -ForegroundColor Yellow
    Write-Host "  (there is no 'uv run sync' — 'sync' and 'self update' are uv subcommands, not scripts to run)" -ForegroundColor DarkYellow
    Write-Host ""
}
Write-Host "Restart your computer to ensure all changes take effect." -ForegroundColor Yellow
Write-Host ""
Pause
