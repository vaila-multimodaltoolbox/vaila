<#
    Script: install_vaila_win_uv.ps1
    Description: Installs the vaila - Multimodal Toolbox on Windows using uv
                 (ultra-fast Python package manager). This replaces the Conda-based
                 installation with a modern uv-based setup.
    Usage:
        1. Download the repository from GitHub manually and extract it.
        2. Open PowerShell (Administrator recommended for full installation)
        3. Navigate to the root directory of the extracted repository
        4. Run: .\install_vaila_win_uv.ps1
    Notes:
        - uv will be automatically installed if not present
        - Python 3.12.12 will be installed via uv if needed
        - This script is 10-100x faster than Conda-based installation
        - Installation location:
          * With admin: C:\Program Files\vaila (Windows standard location)
          * Without admin: C:\Users\<user>\vaila (user directory)
        - Can run without administrator privileges (some features may be skipped)
    Author: Prof. Dr. Paulo R. P. Santiago
    Creation: 17 November 2025
    Updated: 18 November 2025
    Version: 0.2.1
    OS: Windows 11
    Reference: https://docs.astral.sh/uv/
#>

$ErrorActionPreference = "Stop"

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "vaila - Multimodal Toolbox Installation/Update (uv-based)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
If ($isAdmin) {
    $installPath = "${env:ProgramFiles}\vaila"
    Write-Host "Installation location: $installPath" -ForegroundColor Green
    Write-Host "(Administrator privileges detected - using Program Files)" -ForegroundColor Green
} Else {
    $installPath = "$env:USERPROFILE\vaila"
    Write-Host "Installation location: $installPath" -ForegroundColor Yellow
    Write-Host "(No administrator privileges - using user directory)" -ForegroundColor Yellow
    Write-Host "Note: Some features (FFmpeg, Windows Terminal installation) may be skipped." -ForegroundColor Yellow
    Write-Host "Run as administrator for installation to Program Files (recommended)." -ForegroundColor Yellow
}
Write-Host ""
Write-Host "If vaila is already installed, it will be updated with the latest code."
Write-Host ""

# Check Windows version
Write-Host "Checking Windows version..." -ForegroundColor Yellow
$osVersion = [System.Environment]::OSVersion.Version
If ($osVersion.Major -lt 10) {
    Write-Warning "This application is optimized for Windows 10/11. You may experience compatibility issues."
    Write-Host "Current Windows version: $($osVersion.Major).$($osVersion.Minor)" -ForegroundColor Yellow
}

# Check available disk space
Write-Host "Checking available disk space..." -ForegroundColor Yellow
$drive = (Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Root -like "$env:LOCALAPPDATA*" } | Select-Object -First 1)
If ($drive) {
    $freeSpace = $drive.Free / 1GB
    If ($freeSpace -lt 2) {
        Write-Warning "Insufficient disk space. At least 2GB required. Available: $([math]::Round($freeSpace, 2))GB"
        Write-Host "Continuing anyway, but installation may fail..." -ForegroundColor Yellow
    }
}

# Check internet connectivity
Write-Host "Checking internet connectivity..." -ForegroundColor Yellow
Try {
    $null = Invoke-WebRequest -Uri "https://www.google.com" -TimeoutSec 5 -UseBasicParsing
    Write-Host "Internet connection available." -ForegroundColor Green
} Catch {
    Write-Warning "No internet connection detected. Some features may not work properly."
}

# Install uv if not present
Write-Host ""
Write-Host "Checking for uv installation..." -ForegroundColor Yellow
$uvInstalled = Get-Command uv -ErrorAction SilentlyContinue

If (-Not $uvInstalled) {
    Write-Host "uv is not installed. Installing uv..." -ForegroundColor Yellow
    
    $uvInstalledSuccessfully = $false
    
    # Try winget first if available and running as admin
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
    
    # Fallback to official installer if winget failed or not available
    If (-Not $uvInstalledSuccessfully) {
        Write-Host "Using official installer: powershell -ExecutionPolicy ByPass -c `"irm https://astral.sh/uv/install.ps1 | iex`"" -ForegroundColor Cyan
        Try {
            # Install uv using the official installer (recommended method)
            powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
            
            # Refresh PATH to include uv
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
            
            # Verify installation
            Start-Sleep -Seconds 3
            $uvInstalled = Get-Command uv -ErrorAction SilentlyContinue
            
            If (-Not $uvInstalled) {
                Write-Error "uv installation failed. Please install manually:"
                Write-Host "  powershell -ExecutionPolicy ByPass -c `"irm https://astral.sh/uv/install.ps1 | iex`"" -ForegroundColor Yellow
                Write-Host "  Or via winget: winget install --id=astral-sh.uv -e" -ForegroundColor Yellow
                Write-Host "  Or visit: https://docs.astral.sh/uv/getting-started/installation/" -ForegroundColor Yellow
                Exit 1
            }
            Write-Host "uv installed successfully!" -ForegroundColor Green
        } Catch {
            Write-Error "Failed to install uv. Please install manually:"
            Write-Host "  powershell -ExecutionPolicy ByPass -c `"irm https://astral.sh/uv/install.ps1 | iex`"" -ForegroundColor Yellow
            Write-Host "  Or via winget: winget install --id=astral-sh.uv -e" -ForegroundColor Yellow
            Write-Host "  Or visit: https://docs.astral.sh/uv/getting-started/installation/" -ForegroundColor Yellow
            Exit 1
        }
    }
} Else {
    Write-Host "uv is already installed." -ForegroundColor Green
}

# Get uv version
$uvVersion = & uv --version 2>$null
If ($uvVersion) {
    Write-Host "uv version: $uvVersion" -ForegroundColor Green
} Else {
    Write-Host "uv version: unknown" -ForegroundColor Yellow
}
Write-Host ""

# Configure PowerShell autocompletion for uv
Write-Host "Configuring PowerShell autocompletion for uv..." -ForegroundColor Yellow
Try {
    If (-Not (Test-Path -Path $PROFILE)) {
        New-Item -ItemType File -Path $PROFILE -Force | Out-Null
        Write-Host "Created PowerShell profile: $PROFILE" -ForegroundColor Green
    }
    
    # Check if autocompletion is already configured
    $profileContent = Get-Content -Path $PROFILE -ErrorAction SilentlyContinue
    $hasUvCompletion = $profileContent | Select-String -Pattern "uv generate-shell-completion"
    
    If (-Not $hasUvCompletion) {
        Add-Content -Path $PROFILE -Value "`n# uv shell autocompletion`n(& uv generate-shell-completion powershell) | Out-String | Invoke-Expression"
        Write-Host "PowerShell autocompletion for uv configured successfully." -ForegroundColor Green
        Write-Host "Note: Restart PowerShell or run: . `$PROFILE" -ForegroundColor Cyan
    } Else {
        Write-Host "PowerShell autocompletion for uv is already configured." -ForegroundColor Green
    }
} Catch {
    Write-Warning "Failed to configure PowerShell autocompletion for uv: $_"
    Write-Host "You can configure it manually by running:" -ForegroundColor Yellow
    Write-Host "  if (!(Test-Path -Path `$PROFILE)) { New-Item -ItemType File -Path `$PROFILE -Force }" -ForegroundColor Cyan
    Write-Host "  Add-Content -Path `$PROFILE -Value '(& uv generate-shell-completion powershell) | Out-String | Invoke-Expression'" -ForegroundColor Cyan
}
Write-Host ""

# Install Python 3.12.12 via uv if needed
Write-Host "Checking Python version..." -ForegroundColor Yellow
Try {
    $pythonVersion = & uv python list 2>$null | Select-String "3.12.12"
    If (-Not $pythonVersion) {
        Write-Host "Python 3.12.12 not found. Installing via uv..." -ForegroundColor Yellow
        & uv python install 3.12.12
        Write-Host "Python 3.12.12 installed successfully." -ForegroundColor Green
    } Else {
        Write-Host "Python 3.12.12 found." -ForegroundColor Green
    }
} Catch {
    Write-Warning "Could not verify Python 3.12.12 installation. Continuing..."
}

# Define paths
# First, try to detect if we're running from an installed location
$scriptPathForDetection = Split-Path -Parent $MyInvocation.MyCommand.Path

# Check if script is in a vaila installation directory
# Only check 64-bit Program Files location (x86 is not used for vaila)
$possibleInstallPaths = @(
    "${env:ProgramFiles}\vaila",
    "$env:USERPROFILE\vaila"
)

# Check if script is in any of the installation directories
$detectedPath = $null
ForEach ($path in $possibleInstallPaths) {
    If ($scriptPathForDetection -eq $path -and (Test-Path "$path\pyproject.toml")) {
        $detectedPath = $path
        Break
    }
}

If ($detectedPath) {
    # Reject x86 paths - vaila should only be installed in 64-bit Program Files
    If ($detectedPath -like "*Program Files (x86)*") {
        Write-Warning "Detected x86 installation path, but vaila should be installed in 64-bit Program Files."
        Write-Host "Switching to correct 64-bit installation directory..." -ForegroundColor Yellow
        If ($isAdmin) {
            $vailaProgramPath = "${env:ProgramFiles}\vaila"
        } Else {
            $vailaProgramPath = "$env:USERPROFILE\vaila"
        }
    } Else {
        $vailaProgramPath = $detectedPath
    }
    Write-Host "Using installation directory: $vailaProgramPath" -ForegroundColor Green
} Else {
    # Use C:\Program Files\vaila if admin (Windows standard 64-bit), otherwise use user directory
    If ($isAdmin) {
        $vailaProgramPath = "${env:ProgramFiles}\vaila"
        Write-Host "Using standard 64-bit installation directory: $vailaProgramPath" -ForegroundColor Green
    } Else {
        $vailaProgramPath = "$env:USERPROFILE\vaila"
        Write-Host "Using user installation directory: $vailaProgramPath" -ForegroundColor Green
    }
}

# Final safety check: ensure we never use x86 path
If ($vailaProgramPath -like "*Program Files (x86)*") {
    Write-Error "ERROR: Installation path cannot be in Program Files (x86). vaila must be installed in 64-bit Program Files."
    Write-Host "Correct path should be: ${env:ProgramFiles}\vaila" -ForegroundColor Yellow
    Exit 1
}
$sourcePath = (Get-Location).Path
$projectDir = $sourcePath

# Check if pyproject.toml exists
If (-Not (Test-Path "$projectDir\pyproject.toml")) {
    Write-Error "pyproject.toml not found in $projectDir"
    Write-Host "Please ensure you're running this script from the vaila project root." -ForegroundColor Yellow
    Exit 1
}

# Check if we're already in the installation directory (e.g., running from installer)
# Normalize paths for comparison (resolve to absolute paths and compare)
$normalizedProjectDir = (Resolve-Path $projectDir -ErrorAction SilentlyContinue).Path
$normalizedVailaPath = (Resolve-Path $vailaProgramPath -ErrorAction SilentlyContinue).Path
If (-Not $normalizedProjectDir) { $normalizedProjectDir = $projectDir }
If (-Not $normalizedVailaPath) { $normalizedVailaPath = $vailaProgramPath }
$isAlreadyInstalled = ($normalizedProjectDir -eq $normalizedVailaPath)

If ($isAlreadyInstalled) {
    Write-Host "Script is running from installation directory. Files are already in place." -ForegroundColor Green
    Write-Host "Skipping file copy step." -ForegroundColor Green
} Else {
    # Clean destination directory and copy the entire vaila program
    Write-Host ""
    If (Test-Path $vailaProgramPath) {
        Write-Host "Updating existing vaila installation in $vailaProgramPath..." -ForegroundColor Yellow
        Write-Host "Removing old files (keeping .venv to be recreated)..." -ForegroundColor Yellow
        
        # Remove all files except .venv directory
        Get-ChildItem -Path $vailaProgramPath -Exclude ".venv" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    } Else {
        Write-Host "Installing vaila to $vailaProgramPath..." -ForegroundColor Yellow
        Try {
            New-Item -ItemType Directory -Force -Path $vailaProgramPath | Out-Null
            If (-Not (Test-Path $vailaProgramPath)) {
                Write-Error "Failed to create directory $vailaProgramPath. Check permissions."
                If ($vailaProgramPath -like "${env:ProgramFiles}*" -and -Not $isAdmin) {
                    Write-Host "Note: Installing to Program Files requires administrator privileges." -ForegroundColor Yellow
                    Write-Host "Please run this script as Administrator or it will use $env:USERPROFILE\vaila instead." -ForegroundColor Yellow
                }
                Exit 1
            }
        } Catch {
            Write-Error "Failed to create directory $vailaProgramPath`: $($_.Exception.Message)"
            If ($vailaProgramPath -like "${env:ProgramFiles}*" -and -Not $isAdmin) {
                Write-Host "Note: Installing to Program Files requires administrator privileges." -ForegroundColor Yellow
                Write-Host "Please run this script as Administrator." -ForegroundColor Yellow
            }
            Exit 1
        }
    }

    # Copy all files and directories except .venv, __pycache__, and build artifacts
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

# Ensure no stale uv.lock is copied over
If (Test-Path "$vailaProgramPath\uv.lock") {
    Write-Host "Removing existing uv.lock to avoid stale dependency locks..." -ForegroundColor Yellow
    Remove-Item -Path "$vailaProgramPath\uv.lock" -Force -ErrorAction SilentlyContinue
}

# Change to vaila home directory for uv operations
Set-Location $vailaProgramPath

# Check and set permissions for the installation directory if needed
# This is critical when installing to Program Files, as even admin processes
# may not have write permissions by default
If ($isAdmin -and $vailaProgramPath -like "*Program Files*") {
    Write-Host ""
    Write-Host "Setting permissions for installation directory..." -ForegroundColor Yellow
    Try {
        $acl = Get-Acl $vailaProgramPath
        
        # Grant FullControl to the current user
        $currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
        $permission = $currentUser, "FullControl", "ContainerInherit,ObjectInherit", "None", "Allow"
        $accessRule = New-Object System.Security.AccessControl.FileSystemAccessRule $permission
        $acl.SetAccessRule($accessRule)
        
        # Also grant FullControl to Administrators group (for robustness)
        $adminGroup = "BUILTIN\Administrators"
        $adminPermission = $adminGroup, "FullControl", "ContainerInherit,ObjectInherit", "None", "Allow"
        $adminAccessRule = New-Object System.Security.AccessControl.FileSystemAccessRule $adminPermission
        $acl.SetAccessRule($adminAccessRule)
        
        Set-Acl $vailaProgramPath $acl
        Write-Host "Permissions set successfully for $currentUser and Administrators group." -ForegroundColor Green
    } Catch {
        Write-Warning "Could not set permissions: $_"
        Write-Host "Attempting to continue anyway..." -ForegroundColor Yellow
    }
}

# Initialize uv project (if not already initialized)
Write-Host ""
Write-Host "Initializing uv project..." -ForegroundColor Yellow
If (-Not (Test-Path ".python-version")) {
    & uv python pin 3.12.12
}

# Create virtual environment explicitly
Write-Host ""
Write-Host "Creating virtual environment (.venv)..." -ForegroundColor Yellow
If (Test-Path ".venv") {
    Write-Host "Virtual environment already exists. Removing old one..." -ForegroundColor Yellow
    Try {
        Remove-Item -Path ".venv" -Recurse -Force -ErrorAction Stop
    } Catch {
        Write-Warning "Could not remove existing .venv: $_"
        Write-Host "Attempting to create new .venv anyway..." -ForegroundColor Yellow
    }
}

Try {
    & uv venv --python 3.12.12
    If ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment. Exit code: $LASTEXITCODE"
        Write-Host "You may need to run this script as Administrator." -ForegroundColor Yellow
        Exit 1
    }
    Write-Host "Virtual environment created successfully." -ForegroundColor Green
} Catch {
    Write-Error "Failed to create virtual environment: $_"
    Write-Host "You may need to run this script as Administrator." -ForegroundColor Yellow
    Exit 1
}

# Generate lock file
Write-Host ""
Write-Host "Generating lock file (uv.lock)..." -ForegroundColor Yellow
& uv lock

# Sync dependencies (install all packages from pyproject.toml)
Write-Host ""
Write-Host "Installing vaila dependencies with uv..." -ForegroundColor Yellow
Write-Host "This may take a few minutes on first run..." -ForegroundColor Yellow
& uv sync

# Prompt user about installing PyTorch/YOLO stack
Write-Host ""
Write-Host "---------------------------------------------" -ForegroundColor Cyan
Write-Host "PyTorch / YOLO installation options" -ForegroundColor Cyan
Write-Host "  [1] Skip (default)" -ForegroundColor Yellow
Write-Host "  [2] Install PyTorch + YOLO (ultralytics/boxmot)" -ForegroundColor Yellow
Write-Host "---------------------------------------------" -ForegroundColor Cyan
$installOption = Read-Host "Choose an option [1-2]"
If ([string]::IsNullOrWhiteSpace($installOption)) {
    $installOption = "1"
}

If ($installOption -eq "2") {
    $pytorchInstalled = $false
    Write-Host ""
    Write-Host "Select PyTorch build:" -ForegroundColor Cyan
    Write-Host "  [1] CPU-only (default)" -ForegroundColor Yellow
    Write-Host "  [2] CUDA (requires NVIDIA GPU + drivers)" -ForegroundColor Yellow
    $pytorchOption = Read-Host "Choose an option [1-2]"
    If ([string]::IsNullOrWhiteSpace($pytorchOption)) {
        $pytorchOption = "1"
    }

    If ($pytorchOption -eq "2") {
        Write-Host ""
        Write-Host "Installing PyTorch with CUDA support..." -ForegroundColor Yellow
        Try {
            & uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            $pytorchInstalled = $true
            Write-Host "PyTorch with CUDA installed successfully." -ForegroundColor Green
        } Catch {
            Write-Warning "Failed to install CUDA-enabled PyTorch. You can retry later with:"
            Write-Host "  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121" -ForegroundColor Yellow
        }
    } Else {
        Write-Host ""
        Write-Host "Installing CPU-only PyTorch..." -ForegroundColor Yellow
        Try {
            & uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            $pytorchInstalled = $true
            Write-Host "CPU-only PyTorch installed successfully." -ForegroundColor Green
        } Catch {
            Write-Warning "Failed to install CPU-only PyTorch. You can retry later with:"
            Write-Host "  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu" -ForegroundColor Yellow
        }
    }

    If ($pytorchInstalled) {
        Write-Host ""
        Write-Host "Installing YOLO dependencies (ultralytics, boxmot)..." -ForegroundColor Yellow
        Try {
            & uv pip install ultralytics boxmot
            Write-Host "YOLO dependencies installed successfully." -ForegroundColor Green
        } Catch {
            Write-Warning "Failed to install YOLO dependencies. You can install later with:"
            Write-Host "  uv pip install ultralytics boxmot" -ForegroundColor Yellow
        }
    } Else {
        Write-Host "Skipping YOLO packages because PyTorch installation failed." -ForegroundColor Yellow
    }
} Else {
    Write-Host ""
    Write-Host "Skipping PyTorch/YOLO installation. You can install later using:" -ForegroundColor Yellow
    Write-Host "  CUDA PyTorch: uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121" -ForegroundColor Cyan
    Write-Host "  CPU PyTorch : uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu" -ForegroundColor Cyan
    Write-Host "  YOLO stack  : uv pip install ultralytics boxmot" -ForegroundColor Cyan
}

# Install pycairo (Cairo dependencies on Windows are typically handled via pip)
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

# Check/Install system dependencies
Write-Host ""
Write-Host "Checking/installing system dependencies (FFmpeg, Windows Terminal)..." -ForegroundColor Yellow

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
            Write-Warning "Failed to install FFmpeg via winget. You can install manually from: https://ffmpeg.org/download.html"
        }
    } Else {
        Write-Host "FFmpeg is not installed. Administrator privileges required for installation." -ForegroundColor Yellow
        Write-Host "You can install manually from: https://ffmpeg.org/download.html" -ForegroundColor Cyan
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
        Write-Host "You can install manually from Microsoft Store or: https://aka.ms/terminal" -ForegroundColor Cyan
    }
}

# Create a run_vaila.ps1 script using uv
$runScript = Join-Path $vailaProgramPath "run_vaila.ps1"
Write-Host ""
Write-Host "Creating run_vaila.ps1 script..." -ForegroundColor Yellow
@"
# Run vaila using uv
Set-Location "$vailaProgramPath"
& uv run "$vailaProgramPath\vaila.py"
# Keep terminal open after execution
Write-Host ""
Write-Host "Program finished. Press Enter to close this window..." -ForegroundColor Yellow
Read-Host
"@ | Out-File -FilePath $runScript -Encoding UTF8

# Create a run_vaila.bat script for easier double-click execution
$runScriptBat = Join-Path $vailaProgramPath "run_vaila.bat"
Write-Host "Creating run_vaila.bat script..." -ForegroundColor Yellow
@"
@echo off
cd /d "$vailaProgramPath"
pwsh.exe -ExecutionPolicy Bypass -File "run_vaila.ps1"
pause
"@ | Out-File -FilePath $runScriptBat -Encoding ASCII

# -------- Windows Terminal Profile Setup --------
# Find icon for Windows Terminal (can use PNG or ICO)
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

$wtPath = "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe"
If (Test-Path $wtPath) {
    Write-Host "Configuring the 'vaila' profile in Windows Terminal..." -ForegroundColor Yellow
    $settingsPath = "$wtPath\LocalState\settings.json"
    $settingsBackupPath = "$wtPath\LocalState\settings_backup.json"

    # Backup current settings.json
    If (Test-Path $settingsPath) {
        Copy-Item -Path $settingsPath -Destination $settingsBackupPath -Force -ErrorAction SilentlyContinue
        $settingsContent = Get-Content -Path $settingsPath -Raw
        $settingsJson = $settingsContent | ConvertFrom-Json

        # Remove any existing 'vaila' profile
        $settingsJson.profiles.list = $settingsJson.profiles.list | Where-Object { $_.name -ne "vaila" }

        # Add new 'vaila' profile
        $vailaProfile = @{
            name = "vaila"
            commandline = "pwsh.exe -ExecutionPolicy Bypass -NoExit -File `"$runScript`""
            startingDirectory = "$vailaProgramPath"
            guid = "{17ce5bfe-17ed-5f3a-ab15-5cd5baafed5b}"
            hidden = $false
        }
        # Add icon only if found
        If ($wtIconPath) {
            $vailaProfile.icon = $wtIconPath
        }

        $settingsJson.profiles.list += $vailaProfile
        $settingsJson | ConvertTo-Json -Depth 100 | Out-File -FilePath $settingsPath -Encoding UTF8
        Write-Host "'vaila' profile added to Windows Terminal successfully." -ForegroundColor Green
    } Else {
        Write-Host "Windows Terminal settings.json not found. Skipping profile setup." -ForegroundColor Yellow
    }
}

# --------- Desktop Shortcut ---------
Write-Host "Creating Desktop shortcut for 'vaila'..." -ForegroundColor Yellow
$desktopPath = [Environment]::GetFolderPath("Desktop")
$desktopShortcutPath = Join-Path $desktopPath "vaila.lnk"
$wshell = New-Object -ComObject WScript.Shell
$desktopShortcut = $wshell.CreateShortcut($desktopShortcutPath)
$desktopShortcut.TargetPath = "pwsh.exe"
$desktopShortcut.Arguments = "-ExecutionPolicy Bypass -NoExit -File `"$runScript`""
# Find icon file (.ico format required for Windows shortcuts)
$iconPath = $null
$possibleIconPaths = @(
    "$vailaProgramPath\vaila\images\vaila.ico",
    "$vailaProgramPath\vaila\images\vaila_ico_trans.ico",
    "$vailaProgramPath\vaila\images\vaila_icon_win_original.ico",
    "$vailaProgramPath\docs\images\vaila_ico.ico",
    "$vailaProgramPath\docs\images\vaila_ico_trans.ico"
)
ForEach ($path in $possibleIconPaths) {
    If (Test-Path $path) {
        $iconPath = $path
        Break
    }
}
If ($iconPath) {
    # IconLocation format: "path,index" (index 0 = first icon in file)
    $desktopShortcut.IconLocation = "$iconPath,0"
    Write-Host "Using icon: $iconPath" -ForegroundColor Green
} Else {
    Write-Warning "Icon file not found. Shortcut will use default icon."
}
$desktopShortcut.WorkingDirectory = "$vailaProgramPath"
$desktopShortcut.Save()
Write-Host "Desktop shortcut for 'vaila' created at $desktopShortcutPath." -ForegroundColor Green

# --------- Start Menu Shortcut ---------
Write-Host "Creating Start Menu shortcut for 'vaila'..." -ForegroundColor Yellow
# Use user's Start Menu if not admin, otherwise use system-wide Start Menu
If ($isAdmin) {
    $startMenuPath = "C:\ProgramData\Microsoft\Windows\Start Menu\Programs\vaila"
} Else {
    $startMenuPath = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\vaila"
}
If (-Not (Test-Path $startMenuPath)) {
    New-Item -ItemType Directory -Force -Path $startMenuPath | Out-Null
}
$startMenuShortcutPath = "$startMenuPath\vaila.lnk"
$startMenuShortcut = $wshell.CreateShortcut($startMenuShortcutPath)
$startMenuShortcut.TargetPath = "pwsh.exe"
$startMenuShortcut.Arguments = "-ExecutionPolicy Bypass -NoExit -File `"$runScript`""
If ($iconPath) {
    # IconLocation format: "path,index" (index 0 = first icon in file)
    $startMenuShortcut.IconLocation = "$iconPath,0"
}
$startMenuShortcut.WorkingDirectory = "$vailaProgramPath"
$startMenuShortcut.Save()
Write-Host "Start Menu shortcut for 'vaila' created at $startMenuShortcutPath." -ForegroundColor Green

# Ensure the application directory has the correct permissions
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
Write-Host "You can now launch vaila using:" -ForegroundColor Cyan
Write-Host "  - Desktop shortcut" -ForegroundColor Yellow
Write-Host "  - Start Menu shortcut" -ForegroundColor Yellow
Write-Host "  - Windows Terminal profile 'vaila'" -ForegroundColor Yellow
Write-Host "  - Double-click run_vaila.bat" -ForegroundColor Yellow
Write-Host ""
Write-Host "Restart your computer to ensure all changes take effect." -ForegroundColor Yellow
Write-Host ""
Pause

