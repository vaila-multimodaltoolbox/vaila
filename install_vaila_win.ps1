<#
    Script: install_vaila_win.ps1
    Description: Installs the vaila - Multimodal Toolbox on Windows
                 Supports both uv (recommended) and Conda (legacy) installation methods
    Usage:
        1. Download the repository from GitHub manually and extract it.
        2. Open PowerShell (Administrator recommended for full installation)
        3. Navigate to the root directory of the extracted repository
        4. Run: .\install_vaila_win.ps1
    Notes:
        - uv method: uv will be automatically installed if not present
        - conda method: Requires Conda (Anaconda or Miniconda) to be installed
        - Python 3.12.12 will be installed via uv or conda depending on method chosen
        - Installation location:
          * With admin: C:\Program Files\vaila (Windows standard location)
          * Without admin: C:\Users\<user>\vaila (user directory)
        - Can run without administrator privileges (some features may be skipped)
    Author: Prof. Dr. Paulo R. P. Santiago
    Creation: 17 December 2024
    Updated: 11 January 2026
    Version: 0.3.0
    OS: Windows 10/11
    Reference: https://docs.astral.sh/uv/
#>

$ErrorActionPreference = "Stop"

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "vaila - Multimodal Toolbox Installation/Update" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This script will install or update vaila."
Write-Host "If vaila is already installed, it will be updated with the latest code."
Write-Host ""

# Prompt user to choose installation method
Write-Host "---------------------------------------------" -ForegroundColor Cyan
Write-Host "Installation Method Selection" -ForegroundColor Cyan
Write-Host "  [1] uv (recommended - modern, fast)" -ForegroundColor Yellow
Write-Host "  [2] Conda (legacy - for compatibility)" -ForegroundColor Yellow
Write-Host "---------------------------------------------" -ForegroundColor Cyan
$installMethod = Read-Host "Choose an option [1-2] (default: 1)"
If ([string]::IsNullOrWhiteSpace($installMethod)) {
    $installMethod = "1"
}
If ($installMethod -notin @("1", "2")) {
    Write-Host "Invalid option. Defaulting to uv (option 1)." -ForegroundColor Yellow
    $installMethod = "1"
}

# Define paths (common to both methods)
If ($isAdmin) {
    $vailaProgramPath = "${env:ProgramFiles}\vaila"
    Write-Host "Installation location: $vailaProgramPath" -ForegroundColor Green
    Write-Host "(Administrator privileges detected - using Program Files)" -ForegroundColor Green
} Else {
    $vailaProgramPath = "$env:USERPROFILE\vaila"
    Write-Host "Installation location: $vailaProgramPath" -ForegroundColor Yellow
    Write-Host "(No administrator privileges - using user directory)" -ForegroundColor Yellow
    If ($installMethod -eq "1") {
        Write-Host "Note: Some features (FFmpeg, Windows Terminal installation) may be skipped." -ForegroundColor Yellow
    }
    Write-Host "Run as administrator for installation to Program Files (recommended)." -ForegroundColor Yellow
}
Write-Host ""

$sourcePath = (Get-Location).Path
$projectDir = $sourcePath

# Check if pyproject.toml exists (for uv method)
If ($installMethod -eq "1") {
    If (-Not (Test-Path "$projectDir\pyproject.toml")) {
        Write-Error "pyproject.toml not found in $projectDir"
        Write-Host "Please ensure you're running this script from the vaila project root." -ForegroundColor Yellow
        Exit 1
    }
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
# COMMON FUNCTIONS (used by both methods)
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
    
    # Find icon file (.ico format required for Windows shortcuts)
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
    
    # Find icon file
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
                commandline = $CommandLine
                startingDirectory = "$vailaProgramPath"
                guid = "{17ce5bfe-17ed-5f3a-ab15-5cd5baafed5b}"
                hidden = $false
            }
            # Add icon only if found
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
# UV INSTALLATION METHOD
# ============================================================================

function Install-WithUv {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "Installing vaila using uv (recommended method)" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""

    # Install uv if not present
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

    # Check if we're already in the installation directory
    $normalizedProjectDir = (Resolve-Path $projectDir -ErrorAction SilentlyContinue).Path
    $normalizedVailaPath = (Resolve-Path $vailaProgramPath -ErrorAction SilentlyContinue).Path
    If (-Not $normalizedProjectDir) { $normalizedProjectDir = $projectDir }
    If (-Not $normalizedVailaPath) { $normalizedVailaPath = $vailaProgramPath }
    $isAlreadyInstalled = ($normalizedProjectDir -eq $normalizedVailaPath)

    If ($isAlreadyInstalled) {
        Write-Host "Script is running from installation directory. Files are already in place." -ForegroundColor Green
        Write-Host "Skipping file copy step." -ForegroundColor Green
    } Else {
        # Clean destination directory and copy files
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

    # Ensure no stale uv.lock is copied over
    If (Test-Path "$vailaProgramPath\uv.lock") {
        Write-Host "Removing existing uv.lock to avoid stale dependency locks..." -ForegroundColor Yellow
        Remove-Item -Path "$vailaProgramPath\uv.lock" -Force -ErrorAction SilentlyContinue
    }

    # Change to vaila home directory for uv operations
    Set-Location $vailaProgramPath

    # Set permissions for installation directory if needed
    If ($isAdmin -and $vailaProgramPath -like "*Program Files*") {
        Write-Host "Setting permissions for installation directory..." -ForegroundColor Yellow
        Try {
            $acl = Get-Acl $vailaProgramPath
            # Grant FullControl to 'Users' group so standard users can update .venv and run uv
            $userGroup = "BUILTIN\Users"
            $permission = $userGroup, "FullControl", "ContainerInherit,ObjectInherit", "None", "Allow"
            $accessRule = New-Object System.Security.AccessControl.FileSystemAccessRule $permission
            $acl.AddAccessRule($accessRule)
            
            # Ensure Administrators also have full control
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

    # Initialize uv project
    Write-Host ""
    Write-Host "Initializing uv project..." -ForegroundColor Yellow
    If (-Not (Test-Path ".python-version")) {
        & uv python pin 3.12.12
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
        & uv venv --python 3.12.12
        If ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to create virtual environment."
            Exit 1
        }
        Write-Host "Virtual environment created successfully." -ForegroundColor Green
    } Catch {
        Write-Error "Failed to create virtual environment: $_"
        Exit 1
    }

    # Generate lock file
    Write-Host ""
    Write-Host "Generating lock file (uv.lock)..." -ForegroundColor Yellow
    & uv lock --upgrade

    # Sync dependencies
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
                Write-Warning "Failed to install CUDA-enabled PyTorch."
            }
        } Else {
            Write-Host ""
            Write-Host "Installing CPU-only PyTorch..." -ForegroundColor Yellow
            Try {
                & uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
                $pytorchInstalled = $true
                Write-Host "CPU-only PyTorch installed successfully." -ForegroundColor Green
            } Catch {
                Write-Warning "Failed to install CPU-only PyTorch."
            }
        }

        If ($pytorchInstalled) {
            Write-Host ""
            Write-Host "Installing YOLO dependencies (ultralytics, boxmot)..." -ForegroundColor Yellow
            Try {
                & uv pip install ultralytics boxmot
                Write-Host "YOLO dependencies installed successfully." -ForegroundColor Green
            } Catch {
                Write-Warning "Failed to install YOLO dependencies."
            }
        }
    }

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
pwsh.exe -ExecutionPolicy Bypass -File "run_vaila.ps1"
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
    $wtCommandLine = "pwsh.exe -ExecutionPolicy Bypass -NoExit -File `"$runScript`""
    Set-WindowsTerminalProfile -CommandLine $wtCommandLine -IconPath $wtIconPath

    # Create shortcuts
    New-DesktopShortcut -TargetPath "pwsh.exe" -Arguments "-ExecutionPolicy Bypass -NoExit -File `"$runScript`"" -WorkingDirectory $vailaProgramPath
    New-StartMenuShortcut -TargetPath "pwsh.exe" -Arguments "-ExecutionPolicy Bypass -NoExit -File `"$runScript`"" -WorkingDirectory $vailaProgramPath

    return $runScript
}

# ============================================================================
# CONDA INSTALLATION METHOD
# ============================================================================

function Install-WithConda {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "Installing vaila using Conda (legacy method)" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""

    # Check if conda is installed
    If (-Not (Get-Command conda -ErrorAction SilentlyContinue)) {
        Write-Error "Conda is not installed or not in the PATH. Please install Conda first."
        Write-Host "Please install Conda (Anaconda or Miniconda) first:" -ForegroundColor Yellow
        Write-Host "  https://www.anaconda.com/products/individual" -ForegroundColor Cyan
        Exit 1
    }

    # Check if running as administrator (required for Conda method)
    If (-NOT $isAdmin) {
        Write-Warning "Conda installation method requires Administrator privileges."
        Write-Host "Please run this script as Administrator." -ForegroundColor Yellow
        Exit 1
    }

    # Update conda
    Write-Host "Updating conda and all base packages..." -ForegroundColor Yellow
    Try {
        conda update conda -y
        conda update --all -y
        Write-Host "Conda and base packages updated." -ForegroundColor Green
    } Catch {
        Write-Warning "Failed to update conda or packages. Continuing anyway."
    }

    # Check if 'vaila' environment exists
    Write-Host "Checking for existing 'vaila' environment..." -ForegroundColor Yellow
    $envExists = conda env list | Select-String -Pattern "^vaila"
    $choice = ""

    If ($envExists) {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host "vaila environment already exists!" -ForegroundColor Cyan
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Choose installation type:"
        Write-Host "1. UPDATE - Keep existing environment and update vaila files only"
        Write-Host "   (Preserves NVIDIA CUDA installations and other custom packages)"
        Write-Host "2. RESET - Remove existing environment and create fresh installation"
        Write-Host "   (Will require reinstalling NVIDIA CUDA and other custom packages)"
        Write-Host ""
        
        Do {
            $choice = Read-Host "Enter your choice (1 for UPDATE, 2 for RESET)"
        } While ($choice -notin @("1", "2"))
        
        If ($choice -eq "1") {
            Write-Host ""
            Write-Host "Selected: UPDATE - Keeping existing environment" -ForegroundColor Green
            Write-Host "Updating existing 'vaila' environment..."
            conda activate vaila
            
            Try {
                & conda env update -f "$vailaProgramPath\yaml_for_conda_env\vaila_win.yaml"
                Write-Host "Environment updated successfully." -ForegroundColor Green
            } Catch {
                Write-Warning "Failed to update environment from YAML. Continuing anyway."
            }
        } Else {
            Write-Host ""
            Write-Host "Selected: RESET - Creating fresh environment" -ForegroundColor Green
            Write-Host "Removing old 'vaila' environment..."
            Try {
                conda deactivate
                conda env remove -n vaila -y
                Write-Host "Old 'vaila' environment removed successfully." -ForegroundColor Green
            } Catch {
                Write-Warning "Could not remove old environment. Continuing anyway."
            }
            
            Write-Host "Cleaning conda cache..." -ForegroundColor Yellow
            conda clean --all -y
            
            Try {
                Write-Host "Creating 'vaila' environment from YAML..." -ForegroundColor Yellow
                & conda env create -f "$vailaProgramPath\yaml_for_conda_env\vaila_win.yaml"
                Write-Host "'vaila' environment created successfully." -ForegroundColor Green
            } Catch {
                Write-Error "Failed to create the 'vaila' environment."
                Exit 1
            }
        }
    } Else {
        Write-Host "'vaila' environment does not exist. Creating new environment..." -ForegroundColor Yellow
        
        Write-Host "Cleaning conda cache..." -ForegroundColor Yellow
        conda clean --all -y
        
        Try {
            Write-Host "Creating 'vaila' environment from YAML..." -ForegroundColor Yellow
            & conda env create -f "$vailaProgramPath\yaml_for_conda_env\vaila_win.yaml"
            Write-Host "'vaila' environment created successfully." -ForegroundColor Green
        } Catch {
            Write-Error "Failed to create the 'vaila' environment."
            Exit 1
        }
    }

    # Check for NVIDIA GPU and install PyTorch with CUDA if available
    If (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        Write-Host "NVIDIA GPU detected. Installing PyTorch with CUDA support..." -ForegroundColor Yellow
        Try {
            conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia -n vaila -y
            Write-Host "PyTorch with CUDA support installed successfully." -ForegroundColor Green
        } Catch {
            Write-Warning "Failed to install PyTorch with CUDA support."
        }
    } Else {
        Write-Host "No NVIDIA GPU detected. Skipping PyTorch with CUDA installation." -ForegroundColor Yellow
    }

    # Clean destination directory and copy files
    Write-Host ""
    Write-Host "Cleaning destination directory and copying vaila program..." -ForegroundColor Yellow
    If (Test-Path $vailaProgramPath) {
        Write-Host "Removing existing files from destination directory..." -ForegroundColor Yellow
        Get-ChildItem -Path $vailaProgramPath -Recurse -Force | Remove-Item -Force -Recurse
    } Else {
        New-Item -ItemType Directory -Force -Path $vailaProgramPath | Out-Null
    }
    Get-ChildItem -Path $sourcePath -Recurse -Force | Where-Object {
        -not ($_.PSIsContainer -and $_.Name -eq '__pycache__') -and
        -not ($_.Extension -eq '.pyc')
    } | ForEach-Object {
        $target = $_.FullName.Replace($sourcePath, $vailaProgramPath)
        if ($_.PSIsContainer) {
            if (-not (Test-Path $target)) {
                New-Item -ItemType Directory -Force -Path $target | Out-Null
            }
        } else {
            Copy-Item -Path $_.FullName -Destination $target -Force
        }
    }

    # Remove ffmpeg from conda
    Write-Host "Removing ffmpeg from conda (if present)..." -ForegroundColor Yellow
    conda remove -n vaila ffmpeg -y

    # Activate vaila environment
    If ($choice -eq "1") {
        Write-Host "Environment already activated for UPDATE mode." -ForegroundColor Green
    } Else {
        Write-Host "Activating 'vaila' environment..." -ForegroundColor Yellow
        conda activate vaila
    }

    # Upgrade pip and install dependencies
    Write-Host "Upgrading pip and required pip packages..." -ForegroundColor Yellow
    Try {
        python -m pip install --upgrade pip
        pip install --upgrade mediapipe moviepy
        Write-Host "pip, mediapipe, and moviepy installed/upgraded successfully." -ForegroundColor Green
    } Catch {
        Write-Warning "Error upgrading pip or pip dependencies."
    }

    # Install Cairo dependencies and pycairo
    Write-Host "Installing pycairo..." -ForegroundColor Yellow
    Try {
        pip install pycairo
        Write-Host "pycairo installed successfully." -ForegroundColor Green
    } Catch {
        Write-Warning "pycairo installation failed. Trying with force-reinstall..."
        Try {
            pip install --force-reinstall --no-cache-dir pycairo
            Write-Host "pycairo installed successfully." -ForegroundColor Green
        } Catch {
            Write-Warning "pycairo installation failed. This may cause issues with the application."
        }
    }

    # Get Conda base path
    $condaPath = (& conda info --base).Trim()

    # Create run_vaila.ps1 script
    $runScript = Join-Path $vailaProgramPath "run_vaila.ps1"
    Write-Host ""
    Write-Host "Creating run_vaila.ps1 script..." -ForegroundColor Yellow
    @"
# Run vaila using Conda
& '$condaPath\shell\condabin\conda-hook.ps1'
conda activate vaila
Set-Location "$vailaProgramPath"
python "$vailaProgramPath\vaila.py"
# Keep terminal open after execution
Write-Host ""
Write-Host "Program finished. Press Enter to close this window..." -ForegroundColor Yellow
Read-Host
"@ | Out-File -FilePath $runScript -Encoding UTF8

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
    $wtCommandLine = "pwsh.exe -ExecutionPolicy Bypass -NoExit -Command `"& `'$condaPath\shell\condabin\conda-hook.ps1`'; conda activate `'vaila`'; cd `'$vailaProgramPath`'; python `'vaila.py`'`""
    Set-WindowsTerminalProfile -CommandLine $wtCommandLine -IconPath $wtIconPath

    # Create shortcuts
    New-DesktopShortcut -TargetPath "pwsh.exe" -Arguments "-ExecutionPolicy Bypass -NoExit -Command `"& `'$condaPath\shell\condabin\conda-hook.ps1`'; conda activate `'vaila`'; cd `'$vailaProgramPath`'; python `'vaila.py`'`"" -IconPath "$vailaProgramPath\docs\images\vaila_ico.ico" -WorkingDirectory $vailaProgramPath
    New-StartMenuShortcut -TargetPath "pwsh.exe" -Arguments "-ExecutionPolicy Bypass -NoExit -Command `"& `'$condaPath\shell\condabin\conda-hook.ps1`'; conda activate `'vaila`'; cd `'$vailaProgramPath`'; python `'vaila.py`'`"" -IconPath "$vailaProgramPath\docs\images\vaila_ico.ico" -WorkingDirectory $vailaProgramPath

    # Adjust site-packages permissions
    $vailaSitePackagesDir = Join-Path $condaPath "envs\vaila\Lib\site-packages"
    Write-Host "Adjusting permissions on site-packages directory..." -ForegroundColor Yellow
    If (Test-Path $vailaSitePackagesDir) {
        Try {
            Start-Process "icacls.exe" -ArgumentList "`"$vailaSitePackagesDir`" /grant Users:(OI)(CI)F /T" -Wait -NoNewWindow
            Write-Host "Permissions successfully adjusted." -ForegroundColor Green
        } Catch {
            Write-Warning "Failed to adjust permissions."
        }
    }

    return $runScript
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Execute installation based on chosen method
If ($installMethod -eq "1") {
    $runScript = Install-WithUv
} Else {
    $runScript = Install-WithConda
}

# Install system dependencies (common to both methods)
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

# rsync (for transfer scripts)
If ($isAdmin) {
    Write-Host "Checking for rsync..." -ForegroundColor Yellow
    $rsyncInstalled = Get-Command rsync -ErrorAction SilentlyContinue
    If (-Not $rsyncInstalled) {
        # Try Chocolatey first
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
                Write-Host "rsync installed successfully via Chocolatey." -ForegroundColor Green
            } Catch {
                Write-Warning "Failed to install rsync via Chocolatey."
            }
        }
    } Else {
        Write-Host "rsync is already installed." -ForegroundColor Green
    }
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
Write-Host "You can now launch vaila using:" -ForegroundColor Cyan
Write-Host "  - Desktop shortcut" -ForegroundColor Yellow
Write-Host "  - Start Menu shortcut" -ForegroundColor Yellow
Write-Host "  - Windows Terminal profile 'vaila'" -ForegroundColor Yellow
If ($installMethod -eq "1") {
    Write-Host "  - Double-click run_vaila.bat" -ForegroundColor Yellow
}
Write-Host ""
Write-Host "Restart your computer to ensure all changes take effect." -ForegroundColor Yellow
Write-Host ""
Pause
