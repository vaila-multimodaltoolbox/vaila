<#
.SYNOPSIS
    Unified interactive bootstrap for vailá on Windows (PowerShell).

.DESCRIPTION
    Detects OS / architecture / NVIDIA GPU, picks the PyTorch dependency group
    (`cpu` or `cuda`) and the optional extras, then runs `uv sync`.

    There is a SINGLE committed pyproject.toml / uv.lock for every machine and
    OS: the hardware choice is a uv dependency group, not a file swap, so this
    script never modifies anything tracked by git.

      cpu   -> torch trio from https://download.pytorch.org/whl/cpu
               (on macOS: the PyPI wheel, i.e. the Metal/MPS build)
      cuda  -> torch trio from https://download.pytorch.org/whl/cu128,
               plus tensorrt + nvidia-ml-py

    Linux / macOS / WSL / Git Bash: use bin/setup_pyproject.sh instead.

.PARAMETER Target
    auto | cpu | cuda. Default: auto. Legacy names linux-cuda, win-cuda, macos
    and gpu are still accepted.

.PARAMETER Extras
    Comma-separated list of extras (sam, fifa, sapiens, upscaler, dev).

.PARAMETER NonInteractive
    Skip all prompts; use detected / supplied values.

.PARAMETER Yes
    Accept all suggested defaults (no prompts).

.PARAMETER Lock
    Re-run `uv lock` before syncing (normally unnecessary).

.PARAMETER NoLock
    Skip `uv lock` (default).

.PARAMETER NoSync
    Skip `uv sync`.

.EXAMPLE
    pwsh bin/setup_pyproject.ps1
    pwsh bin/setup_pyproject.ps1 -Target cuda -Extras sam
    pwsh bin/setup_pyproject.ps1 -Target cpu -NonInteractive -Yes
#>

[CmdletBinding()]
param(
    [ValidateSet('auto','cpu','cuda','gpu','linux-cuda','win-cuda','macos')]
    [string]$Target = 'auto',
    [string]$Extras = '',
    [switch]$Full,
    [switch]$NonInteractive,
    [switch]$Yes,
    [switch]$Lock,
    [switch]$NoLock,
    [switch]$NoSync,
    [switch]$SkipWorktree,
    [switch]$NoSkipWorktree
)

$ErrorActionPreference = 'Stop'

# A bare `uv run` re-resolves the project with the default dependency groups
# (dev + cpu), which on a CUDA machine silently replaces the cu128 wheels with
# the CPU ones. Use the environment exactly as the installer left it.
If (-not $env:UV_NO_SYNC) { $env:UV_NO_SYNC = "1" }

$Root = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $Root

# ---------- detection ----------
function Get-OsKind {
    if ($IsMacOS)         { return 'macos' }
    if ($IsLinux) {
        if (Test-Path /proc/version) {
            $v = Get-Content /proc/version -ErrorAction SilentlyContinue
            if ($v -match '(?i)microsoft|wsl') { return 'wsl' }
        }
        return 'linux'
    }
    if ($IsWindows -or $env:OS -eq 'Windows_NT') { return 'windows' }
    return 'unknown'
}

function Get-Arch {
    try { return (Get-CimInstance -ClassName Win32_Processor -ErrorAction Stop).Architecture.ToString() } catch {}
    if ($env:PROCESSOR_ARCHITECTURE) { return $env:PROCESSOR_ARCHITECTURE }
    return 'unknown'
}

function Get-NvidiaInfo {
    $nv = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nv) { return $null }
    try {
        $line = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null |
                Select-Object -First 1
        if ($line) { return $line.Trim() }
    } catch { return $null }
    return $null
}

$OsKind  = Get-OsKind
$Arch    = Get-Arch
$GpuInfo = Get-NvidiaInfo
$HasNvidia = [bool]$GpuInfo

# ---------- auto target ----------
function Resolve-Target($t) {
    if ($t -ne 'auto') { return $t }
    switch ($OsKind) {
        'macos'   { 'cpu' }   # PyPI wheel = Metal/MPS build
        'windows' { if ($HasNvidia) { 'cuda' } else { 'cpu' } }
        'linux'   { if ($HasNvidia) { 'cuda' } else { 'cpu' } }
        'wsl'     { if ($HasNvidia) { 'cuda' } else { 'cpu' } }
        default   { 'cpu' }
    }
}

# Legacy target names map onto the two dependency groups in pyproject.toml.
function Normalize-Target($t) {
    switch ($t) {
        'cpu'        { 'cpu' }
        'macos'      { 'cpu' }
        'cuda'       { 'cuda' }
        'gpu'        { 'cuda' }
        'linux-cuda' { 'cuda' }
        'win-cuda'   { 'cuda' }
        default      { throw "unknown target: $t" }
    }
}

$Target = Normalize-Target (Resolve-Target $Target)
if ($Target -eq 'cuda' -and $OsKind -eq 'macos') {
    throw "target 'cuda' is not available on macOS (no CUDA wheels); use -Target cpu."
}

function Detect-InstalledExtras {
    $detected = @()
    $py = if ($IsWindows) { Join-Path $Root '.venv\Scripts\python.exe' } else { Join-Path $Root '.venv/bin/python3' }
    if (Test-Path $py) {
        if (& $py -c "import sam3" 2>$null) { $detected += 'sam' }
        if (& $py -c "import sapiens" 2>$null) { $detected += 'sapiens' }
        if (& $py -c "import pytorch_lightning" 2>$null) { $detected += 'fifa' }
        if (& $py -c "import diffusers" 2>$null) { $detected += 'upscaler' }
        if (& $py -c "import pytest" 2>$null) { $detected += 'dev' }
    }
    return $detected
}

function Suggested-Extras($t) {
    # tensorrt / nvidia-ml-py live in the `cuda` dependency group, not an extra.
    $base = @()
    $installed = Detect-InstalledExtras
    $all = ($base + $installed) | Select-Object -Unique
    return ($all -join ' ')
}

$SuggestedExtras = Suggested-Extras $Target

# ---------- summary ----------
Write-Host ''
Write-Host '== vailá pyproject setup ==' -ForegroundColor Cyan
Write-Host ("  {0,-18} {1} ({2})" -f 'OS detected:', $OsKind, $Arch)
if ($GpuInfo) {
    Write-Host ("  {0,-18} {1}" -f 'NVIDIA GPU:', $GpuInfo)
} else {
    Write-Host ("  {0,-18} none detected" -f 'NVIDIA GPU:') -ForegroundColor DarkGray
}
Write-Host ("  {0,-18} {1}" -f 'Target:', $Target)
Write-Host ("  {0,-18} {1}" -f 'PyTorch group:', "$Target (uv sync --group $Target)")
$shown = if ($SuggestedExtras) { $SuggestedExtras } else { '(none)' }
Write-Host ("  {0,-18} {1}" -f 'Suggested extras:', $shown)
Write-Host ''

# ---------- target confirmation ----------
function Confirm-Default($msg, [bool]$default) {
    if ($NonInteractive -or $Yes) { return $default }
    $hint = if ($default) { '[Y/n]' } else { '[y/N]' }
    $r = Read-Host "$msg $hint"
    if ([string]::IsNullOrWhiteSpace($r)) { return $default }
    return ($r -match '^[Yy]')
}

if (-not $NonInteractive -and -not $Yes) {
    if (-not (Confirm-Default "Use target '$Target'?" $true)) {
        Write-Host 'Available targets: cpu (CPU wheels / macOS Metal), cuda (NVIDIA CUDA 12.8)'
        $new = Read-Host 'Pick target'
        if ($new) {
            $Target = Normalize-Target $new
            $SuggestedExtras = Suggested-Extras $Target
            Write-Host "Switched to target=$Target, suggested extras='$SuggestedExtras'" -ForegroundColor Cyan
        }
    }
}

# ---------- extras ----------
# Identical on every platform, because there is only one pyproject.toml.
$AvailableExtras = @('sam','fifa','sapiens','upscaler','dev')

if ($Full -or $Extras -eq 'all') {
    $Chosen = $AvailableExtras
} elseif ($Extras) {
    $Chosen = $Extras -split '[,\s]+' | Where-Object { $_ }
} elseif ($NonInteractive -or $Yes) {
    $Chosen = $SuggestedExtras -split '[,\s]+' | Where-Object { $_ }
} else {
    Write-Host ''
    Write-Host "Available extras: $($AvailableExtras -join ', ')" -ForegroundColor Cyan
    Write-Host '  sam      = SAM 3 video segmentation (sam3==0.1.3; CUDA at runtime)'
    Write-Host '  fifa     = FIFA Skeletal Tracking Light (pytorch-lightning, timm, ...)'
    Write-Host '  sapiens  = Sapiens2 Pose (transformers + safetensors; CUDA; then bin/setup_sapiens2.ps1)'
    Write-Host '  upscaler = diffusers (image upscaling)'
    Write-Host '  dev      = ruff, ty, pytest (developer tooling)'
    $r = Read-Host "Extras to install [default: '$SuggestedExtras']"
    if ([string]::IsNullOrWhiteSpace($r)) { $r = $SuggestedExtras }
    $Chosen = $r -split '[,\s]+' | Where-Object { $_ }
}

$Valid = @(); $Invalid = @()
foreach ($e in $Chosen) {
    if ($AvailableExtras -contains $e) { $Valid += $e } else { $Invalid += $e }
}
if ($Invalid.Count -gt 0) {
    Write-Warning "Ignoring extras not defined in pyproject.toml: $($Invalid -join ', ')"
}
$ExtrasList = $Valid

# ---------- lock ----------
if ($Lock -and -not $NoLock) {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Error 'uv not found in PATH. Install: https://docs.astral.sh/uv/getting-started/installation/'
        exit 1
    }
    Write-Host 'Running: uv lock' -ForegroundColor Cyan
    & uv lock
} else {
    Write-Host 'Using the committed uv.lock (pass -Lock to re-resolve)' -ForegroundColor Cyan
}

# ---------- sync ----------
if (-not $NoSync) {
    # `cpu` is a default group in pyproject.toml, so the CUDA build must both
    # drop it and request `cuda` (the two groups are declared as conflicting).
    $argList = @('sync')
    if ($Target -eq 'cuda') { $argList += @('--no-group', 'cpu', '--group', 'cuda') }
    foreach ($e in $ExtrasList) { $argList += @('--extra', $e) }
    Write-Host "Running: uv $($argList -join ' ')" -ForegroundColor Cyan
    & uv @argList
    Write-Host ''
    Write-Host "Done. vailá ready for target='$Target' with extras=[$($ExtrasList -join ' ')]." -ForegroundColor Green
    Write-Host 'Run the GUI:   uv run --no-sync vaila.py'
} else {
    Write-Host "Skipping 'uv sync' (-NoSync)" -ForegroundColor Cyan
    Write-Host ''
    Write-Host 'Next, run manually:'
    $groupHint = if ($Target -eq 'cuda') { '--no-group cpu --group cuda ' } else { '' }
    if ($ExtrasList.Count -gt 0) {
        $tail = ($ExtrasList | ForEach-Object { "--extra $_" }) -join ' '
        Write-Host "  uv sync $groupHint$tail"
    } else {
        Write-Host "  uv sync $groupHint".TrimEnd()
    }
}

# ---------- clear any legacy skip-worktree bits ----------
# Older versions of this script hid pyproject.toml / uv.lock from git status.
# Both files are portable now, so make sure they are tracked normally again.
if (Get-Command git -ErrorAction SilentlyContinue) {
    git update-index --no-skip-worktree pyproject.toml uv.lock 2>$null
    if ($SkipWorktree -or $NoSkipWorktree) {
        Write-Host 'note: -SkipWorktree / -NoSkipWorktree are obsolete (single portable pyproject.toml/uv.lock).' -ForegroundColor Yellow
    }
}
