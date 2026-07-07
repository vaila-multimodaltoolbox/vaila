<#
.SYNOPSIS
    Unified interactive bootstrap for vailá on Windows (PowerShell).

.DESCRIPTION
    Detects OS / architecture / NVIDIA GPU, picks the right pyproject_*.toml
    template, lets the user confirm extras, then runs `uv lock` + `uv sync --extra ...`.

    Linux / macOS / WSL / Git Bash: use bin/setup_pyproject.sh instead.

.PARAMETER Target
    auto | cpu | linux-cuda | win-cuda | macos. Default: auto.

.PARAMETER Extras
    Comma-separated list of extras (gpu, sam, fifa, upscaler, dev).

.PARAMETER NonInteractive
    Skip all prompts; use detected / supplied values.

.PARAMETER Yes
    Accept all suggested defaults (no prompts).

.PARAMETER NoLock
    Skip `uv lock`.

.PARAMETER NoSync
    Skip `uv sync`.

.EXAMPLE
    pwsh bin/setup_pyproject.ps1
    pwsh bin/setup_pyproject.ps1 -Target win-cuda -Extras gpu,sam
    pwsh bin/setup_pyproject.ps1 -Target cpu -NonInteractive -Yes
#>

[CmdletBinding()]
param(
    [ValidateSet('auto','cpu','linux-cuda','win-cuda','macos')]
    [string]$Target = 'auto',
    [string]$Extras = '',
    [switch]$NonInteractive,
    [switch]$Yes,
    [switch]$NoLock,
    [switch]$NoSync
)

$ErrorActionPreference = 'Stop'

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
        'macos'   { 'macos' }
        'windows' { if ($HasNvidia) { 'win-cuda' }   else { 'cpu' } }
        'linux'   { if ($HasNvidia) { 'linux-cuda' } else { 'cpu' } }
        'wsl'     { if ($HasNvidia) { 'linux-cuda' } else { 'cpu' } }
        default   { 'cpu' }
    }
}

$Target = Resolve-Target $Target

function Template-For($t) {
    switch ($t) {
        'cpu'        { 'pyproject_universal_cpu.toml' }
        'linux-cuda' { 'pyproject_linux_cuda12.toml' }
        'win-cuda'   { 'pyproject_win_cuda12.toml' }
        'macos'      { 'pyproject_macos.toml' }
        default      { throw "unknown target: $t" }
    }
}

function Suggested-Extras($t) {
    switch ($t) {
        'cpu'        { '' }
        'linux-cuda' { 'gpu' }
        'win-cuda'   { 'gpu' }
        'macos'      { '' }
    }
}

$Src = Template-For $Target
if (-not (Test-Path (Join-Path $Root $Src))) {
    Write-Error "template not found: $Src"
    exit 1
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
Write-Host ("  {0,-18} {1}" -f 'Template:', $Src)
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
        Write-Host 'Available targets: cpu, linux-cuda, win-cuda, macos'
        $new = Read-Host 'Pick target'
        if ($new) {
            $Target = $new
            $Src = Template-For $Target
            if (-not (Test-Path (Join-Path $Root $Src))) { throw "template not found: $Src" }
            $SuggestedExtras = Suggested-Extras $Target
            Write-Host "Switched to target=$Target, template=$Src, suggested extras='$SuggestedExtras'" -ForegroundColor Cyan
        }
    }
}

# ---------- extras ----------
$AvailableExtras = switch ($Target) {
    'linux-cuda' { @('gpu','sam','fifa','sapiens','upscaler','dev') }
    'win-cuda'   { @('gpu','sam','fifa','sapiens','upscaler','dev') }
    default      { @('sam','fifa','sapiens','upscaler','dev') }
}

if ($Extras) {
    $Chosen = $Extras -split '[,\s]+' | Where-Object { $_ }
} elseif ($NonInteractive -or $Yes) {
    $Chosen = $SuggestedExtras -split '[,\s]+' | Where-Object { $_ }
} else {
    Write-Host ''
    Write-Host "Available extras for ${Target}: $($AvailableExtras -join ', ')" -ForegroundColor Cyan
    Write-Host '  gpu      = tensorrt + nvidia-ml-py (CUDA only)'
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
    Write-Warning "Ignoring extras not defined in ${Src}: $($Invalid -join ', ')"
}
$ExtrasList = $Valid

# ---------- apply template ----------
Write-Host ''
Write-Host "Copying template: $Src -> pyproject.toml" -ForegroundColor Cyan
Copy-Item (Join-Path $Root $Src) (Join-Path $Root 'pyproject.toml') -Force

# ---------- lock ----------
if (-not $NoLock) {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Error 'uv not found in PATH. Install: https://docs.astral.sh/uv/getting-started/installation/'
        exit 1
    }
    Write-Host 'Running: uv lock' -ForegroundColor Cyan
    & uv lock
} else {
    Write-Host "Skipping 'uv lock' (-NoLock)" -ForegroundColor Cyan
}

# ---------- sync ----------
if (-not $NoSync) {
    $argList = @('sync')
    foreach ($e in $ExtrasList) { $argList += @('--extra', $e) }
    Write-Host "Running: uv $($argList -join ' ')" -ForegroundColor Cyan
    & uv @argList
    Write-Host ''
    Write-Host "Done. vailá ready for target='$Target' with extras=[$($ExtrasList -join ' ')]." -ForegroundColor Green
    Write-Host 'Run the GUI:   uv run vaila.py'
} else {
    Write-Host "Skipping 'uv sync' (-NoSync)" -ForegroundColor Cyan
    Write-Host ''
    Write-Host 'Next, run manually:'
    if ($ExtrasList.Count -gt 0) {
        $tail = ($ExtrasList | ForEach-Object { "--extra $_" }) -join ' '
        Write-Host "  uv sync $tail"
    } else {
        Write-Host '  uv sync'
    }
}
