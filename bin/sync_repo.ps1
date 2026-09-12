<#
.SYNOPSIS
    Safe cross-platform git sync for vailá on Windows (PowerShell).

.DESCRIPTION
    1. Protects your local uncommitted work (aborts if unrelated files are dirty).
    2. Temporarily resets local pyproject.toml / uv.lock to the clean CPU version in Git.
    3. Pulls latest changes from GitHub (git pull --rebase) without merge conflicts.
    4. Automatically re-applies the right hardware template and runs `uv sync` for Windows.

.EXAMPLE
    pwsh bin/sync_repo.ps1
    pwsh bin/sync_repo.ps1 -Target win-cuda -Extras gpu,sam
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

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "git is required but not found in PATH."
    exit 1
}

# 1. Unset skip-worktree temporarily to inspect status
git update-index --no-skip-worktree pyproject.toml uv.lock 2>$null

# Check for dirty files other than pyproject.toml and uv.lock
$status = git status --porcelain
$dirty = $status | Where-Object { $_ -notmatch 'pyproject\.toml|uv\.lock' }
if ($dirty) {
    Write-Warning "=========================================================================="
    Write-Warning "Uncommitted changes detected in repository:"
    $dirty | ForEach-Object { Write-Host $_ -ForegroundColor Yellow }
    Write-Warning ""
    Write-Warning "Please commit or stash your code changes before syncing:"
    Write-Warning "  git add <files>; git commit -m 'your message'"
    Write-Warning "  or: git stash"
    Write-Warning "=========================================================================="
    exit 1
}

# 2. Reset local hardware template to clean tracking state
Write-Host "Resetting local pyproject.toml & uv.lock to clean tracking state..." -ForegroundColor Cyan
git checkout pyproject.toml uv.lock 2>$null

# 3. Pull latest changes
Write-Host "Pulling latest changes from remote (git pull --rebase)..." -ForegroundColor Cyan
git pull --rebase

# 4. Re-apply hardware setup
Write-Host "Applying hardware configuration for current machine..." -ForegroundColor Cyan
$setupArgs = @()
if ($Target) { $setupArgs += "-Target", $Target }
if ($Extras) { $setupArgs += "-Extras", $Extras }
if ($NonInteractive) { $setupArgs += "-NonInteractive" }
if ($Yes) { $setupArgs += "-Yes" }
if ($NoLock) { $setupArgs += "-NoLock" }
if ($NoSync) { $setupArgs += "-NoSync" }

& (Join-Path $PSScriptRoot "setup_pyproject.ps1") @setupArgs

Write-Host "Repository successfully synchronized and environment is up to date!" -ForegroundColor Green
