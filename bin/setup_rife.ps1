# Practical-RIFE 4.25 installer. Updated: 30 October 2026; Version: 0.4.13.
# Uses the current vailá environment without changing CUDA groups or extras.
# Optional and relocatable: --home DIR installs into DIR, --export DIR copies it (no network).
$ErrorActionPreference = 'Stop'
$rifeRepoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
# No Set-Location: a relative --home/--export DIR is relative to where you ran this, not the repo.
& uv run --no-sync --project $rifeRepoRoot python (Join-Path $rifeRepoRoot 'bin/setup_rife.py') @args
if ($LASTEXITCODE -ne 0) { throw "RIFE setup failed (exit $LASTEXITCODE). See output above." }
