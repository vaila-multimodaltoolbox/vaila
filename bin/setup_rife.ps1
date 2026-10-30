# Practical-RIFE 4.25 installer. Updated: 29 October 2026; Version: 0.4.12.
# Uses the current vailá environment without changing CUDA groups or extras.
$ErrorActionPreference = 'Stop'
$rifeRepoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
Push-Location $rifeRepoRoot
try {
    & uv run --no-sync python bin/setup_rife.py @args
    if ($LASTEXITCODE -ne 0) { throw "RIFE setup failed (exit $LASTEXITCODE). See output above." }
} finally {
    Pop-Location
}
