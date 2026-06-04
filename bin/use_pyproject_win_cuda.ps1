# Thin wrapper -> bin/setup_pyproject.ps1 -Target win-cuda
# Kept for backward compatibility. Prefer running setup_pyproject.ps1 directly
# (interactive + cross-platform detection).
$Root = Resolve-Path (Join-Path $PSScriptRoot '..')
& (Join-Path $Root 'bin/setup_pyproject.ps1') `
    -Target win-cuda `
    -Extras gpu `
    -NonInteractive `
    -NoSync @args
