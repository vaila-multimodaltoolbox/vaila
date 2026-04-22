# Switch vailá to Windows + NVIDIA CUDA 12.1 manifest.
# Regenerates uv.lock; run: uv sync --extra gpu [--extra sam]
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root
$Src = Join-Path $Root "pyproject_win_cuda12.toml"
if (-not (Test-Path $Src)) { throw "missing $Src" }
Copy-Item $Src (Join-Path $Root "pyproject.toml") -Force
Write-Host "pyproject.toml <- pyproject_win_cuda12.toml"
Write-Host "Running uv lock ..."
uv lock
Write-Host ""
Write-Host "Next: uv sync --extra gpu   or   uv sync --extra gpu --extra sam"
