# Switch vailá to portable CPU manifest (Windows / no NVIDIA CUDA workflow).
# Regenerates uv.lock; run: uv sync [--extra sam]
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root
$Src = Join-Path $Root "pyproject_universal_cpu.toml"
if (-not (Test-Path $Src)) { throw "missing $Src" }
Copy-Item $Src (Join-Path $Root "pyproject.toml") -Force
Write-Host "pyproject.toml <- pyproject_universal_cpu.toml"
Write-Host "Running uv lock ..."
uv lock
Write-Host ""
Write-Host "Next: uv sync   or   uv sync --extra sam"
