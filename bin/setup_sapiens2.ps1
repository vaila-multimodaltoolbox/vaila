# ============================================================================
# setup_sapiens2.ps1 — Sapiens2 Pose + DETR bootstrap (Windows PowerShell)
# ============================================================================
$ErrorActionPreference = "Stop"

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot    = Resolve-Path (Join-Path $ScriptDir "..")
$Sapiens2Dir = Join-Path $RepoRoot ".local\third_party\sapiens2"
$WeightsDir  = Join-Path $RepoRoot "vaila\models\sapiens2"

New-Item -ItemType Directory -Path (Split-Path $Sapiens2Dir -Parent) -Force | Out-Null

Write-Host ">> vaila - Sapiens2 Pose setup"
Write-Host "   repo root: $RepoRoot"

if (Test-Path (Join-Path $Sapiens2Dir ".git")) {
  Write-Host ">> [1/4] sapiens2 already cloned; pulling latest..."
  try {
    git -C $Sapiens2Dir pull --ff-only | Out-Null
  } catch {
    Write-Warning "git pull failed; continuing with existing checkout"
  }
} else {
  Write-Host ">> [1/4] Cloning facebookresearch/sapiens2 into $Sapiens2Dir..."
  git clone --depth 1 https://github.com/facebookresearch/sapiens2.git $Sapiens2Dir
}

Write-Host ">> [2/4] Installing sapiens2 (editable) into the uv environment..."
Push-Location $RepoRoot
try {
  uv pip install -e $Sapiens2Dir
} finally {
  Pop-Location
}

Write-Host ">> [3/4] Downloading pose + detector weights to $WeightsDir..."
New-Item -ItemType Directory -Path (Join-Path $WeightsDir "pose") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $WeightsDir "detector") -Force | Out-Null
Push-Location $RepoRoot
try {
  # huggingface_hub API via vaila_sapiens (hf CLI can spuriously exit 1 via click.Exit(0))
  uv run vaila/vaila_sapiens.py --download-weights --model 1b
} finally {
  Pop-Location
}

Write-Host ">> [4/4] Validating layout..."
$missing = 0
@(
  (Join-Path $WeightsDir "pose\sapiens2_1b_pose.safetensors"),
  (Join-Path $WeightsDir "detector\detr-resnet-101-dc5\config.json")
) | ForEach-Object {
  if (-not (Test-Path $_)) {
    Write-Host "   MISSING: $_"
    $missing = 1
  } else {
    Write-Host "   OK:      $_"
  }
}

if ($missing -eq 0) {
  Write-Host ""
  Write-Host ">> Done. Sapiens2 Pose is ready."
  Write-Host "   uv run vaila/vaila_sapiens.py -i tests/markerless_2d_analysis/ -o C:\temp\sapiens_out --model 1b"
  exit 0
} else {
  Write-Host ""
  Write-Host ">> Some weights are missing. Run: uv run hf auth login"
  exit 1
}
