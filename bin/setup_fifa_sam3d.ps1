# ============================================================================
# setup_fifa_sam3d.ps1 — SAM 3D Body + weights bootstrap (Windows PowerShell)
# ============================================================================
# - Clones Meta sam_3d_body at the repo root.
# - Downloads facebook/sam-3d-body-dinov3 weights into vaila\models\sam-3d-dinov3\.
# - Validates the layout and prints next-step commands.
#
# Requirements: git, uv, Hugging Face CLI (`uv run hf auth login`) and license
# acceptance on https://huggingface.co/facebook/sam-3d-body-dinov3.
# ============================================================================
$ErrorActionPreference = "Stop"

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot   = Resolve-Path (Join-Path $ScriptDir "..")
$Sam3dDir   = Join-Path $RepoRoot "sam_3d_body"
$WeightsDir = Join-Path $RepoRoot "vaila\models\sam-3d-dinov3"

Write-Host ">> vaila FIFA - SAM 3D Body setup"
Write-Host "   repo root: $RepoRoot"

# --- [1/3] sam_3d_body ------------------------------------------------------
if (Test-Path (Join-Path $Sam3dDir ".git")) {
  Write-Host ">> [1/3] sam_3d_body already cloned; pulling latest..."
  try {
    git -C $Sam3dDir pull --ff-only | Out-Null
  } catch {
    Write-Warning "git pull failed; continuing with existing checkout"
  }
} else {
  Write-Host ">> [1/3] Cloning facebookresearch/sam_3d_body into $Sam3dDir..."
  git clone --depth 1 https://github.com/facebookresearch/sam_3d_body.git $Sam3dDir
}

Write-Host ">> [1/3] Installing sam_3d_body (editable) into the uv environment..."
Push-Location $RepoRoot
try {
  uv pip install -e $Sam3dDir
} finally {
  Pop-Location
}

# --- [2/3] weights download -------------------------------------------------
Write-Host ">> [2/3] Downloading facebook/sam-3d-body-dinov3 weights to $WeightsDir..."
New-Item -ItemType Directory -Path $WeightsDir -Force | Out-Null
Push-Location $RepoRoot
try {
  uv run hf download facebook/sam-3d-body-dinov3 --local-dir $WeightsDir
} finally {
  Pop-Location
}

# --- [3/3] validation -------------------------------------------------------
Write-Host ">> [3/3] Validating layout..."
$missing = $false
foreach ($f in @(
  (Join-Path $WeightsDir "model.ckpt"),
  (Join-Path $WeightsDir "assets\mhr_model.pt")
)) {
  if (-not (Test-Path $f)) {
    Write-Host "   MISSING: $f"
    $missing = $true
  } else {
    Write-Host "   OK:      $f"
  }
}

if (-not $missing) {
  Write-Host ""
  Write-Host ">> Done. SAM 3D Body is ready for the FIFA Skeletal Tracking Light pipeline."
  Write-Host "   Next steps:"
  Write-Host "     uv run vaila\vaila_sam.py fifa bootstrap `"
  Write-Host "       --videos-dir <...>\FIFA_Challenge_2026_Video_Data\Videos `"
  Write-Host "       --data-root  <...>\FIFA\data"
  Write-Host "     uv run vaila\vaila_sam.py fifa prepare --data-root <...>\FIFA\data `"
  Write-Host "       --video-source <...>\FIFA_Challenge_2026_Video_Data\Videos"
  Write-Host "     uv run vaila\vaila_sam.py fifa preprocess --data-root <...>\FIFA\data `"
  Write-Host "       --sequences <...>\FIFA\data\sequences_full.txt"
  Write-Host "     uv run vaila\vaila_sam.py fifa baseline   --data-root <...>\FIFA\data `"
  Write-Host "       --sequences <...>\FIFA\data\sequences_full.txt -o outputs\submission_full.npz"
  exit 0
} else {
  Write-Host ""
  Write-Host ">> Weights are missing. Most likely:"
  Write-Host "   1) You have not accepted the license on https://huggingface.co/facebook/sam-3d-body-dinov3"
  Write-Host "   2) You have not run 'uv run hf auth login' with an account that has access."
  exit 2
}
