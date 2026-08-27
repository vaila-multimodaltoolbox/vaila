# Hugging Face setup (any PC)

Gated Meta weights (`facebook/sam3`, `facebook/sam-3d-body-dinov3`, Sapiens2) need a
Hugging Face account with license accepted **and** a working local login. Weights
are **gitignored** — each machine must download them once.

## 1. Sync the project (pulls the fixed Hub CLI)

From the repo root, use the bootstrap so `uv.lock` matches this machine:

```bash
# Linux / macOS / WSL
bash bin/setup_pyproject.sh --target=linux-cuda --extras=gpu,sam,fifa,sapiens --yes
# CPU laptop: --target=cpu --extras=sam,fifa,sapiens
# Windows: pwsh bin/setup_pyproject.ps1 -Target win-cuda -Extras gpu,sam,fifa,sapiens -Yes
```

`pyproject.toml` overrides require **`huggingface-hub>=1.22`** and **`click>=8.4.2`**.
That fixes the old bug where `uv run hf auth login` succeeded but then printed
`click.exceptions.Exit: 0` and returned exit code **1** (and where large downloads
could stall at `Downloading (incomplete total...): 0.00B`).

Check:

```bash
uv run python -c "from importlib.metadata import version; print(version('huggingface-hub'), version('click'))"
# expect hub >= 1.22 (often 1.29.x) and click >= 8.4
uv run hf auth whoami   # must exit 0, no traceback
```

If hub is still `<1.22`, re-lock from a current checkout: `uv lock && uv sync --extra …`.

## 2. Accept licenses (browser, once per account)

| Model | Accept access |
| ----- | ------------- |
| SAM 3 video | https://huggingface.co/facebook/sam3 |
| SAM 3D Body (FIFA / DINOv3 3D) | https://huggingface.co/facebook/sam-3d-body-dinov3 |
| Sapiens2 Pose | https://huggingface.co/facebook/sapiens2 (and linked pose / DETR repos) |

Use the **same** HF user you will log in with on this PC.

## 3. Log in on this PC

```bash
uv run hf auth login
# or: uv run hf auth login --force
uv run hf auth whoami   # e.g. user=yourname  — exit 0
```

Token: https://huggingface.co/settings/tokens (Read is enough).  
Optional: `export HF_TOKEN=hf_…` for headless CI (never commit the token).

## 4. Download weights (per machine)

**Recommended (scripts clear stale locks and use the Hub API):**

```bash
# SAM 3 → vaila/models/sam3/sam3.pt
uv run vaila/vaila_sam.py --download-weights

# SAM 3D Body → vaila/models/sam-3d-dinov3/{model.ckpt,assets/mhr_model.pt}
bash bin/setup_fifa_sam3d.sh          # Linux / macOS
# pwsh bin/setup_fifa_sam3d.ps1       # Windows

# Sapiens2 → vaila/models/sapiens2/
bash bin/setup_sapiens2.sh
```

**Manual API fallback** (if the CLI misbehaves on an old lockfile):

```bash
uv run python -c "
from pathlib import Path
from huggingface_hub import snapshot_download
root = Path('vaila/models/sam-3d-dinov3')
root.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id='facebook/sam-3d-body-dinov3', local_dir=str(root))
"
```

### If download sticks at `0.00B` / incomplete total

1. Stop the download (Ctrl+C).
2. Remove stale locks only (not the whole model tree unless you want a full re-download):

```bash
find vaila/models/sam-3d-dinov3/.cache/huggingface/download -name '*.lock' -delete
find vaila/models/sam3/.cache/huggingface/download -name '*.lock' -delete 2>/dev/null || true
```

3. Confirm hub version (`>=1.22`), then re-run the setup script or `snapshot_download` above.
4. Last resort: `HF_HUB_DISABLE_XET=1` then retry the same download command.

## 5. Quick verify

```bash
test -f vaila/models/sam3/sam3.pt && ls -lh vaila/models/sam3/sam3.pt
test -f vaila/models/sam-3d-dinov3/model.ckpt && ls -lh vaila/models/sam-3d-dinov3/model.ckpt vaila/models/sam-3d-dinov3/assets/mhr_model.pt
```

Do **not** copy multi‑GB `.pt` / `.ckpt` between PCs via git — they are ignored and blocked by the ≥20 MiB pre-commit hook. Prefer Hub download on each machine (or an external disk copy outside the repo).

## Related

- Folders: `vaila/models/sam3/instruction_to_download_models.txt`, `vaila/models/sam-3d-dinov3/instruction_to_download_models.txt`
- [CONTRIBUTING.md](../CONTRIBUTING.md) — *vaila models directory*
- [vaila/help/gpu_guide.md](../vaila/help/gpu_guide.md) — CUDA extras + gated auth
- [vaila/help/sam3dinov3.md](../vaila/help/sam3dinov3.md) — SAM3+DINOv3 3D pipeline
