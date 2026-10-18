# Prompt: FreeKiki HRNet-W32 student (distill from active YOLO `l`)

Paste this whole file into **Claude Code** at repo root `/home/preto/data/vaila`.
Scope is intentionally narrow: **HRNet-W32 only**, **fixed distill λ = 0.5**, concrete workspace below.

---

## Goal

Add an **HRNet-W32 heatmap** FreeKiki backend that trains with **knowledge distillation** from the current promoted YOLO model, so the new net inherits what `active` already learned. YOLO → HRNet **cannot** copy weights; transfer = teacher–student soft targets + GT labels.

## Concrete environment (this machine)

| Item | Value |
|------|--------|
| Repo | `/home/preto/data/vaila` |
| Workspace `WS` | `/home/preto/data/FreeKiki` |
| Teacher (`active`) | slot **`l`** → `models/freekiki_l.pt` (`yolo26l-pose`, pose mAP50-95 ≈ 0.887) |
| Dataset | `$WS/datasets/kiki49` |
| Prefer manifest | `v012` (latest under `$WS/manifests/`; fall back to `v011` if needed) |
| Toolchain | `uv run --no-sync …` (CUDA machine — never bare `uv run`) |

Verify before coding:

```bash
cd /home/preto/data/vaila
test -f /home/preto/data/FreeKiki/freekiki.toml
test -f /home/preto/data/FreeKiki/models/freekiki_l.pt
uv run --no-sync vaila/freekiki.py models -w /home/preto/data/FreeKiki
```

## Read first

- `AGENTS.md`, `.cursor/rules/vaila.mdc`
- `vaila/freekiki.py` — slots, `train`, `load_predictor`, `evaluate`, `compare`, HEATMAP_SLOTS
- `vaila/freekiki_heatmap.py` — `KikiHeatNet`, letterbox, Gaussian targets, decode, `train_heatmap`, checkpoints (`backend: freekiki_heatmap_v1`)

## Fixed design choices (do not expand scope)

1. **Backbone:** only `hrnet_w32` (no W48 in this task).
2. **Distill weight λ:** fixed default **`0.5`** (CLI flag still exists so smoke can set `0`, but default and docs = 0.5).
3. **Teacher:** default `--teacher active` → resolves to YOLO `l` in this workspace.
4. **No PnLCalib / line network / PnL refine** in this task.
5. **Do not** change YOLO train path or replace default slot without beating `compare`.
6. **Do not commit** unless asked.

## Transfer learning (required)

```
L = L_gt + 0.5 * L_distill
```

- `L_gt`: MSE vs GT heatmaps (vis=0 → zero map), same as current heatmap.
- `L_distill`: MSE vs soft heatmaps from teacher on the **same augmented canvas**
  (teacher in eval / no grad; YOLO preds → xy + conf → render Gaussians / soft maps
  with same σ/stride as GT).
- Where GT vis=0 but teacher conf ≥ 0.2, still distill (inherit teacher on hard/missing labels).
- Where GT vis>0, keep GT dominant; distill term still on at λ=0.5.
- Log teacher path, λ, backbone in `args.yaml` + `train_manifest.json`.

Init student from ImageNet-style HRNet trunk if available locally without network download; else random trunk + distill still OK. Never download gated weights without saying so.

## Implementation sketch

1. Extend heatmap family: `backbone=hrnet_w32` builds HRNet-W32 → 49 heatmaps @ stride 4 (reuse letterbox / decode / dataset from `freekiki_heatmap.py`). Prefer torch-only implementation in `freekiki_heatmap.py` or new `freekiki_hrnet.py` + dual-import pattern.
2. Slot: reuse `hm_m` **or** add `hm_hr32` with `arch=heatmap-hrnet_w32`. One clear choice; heatmap stays non-auto-promote (`compare --promote`).
3. CLI (`freekiki.py train`):
   - `--backend heatmap --backbone hrnet_w32`
   - `--teacher active` (default for hrnet_w32)
   - `--distill-weight 0.5` (default)
4. Wire GUI Heatmap preset note: HRNet-W32 + distill from active.
5. Tests (CPU): forward shape `(B,49,H/4,W/4)`, distill loss with mock teacher, checkpoint round-trip + `model_backend`.
6. Metadata: script headers (date/version = global vailá), `vaila/help/freekiki.md` + `.html`.
7. QA: `uv run --no-sync ruff check … --fix && ruff format && ty check` on touched files; pytest for new tests.

## Acceptance criteria

- [ ] `train --backend heatmap --backbone hrnet_w32 --teacher active --distill-weight 0.5` writes heatmap `best.pt`.
- [ ] ResNet heatmap + YOLO paths unbroken.
- [ ] `evaluate` / `compare` work on the HRNet candidate vs active/`hm_*`.
- [ ] Docs state: no YOLO weight copy; transfer = distillation from `active` (slot `l` here).
- [ ] Tests pass for forward + distill + checkpoint.

## Commands after implementation

```bash
export WS=/home/preto/data/FreeKiki
cd /home/preto/data/vaila

# smoke (~minutes)
uv run --no-sync vaila/freekiki.py train -w "$WS" --backend heatmap --backbone hrnet_w32 \
  --teacher active --distill-weight 0.5 --epochs 1 --fraction 0.01 \
  --name smoke_hrnet_w32_distill

# real run (tune --batch to VRAM; start 4–8 @ 1024)
uv run --no-sync vaila/freekiki.py train -w "$WS" --backend heatmap --backbone hrnet_w32 \
  --teacher active --distill-weight 0.5 --epochs 60 --imgsz 1024 --batch 8 \
  --manifest v012 --name hrnet_w32_distill_v012

uv run --no-sync vaila/freekiki.py evaluate -w "$WS" \
  --model runs/hrnet_w32_distill_v012/weights/best.pt

uv run --no-sync vaila/freekiki.py compare -w "$WS" \
  --candidate runs/hrnet_w32_distill_v012/weights/best.pt
# promote only if gate says so:
# uv run --no-sync vaila/freekiki.py compare -w "$WS" \
#   --candidate runs/hrnet_w32_distill_v012/weights/best.pt --promote
```

## Process

1. Short design note (files + distill formula) **before** coding.
2. Implement narrowly as above.
3. Smoke train on `$WS`.
4. Stop after smoke + tests unless I ask for the full 60-epoch run.

Start now.
```
