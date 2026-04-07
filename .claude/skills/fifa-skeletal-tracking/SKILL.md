# FIFA Skeletal Tracking Light (vailá)

Use when the user works on **FIFA Skeletal Tracking Light 2026**, monocular broadcast pose, or `vaila_sam.py fifa`.

## Setup

- `uv sync --extra fifa` — adds Lightning/timm/omegaconf/etc. For GPU inference use CUDA `pyproject` template + `uv sync --extra gpu --extra fifa`.
- Weights: `uv run hf download facebook/sam-3d-body-dinov3 --local-dir vaila/models/sam-3d-dinov3` (after HF license acceptance).
- `sam_3d_body/` at repo root is vendored Meta code (`VENDOR_vaila.txt`).

## CLI (subcommand of SAM entrypoint)

```bash
uv run vaila/vaila_sam.py fifa prepare --video-source DIR --data-root data/
uv run vaila/vaila_sam.py fifa boxes --data-root data/ --sequences data/sequences_val.txt
uv run vaila/vaila_sam.py fifa preprocess --data-root data/ --sequences data/sequences_val.txt
uv run vaila/vaila_sam.py fifa baseline --data-root data/ --sequences data/sequences_val.txt -o outputs/submission_full.npz
uv run vaila/vaila_sam.py fifa pack --submission-full outputs/submission_full.npz --data-root data/ --output-dir outputs/ --split val
```

`prepare` does not create `cameras/`, `pitch_points.txt`, or official `boxes/` — those come from the [starter kit](https://github.com/FIFA-Skeletal-Light-Tracking-Challenge/FIFA-Skeletal-Tracking-Starter-Kit-2026) / linked Hugging Face data.

## Code map

- `vaila/fifa_skeletal_pipeline.py` — orchestration + baseline logic
- `vaila/fifa_starter_lib/` — MIT-ported `camera_tracker` + `postprocess`
- Tests: `tests/test_fifa_skeletal_pipeline.py`

See [AGENTS.md](../../../AGENTS.md) for the canonical agent summary.
