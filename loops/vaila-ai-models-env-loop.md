---
name: vaila-ai-models-env-loop
category: Vailá
trigger: manual
verification-level: 1
theory-base: arXiv:2607.00038
---

# vailá AI Models & Environment Diagnostic Loop

## Description
Audit, discover, and display vailá application version/date, all local AI tracking models
(MediaPipe tasks/tflite, YOLOv26/YOLO11 weights/pose/segmentation/engines, SAM 3, SAM-3D-DINOv3,
Sapiens2), Hugging Face Hub authentication and cache status, and hardware accelerators
(NVIDIA GPU via nvidia-smi and PyTorch CUDA) across both CLI (`vaila/vaila_env.py`, `vaila.py --env-info`)
and the interactive Tkinter GUI ("imagination!" dialog).

## Use When
- Inspecting vailá system state, version, and date across developer workstations and laptops.
- Verifying whether gated or external AI weights are downloaded and accessible on disk:
  - `facebook/sam3` (`sam3.pt`, `model.safetensors`, `config.json`)
  - `facebook/sam-3d-body-dinov3` (`model.ckpt`, `model_config.yaml`)
  - `facebook/sapiens2-pose-*` (0.4B, 1B, 5B) & `detr-resnet-101-dc5` detector
  - MediaPipe landmarkers (`pose_landmarker_heavy.task`, `pose_landmarker_lite.task`, `face_landmarker.task`, `hand_landmarker.task`, `face_detector.task`, selfie segmenters)
  - Ultralytics YOLO models (`yolo26x-pose.pt`, `yolo26l-pose.pt`, `yolo26m-pose.pt`, `yolo26n-pose.pt`, `yolo11n.pt`, `best.pt`, TensorRT `.engine`, ONNX `.onnx`)
- Checking Hugging Face token presence and authenticated user name (`huggingface_hub.whoami()`).
- Detecting real NVIDIA hardware capabilities via `nvidia-smi` even when PyTorch CUDA encounters shared library issues (e.g. `libcusparseLt.so.0`).
- Providing guided instructions for activating `.venv` and running interactive IPython biomechanical sessions.
- Not for: training YOLO or SAM models (use `vaila/yolotrain.py`); not for performing batch video inference (use `vaila/vaila_sam.py` or `vaila/vaila_sapiens.py`).

## Inputs
1. `vaila_env` — `vaila/vaila_env.py` (CLI formatter, AI model scanner, and Tkinter GUI).
2. `vaila_main` — `vaila.py` (version banner, `--env-info` dispatcher, and `imagination!` button).
3. `models_dir` — `vaila/models/` (local model directories and weight files).
4. `hf_hub` — `huggingface_hub` (package, auth token, user info, cache directory).
5. `tests` — `tests/test_vaila_env.py`.
6. `help` — `vaila/help/vaila_env.md` and `vaila/help/vaila_env.html`.

## Goal
An objectively verifiable diagnostic state:
1. `uv run vaila/vaila_env.py` and `uv run vaila.py --env-info` print a complete, structured report containing:
   - Header with `vailá Version` (0.3.131) and `Update Date` (09 September 2026), plus Git branch/commit.
   - Operating system, Python version, executable path, and `.venv` status.
   - Hardware detection reporting real NVIDIA GPU (e.g. RTX 4090, VRAM, driver version) via `nvidia-smi` and PyTorch CUDA status.
   - 16+ core Python scientific libraries with their installed versions (including `ezc3d 1.7.2`).
   - Hugging Face Hub status (version, token presence, authenticated username, cache directory).
   - AI models inventory with file existence and byte sizes:
     - MediaPipe tasks (`pose_landmarker_heavy.task`, `lite`, `face`, `hand`, `face_detector`, selfie segmenters).
     - YOLO models (`yolo26*-pose.pt`, `yolo11n.pt`, `best.pt`, `.onnx`, `.engine`).
     - SAM 3 (`vaila/models/sam3/` weights).
     - SAM-3D-DINOv3 (`vaila/models/sam-3d-dinov3/` weights).
     - Sapiens2 (`vaila/models/sapiens2/` pose 0.4B/1B/5B and DETR detector).
   - Platform-specific `.venv` activation commands.
   - IPython biomechanics launch commands and code snippet.
2. The Tkinter GUI dialog (`VailaEnvGUI`) displays all diagnostic sections clearly with tabs/notebook or scrollable categories.
3. Automated test suite passes and ruff/ty static checks are clean.

**Targets (worst-first):**
1. **vailá Version & Date in Header:** Read `vaila.py` version and date constants, git branch and commit, and format in top header.
2. **Hardware & GPU detection resilience:** Detect NVIDIA GPU via `nvidia-smi` query (`name, memory.total, driver_version`) as fallback when `torch.cuda` raises or is CPU-only.
3. **Hugging Face Hub integration:** Query `huggingface_hub` version, token presence, cache folder, and `whoami()` safely with exception handling.
4. **AI Models inventory scanner:** Implement `get_ai_models_status()` scanning `vaila/models/` for MediaPipe, YOLO, SAM 3, SAM-3D-DINOv3, and Sapiens2.
5. **CLI output formatting:** Expand `format_env_summary()` to include all newly audited sections in clean terminal tables.
6. **GUI dialog update:** Enhance `VailaEnvGUI` with Notebook tabs: "Packages & System", "AI Models & HuggingFace", "IPython & .venv".
7. **Test suite & Documentation:** Expand `tests/test_vaila_env.py` with tests for AI models, HF info, and vailá version; update help docs and index.

## Verification (Governing Check)
- **True level:** 1 (deterministic pytest assertions, CLI exit codes, and output string checks) with level-2 ruff/ty static verification.
- **Check (every iteration):**
  ```bash
  uv run ruff check vaila/vaila_env.py tests/test_vaila_env.py --fix
  uv run ruff format vaila/vaila_env.py tests/test_vaila_env.py
  uv run ty check vaila/vaila_env.py tests/test_vaila_env.py
  uv run pytest tests/test_vaila_env.py -v
  uv run vaila/vaila_env.py
  ```
- **Evidence:** Raw stdout/stderr of `uv run vaila/vaila_env.py` and pytest test results recorded in `loops/state/vaila-ai-models-env-loop-state.json`.
- **Completion criterion:**
  - `vaila_env.py` outputs vailá version/date, GPU info, Hugging Face user, MediaPipe tasks, YOLO weights, SAM3, SAM3D, and Sapiens2.
  - All tests in `tests/test_vaila_env.py` pass without errors.
  - Ruff and Ty type checking pass with 0 diagnostics.

## Trigger
Manual invocation by developer or CI pipeline when auditing installed AI weights or releasing new vailá versions.

## Iteration
0. Validate local `vaila/models/` structure.
1. Run baseline check.
2. Select highest unfulfilled target from Targets list.
3. Apply atomic code edit in `vaila/vaila_env.py` or tests.
4. Run governing check and capture stdout/stderr.
5. Retain edit only if governing check passes; rollback on failure.
6. Record state atomically in `loops/state/vaila-ai-models-env-loop-state.json`.
7. Evaluate terminal states; stop on success.

## Terminal States
- **success:** All 7 targets verified, tests passing, ruff/ty clean, complete terminal and GUI reports generated.
- **no-op:** All AI models, packages, and version headers already detected and displayed accurately.
- **no-progress/stalled:** Two consecutive iterations fail to improve test coverage or produce clean reports.
- **blocked:** Missing critical Python modules or broken disk permissions on `vaila/models/`.
- **exhausted:** Maximum allocated turn limit (10 iterations) reached.

## Guardrails
- **Maximum allocation:** 10 iterations, 30 tool calls.
- **Human approval required:** Deleting or modifying existing downloaded weight files in `vaila/models/`.
- **Protected verifier:** Test assertions cannot be hard-coded to ignore missing information.
- **Rollback:** Single-iteration `git checkout vaila/vaila_env.py` or `replace_file_content`.

## State Memory
- **Path:** `loops/state/vaila-ai-models-env-loop-state.json`.
- **Persist:** Targets accepted, model inventory counts, HF auth status, and test results.
- **Recovery:** Resumes from highest unfinished target upon re-invocation.

## Skills
- `$safe-refactor` — Extend `VailaEnvGUI` with Tkinter ttk.Notebook tabs without breaking existing button actions.
- `$surgical-patch` — Patch `vaila_env.py` hardware and model scanner functions.
- `$verify-and-stop` — Validate test suite and CLI execution before concluding.

## Why It Works
- **Dynamic filesystem discovery:** Directly checks files on disk rather than assuming fixed environments.
- **Hardware resilience:** Queries both `nvidia-smi` and PyTorch so GPU hardware is never falsely reported as absent.
- **Safe HF inspection:** Wraps Hugging Face calls in try/except blocks so offline/air-gapped systems never crash.
- **Clear GUI/CLI parity:** Terminal report and GUI tabs show identical data.

## Health Metrics
- **Cost per accepted change:** `total tokens or currency / verified changes retained`.
- **AI model discovery coverage:** 5/5 model families detected (MediaPipe, YOLO, SAM3, SAM3D, Sapiens2).
