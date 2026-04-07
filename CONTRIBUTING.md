# Contributing to _vailá_

**English** · [Português](#como-contribuir)

Thank you for helping improve **vailá** (AGPL-3.0). This document complements [AGENTS.md](AGENTS.md) and [CLAUDE.md](CLAUDE.md) (tooling and conventions for AI-assisted workflows).

## Before you start

- **License:** By contributing, you agree your contributions are under the [same license as the project](LICENSE) (AGPL-3.0).
- **Security:** Read [SECURITY.md](SECURITY.md) — never commit API keys, tokens, or local credential files.

## Development setup

```bash
uv sync                    # or uv sync --extra gpu on supported NVIDIA setups
uv run ruff check vaila/ --fix && uv run ruff format vaila/
uv run ty check vaila/
uv run pytest tests/ -v
```

See [AGENTS.md](AGENTS.md) for the full QA pipeline and project layout.

## Pull requests

1. Fork and create a branch from `main`.
2. Keep changes focused; add or update **tests** when you change analysis logic.
3. Run Ruff, `ty`, and pytest before opening a PR.
4. Describe **what** changed and **why** (biomechanics context helps reviewers).

## Code style

- Python **3.12** only (`>=3.12,<3.13`).
- **Tkinter** for GUI — do not introduce other GUI frameworks.
- Use the **dual-import pattern** documented in [AGENTS.md](AGENTS.md) for `vaila/` modules.
- Astral toolchain: `uv`, `ruff`, `ty` (not bare `pip` / `black` / `mypy`).

## Versioning and GitHub releases

**vailá** uses two related labels: a **package version** (for installers) and an optional **GitHub release codename** (for project milestones).

- **Package version** — Source of truth is [`pyproject.toml`](pyproject.toml) under `[project].version` (PEP 621). This is what **`uv`** and **`pip`** report and what must match **wheels / sdist** metadata. Example form: `0.3.34` (often written as **v0.3.34** in prose).
- **GitHub release codename** — Human-facing name for a milestone: **`rp`** stands for **Ribeirão Preto**, followed by the date as **day + abbreviated English month + two-digit year**, e.g. **`rp23mar26`** = 23 Mar 2026. This does not replace the package version.
- **Release notes** — Prefer stating **both** so installers and GitHub readers stay aligned, for example:
  - `Package version: v0.3.34`
  - `Release codename: rp23mar26 (Ribeirão Preto — 23 Mar 2026)`
- **Git tags** — Pick **one** convention and use it consistently:
  - **Option A:** Git tag = **`v0.3.34`** (semver); GitHub **release title** = **`rp23mar26`** or **`rp23mar26 — v0.3.34`**.
  - **Option B:** Git tag = **`rp23mar26`**; the release description **must** clearly state the **package version** (e.g. **v0.3.34**).

## Security reminders

- Do not commit `.env`, secrets, or real API examples. See [SECURITY.md](SECURITY.md).
- Large files: the repo enforces a **20 MiB** limit via git hooks (`install-hooks.sh`); staging a file **≥ 20 MiB** fails the pre-commit hook.

## vaila models directory

- **Tracked in Git:** small reference data such as **`.csv`** field/court models and similar lightweight assets under `vaila/models/` (and small **`.json`** / config files under **20 MiB**).
- **Not tracked:** downloaded weights and heavy binaries — **`.pt`**, **`.ckpt`**, **`.onnx`**, **`.engine`**, **`.task`** (MediaPipe), **`.pth`**, **`.bin`**, **`.safetensors`**, plus **`vaila/models/**/.cache/`** (e.g. Hugging Face [facebook/sam3](https://huggingface.co/facebook/sam3) downloads) — are listed in [`.gitignore`](.gitignore). They are created under `vaila/models/` on **first run** or via **`hf download`** when a module needs them (e.g. Ultralytics YOLO, SAM3 in `vaila_sam.py`, **SAM 3D Body** `vaila/models/sam-3d-dinov3/` — run `uv run hf download facebook/sam-3d-body-dinov3 --local-dir vaila/models/sam-3d-dinov3` after license acceptance).
- **Small `*.pkl`** under `vaila/models/` (default **walkway** sklearn/joblib models) may remain **tracked** if each file stays **under 20 MiB**; the pre-commit hook rejects larger files.
- **Repository root:** do not leave weight files at the project root; use `vaila/models/` (root `*.pt`, `*.ckpt`, `*.onnx`, `*.pkl`, `*.safetensors` are gitignored).
- Anything **≥ 20 MiB** must not be committed; the hook blocks it even if a pattern were wrong. (`.gitignore` cannot filter by file size; the hook is the backstop.)
- If a large binary was **already pushed**, removing it from the current tree is not enough: rewrite history (e.g. [`git filter-repo`](https://github.com/newren/git-filter-repo)) or ask a maintainer to purge it, then force-push; otherwise clones stay huge and hosts may reject pushes.

---

## Como contribuir

Obrigado por contribuir para o **vailá**. Este ficheiro complementa [AGENTS.md](AGENTS.md).

### Modelos em `vaila/models`

- **Tracked (Git):** ficheiros pequenos de referência (ex. **`.csv`**). **Não tracked:** pesos grandes (**.pt**, **.ckpt**, **.safetensors**, **.engine**, etc.) — descarregados na primeira execução; ver [vaila models directory](#vaila-models-directory). **Pre-commit:** ficheiros **≥ 20 MiB** ao stage falham.

### Versões e releases no GitHub

- **Versão do pacote:** definida em `pyproject.toml` (`[project].version`); é a versão que **`uv`** / **`pip`** mostram (ex.: `0.3.34`).
- **Codename de release (GitHub):** formato **`rp` + data** — **rp** = Ribeirão Preto; data = dia + mês abreviado (inglês) + ano com dois dígitos, ex.: **`rp23mar26`** = 23 mar 2026.
- **Notas de release:** indiquem sempre as duas coisas (versão do pacote + codename) quando usarem codenames.
- **Tags Git:** ou tag semântica **`v0.3.34`** com título da release **`rp...`**, ou tag **`rp...`** com a versão do pacote explícita no texto — ver [Versioning and GitHub releases](#versioning-and-github-releases) (inglês).

- **Licença:** as contribuições são aceites sob [AGPL-3.0](LICENSE).
- **Segurança:** leia [SECURITY.md](SECURITY.md) — não commite chaves nem credenciais.
- **Ambiente:** `uv sync`, depois Ruff, `ty` e `pytest` como em [AGENTS.md](AGENTS.md).
- **PRs:** branch a partir de `main`, alterações focadas, testes quando mudar lógica de análise.
