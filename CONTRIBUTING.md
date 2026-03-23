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

- **Package version** — Source of truth is [`pyproject.toml`](pyproject.toml) under `[project].version` (PEP 621). This is what **`uv`** and **`pip`** report and what must match **wheels / sdist** metadata. Example form: `0.3.31` (often written as **v0.3.31** in prose).
- **GitHub release codename** — Human-facing name for a milestone: **`rp`** stands for **Ribeirão Preto**, followed by the date as **day + abbreviated English month + two-digit year**, e.g. **`rp23mar26`** = 23 Mar 2026. This does not replace the package version.
- **Release notes** — Prefer stating **both** so installers and GitHub readers stay aligned, for example:
  - `Package version: v0.3.31`
  - `Release codename: rp23mar26 (Ribeirão Preto — 23 Mar 2026)`
- **Git tags** — Pick **one** convention and use it consistently:
  - **Option A:** Git tag = **`v0.3.31`** (semver); GitHub **release title** = **`rp23mar26`** or **`rp23mar26 — v0.3.31`**.
  - **Option B:** Git tag = **`rp23mar26`**; the release description **must** clearly state the **package version** (e.g. **v0.3.31**).

## Security reminders

- Do not commit `.env`, secrets, or real API examples. See [SECURITY.md](SECURITY.md).
- Large files: the repo enforces a **24MB** limit via git hooks (`install-hooks.sh`).

---

## Como contribuir

Obrigado por contribuir para o **vailá**. Este ficheiro complementa [AGENTS.md](AGENTS.md).

### Versões e releases no GitHub

- **Versão do pacote:** definida em `pyproject.toml` (`[project].version`); é a versão que **`uv`** / **`pip`** mostram (ex.: `0.3.31`).
- **Codename de release (GitHub):** formato **`rp` + data** — **rp** = Ribeirão Preto; data = dia + mês abreviado (inglês) + ano com dois dígitos, ex.: **`rp23mar26`** = 23 mar 2026.
- **Notas de release:** indiquem sempre as duas coisas (versão do pacote + codename) quando usarem codenames.
- **Tags Git:** ou tag semântica **`v0.3.31`** com título da release **`rp...`**, ou tag **`rp...`** com a versão do pacote explícita no texto — ver [Versioning and GitHub releases](#versioning-and-github-releases) (inglês).

- **Licença:** as contribuições são aceites sob [AGPL-3.0](LICENSE).
- **Segurança:** leia [SECURITY.md](SECURITY.md) — não commite chaves nem credenciais.
- **Ambiente:** `uv sync`, depois Ruff, `ty` e `pytest` como em [AGENTS.md](AGENTS.md).
- **PRs:** branch a partir de `main`, alterações focadas, testes quando mudar lógica de análise.
