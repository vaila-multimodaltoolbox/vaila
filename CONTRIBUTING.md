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

## Security reminders

- Do not commit `.env`, secrets, or real API examples. See [SECURITY.md](SECURITY.md).
- Large files: the repo enforces a **24MB** limit via git hooks (`install-hooks.sh`).

---

## Como contribuir

Obrigado por contribuir para o **vailá**. Este ficheiro complementa [AGENTS.md](AGENTS.md).

- **Licença:** as contribuições são aceites sob [AGPL-3.0](LICENSE).
- **Segurança:** leia [SECURITY.md](SECURITY.md) — não commite chaves nem credenciais.
- **Ambiente:** `uv sync`, depois Ruff, `ty` e `pytest` como em [AGENTS.md](AGENTS.md).
- **PRs:** branch a partir de `main`, alterações focadas, testes quando mudar lógica de análise.
