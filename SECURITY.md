# Security policy — _vailá_

**English** · [Português](#política-de-segurança)

## Supported versions

Security fixes are applied to the default development branch (`main`) and released as tagged versions when appropriate. Use the latest release or `main` for the most up-to-date fixes.

## Reporting a vulnerability

If you discover a security issue in **this repository** (code, scripts, or documented install paths):

1. **Prefer** [GitHub Security Advisories](https://github.com/vaila-multimodaltoolbox/vaila/security/advisories) (private report to maintainers), if enabled on the repo.
2. **Alternatively**, contact the maintainers through the contact options listed in [`pyproject.toml`](pyproject.toml) (authors / project metadata) or via an issue **without** including exploit details publicly until fixed.

Please include: affected component, steps to reproduce, and impact assessment when possible.

**Do not** open a public issue with exploit details before a fix is coordinated.

## Secrets and credentials (contributors)

This project is **open source (AGPL-3.0)**. The following must **never** be committed, pasted into issues/PRs, or embedded in documentation as real values:

- API keys (OpenAI, Groq, OpenRouter, Anthropic, cloud providers, etc.)
- Passwords, tokens, OAuth secrets, private SSH keys, or `.pem` / certificate private material
- Contents of local tool auth files (e.g. OpenCode / IDE credential stores under your home directory)
- Personal or institutional credentials

**Use instead:**

- Environment variables (e.g. load from a local `.env` that is **gitignored** — see [`.gitignore`](.gitignore))
- Placeholders in docs: `your_api_key_here`, `export MY_API_KEY=...` without real values
- [GitHub Encrypted Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets) for CI only

If you accidentally committed a secret:

1. **Revoke/rotate** the credential immediately at the provider.
2. Remove it from git history if it was ever pushed (e.g. `git filter-repo` or GitHub support) — a new commit alone is not enough if history is public.

## Dependency and supply chain

- Lockfile [`uv.lock`](uv.lock) is committed for reproducible installs.
- Prefer `uv add` / `uv sync` over ad-hoc `pip install` in documentation for this repo.

---

## Política de segurança

**Versões suportadas:** correções de segurança entram em `main` e em releases quando fizer sentido. Use a versão mais recente.

**Reportar vulnerabilidade:** use [Security Advisories](https://github.com/vaila-multimodaltoolbox/vaila/security/advisories) no GitHub ou contacto dos autores em [`pyproject.toml`](pyproject.toml). Não divulgar publicamente detalhes de exploração antes de coordenação com os mantenedores.

**Segredos:** nunca commitar chaves de API, palavras-passe, tokens ou ficheiros de autenticação locais (ex.: credenciais de ferramentas no diretório home). O projeto é AGPL e público — trate qualquer chave como comprometida se tiver sido exposta e **rode-a** no fornecedor.

**Se expuser uma chave por engano:** revogue-a de imediato e limpe o histórico do Git se já tiver sido enviado para o remoto.

Consulte também [CONTRIBUTING.md](CONTRIBUTING.md).
