# Guia de Releases e Geração de Instaladores Multi-OS (GitHub Actions)
# GitHub Release & Multi-OS Installer Guide

**Português** · [English](#english-version)

Este guia documenta o fluxo passo a passo para criar uma tag a partir da branch `main` e gerar automaticamente uma **Release no GitHub** contendo os instaladores de **Windows (`.exe`)** e **macOS (`.dmg`)** compilados nas máquinas virtuais (runners) do GitHub Actions.

---

## Como Funciona a Automação

O repositório possui o workflow configurado em [`.github/workflows/release-installers.yml`](../.github/workflows/release-installers.yml).

Quando uma tag com prefixo **`v*`** (ex: `v0.4.3`) ou **`rp*`** (ex: `rp15Sep2026`) é enviada (`git push`) para o GitHub, o GitHub Actions dispara automaticamente:

1. **Job `resolve` (Ubuntu)**:
   - Identifica a tag e a versão do pacote a partir de `vaila.py` / `pyproject.toml`.
2. **Job `build-macos` (VM macOS `macos-latest`)**:
   - Executa [`create_dmg_installer.sh`](../create_dmg_installer.sh).
   - Compila e empacota a imagem de disco **`vaila_installer.dmg`**.
   - Faz upload do artefato para a release.
3. **Job `build-windows` (VM Windows `windows-latest`)**:
   - Instala o Inno Setup via Chocolatey.
   - Atualiza a versão no script [`vaila_installer.iss`](../vaila_installer.iss).
   - Compila o instalador executável **`vaila_installer.exe`**.
   - Faz upload do artefato para a release.
4. **Job `publish-release` (Ubuntu)**:
   - Baixa os artefatos `.dmg` e `.exe`.
   - Cria (ou atualiza) a **GitHub Release** correspondente à tag.
   - Anexa ambos os instaladores e publica as notas da release.

---

## Passo a Passo para Criar uma Release

### 1. Checklist Pré-Release (Verificação Local)

Antes de criar a tag, garanta que a branch `main` está atualizada e limpa:

```bash
# 1. Certifique-se de estar na branch main atualizada
git checkout main
git pull origin main

# 2. Execute a suíte de testes rápidos e o linter
uv run ruff check vaila/
uv run pytest tests/test_planar_geometry_tracker.py tests/test_video_stabilizer.py tests/test_dlt_rec.py -v

# 3. Verifique se a versão está alinhada em vaila.py e pyproject.toml
grep -E "^Version:" vaila.py
grep -E "^version *=" pyproject.toml
```

Se houver alterações pendentes, faça commit e envie para `main`:

```bash
git add pyproject.toml vaila.py README.md
git commit -m "chore: bump version to v0.4.3 for release"
git push origin main
```

---

### 2. Criando e Enviando a Tag Git

Escolha o formato da tag:
- **Semântico (Recomendado):** `v0.4.3`
- **Codename vailá:** `rp15Sep2026` (onde `rp` = Ribeirão Preto, seguido de dia + mês em inglês + ano)

Execute no terminal:

```bash
# Criar tag anotada na HEAD da branch main
git tag -a v0.4.3 -m "Release v0.4.3 — rp15Sep2026"

# Enviar a tag para o GitHub (isso dispara o GitHub Actions!)
git push origin v0.4.3
```

> [!IMPORTANT]
> O envio da tag (`git push origin <tag>`) **inicia imediatamente** as VMs do GitHub Actions. Não é necessário compilar o `.exe` ou `.dmg` localmente.

---

### 3. Acompanhando a Compilação dos Instaladores

Assim que a tag for enviada:
1. Abra no navegador: **`https://github.com/vaila-multimodaltoolbox/vaila/actions`**
2. Clique na execução ativa do workflow **"Release installers"**.
3. Você verá os jobs paralelos rodando:
   - `build-macos` (compilando `vaila_installer.dmg` no macOS)
   - `build-windows` (compilando `vaila_installer.exe` no Windows)
4. Ao final, o job `publish-release` publicará a release em:
   **`https://github.com/vaila-multimodaltoolbox/vaila/releases`**

Se preferir acompanhar pelo terminal via GitHub CLI (`gh`):
```bash
gh run list --workflow=release-installers.yml
gh run watch
```

---

### 4. Re-gerar Instaladores Manualmente (Sem Nova Tag)

Se uma compilação falhar na VM ou se você precisar regerar os instaladores para uma tag existente sem criar uma nova tag:

1. Acesse: **Actions** → **Release installers** → Botão **"Run workflow"**.
2. Digite a tag desejada (ex: `v0.4.3`) e confirme.

Pelo terminal (GitHub CLI):
```bash
gh workflow run release-installers.yml -f tag=v0.4.3
```

---

### 5. Como Corrigir / Deletar uma Tag Criada por Engano

Se você criou uma tag com erro e precisa removê-la:

```bash
# 1. Deletar a tag localmente
git tag -d v0.4.3

# 2. Deletar a tag no GitHub
git push origin --delete v0.4.3

# 3. (Opcional) Se a release já tiver sido criada, exclua-a via gh:
gh release delete v0.4.3 --yes
```

---

### 6. Script de Automação Rápida (`bin/create_release.sh`)

Para facilitar o processo no dia a dia, use o script helper incluído no repositório:

```bash
# Modo interativo (guia você por todas as etapas)
bash bin/create_release.sh

# Modo direto especificando a tag
bash bin/create_release.sh --tag=v0.4.3 --yes
```

---
---

<a name="english-version"></a>
# English Version

This guide documents the end-to-end workflow to tag `main` and trigger automated **GitHub Releases** with **Windows (`.exe`)** and **macOS (`.dmg`)** installers built inside GitHub Actions virtual machines.

---

## Workflow Overview

The repository workflow is defined in [`.github/workflows/release-installers.yml`](../.github/workflows/release-installers.yml).

Pushing a tag matching **`v*`** (e.g. `v0.4.3`) or **`rp*`** (e.g. `rp15Sep2026`) triggers:
1. **`resolve`** (Ubuntu): extracts the release tag and package version from `vaila.py` / `pyproject.toml`.
2. **`build-macos`** (`macos-latest` VM): runs [`create_dmg_installer.sh`](../create_dmg_installer.sh) and builds `vaila_installer.dmg`.
3. **`build-windows`** (`windows-latest` VM): installs Inno Setup and compiles [`vaila_installer.iss`](../vaila_installer.iss) into `vaila_installer.exe`.
4. **`publish-release`** (Ubuntu): downloads artifacts, creates or updates the GitHub Release, and publishes the release notes.

---

## Step-by-Step Release Instructions

### 1. Pre-Release Checklist

Ensure `main` is clean, synced, and passes tests:

```bash
git checkout main
git pull origin main
uv run ruff check vaila/
uv run pytest tests/test_planar_geometry_tracker.py tests/test_video_stabilizer.py tests/test_dlt_rec.py -v
```

Ensure version consistency across files:
```bash
grep -E "^Version:" vaila.py
grep -E "^version *=" pyproject.toml
```

Commit and push any version updates:
```bash
git add pyproject.toml vaila.py README.md
git commit -m "chore: bump version to v0.4.3 for release"
git push origin main
```

---

### 2. Create and Push the Tag

```bash
# Create annotated tag
git tag -a v0.4.3 -m "Release v0.4.3 — rp15Sep2026"

# Push tag to GitHub (triggers GitHub Actions)
git push origin v0.4.3
```

---

### 3. Monitor Build & Release

1. Visit: **`https://github.com/vaila-multimodaltoolbox/vaila/actions`**
2. Watch the **"Release installers"** workflow.
3. Once completed, inspect the release and download assets at:
   **`https://github.com/vaila-multimodaltoolbox/vaila/releases`**

---

### 4. Manual Workflow Dispatch (Rebuild Existing Tag)

Via GitHub Web UI:
- Go to **Actions** → **Release installers** → **Run workflow** → enter tag name.

Via CLI:
```bash
gh workflow run release-installers.yml -f tag=v0.4.3
```

---

### 5. Deleting a Mistaken Tag

```bash
git tag -d v0.4.3
git push origin --delete v0.4.3
gh release delete v0.4.3 --yes
```

---

### 6. One-Command Helper Script

```bash
bash bin/create_release.sh
```
