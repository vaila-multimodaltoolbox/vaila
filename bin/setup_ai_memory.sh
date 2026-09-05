#!/usr/bin/env bash
# ============================================================================
# setup_ai_memory.sh — ai-memory cross-agent shared-memory bootstrap (Linux / macOS)
# ============================================================================
# Installs and wires up https://github.com/akitaonrails/ai-memory as shared,
# long-term memory for this repo (vaila-multimodaltoolbox/vaila), usable by
# Claude Code, Cursor Agent, OpenAI Codex, OpenCode, Gemini CLI, and other
# MCP-compatible harnesses. Mirrors the plan in installmemories.md; this
# script is the idempotent, repeatable, cross-platform (Linux/macOS/WSL/Git
# Bash) version of those steps. For native Windows PowerShell, use
# bin/setup_ai_memory.ps1 instead.
#
# Safe to re-run: every step checks current state before acting.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DAEMON_HOST="127.0.0.1"
DAEMON_PORT="49374"
DAEMON_URL="http://${DAEMON_HOST}:${DAEMON_PORT}"

OS_NAME="$(uname -s)"
case "${OS_NAME}" in
  Linux*)  PLATFORM="linux" ;;
  Darwin*) PLATFORM="macos" ;;
  *)       PLATFORM="other" ;;
esac

echo ">> vaila — ai-memory setup (${PLATFORM}, repo root: ${REPO_ROOT})"

# ---------------------------------------------------------- shell rc for PATH
# Used only if we need to export ~/.cargo/bin ourselves (rustup's own
# installer usually does this already via ~/.cargo/env).
shell_rc() {
  case "$(basename "${SHELL:-bash}")" in
    zsh)  echo "${HOME}/.zshrc" ;;
    bash) if [[ "${PLATFORM}" == "macos" ]]; then echo "${HOME}/.bash_profile"; else echo "${HOME}/.bashrc"; fi ;;
    *)    echo "${HOME}/.profile" ;;
  esac
}

ensure_cargo_bin_on_path() {
  local rc; rc="$(shell_rc)"
  if ! grep -q '\.cargo/bin' "${rc}" 2>/dev/null; then
    echo ">>   Adding ~/.cargo/bin to PATH in ${rc}"
    { echo ''; echo '# ai-memory / rustup (added by bin/setup_ai_memory.sh)'; \
      echo 'export PATH="$HOME/.cargo/bin:$PATH"'; } >> "${rc}"
  fi
  export PATH="${HOME}/.cargo/bin:${PATH}"
}

# ------------------------------------------------- [1/4] binary + daemon
echo ">> [1/4] Binary installation and service verification"

if ! command -v cargo >/dev/null 2>&1; then
  echo ">>   cargo not found; installing Rust toolchain via rustup..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  # shellcheck disable=SC1090
  source "${HOME}/.cargo/env"
fi
ensure_cargo_bin_on_path

if command -v ai-memory >/dev/null 2>&1; then
  echo ">>   ai-memory already installed: $(command -v ai-memory)"
else
  echo ">>   Installing ai-memory via cargo install --git ..."
  cargo install --git https://github.com/akitaonrails/ai-memory.git
fi

if curl -fsS -m 3 "${DAEMON_URL}/healthz" >/dev/null 2>&1; then
  echo ">>   ai-memory daemon already responding at ${DAEMON_URL}"
else
  echo ">>   Starting ai-memory daemon (background)..."
  ai-memory serve --daemon
  sleep 1
  curl -fsS -m 5 "${DAEMON_URL}/healthz" >/dev/null 2>&1 \
    && echo ">>   Daemon is up." \
    || echo ">>   (warning) Daemon did not respond yet; check 'ai-memory status' manually."
fi

# ------------------------------------------------- [2/4] workspace init
echo ">> [2/4] Repository workspace initialization"
(cd "${REPO_ROOT}" && ai-memory init --project vaila)

GITIGNORE="${REPO_ROOT}/.gitignore"
if ! grep -q '^\.ai-memory/\*\.db$' "${GITIGNORE}" 2>/dev/null; then
  echo ">>   Adding ai-memory index files to .gitignore"
  { echo ''; echo '# ai-memory local index'; \
    echo '.ai-memory/*.db'; echo '.ai-memory/*.db-wal'; echo '.ai-memory/*.db-shm'; } >> "${GITIGNORE}"
fi

# ------------------------------------------------- [3/4] harness configs
echo ">> [3/4] Multi-agent harness configuration"

if command -v claude >/dev/null 2>&1; then
  echo ">>   Registering ai-memory MCP server with Claude Code..."
  claude mcp add ai-memory "${DAEMON_URL}/mcp" || echo ">>   (note) claude mcp add reported an issue; check manually."
  ai-memory install-hooks --harness claude-code || echo ">>   (note) install-hooks --harness claude-code reported an issue."
else
  echo ">>   'claude' CLI not found on PATH; skipping Claude Code MCP registration."
fi

mkdir -p "${REPO_ROOT}/.cursor"
if [[ ! -f "${REPO_ROOT}/.cursor/mcp.json" ]]; then
  echo ">>   Writing .cursor/mcp.json"
  cat > "${REPO_ROOT}/.cursor/mcp.json" <<EOF
{
  "mcpServers": {
    "ai-memory": {
      "url": "${DAEMON_URL}/mcp"
    }
  }
}
EOF
fi

mkdir -p "${REPO_ROOT}/.cursor/rules"
if [[ ! -f "${REPO_ROOT}/.cursor/rules/ai-memory.mdc" ]]; then
  echo ">>   Writing .cursor/rules/ai-memory.mdc"
  cat > "${REPO_ROOT}/.cursor/rules/ai-memory.mdc" <<'EOF'
---
description: ai-memory cross-agent shared memory integration
alwaysApply: true
---

# ai-memory Integration

This repository uses [`ai-memory`](https://github.com/akitaonrails/ai-memory) as
shared, long-term memory across coding harnesses (Claude Code, Cursor Agent,
OpenAI Codex, OpenCode, Gemini CLI). The daemon runs locally at
`http://127.0.0.1:49374` and is registered as an MCP server (`.cursor/mcp.json`
for Cursor, root `mcp.json` for other MCP-compatible CLIs).

- **At the start of a session**: check `ai-memory` via MCP tools
  (`search_memory`, `get_handoff`) for prior context, unresolved edge cases,
  and architectural decisions relevant to the current task.
- **Before exiting or concluding a major task**: summarize unresolved edge
  cases, architectural decisions, and hardware/environment dependencies using
  `create_handoff`, so the next agent (regardless of harness) can pick up
  where this session left off.

See `installmemories.md` at the repo root for the full setup plan and
`.ai-memory.toml` for local index/wiki paths.
EOF
fi

if [[ ! -f "${REPO_ROOT}/mcp.json" ]]; then
  echo ">>   Writing root mcp.json (generic MCP discovery for Codex/OpenCode/Gemini CLI)"
  cat > "${REPO_ROOT}/mcp.json" <<EOF
{
  "mcpServers": {
    "ai-memory": {
      "type": "sse",
      "url": "${DAEMON_URL}/mcp"
    }
  }
}
EOF
fi

echo ">>   Running generic hook injection for other harnesses..."
ai-memory install-hooks --harness generic || echo ">>   (note) install-hooks --harness generic reported an issue."

# ------------------------------------------------- [4/4] verification
echo ">> [4/4] Verification and sanity check"
(cd "${REPO_ROOT}" && ai-memory ingest README.md 2>&1 | sed 's/^/>>   /') || true
(cd "${REPO_ROOT}" && ai-memory search "biomechanics pipeline" 2>&1 | sed 's/^/>>   /') || true

echo ""
echo ">> Done. ai-memory should now be running at ${DAEMON_URL} with the"
echo "   Markdown wiki initialized and MCP endpoints declared for this repo."
echo "   Persisting the daemon across reboots is OS-specific and NOT done by"
echo "   this script; see the 'Persistent background service' notes in"
echo "   installmemories.md (systemd --user unit on Linux, launchd agent on"
echo "   macOS) if you want it to survive a logout/reboot."
