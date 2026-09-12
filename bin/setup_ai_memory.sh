#!/usr/bin/env bash
# ============================================================================
# setup_ai_memory.sh — ai-memory cross-agent shared-memory bootstrap (Linux / macOS)
# ============================================================================
# Installs and wires up https://github.com/akitaonrails/ai-memory as shared,
# long-term memory for this repo (vaila-multimodaltoolbox/vaila), usable by
# Claude Code, Cursor Agent, OpenAI Codex, OpenCode, Gemini CLI, and other
# MCP-compatible harnesses. This script is the idempotent, repeatable,
# cross-platform (Linux/macOS/WSL) bootstrap. On native Windows (including
# Git Bash), use bin/setup_ai_memory.ps1 instead.
#
# Safe to re-run: every step checks current state before acting.
# ============================================================================
set -Eeuo pipefail

if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
  echo ">> ERROR: do not run this script with sudo / as root." >&2
  echo "   It installs into YOUR \$HOME (~/Applications or ~/.local) and" >&2
  echo "   configures user-level agent clients. Re-run as your normal user:" >&2
  echo "     bin/setup_ai_memory.sh" >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DAEMON_HOST="127.0.0.1"
DAEMON_PORT="49374"
DAEMON_URL="http://${DAEMON_HOST}:${DAEMON_PORT}"
INSTALL_TMP_DIR=""

OS_NAME="$(uname -s)"
case "${OS_NAME}" in
  Linux*)  PLATFORM="linux" ;;
  Darwin*) PLATFORM="macos" ;;
  *)       PLATFORM="other" ;;
esac

case "$(uname -m)" in
  arm64|aarch64) RELEASE_ARCH="aarch64" ;;
  x86_64|amd64) RELEASE_ARCH="x86_64" ;;
  *) RELEASE_ARCH="unsupported" ;;
esac

if [[ "${PLATFORM}" == "macos" ]]; then
  AI_MEMORY_INSTALL_DIR="${HOME}/Applications/ai-memory"
else
  AI_MEMORY_INSTALL_DIR="${HOME}/.local/opt/ai-memory"
fi

cleanup() {
  if [[ -n "${INSTALL_TMP_DIR}" && -d "${INSTALL_TMP_DIR}" ]]; then
    rm -rf -- "${INSTALL_TMP_DIR}"
  fi
}

trap cleanup EXIT
trap 'echo ">> ERROR: ai-memory setup failed at line ${LINENO}." >&2' ERR

echo ">> vaila — ai-memory setup (${PLATFORM}, repo root: ${REPO_ROOT})"

# ---------------------------------------------------------- shell rc for PATH
# The release binary is exposed through ~/.local/bin so every harness can
# resolve the same stable command after setup.
shell_rc() {
  case "$(basename "${SHELL:-bash}")" in
    zsh)  echo "${HOME}/.zshrc" ;;
    bash) if [[ "${PLATFORM}" == "macos" ]]; then echo "${HOME}/.bash_profile"; else echo "${HOME}/.bashrc"; fi ;;
    *)    echo "${HOME}/.profile" ;;
  esac
}

ensure_local_bin_on_path() {
  local rc; rc="$(shell_rc)"
  mkdir -p "${HOME}/.local/bin"
  if ! grep -q '\.local/bin' "${rc}" 2>/dev/null; then
    echo ">>   Adding ~/.local/bin to PATH in ${rc}"
    { echo ''; echo '# ai-memory (added by bin/setup_ai_memory.sh)'; \
      echo 'export PATH="$HOME/.local/bin:$PATH"'; } >> "${rc}"
  fi
  export PATH="${HOME}/.local/bin:${PATH}"
}

install_release_binary() {
  local archive_name archive_url checksum_name command_name

  if [[ "${PLATFORM}" == "other" || "${RELEASE_ARCH}" == "unsupported" ]]; then
    echo ">> ERROR: no ai-memory release is published for ${OS_NAME}/$(uname -m)." >&2
    echo ">>        Follow the upstream source-build instructions instead." >&2
    return 1
  fi

  for command_name in curl tar; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
      echo ">> ERROR: required command not found: ${command_name}" >&2
      return 1
    fi
  done

  archive_name="ai-memory-${PLATFORM}-${RELEASE_ARCH}.tar.gz"
  checksum_name="${archive_name}.sha256"
  archive_url="https://github.com/akitaonrails/ai-memory/releases/latest/download/${archive_name}"
  INSTALL_TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/vaila-ai-memory.XXXXXX")"

  echo ">>   Downloading the latest native release (${PLATFORM}/${RELEASE_ARCH})..."
  curl -fsSL --retry 3 "${archive_url}" -o "${INSTALL_TMP_DIR}/${archive_name}"
  curl -fsSL --retry 3 "${archive_url}.sha256" -o "${INSTALL_TMP_DIR}/${checksum_name}"

  echo ">>   Verifying the published SHA-256 checksum..."
  if command -v sha256sum >/dev/null 2>&1; then
    (cd "${INSTALL_TMP_DIR}" && sha256sum -c "${checksum_name}")
  elif command -v shasum >/dev/null 2>&1; then
    (cd "${INSTALL_TMP_DIR}" && shasum -a 256 -c "${checksum_name}")
  else
    echo ">> ERROR: sha256sum or shasum is required to verify the download." >&2
    return 1
  fi

  mkdir -p "${AI_MEMORY_INSTALL_DIR}"
  tar -xzf "${INSTALL_TMP_DIR}/${archive_name}" -C "${AI_MEMORY_INSTALL_DIR}"
  if [[ ! -x "${AI_MEMORY_INSTALL_DIR}/ai-memory" ]]; then
    echo ">> ERROR: release did not contain an executable ai-memory binary." >&2
    return 1
  fi

  ln -sfn "${AI_MEMORY_INSTALL_DIR}/ai-memory" "${HOME}/.local/bin/ai-memory"
  AI_MEMORY_BIN="${AI_MEMORY_INSTALL_DIR}/ai-memory"
}

# --------------------------------------------------------- [1/4] native binary
echo ">> [1/4] Native binary installation"

ensure_local_bin_on_path

if command -v ai-memory >/dev/null 2>&1; then
  AI_MEMORY_BIN="$(command -v ai-memory)"
  echo ">>   ai-memory already installed: ${AI_MEMORY_BIN}"
else
  install_release_binary
  echo ">>   Installed ai-memory: ${AI_MEMORY_BIN}"
fi
"${AI_MEMORY_BIN}" --version

if ! command -v curl >/dev/null 2>&1; then
  echo ">> ERROR: curl is required for the local-service readiness check." >&2
  exit 1
fi

# ------------------------------------------- [2/4] data init + local service
echo ">> [2/4] Data initialization and local service verification"
# `init` only lays out the data directory (no --project flag); ai-memory
# scopes projects per-cwd automatically (basename strategy) via its hooks.
(cd "${REPO_ROOT}" && "${AI_MEMORY_BIN}" init)

# The MCP endpoint is POST-only, so GET normally returns 405. `curl` without
# `-f` succeeds on any HTTP response and fails on a refused connection, which
# makes it a reliable readiness probe without depending on a `/healthz` route.
if curl -sS -m 3 -o /dev/null "${DAEMON_URL}/mcp" 2>/dev/null; then
  echo ">>   ai-memory daemon already responding at ${DAEMON_URL}"
else
  echo ">>   Starting ai-memory daemon (background)..."
  nohup "${AI_MEMORY_BIN}" serve --transport http --bind "${DAEMON_HOST}:${DAEMON_PORT}" --enable-web \
    > "${REPO_ROOT}/.ai-memory-daemon.log" 2>&1 < /dev/null &
  disown
  sleep 2
  if curl -sS -m 5 -o /dev/null "${DAEMON_URL}/mcp" 2>/dev/null; then
    echo ">>   Daemon is up."
  else
    echo ">> ERROR: daemon did not respond at ${DAEMON_URL}." >&2
    echo ">>        See ${REPO_ROOT}/.ai-memory-daemon.log for details." >&2
    exit 1
  fi
fi

GITIGNORE="${REPO_ROOT}/.gitignore"
if ! grep -q '^\.ai-memory/\*\.db$' "${GITIGNORE}" 2>/dev/null; then
  echo ">>   Adding ai-memory index files to .gitignore"
  { echo ''; echo '# ai-memory local index'; \
    echo '.ai-memory/*.db'; echo '.ai-memory/*.db-wal'; echo '.ai-memory/*.db-shm'; } >> "${GITIGNORE}"
fi

# ------------------------------------------------- [3/4] harness configs
echo ">> [3/4] Multi-agent harness configuration"

# Cross-harness memory: same daemon + same project wiki. Handoffs created by
# one agent's SessionEnd (or memory_handoff_begin shared=true) are consumed by
# the next agent's SessionStart — Claude ↔ Cursor Agent (`agent`) ↔ Codex ↔
# Antigravity (`agy`). Cursor is NOT an `ai-memory run` harness; launch with
# `agent` after MCP/hooks are wired.
#
# install-mcp --client <…>  |  install-hooks --agent <…>
# Flag for hooks is `--agent`, not `--harness`.
AI_MEMORY_HOOKS_DIR="$(find "${HOME}/.cargo/git/checkouts" -maxdepth 3 -type d \
  -path '*/ai-memory-*/*/hooks' 2>/dev/null | head -n1)"

wire_harness() {
  local mcp_client="$1"
  local hook_agent="$2"
  echo ">>   Wiring MCP client=${mcp_client} hooks-agent=${hook_agent}..."
  "${AI_MEMORY_BIN}" install-mcp --client "${mcp_client}" --apply \
    || echo ">>   (note) install-mcp --client ${mcp_client} reported an issue; check manually."
  if [[ -n "${AI_MEMORY_HOOKS_DIR}" ]]; then
    "${AI_MEMORY_BIN}" install-hooks --agent "${hook_agent}" --apply --hooks-dir "${AI_MEMORY_HOOKS_DIR}" \
      || echo ">>   (note) install-hooks --agent ${hook_agent} reported an issue; check manually."
  else
    "${AI_MEMORY_BIN}" install-hooks --agent "${hook_agent}" --apply \
      || echo ">>   (note) install-hooks --agent ${hook_agent} reported an issue; check manually."
  fi
}

wire_harness claude-code claude-code
wire_harness cursor cursor
wire_harness codex codex
wire_harness antigravity-cli antigravity-cli

echo ">>   Notes:"
echo "      - Cursor Agent CLI: cd <repo> && agent --approve-mcps   (NOT: ai-memory run agent)"
echo "      - Claude Code:      ai-memory run claude   OR   claude"
echo "      - Codex:            ai-memory run codex    OR   codex  (trust hooks once in TUI)"
echo "      - Antigravity:      ai-memory run antigravity OR agy"
echo "      - After final agy turn: ai-memory finalize-session --agent antigravity-cli"

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
# Always refresh the rule so tool names / cross-harness notes stay current.
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
  (`memory_query`, `memory_handoff_accept` / SessionStart handoff block) for
  prior context, unresolved edge cases, and architectural decisions.
- **Before exiting or concluding a major task**: summarize with
  `memory_handoff_begin` (`shared: true` when the next operator/harness should
  pick it up), so Claude / `agent` / Codex / `agy` share the same baton.
- **Cursor Agent CLI** is launched with `agent`, not `ai-memory run agent`
  (Cursor is not an `ai-memory run` harness). Wire via
  `install-mcp --client cursor` + `install-hooks --agent cursor`.
- **Antigravity (`agy`)**: after the final turn run
  `ai-memory finalize-session --agent antigravity-cli` so the handoff is closed.

See `bin/setup_ai_memory.sh` (or `bin/setup_ai_memory.ps1`) for setup details
and `.ai-memory.toml` for local index/wiki paths.
EOF

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

# install-instructions covers remaining agents agent-agnostically: drops an
# idempotent, marker-delimited usage snippet + managed Agent Skills into the
# project itself, readable by any harness regardless of native hook support.
echo ">>   Previewing agent-agnostic ai-memory usage instructions..."
# --print only: by default this command MUTATES CLAUDE.md/AGENTS.md (both
# exist in this repo, so it would write to both) by inserting a marker-
# delimited snippet. That's a real edit to two curated, hand-maintained
# docs — preview it and apply by hand (drop --print) after reviewing the
# diff, rather than have a bootstrap script silently rewrite them.
(cd "${REPO_ROOT}" && "${AI_MEMORY_BIN}" install-instructions --print) \
  || echo ">>   (note) install-instructions --print reported an issue; check manually."
echo ">>   (review the snippet above; re-run 'ai-memory install-instructions' without --print to apply)"

# ------------------------------------------------- [4/4] verification
echo ">> [4/4] Verification and sanity check"
# No `ingest` subcommand exists in this CLI — the wiki fills organically via
# lifecycle hooks during real sessions, or via `bootstrap` (needs an LLM
# provider configured). Sanity-check with commands that always work instead.
(cd "${REPO_ROOT}" && "${AI_MEMORY_BIN}" status 2>&1 | sed 's/^/>>   /') || true
# A 404 "project 'vaila' not found" here is expected on a brand-new install:
# the project is created lazily by the first captured session (via the
# hooks above), not by `init`. Re-run this search after your next real
# Claude Code session in this repo.
(cd "${REPO_ROOT}" && "${AI_MEMORY_BIN}" search "vaila" 2>&1 | sed 's/^/>>   /') || true

echo ""
echo ">> Done. ai-memory should now be running at ${DAEMON_URL} with the"
echo "   Markdown wiki initialized and MCP endpoints declared for this repo."
echo "   Persisting the daemon across reboots is OS-specific and NOT done by"
echo "   this script (e.g. systemd --user unit on Linux, launchd agent on"
echo "   macOS) if you want it to survive a logout/reboot."
