#!/usr/bin/env bash
# ============================================================================
# setup_ai_memory.sh — ai-memory cross-agent shared-memory bootstrap (Linux / macOS)
# ============================================================================
# Installs and wires up https://github.com/akitaonrails/ai-memory as shared,
# long-term memory for this repo (vaila-multimodaltoolbox/vaila), usable by
# Claude Code, Cursor Agent, OpenAI Codex, OpenCode, Gemini CLI, and other
# MCP-compatible harnesses. This script is the idempotent, repeatable,
# cross-platform (Linux/macOS/WSL/Git Bash) bootstrap. For native Windows
# PowerShell, use bin/setup_ai_memory.ps1 instead.
#
# Safe to re-run: every step checks current state before acting.
# ============================================================================
set -euo pipefail

if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
  echo ">> ERROR: do not run this script with sudo / as root." >&2
  echo "   It installs into YOUR \$HOME (~/.cargo, ~/.bashrc) and expects your" >&2
  echo "   own rustup toolchain there. Under sudo, \$HOME becomes /root, so it" >&2
  echo "   silently picks up root's (often older, distro) cargo instead —" >&2
  echo "   e.g. missing the 'edition2024' feature. Re-run as your normal user:" >&2
  echo "     bin/setup_ai_memory.sh" >&2
  exit 1
fi

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
  # Workspace repo has multiple binaries (cli/eval/importer); pin the CLI
  # package explicitly, which is what produces the `ai-memory` binary.
  cargo install --git https://github.com/akitaonrails/ai-memory.git ai-memory-cli
fi

# ai-memory 2.0.3 has no `/healthz` route and no `serve --daemon` flag — the
# server has a real MCP endpoint at `/mcp` instead (GET there returns 405,
# not 200, since MCP is POST-only; that 405 is itself proof the port is up).
# `curl` without `-f` exits 0 on any HTTP response, only failing on a refused
# connection, which is exactly the "is something listening" check we want.
if curl -sS -m 3 -o /dev/null "${DAEMON_URL}/mcp" 2>/dev/null; then
  echo ">>   ai-memory daemon already responding at ${DAEMON_URL}"
else
  echo ">>   Starting ai-memory daemon (background)..."
  nohup ai-memory serve --transport http --bind "${DAEMON_HOST}:${DAEMON_PORT}" --enable-web \
    > "${REPO_ROOT}/.ai-memory-daemon.log" 2>&1 < /dev/null &
  disown
  sleep 2
  curl -sS -m 5 -o /dev/null "${DAEMON_URL}/mcp" 2>/dev/null \
    && echo ">>   Daemon is up." \
    || echo ">>   (warning) Daemon did not respond yet; check 'ai-memory status' or ${REPO_ROOT}/.ai-memory-daemon.log manually."
fi

# ------------------------------------------------- [2/4] workspace init
echo ">> [2/4] Repository workspace initialization"
# `init` only lays out the data directory (no --project flag); ai-memory
# scopes projects per-cwd automatically (basename strategy) via its hooks.
(cd "${REPO_ROOT}" && ai-memory init)

GITIGNORE="${REPO_ROOT}/.gitignore"
if ! grep -q '^\.ai-memory/\*\.db$' "${GITIGNORE}" 2>/dev/null; then
  echo ">>   Adding ai-memory index files to .gitignore"
  { echo ''; echo '# ai-memory local index'; \
    echo '.ai-memory/*.db'; echo '.ai-memory/*.db-wal'; echo '.ai-memory/*.db-shm'; } >> "${GITIGNORE}"
fi

# ------------------------------------------------- [3/4] harness configs
echo ">> [3/4] Multi-agent harness configuration"

# `cargo install` only builds the binary — it doesn't copy the repo's
# hooks/ scripts anywhere ai-memory looks by default (/usr/local/share/...
# etc.), so install-hooks can't find them unless pointed at the checkout.
# Keep the path on ONE line when pasting manually — shell line-wrap glyphs
# like │ break --hooks-dir and yield "hooks directory │/cursor does not exist".
AI_MEMORY_HOOKS_DIR="$(find "${HOME}/.cargo/git/checkouts" -maxdepth 3 -type d \
  -path '*/ai-memory-*/*/hooks' 2>/dev/null | head -n1)"

# Cross-harness memory: same daemon + same project wiki. Handoffs created by
# one agent's SessionEnd (or memory_handoff_begin shared=true) are consumed by
# the next agent's SessionStart — Claude ↔ Cursor Agent (`agent`) ↔ Codex ↔
# Antigravity (`agy`). Cursor is NOT an `ai-memory run` harness; launch with
# `agent` after MCP/hooks are wired.
#
# install-mcp --client <…>  |  install-hooks --agent <…>
# Flag for hooks is `--agent`, not `--harness`.
wire_harness() {
  local mcp_client="$1"
  local hook_agent="$2"
  echo ">>   Wiring MCP client=${mcp_client} hooks-agent=${hook_agent}..."
  ai-memory install-mcp --client "${mcp_client}" --apply \
    || echo ">>   (note) install-mcp --client ${mcp_client} reported an issue; check manually."
  if [[ -n "${AI_MEMORY_HOOKS_DIR}" ]]; then
    ai-memory install-hooks --agent "${hook_agent}" --apply --hooks-dir "${AI_MEMORY_HOOKS_DIR}" \
      || echo ">>   (note) install-hooks --agent ${hook_agent} reported an issue; check manually."
  else
    echo ">>   (note) couldn't find ai-memory's hooks/ dir under ~/.cargo/git/checkouts;"
    echo "        re-run: ai-memory install-hooks --agent ${hook_agent} --apply --hooks-dir <path-to-hooks>"
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
(cd "${REPO_ROOT}" && ai-memory install-instructions --print) \
  || echo ">>   (note) install-instructions --print reported an issue; check manually."
echo ">>   (review the snippet above; re-run 'ai-memory install-instructions' without --print to apply)"

# ------------------------------------------------- [4/4] verification
echo ">> [4/4] Verification and sanity check"
# No `ingest` subcommand exists in this CLI — the wiki fills organically via
# lifecycle hooks during real sessions, or via `bootstrap` (needs an LLM
# provider configured). Sanity-check with commands that always work instead.
(cd "${REPO_ROOT}" && ai-memory status 2>&1 | sed 's/^/>>   /') || true
# A 404 "project 'vaila' not found" here is expected on a brand-new install:
# the project is created lazily by the first captured session (via the
# hooks above), not by `init`. Re-run this search after your next real
# Claude Code session in this repo.
(cd "${REPO_ROOT}" && ai-memory search "vaila" 2>&1 | sed 's/^/>>   /') || true

echo ""
echo ">> Done. ai-memory should now be running at ${DAEMON_URL} with the"
echo "   Markdown wiki initialized and MCP endpoints declared for this repo."
echo "   Persisting the daemon across reboots is OS-specific and NOT done by"
echo "   this script (e.g. systemd --user unit on Linux, launchd agent on"
echo "   macOS) if you want it to survive a logout/reboot."
