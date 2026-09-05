# ============================================================================
# setup_ai_memory.ps1 — ai-memory cross-agent shared-memory bootstrap (Windows)
# ============================================================================
# Installs and wires up https://github.com/akitaonrails/ai-memory as shared,
# long-term memory for this repo (vaila-multimodaltoolbox/vaila), usable by
# Claude Code, Cursor Agent, OpenAI Codex, OpenCode, Gemini CLI, and other
# MCP-compatible harnesses. Mirrors the plan in installmemories.md; this is
# the idempotent, repeatable Windows PowerShell version of those steps. For
# Linux/macOS/WSL/Git Bash, use bin/setup_ai_memory.sh instead.
#
# Safe to re-run: every step checks current state before acting.
# ============================================================================
$ErrorActionPreference = "Stop"

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot   = Resolve-Path (Join-Path $ScriptDir "..")
$DaemonHost = "127.0.0.1"
$DaemonPort = 49374
$DaemonUrl  = "http://${DaemonHost}:${DaemonPort}"

Write-Host ">> vaila - ai-memory setup (windows, repo root: $RepoRoot)"

# ai-memory 2.0.3 has no `/healthz` route; the real MCP endpoint is `/mcp`
# (POST-only, so a GET there returns 405 — that 405 itself proves the port
# is up). Any response, including a non-2xx one, means "listening"; only a
# connection failure means "not up".
function Test-Daemon {
  try {
    Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 -Uri "$DaemonUrl/mcp" | Out-Null
    return $true
  } catch [System.Net.WebException] {
    if ($_.Exception.Response) { return $true }  # got an HTTP response (e.g. 405) = server is up
    return $false
  } catch {
    return $false
  }
}

# --- [1/4] binary + daemon ---------------------------------------------------
Write-Host ">> [1/4] Binary installation and service verification"

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
  Write-Host ">>   cargo not found. Install Rust first: https://rustup.rs (rustup-init.exe), then re-run this script."
  Write-Host "     Ensure %USERPROFILE%\.cargo\bin is on PATH afterwards."
  exit 1
}

if (Get-Command ai-memory -ErrorAction SilentlyContinue) {
  Write-Host ">>   ai-memory already installed: $((Get-Command ai-memory).Source)"
} else {
  Write-Host ">>   Installing ai-memory via cargo install --git ..."
  # Workspace repo has multiple binaries (cli/eval/importer); pin the CLI
  # package explicitly, which is what produces the `ai-memory` binary.
  cargo install --git https://github.com/akitaonrails/ai-memory.git ai-memory-cli
}

if (Test-Daemon) {
  Write-Host ">>   ai-memory daemon already responding at $DaemonUrl"
} else {
  Write-Host ">>   Starting ai-memory daemon (background)..."
  # `serve --daemon` doesn't exist; the CLI takes an explicit transport/bind
  # and runs in the foreground, so background it via Start-Process instead.
  Start-Process -FilePath "ai-memory" `
    -ArgumentList @("serve", "--transport", "http", "--bind", "${DaemonHost}:${DaemonPort}", "--enable-web") `
    -RedirectStandardOutput (Join-Path $RepoRoot ".ai-memory-daemon.log") `
    -RedirectStandardError  (Join-Path $RepoRoot ".ai-memory-daemon.err.log") `
    -WindowStyle Hidden
  Start-Sleep -Seconds 2
  if (Test-Daemon) {
    Write-Host ">>   Daemon is up."
  } else {
    Write-Warning "Daemon did not respond yet; check 'ai-memory status' or the .ai-memory-daemon*.log files manually."
  }
}

# --- [2/4] workspace init -----------------------------------------------------
Write-Host ">> [2/4] Repository workspace initialization"
# `init` only lays out the data directory (no --project flag); ai-memory
# scopes projects per-cwd automatically (basename strategy) via its hooks.
Push-Location $RepoRoot
try {
  ai-memory init
} finally {
  Pop-Location
}

$GitignorePath = Join-Path $RepoRoot ".gitignore"
$gitignoreText = if (Test-Path $GitignorePath) { Get-Content $GitignorePath -Raw } else { "" }
if ($gitignoreText -notmatch [regex]::Escape('.ai-memory/*.db')) {
  Write-Host ">>   Adding ai-memory index files to .gitignore"
  Add-Content -Path $GitignorePath -Value "`n# ai-memory local index`n.ai-memory/*.db`n.ai-memory/*.db-wal`n.ai-memory/*.db-shm"
}

# --- [3/4] harness configs -----------------------------------------------------
Write-Host ">> [3/4] Multi-agent harness configuration"

Write-Host ">>   Registering ai-memory MCP server + hooks with Claude Code..."
# ai-memory ships its own client-aware writers for this — no need to shell
# out to `claude mcp add`. install-mcp --apply edits %USERPROFILE%\.claude.json
# in place (name defaults to "ai-memory", timestamped backup written first);
# install-hooks --apply wires the lifecycle hooks the same way. Flag is
# `--agent`, not `--harness`.
try { ai-memory install-mcp --client claude-code --apply } catch { Write-Warning "install-mcp --client claude-code reported an issue; check manually." }

# `cargo install` only builds the binary — it doesn't copy the repo's
# hooks/ scripts anywhere ai-memory looks by default, so install-hooks
# can't find them unless pointed at the checkout.
$HooksDir = Get-ChildItem -Path (Join-Path $env:USERPROFILE ".cargo\git\checkouts") -Recurse -Directory -Filter "hooks" -ErrorAction SilentlyContinue |
  Where-Object { $_.FullName -match "ai-memory-" } | Select-Object -First 1 -ExpandProperty FullName
if ($HooksDir) {
  try { ai-memory install-hooks --agent claude-code --apply --hooks-dir $HooksDir } catch { Write-Warning "install-hooks --agent claude-code reported an issue; check manually." }
} else {
  Write-Host ">>   (note) couldn't find ai-memory's hooks\ dir under %USERPROFILE%\.cargo\git\checkouts;"
  Write-Host "        re-run: ai-memory install-hooks --agent claude-code --apply --hooks-dir <path-to-hooks>"
}

$CursorDir = Join-Path $RepoRoot ".cursor"
New-Item -ItemType Directory -Force -Path $CursorDir | Out-Null
$CursorMcp = Join-Path $CursorDir "mcp.json"
if (-not (Test-Path $CursorMcp)) {
  Write-Host ">>   Writing .cursor\mcp.json"
  @"
{
  "mcpServers": {
    "ai-memory": {
      "url": "$DaemonUrl/mcp"
    }
  }
}
"@ | Set-Content -Path $CursorMcp -Encoding utf8
}

$CursorRulesDir = Join-Path $CursorDir "rules"
New-Item -ItemType Directory -Force -Path $CursorRulesDir | Out-Null
$CursorRule = Join-Path $CursorRulesDir "ai-memory.mdc"
if (-not (Test-Path $CursorRule)) {
  Write-Host ">>   Writing .cursor\rules\ai-memory.mdc"
  @'
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
'@ | Set-Content -Path $CursorRule -Encoding utf8
}

$RootMcp = Join-Path $RepoRoot "mcp.json"
if (-not (Test-Path $RootMcp)) {
  Write-Host ">>   Writing root mcp.json (generic MCP discovery for Codex/OpenCode/Gemini CLI)"
  @"
{
  "mcpServers": {
    "ai-memory": {
      "type": "sse",
      "url": "$DaemonUrl/mcp"
    }
  }
}
"@ | Set-Content -Path $RootMcp -Encoding utf8
}

# There is no single "generic" --agent value — every other harness (codex,
# gemini-cli, open-code, ...) needs its own install-hooks/install-mcp call.
# install-instructions covers them agent-agnostically instead: it drops an
# idempotent, marker-delimited usage snippet + managed Agent Skills into the
# project itself, readable by any harness regardless of native hook support.
# --print only: by default this command MUTATES CLAUDE.md/AGENTS.md (both
# exist in this repo, so it writes to both) by inserting the snippet. That's
# a real edit to two curated, hand-maintained docs — preview it and apply by
# hand (drop --print) after reviewing the diff.
Write-Host ">>   Previewing agent-agnostic ai-memory usage instructions..."
Push-Location $RepoRoot
try {
  try { ai-memory install-instructions --print } catch { Write-Warning "install-instructions --print reported an issue; check manually." }
} finally {
  Pop-Location
}
Write-Host ">>   (review the snippet above; re-run 'ai-memory install-instructions' without --print to apply)"

# --- [4/4] verification --------------------------------------------------------
Write-Host ">> [4/4] Verification and sanity check"
# No `ingest` subcommand exists in this CLI — the wiki fills organically via
# lifecycle hooks during real sessions, or via `bootstrap` (needs an LLM
# provider configured). Sanity-check with commands that always work instead.
Push-Location $RepoRoot
try {
  try { ai-memory status } catch { Write-Warning "ai-memory status reported an issue." }
  # A 404 "project 'vaila' not found" here is expected on a brand-new
  # install: the project is created lazily by the first captured session
  # (via the hooks above), not by `init`. Re-run after your next real
  # Claude Code session in this repo.
  try { ai-memory search "vaila" } catch { Write-Warning "ai-memory search reported an issue (expected on a brand-new install; see comment above)." }
} finally {
  Pop-Location
}

Write-Host ""
Write-Host ">> Done. ai-memory should now be running at $DaemonUrl with the"
Write-Host "   Markdown wiki initialized and MCP endpoints declared for this repo."
Write-Host "   Persisting the daemon across reboots (e.g. a Scheduled Task or"
Write-Host "   Windows service) is not done by this script; see installmemories.md."
