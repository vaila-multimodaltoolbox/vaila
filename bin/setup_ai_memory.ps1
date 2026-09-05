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

function Test-Daemon {
  try {
    Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 -Uri "$DaemonUrl/healthz" | Out-Null
    return $true
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
  cargo install --git https://github.com/akitaonrails/ai-memory.git
}

if (Test-Daemon) {
  Write-Host ">>   ai-memory daemon already responding at $DaemonUrl"
} else {
  Write-Host ">>   Starting ai-memory daemon (background)..."
  ai-memory serve --daemon
  Start-Sleep -Seconds 1
  if (Test-Daemon) {
    Write-Host ">>   Daemon is up."
  } else {
    Write-Warning "Daemon did not respond yet; check 'ai-memory status' manually."
  }
}

# --- [2/4] workspace init -----------------------------------------------------
Write-Host ">> [2/4] Repository workspace initialization"
Push-Location $RepoRoot
try {
  ai-memory init --project vaila
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

if (Get-Command claude -ErrorAction SilentlyContinue) {
  Write-Host ">>   Registering ai-memory MCP server with Claude Code..."
  try { claude mcp add ai-memory "$DaemonUrl/mcp" } catch { Write-Warning "claude mcp add reported an issue; check manually." }
  try { ai-memory install-hooks --harness claude-code } catch { Write-Warning "install-hooks --harness claude-code reported an issue." }
} else {
  Write-Host ">>   'claude' CLI not found on PATH; skipping Claude Code MCP registration."
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

Write-Host ">>   Running generic hook injection for other harnesses..."
try { ai-memory install-hooks --harness generic } catch { Write-Warning "install-hooks --harness generic reported an issue." }

# --- [4/4] verification --------------------------------------------------------
Write-Host ">> [4/4] Verification and sanity check"
Push-Location $RepoRoot
try {
  try { ai-memory ingest README.md } catch { Write-Warning "ai-memory ingest reported an issue." }
  try { ai-memory search "biomechanics pipeline" } catch { Write-Warning "ai-memory search reported an issue." }
} finally {
  Pop-Location
}

Write-Host ""
Write-Host ">> Done. ai-memory should now be running at $DaemonUrl with the"
Write-Host "   Markdown wiki initialized and MCP endpoints declared for this repo."
Write-Host "   Persisting the daemon across reboots (e.g. a Scheduled Task or"
Write-Host "   Windows service) is not done by this script; see installmemories.md."
