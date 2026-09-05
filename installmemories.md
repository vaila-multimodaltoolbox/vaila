```markdown
Configure and integrate `ai-memory` (https://github.com/akitaonrails/ai-memory) into this repository (`vaila-multimodaltoolbox/vaila`) to establish shared, long-term memory and cross-agent handoffs across multiple coding harnesses (Claude Code, OpenAI Codex, Cursor Agent, OpenCode, Gemini CLI, and custom agents).

Follow these exact execution steps:

### 1. Binary Installation and Service Verification
1. Check if the `ai-memory` binary exists in `$PATH` (`which ai-memory`).
2. If missing, install it via Cargo:
   ```bash
   cargo install --git [https://github.com/akitaonrails/ai-memory.git](https://github.com/akitaonrails/ai-memory.git)

```

*(Ensure `~/.cargo/bin` is exported in the user profile).*
3. Verify that the `ai-memory` daemon is running on port `49374` (`curl -s http://127.0.0.1:49374/healthz` or `ai-memory status`). If not running, start it as a background service:

```bash
ai-memory serve --daemon

```

### 2. Repository Workspace Initialization

1. From the root of `vaila`, run the repository initialization command:
```bash
ai-memory init --project vaila

```


2. Verify that `.ai-memory.toml` is created at the repository root. Ensure it includes the repository metadata, local SQLite/FTS5 indexing path, and Markdown storage path (default: `.ai-memory/wiki`).
3. Add the ephemeral index database to `.gitignore` while keeping `.ai-memory.toml` and documentation tracking intact:
```gitignore
# ai-memory local index
.ai-memory/*.db
.ai-memory/*.db-wal
.ai-memory/*.db-shm

```



### 3. Multi-Agent Harness Configuration

Configure the universal protocol integrations so any active agent can consume and update the memory layer:

#### A. Claude Code CLI

1. Configure lifecycle hooks and the MCP server for Claude Code:
```bash
ai-memory install-hooks --harness claude-code

```


2. Register the MCP server in Claude Code:
```bash
claude mcp add ai-memory [http://127.0.0.1:49374/mcp](http://127.0.0.1:49374/mcp)

```



#### B. Cursor Agent

1. Create or update `.cursor/mcp.json` at the project root:
```json
{
  "mcpServers": {
    "ai-memory": {
      "url": "[http://127.0.0.1:49374/mcp](http://127.0.0.1:49374/mcp)"
    }
  }
}

```


2. Append integration instructions into `.cursorrules` (or `.cursor/rules/ai-memory.mdc`):
```markdown
Always check `ai-memory` via MCP tools (`search_memory`, `get_handoff`) at the start of a session. Before exiting or concluding a major task, summarize unresolved edge cases, architectural decisions, and hardware/environment dependencies using `create_handoff`.

```



#### C. OpenAI Codex, OpenCode, and Generic CLI Agents (agy / Gemini CLI)

1. Add the root `mcp.json` (Model Context Protocol standard) so compatible tools automatically discover the endpoint:
```json
{
  "mcpServers": {
    "ai-memory": {
      "type": "sse",
      "url": "[http://127.0.0.1:49374/mcp](http://127.0.0.1:49374/mcp)"
    }
  }
}

```


2. Run standard hook injection for other supported harnesses:
```bash
ai-memory install-hooks --harness generic

```



### 4. Verification and Sanity Check

1. Ingest initial repository context into the memory wiki:
```bash
ai-memory ingest README.md ARCHITECTURE.md

```


2. Perform a test memory query:
```bash
ai-memory search "biomechanics pipeline"

```


3. Verify that:
* Daemon responds at `http://127.0.0.1:49374`.
* Markdown wiki files are initialized.
* MCP endpoints are properly declared across Claude, Cursor, and root configs.



```

```

## Cross-Platform Setup (Linux / macOS / Windows)

The step-by-step plan above is shell-agnostic in intent but was written with
Windows/bash mixed together. Two idempotent, repeatable scripts implement it
per platform, safe to re-run (every step checks current state first):

| Script                       | Platform                                | Run with            |
| ----------------------------- | ---------------------------------------- | -------------------- |
| `bin/setup_ai_memory.sh`      | Linux, macOS, WSL, Git Bash on Windows   | `bash bin/setup_ai_memory.sh` |
| `bin/setup_ai_memory.ps1`     | Windows PowerShell                       | `pwsh bin/setup_ai_memory.ps1` (or `powershell -File ...`) |

Both scripts perform all four sections above (binary install, workspace init,
multi-agent harness configuration, verification) and are safe to run more
than once — each step checks whether it's already done (binary present,
daemon already responding, config file already written, `.gitignore` entry
already present) before acting.

### OS-specific notes

- **`~/.cargo/bin` on PATH**: `rustup`'s own installer normally handles this
  via `~/.cargo/env`. `setup_ai_memory.sh` also appends an explicit
  `export PATH="$HOME/.cargo/bin:$PATH"` to `~/.bashrc` (Linux) or
  `~/.bash_profile` (macOS, matching Terminal.app's default login-shell
  behavior) / `~/.zshrc` if it isn't already there. On Windows, the Rust
  installer (`rustup-init.exe`) adds `%USERPROFILE%\.cargo\bin` to the user
  `PATH` automatically; `setup_ai_memory.ps1` does not modify `PATH` itself
  and expects `cargo` to already resolve (open a new terminal after
  installing Rust if it doesn't).
- **Persisting the daemon across reboots** — none of the setup scripts do
  this; `ai-memory serve --daemon` only survives the current login session.
  If you want it to survive a reboot/logout:
  - **Linux (systemd --user)**: create
    `~/.config/systemd/user/ai-memory.service` with an `ExecStart=%h/.cargo/bin/ai-memory serve`
    (no `--daemon`, let systemd supervise it) unit, then
    `systemctl --user enable --now ai-memory`.
  - **macOS (launchd)**: create a `~/Library/LaunchAgents/com.vaila.ai-memory.plist`
    `LaunchAgent` pointing `ProgramArguments` at
    `~/.cargo/bin/ai-memory serve`, then
    `launchctl load ~/Library/LaunchAgents/com.vaila.ai-memory.plist`.
  - **Windows**: register a Scheduled Task ("At log on") running
    `ai-memory.exe serve`, or use `ai-memory serve --daemon` manually each
    session — there is no first-class Windows service wrapper here.
- **Repo-tracked config vs. local index**: `.ai-memory.toml`, `.cursor/mcp.json`,
  `.cursor/rules/ai-memory.mdc`, and root `mcp.json` are meant to be committed
  (they only carry `127.0.0.1` URLs, no secrets) so every clone — on any OS —
  gets the same MCP wiring automatically; only the SQLite index
  (`.ai-memory/*.db*`) and the Markdown wiki content are local/generated and
  covered by the `.gitignore` entries above.

