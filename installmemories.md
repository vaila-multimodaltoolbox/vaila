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
