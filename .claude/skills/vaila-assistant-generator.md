---
name: vaila-assistant-generator
description: Generate new skills, agents, or rules for Claude, Cursor, and OpenCode within the vailá toolbox. Use this when you want to create a new biomechanical analysis component or automate a specific workflow for any of the project's AI assistants.
---

# Vaila Assistant Generator

This skill helps you create new assistant components for the **vailá** toolbox across multiple platforms: **Claude** (.claude/), **Cursor** (.cursor/), and **OpenCode** (.opencode/).

## Assistant Types & Locations

1.  **Claude Skills** (`.claude/skills/[name].md`): Step-by-step workflows with YAML frontmatter.
2.  **Claude Agents** (`.claude/agents/[name].md`): Role definitions for specialized subagents.
3.  **Cursor Rules** (`.cursor/rules/[name].mdc`): Contextual rules with `globs` and `alwaysApply`.
4.  **OpenCode Skills** (`.opencode/skills/[name].md`): Similar to Claude skills, used by the OpenCode interface.

## Workflow: Generating a New Component

### 1. Identify the Platform & Purpose
- **Cursor Rule**: Best for "Always On" linting, formatting, or project-wide architectural constraints (e.g., "Always use `uv run`").
- **Claude/OpenCode Skill**: Best for complex, multi-step procedures (e.g., "Add a new EMG analysis module").
- **Claude Agent**: Best for delegating specialized domain knowledge (e.g., "Biomechanics Math Expert").

### 2. Drafting the Content

#### For Skills (.md)
Use YAML frontmatter:
```yaml
---
name: [identifier]
description: [trigger description]
---
# [Title]
[Instructions...]
```

#### For Cursor Rules (.mdc)
Use Cursor-specific frontmatter:
```yaml
---
description: [What this rule does]
globs: ["vaila/**/*.py", "tests/*.py"]
alwaysApply: true/false
---
# [Rule Title]
[Enforcement logic...]
```

#### For Claude Agents (.md)
Standard Markdown identifying:
- **Role**: Who is this agent?
- **Expertise**: What do they know?
- **When to Invoke**: Trigger conditions.
- **Behavior Rules**: Specific DOs and DON'Ts.

### 3. Vaila-Specific Standards
Ensure every component mentions:
- **Astral Toolchain**: `uv`, `ruff`, `ty`.
- **Dual-Import Pattern**: `try/except` for relative vs. absolute imports.
- **Biochemical Units**: Suffix variables with units (`_n`, `_deg`, `_ms`).
- **Timestamped Outputs**: `processed_[type]_YYYYMMDD_HHMMSS/`.

## verification
Recommend adding test cases or manual verification steps for each new component created.
