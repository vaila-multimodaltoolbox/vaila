# Preto Loop: multi-agent distribution

`preto-loop` is the repository-portable loop-authoring skill for vailá biomechanics and data-science work. Its source package is tracked at `skills/preto-loop/`.

## What is versioned

- `skills/preto-loop/` — canonical portable package.
- `.claude/skills/preto-loop/` — Claude Code discovery path.
- `.agents/skills/preto-loop/` — project-local skills path for Codex-compatible runners.
- `.cursor/rules/preto-loop.mdc` — Cursor rule adapter.
- `.agent/rules/preto-loop.md` — Antigravity-compatible rule adapter.

The copies contain the same `SKILL.md` and references so each agent can discover the skill after cloning the repository. Keep them synchronized when changing the skill; compare them with:

```bash
diff -u skills/preto-loop/SKILL.md .claude/skills/preto-loop/SKILL.md
diff -u skills/preto-loop/SKILL.md .agents/skills/preto-loop/SKILL.md
```

## Activation

- Codex: use `$preto-loop` when the skill is installed globally or loaded from `.agents/skills/preto-loop/`.
- Claude Code: ask for `preto-loop` or use the repository skill under `.claude/skills/preto-loop/`.
- Cursor: use a request such as “use the Preto Loop workflow to design a vailá biomechanics loop”; the `.cursor/rules/preto-loop.mdc` adapter points to the package.
- Antigravity: load `.agent/rules/preto-loop.md` and `skills/preto-loop/SKILL.md` according to the host's project-rule convention.

The exact discovery mechanism varies by host. The Git commit makes the artifacts available; it does not install them into a user's global configuration automatically.

## GitHub installation

After cloning, use the host's project skill/rule discovery or copy the package to its global skills directory. For Codex, the global copy can be installed as:

```bash
cp -R skills/preto-loop ~/.codex/skills/preto-loop
```

For a different host, use its documented project rules/skills directory. Do not assume that a Cursor rule or Claude skill is automatically active in every other product.

## Commit boundary

The files are prepared for Git tracking, but creating a commit or pushing to GitHub remains a separate explicit action. Before committing, verify all synchronized copies and review `git diff --check`.
