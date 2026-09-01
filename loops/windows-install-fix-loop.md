---
name: windows-install-fix
category: Vailá
trigger: manual
verification-level: 2
theory-base: arXiv:2607.00038
---

# Windows Install / GUI-Update / PowerShell One-Liner Fix

## Description
Close four related Windows-onboarding defects in one bounded pass: missing
`.venv` activation docs, an SSH-vs-HTTPS remote collision that breaks the GUI
"Check for Updates" button, any malformed PowerShell one-liner in
`README.md`, and unclear `uv` lifecycle commands in `install_vaila_win.ps1` —
each fixed at the narrowest responsible layer and proven with a deterministic
check, not a visual read-through.

## Use When
- Reopening this specific Windows-onboarding bug cluster (venv docs, SSH/HTTPS
  remote handling, PS one-liner syntax, `uv` command hygiene) across
  `README.md`, `install_vaila_win.ps1`, and `vaila/update_checker.py`.
- **Exclusions:** this loop does not touch `install_vaila_linux.sh`,
  `install_vaila_mac.sh`, `uninstall_vaila_win.ps1`, or any non-Windows
  install path. It does not add a new package manager or change the
  `hatchling`/`uv` build backend. It does not perform a live install run on a
  real Windows box or click the live GUI button itself — those two checklist
  items are explicit human checkpoints (see Verification).

## Goal
Concrete end state, all objectively checkable except the two flagged items:
1. `README.md` documents `.venv` activation for PowerShell, CMD, and Git
   Bash, plus the `uv run vaila` no-activation alternative.
2. Every PowerShell code block in `README.md` (the download-and-run
   one-liner in particular) parses with zero tokenizer errors under both
   Windows PowerShell 5.1 and PowerShell 7 syntax rules.
3. `install_vaila_win.ps1` clones/uses HTTPS remotes only, detects and
   rewrites an existing `git@github.com:...` origin to
   `https://github.com/vaila-multimodaltoolbox/vaila.git`, runs `uv self
   update` then `uv sync --upgrade` (never `uv run sync --upgrade`), and
   prints the activation commands from (1) plus a working `run_vaila.ps1` /
   `run_vaila.bat` wrapper at the end of a successful install.
4. `vaila/update_checker.py`'s git fetch/status path no longer surfaces a
   raw `Permission denied (publickey)` failure to the GUI: it detects an SSH
   origin, offers/performs the HTTPS rewrite (or falls back to
   `git ls-remote https://github.com/vaila-multimodaltoolbox/vaila.git main`
   for a read-only comparison), and any non-zero git exit code reaches the
   GUI as an actionable message, not a raw stack trace.
5. (Human checkpoint, not loop-verifiable) the one-liner and installer were
   actually run on a clean Windows PowerShell 5.1 machine, and the GUI
   **Check for Updates** button was actually clicked against a real SSH
   origin and recovered.

**Baseline note:** do not trust the bug report's example snippets as the
current state of the repo. On this branch (`wininstallfix`, 2026-09-01) the
README one-liner at lines 84–89 already appears well-formed (separate
statements, no missing space in `Join-Path`), and `origin` is genuinely set
to `git@github.com:...` (confirmed live). Iteration 1 must re-derive the
actual defect list from the repo, not from the ticket text, and may find (2)
and part of (3) already satisfied — record that as such, not as invented
work.

## Verification (Governing Check)
- **True level:** Composite. Per-iteration automated check is **level
  1–2** (deterministic parse/lint/test + rule-based structural grep). The
  overall goal's "done" declaration is honestly **level 5** — items 1–4 above
  can be fully automated, but item 5 requires a human to actually run the
  installer on real Windows and click the real GUI button. The loop must
  not claim success on levels 1–2 alone.
- **Check (per iteration, run all, record all):**
  1. Extract every fenced `powershell`-language code block out of
     `README.md` (split on its own fence markers, not embedded here to
     avoid nesting them inside this document's own fences), then for each
     extracted block plus the full text of `install_vaila_win.ps1` run
     `[System.Management.Automation.PSParser]::Tokenize($text, [ref]$errors)`
     via `pwsh -NoProfile -Command` and assert `$errors.Count -eq 0`.
     Windows PowerShell 5.1's grammar is a strict subset of 7's tokenizer
     for the constructs used here (no ternary, no null-coalescing), so a
     5.1-only smoke pass is only required if `pwsh` (7) is unavailable —
     prefer running the same check under `powershell.exe -NoProfile` too
     when both exist on the runner.
  2. `git -C . remote get-url origin` inside a throwaway temp clone (never the
     user's real `origin`) fed a fixture SSH URL, then run the extracted
     `install_vaila_win.ps1` sanitization function in isolation (source the
     script with `$env:VAILA_TEST_REMOTE_SANITIZE_ONLY=1` guarding any
     network/install side effects) — assert the rewritten URL equals
     `https://github.com/vaila-multimodaltoolbox/vaila.git`.
  3. `uv run pytest tests/test_update_checker.py -v` plus new tests added
     this loop for: SSH-origin detection, HTTPS fallback via
     `git ls-remote`, and a non-zero git exit code producing a non-empty,
     non-traceback user-facing message. All must pass; 0 new failures
     elsewhere via `uv run pytest tests/ -v`.
  4. `uv run ruff check vaila/update_checker.py vaila.py --fix-only --diff`
     (must be empty) and `uv run ty check vaila/update_checker.py` (no new
     errors vs. baseline).
  5. Structural grep/parse on `README.md`: a `### Environment Activation` (or
     equivalent) heading exists; it contains fenced examples for
     `powershell`, `bat`/`dos`, and `bash`/`sh` covering
     `.venv\Scripts\Activate.ps1`, `.venv\Scripts\activate.bat`, and
     `source .venv/Scripts/activate` respectively, plus an `uv run vaila`
     no-activation line. Verify the Markdown itself parses cleanly (no
     unterminated fences) with `python -c "import markdown,
     pathlib,sys; markdown.markdown(pathlib.Path('README.md').read_text(encoding='utf-8'))"`
     or equivalent — this loop's own source ticket contained an unterminated
     fence, so treat fence-balance checking as mandatory, not optional.
- **Evidence:** raw stdout/exit codes from all five checks above, appended
  verbatim to the current iteration's entry in the state file — never a
  paraphrase ("looks fixed").
- **Completion criterion:** iteration is *accepted* only when checks 1–5 all
  pass for the file(s) touched that iteration with no regression elsewhere.
  The loop-wide **success** terminal state additionally requires the
  `human_signoff` block in the state file to have both
  `live_windows_ps51_run` and `gui_check_for_updates_live` set `true` with a
  date and the tester's own note (not fabricated by the agent).
- **Verifier protection:** the tokenizer/pytest/ruff/ty commands above are
  the frozen governing check. The maker may add new tests but must not
  delete, skip (`-k 'not ...'` to dodge a real failure), or loosen an
  existing assertion in `tests/test_update_checker.py` to force a pass —
  any such edit requires a one-line justification in the state file's
  `attempts[].change_summary` and is flagged for human review, not
  self-approved. Item 5's human-checkpoint fields can only be set by the
  human, never inferred or defaulted true by the agent.
- **Reproducibility / metadata validity:** per `CLAUDE.md`, any edited
  `.py` file gets its header Update Date + Version bumped to match
  `vaila.py`'s current version; `README.md`'s `Last updated:` line is bumped;
  GUI-visible strings (the updater's error text, the installer's final
  summary) stay consistent with whatever the docs now say. GUI and CLI
  paths for the updater must stay behaviorally equivalent — the same
  SSH-fallback logic must be reachable both from the Tkinter button and from
  a headless `uv run` invocation of the checker functions.

## Trigger
Manual. The developer working this branch (`wininstallfix`) invokes this
loop by name after `/preto-loop` produced it; re-running is idempotent
because iteration 0 always re-derives the live defect list from the repo
(see Baseline note) rather than replaying stale assumptions. Duplicate-run
protection: iteration 0 checks the state file's `status` field — if already
`"success"` or `"blocked"`, the loop reports that instead of restarting.

## Iteration
0. On the first iteration: `git rev-parse HEAD` as `baseline_git_ref`,
   confirm the working tree is on `wininstallfix` (or fail closed and ask),
   confirm the state file doesn't already say `"success"`, and run the full
   governing check once *before any edit* to establish which of items 1–4
   are already true vs. actually broken (per Baseline note).
1. Load this file and `loops/state/windows-install-fix-loop-state.json`
   (create it from the template in State Memory if absent); confirm the
   20-turn budget remaining and that no pending action needs the human
   approvals listed in Guardrails.
2. Snapshot `git diff` (should be empty at the top of each iteration) and
   the check-0 results as the current baseline.
3. Rank the still-failing items from goal list 1–4 worst-first (an
   uncaught SSH permission-denied reaching the GUI as a raw traceback
   outranks a missing doc section outranks a cosmetic `uv` command note),
   and pick exactly one.
4. Invoke `$surgical-patch` to make exactly one attributable change at the
   narrowest layer for that item (one of: `README.md` doc section,
   `install_vaila_win.ps1` sanitization/uv-command block, or
   `vaila/update_checker.py` fallback/error-message logic — plus its test).
5. Run the full governing check (all five sub-checks) and record raw
   evidence in the state file's `attempts[]` for this iteration.
6. Retain the change only if its target check(s) pass with zero regressions
   elsewhere; otherwise `git restore <the one changed file>` (and delete the
   new test file if one was added) — a rollback scoped to this iteration
   only, never touching prior accepted iterations' files.
7. Curate: append a one-line lesson only if evidence supports it (e.g. "PS
   5.1 tokenizer rejects `??`" is a lesson; "seems fine" is not). Atomically
   write the updated state file (write to `.tmp`, then rename).
8. Evaluate terminal states (below); otherwise continue to the next
   ranked item.
9. Once items 1–4 are all accepted, invoke `$verify-and-stop` to run the
   complete governing check one final time end-to-end, then present the
   two human-checkpoint items (live PS 5.1 run, live GUI button click) to
   the user explicitly and stop — do not mark `success` until the human
   records both.

## Terminal States
- **success:** all of goal items 1–4 accepted per the completion criterion
  above, `uv run pytest tests/ -v` fully green, and
  `human_signoff.live_windows_ps51_run` + `human_signoff.gui_check_for_updates_live`
  both `true` with dates/notes in the state file.
- **no-op:** iteration 0's pre-edit check shows all of items 1–4 already
  true on the current branch — record which and why, make no edits, and
  stop pending only the human-checkpoint item 5.
- **no-progress/stalled:** two consecutive iterations targeting the same
  ranked item produce identical governing-check failure output (no
  reduction in failing sub-checks) — stop and surface the item for human
  triage rather than retrying a third time.
- **blocked:** the sanitization/fallback logic needs to touch the real
  `origin` remote or perform a real `git push`/PR/live install to prove
  itself, and human approval (see Guardrails) has not been granted — or
  network access to `github.com`/`raw.githubusercontent.com` is unavailable
  in the current sandbox for the read-only `ls-remote` fallback test.
- **exhausted:** 20 turns consumed (see Guardrails) without reaching
  `success` or a clean `no-op`.

Errors, missing evidence, and budget exhaustion are never success.

## Guardrails
- **Maximum allocation:** 20 turns total for this loop (parent-only; no
  sub-loops are nested — see Sub-Loops).
- **Human approval required (never auto-performed by the loop):**
  - `git push` / opening a pull request for any accepted change.
  - Rewriting `origin`'s URL on the developer's real, non-fixture git
    remote — all sanitization logic is proven against a throwaway fixture
    remote/clone, never the actual `vaila` checkout's `origin`.
  - Running `install_vaila_win.ps1` end-to-end against a real system (the
    loop only statically parses/lints it and unit-tests its extracted
    logic).
- **Isolation and credentials:** work stays on the `wininstallfix` branch,
  local commits only (no push). Network calls limited to read-only
  `git ls-remote` / `raw.githubusercontent.com` fetches needed by check 2/4
  above; no credentials are read or required for any of those.
- **Protected verifier:** see "Verifier protection" above — the five
  governing sub-checks and `tests/test_update_checker.py`'s existing
  assertions are frozen; weakening them requires a logged justification and
  is not self-approved.
- **Rollback:** `git restore <single file>` (plus deleting any new
  not-yet-committed test file from that same iteration) — scoped to the one
  file touched in the rejected iteration.

## State Memory
- **Path:** `loops/state/windows-install-fix-loop-state.json`.
- **Persist:** `baseline_git_ref`, `status` (`in_progress|success|blocked|
  exhausted|stalled`), `budget` (`max_iterations: 20`, turns used), per-item
  status for goal items 1–4, `attempts[]` (iteration, target, change_summary,
  raw `check_evidence` for all five sub-checks, `outcome`:
  `accepted|rejected`), `accepted_changes` (file list), `lessons[]` (only
  evidence-backed), `human_signoff` (`live_windows_ps51_run`,
  `gui_check_for_updates_live`, each `{done: bool, date, note}`), and `cost`
  (`iterations`, `accepted_changes`).
- **Recovery:** a fresh context re-reads this file and the state file. If
  the state file is missing, treat as iteration 0. If the last `attempts[]`
  entry has no `outcome` field, the previous run was interrupted mid-write —
  re-run that entry's governing check from scratch before deciding
  accept/reject; never trust an unterminated attempt's claimed result.

## Skills
- `$surgical-patch` — makes exactly one narrow, regression-proof fix per
  iteration (installer script, updater module, or README section).
- `$verify-and-stop` — runs the final end-to-end governing check and states
  the two human-checkpoint items explicitly before any success claim.

## Why It Works
Ranking worst-first and changing one file per iteration keeps every
governing-check failure attributable to a single edit, so a regression is
always revertible without guessing which of several simultaneous changes
broke it. Freezing the tokenizer/pytest/ruff/ty commands as the governing
check, and requiring raw evidence in the state file, prevents the loop from
declaring victory on a "looks fixed" read-through — which is exactly how
the original bug (a GUI button surfacing a raw SSH stack trace) went
unnoticed. Explicitly separating the level-1/2 automated checks from the
level-5 human checkpoints (live PS 5.1 run, live GUI click) stops the loop
from quietly upgrading its own confidence: static parsing proves the
one-liner is *syntactically* valid, not that it *installs vailá* on a real
machine. Gating remote-URL rewrites and installer execution behind human
approval, and testing sanitization only against a fixture remote, prevents
the loop from ever mutating the developer's real git configuration or
running an uncontrolled installer on their machine.

## How to Trigger
### Context-bound
In Claude Code, on the `wininstallfix` branch: "Follow
`loops/windows-install-fix-loop.md` — run the next iteration." Repeat until
a terminal state is reported. Fits in one context window (20-turn budget).

### Fresh-context / Ralph
If the session is restarted mid-loop, a fresh context must re-read this
file and `loops/state/windows-install-fix-loop-state.json` before acting,
resume from the first `attempts[]` entry lacking a terminal `outcome`, and
stop only on a named terminal state above — never on "seems done."

## Health Metrics
- **Cost per accepted change:** turns spent / entries in `accepted_changes`.
- **Root causes closed:** count of goal items 1–4 with `accepted` status
  (0–4).
- **Regression count:** iterations where a previously-accepted item's check
  flipped from pass to fail (must stay 0; any non-zero value halts the loop
  for human review regardless of budget remaining).
- **Human-checkpoint pending:** boolean pair
  (`live_windows_ps51_run`, `gui_check_for_updates_live`) — `success` is
  impossible while either is `false`.
