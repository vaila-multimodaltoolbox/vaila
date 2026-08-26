---
name: interp-smooth-split-parity-loop
category: Data Science
trigger: manual
verification-level: 1
theory-base: arXiv:2607.00038
---

# Interp / Smooth / Split Parity Loop

## Description
Close deterministic coverage gaps in `vaila/interp_smooth_split.py` and
`vaila/interp_smooth_core.py` so every documented gap-fill and smooth method,
plus padding / split / resample / index-column contracts, has a failing-then-passing
pytest assertion — without regressing the current 35-test green baseline.

## Use When
- Expanding or repairing numerical/CLI contracts for interpolation, smoothing,
  padding, splitting, or resampling of biomechanical CSV time series.
- Enforcing GUI↔CLI numerical parity through the shared `process_file` /
  `interp_smooth_core` path (not a Tk screenshot review).
- Not for: markerless video (SAM3/Sapiens2/YOLO), DLT/reconstruction geometry,
  or unrelated Frame C tools.

## Inputs
1. `module` — `vaila/interp_smooth_split.py` (GUI + CLI + `process_file`).
2. `core` — `vaila/interp_smooth_core.py` (shared numerical path).
3. `tests` — `tests/test_interp_smooth_split.py` (baseline: 35 passed, 2026-08-26).
4. `help` — `vaila/help/interp_smooth_split.md` + `.html` (documented method list
   and sampling-rate semantics are the contract surface).
5. `config_fixture` — synthetic CSVs + `smooth_config.toml` written under
   pytest `tmp_path` each run (do not require a fixed external dataset).

## Goal
Objectively verifiable end state: every method and contract below has at least
one deterministic assertion in `tests/test_interp_smooth_split.py`, the suite
stays fully green, ruff/ty are clean on touched files, and mandatory vailá
Python/help metadata is updated when code changes.

**Coverage targets (worst-first, from baseline audit 2026-08-26):**

1. **Gap fill (`apply_interpolation_1d`):** `linear`, `cubic`, `nearest`,
   `kalman` (core linear fallback), `none`, `skip`, plus `max_gap` leaving
   oversized gaps as NaN.
2. **Smooth helpers path:** `kalman`, `splines`, `arima`, `median`, `hampel`
   via `apply_smoothing_1d(..., helpers=...)` using the real helpers from
   `interp_smooth_split` (or thin test doubles only when the real helper is
   unavailable and the failure mode is asserted).
3. **Split:** `do_split=True` writes the expected pair of output CSVs with
   correct row partition; `do_split=False` remains single-file.
4. **Index / padding / rate contracts (already partially covered — extend,
   do not weaken):** Time/`frame` never smoothed; padding preserves row count
   when resample is off; Butterworth rejects `cutoff >= fs/2`; time-column
   rate override does not silently replace Butterworth `fs`.
5. **CLI / TOML:** flag override of TOML; headless `-i`/`-c`/`-o` run on a
   tmp fixture exits 0 and writes timestamped outputs; `--help` documents
   resample and config flags.
6. **Parity:** same config dict → bit-identical numeric columns through
   `process_file` (shared path). True Tk GUI event-loop testing is out of
   scope; GUI must keep calling the same core.

## Verification (Governing Check)
- **True level:** 1 (pytest assertions + CLI exit codes) with level-2 static
  gates (ruff, ty). Visual plot review is **not** acceptance.
- **Check (every iteration):**
  ```bash
  uv run ruff check vaila/interp_smooth_core.py vaila/interp_smooth_split.py tests/test_interp_smooth_split.py --fix
  uv run ruff format vaila/interp_smooth_core.py vaila/interp_smooth_split.py tests/test_interp_smooth_split.py
  uv run ty check vaila/interp_smooth_core.py vaila/interp_smooth_split.py tests/test_interp_smooth_split.py
  uv run pytest tests/test_interp_smooth_split.py -v
  ```
- **Evidence:** full raw stdout/stderr of the four commands above, appended
  verbatim into `loops/state/interp-smooth-split-parity-loop-state.json`
  under the current attempt (do not summarize away failures).
- **Completion criterion:**
  - All six coverage targets above have dedicated passing tests.
  - `uv run pytest tests/test_interp_smooth_split.py -v` is 100% pass
    (count ≥ baseline 35; new tests only add).
  - Ruff check/format and ty check clean on the three files.
  - Any edited `*.py` has header **Update Date** / **Version** synced to
    global vailá version; `vaila/help/interp_smooth_split.{md,html}` and
    index metadata updated when behavior or CLI surface changes.
  - Stale metadata with a green suite is **incomplete**, not `success`.
- **Verifier protection:**
  - Do not delete or weaken existing tests to reach green.
  - Do not change the governing check commands or silence assertions.
  - Prefer red-before/green-after for each new test (add failing test, then
    fix code only if the failure is a real product bug).
  - New golden numeric tolerances must be documented in the test (absolute
    or relative) and justified (filter known-gain, exact NaN mask, etc.).
- **Scientific validity:**
  - Units: CSV column units are caller-defined; loop must not invent
    physical units. Hz/FPS = samples per second for this tool.
  - Butterworth: `0 < cutoff < fs/2`; invalid cutoffs raise, never clamp.
  - Index columns (`Time`/`t`/`tempo`, `frame`/`frames`/`frame_index`) are
    axes, never signals.
  - Gap fill must not alter non-NaN samples; smooth may alter all samples
    per method contract.
  - Upsampling creates interpolated estimates, not new measurements —
    reports/tests must not claim otherwise.
  - Record `fs`, padding percent, interp/smooth method, and resample
    rates in each accepted attempt's evidence blob.

## Trigger
Manual. Operator or harness starts an iteration. Duplicate runs must load
`loops/state/interp-smooth-split-parity-loop-state.json` and skip already
accepted targets.

## Iteration
0. Validate inputs; freeze baseline: note pytest count, ruff/ty status, and
   the six coverage targets still open.
1. Load this specification and the state file; confirm budget and approvals.
2. Snapshot baseline (`git status` + governing check); record raw evidence.
3. Rank open targets worst-first (prefer missing method tests over polish).
4. Invoke `$surgical-patch` (or `$investigate-first` if a failure is opaque)
   to make **exactly one** attributable change (one new test, or one bugfix
   forced by a new red test).
5. Re-run the governing check; store raw evidence.
6. Retain only if check passes without regression; otherwise rollback that
   iteration's change.
7. Curate lessons (only when supported by evidence); atomically update state.
8. Evaluate terminal states; else next iteration.

## Terminal States
- **success:** All six coverage targets accepted with evidence; suite green;
  ruff/ty clean; metadata current.
- **no-op:** Audit shows all six targets already covered and suite ≥35 green
  with clean static gates (baseline alone is **not** no-op — gaps listed in
  Goal were open as of 2026-08-26).
- **no-progress/stalled:** Two consecutive iterations with no new accepted
  target and no metric improvement (new failing tests that stay red count as
  progress only if the next iteration is a fix attempt).
- **blocked:** Missing optional scientific dependency for a method under
  test that cannot be stubbed without lying about behavior; or environment
  cannot run `uv`/`pytest`.
- **exhausted:** Hard budget reached before success.

Errors, missing evidence, and budget exhaustion are never success.

## Guardrails
- **Maximum allocation:** 12 iterations; 40 agent tool-turns; no currency
  budget (local `uv` only).
- **Human approval required:** git commit/push; deleting or rewriting
  existing fixture files outside `tmp_path`; adding paid/network model deps;
  changing Butterworth/scientific defaults in a way that alters historical
  outputs without a version note in help.
- **Isolation and credentials:** headless CLI/tests only; no second `tk.Tk()`;
  no network required for the governing check.
- **Protected verifier:** existing tests in
  `tests/test_interp_smooth_split.py` may gain cases but must not be gutted;
  governing check command block is frozen in this document.
- **Rollback:** revert only that iteration's files via
  `git checkout -- <touched paths>` or equivalent stash pop of the single
  change set.

## State Memory
- **Path:** `loops/state/interp-smooth-split-parity-loop-state.json`
- **Persist:** baseline counts, open/accepted targets, attempt log with raw
  check excerpts, curated lessons, decisions, cost counters
  (`iterations`, `accepted_changes`, `tokens_or_turns` if known).
- **Recovery:** fresh context re-reads this markdown + state JSON; if JSON
  parse fails, treat as interrupted write — restore from
  `*.bak` if present, else re-run baseline check and rebuild open-target
  list from Goal section (do not trust chat history).

## Skills
- `$investigate-first` — diagnose opaque pytest/CLI failures before editing.
- `$surgical-patch` — one narrow change per iteration (test or fix).
- `$verify-and-stop` — confirm completion criteria; refuse scope expansion.
- `$preto-loop` — maintain this specification; do not execute unless the
  operator explicitly starts the loop.

## Sub-Loops
None. Circular nesting prohibited.

## Why It Works
- External pytest/ruff/ty feedback blocks “looks smoother” self-approval.
- One change per iteration keeps attribution and rollback cheap.
- Worst-first missing-method tests prevent polishing already-covered
  Butterworth/resample paths while kalman/hampel/split stay untested.
- Scientific contracts (Nyquist, index columns, non-mutation of non-NaN on
  gap fill) are asserted, not inferred from plots.
- Metadata gate matches vailá repo rules so a green suite with stale help
  cannot fake completion.

## How to Trigger
### Context-bound
In Cursor: open this file and instruct the agent to execute
`interp-smooth-split-parity-loop` until a named terminal state, re-reading
`loops/state/interp-smooth-split-parity-loop-state.json` each turn.

### Fresh-context / Ralph
```text
1. Read loops/interp-smooth-split-parity-loop.md
2. Read loops/state/interp-smooth-split-parity-loop-state.json
3. Run one Iteration step (exactly one change)
4. Update state atomically
5. Stop only on success | no-op | no-progress/stalled | blocked | exhausted
```
Illustrative only — do not bypass human approval gates.

## Health Metrics
- **Cost per accepted change:** `total_iterations_or_turns / accepted_targets`.
- **Coverage progress:** `accepted_targets / 6`.
- **Suite size:** test count must be ≥ baseline 35 after success.
- **Regression:** any previously passing test failing → retain = false.
