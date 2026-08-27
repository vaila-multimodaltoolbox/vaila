---
name: dmg-installer-macos-path-loop
category: Vailá
trigger: manual
verification-level: 2
theory-base: arXiv:2607.00038
---

# DMG Installer macOS Install-Path Loop

## Description
Add an interactive install-path prompt (local `~/Applications`, default vs. system `/Applications`, sudo) to `install_vaila_mac.sh`, and make sure whichever path is chosen ends up registered in Launchpad and Spotlight — verified in an isolated fake-`$HOME` sandbox before any real-system run.

## Use When
Iterating on `install_vaila_mac.sh` (invoked from `create_dmg_installer.sh`'s "Install vaila.app" bundle) to change install-path behavior or app-registration behavior.
Exclusions: does not touch `create_dmg_installer.sh` packaging logic — confirmed out of scope (it only rsyncs project files + wraps the installer in an app bundle; no path/registration logic lives there). Does not cover Windows/Linux installers.

## Inputs
1. `install_vaila_mac.sh` — file under edit, current state: builds `~/vaila`, always creates `~/Applications/vaila.app`, then *unconditionally* `sudo ln -s` into `/Applications/vaila.app` (line ~733–751 as of 2026-08-26) — no user choice, always requires sudo even for a pure-local run.

## Goal
`install_vaila_mac.sh` supports:
- `--install-path=local|applications` (non-interactive, for the loop's sandboxed check) and an interactive prompt with the same two choices when run without the flag, **default = local**.
- `local`: app bundle lives only under `~/Applications/vaila.app` (or `$HOME/Applications` under test), no sudo, no write to `/Applications`.
- `applications`: app bundle also symlinked into `/Applications/vaila.app` via sudo, as today.
- Both paths call `lsregister -f` on the final bundle path and `mdimport -d1` on it, so Launchpad and Spotlight pick it up without a manual `killall Finder`/reboot.
- A `VAILA_TEST_APPLICATIONS_DIR` env override replaces the hardcoded `/Applications` target so the loop's sandbox never touches the real system dir or invokes real `sudo`.

Objectively verifiable: yes, for path/registration-call correctness (level 1–2). Full visual Launchpad/Spotlight-search confirmation on a real Mac remains a human checkpoint (level 5), done once before release, not every iteration.

## Verification (Governing Check)
- **True level:** 2 (rule/constraint — file-path assertions + registration-command exit codes), with a level-5 human checkpoint gating any real (non-sandboxed) run.
- **Check:** (interface as actually implemented in iteration 1 — env vars, not CLI flags; `create_app_bundle()` was moved near the top of `install_vaila_mac.sh` and gated by `VAILA_TEST_APP_BUNDLE_ONLY=1` so the sandbox never runs `uv sync`/Python install/SAM/FIFA prompts)
  ```bash
  bash -n install_vaila_mac.sh   # syntax gate (shellcheck not installed in this env; ruff/ty do not apply — it's bash, not Python)
  LSREGISTER=/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister
  PROJECT_DIR_REAL="$(pwd)"
  ITER_HOME="/tmp/vaila_loop_test_$$"; rm -rf "$ITER_HOME"; mkdir -p "$ITER_HOME"
  # local (default) case
  HOME="$ITER_HOME" VAILA_TEST_APP_BUNDLE_ONLY=1 VAILA_INSTALL_PATH_CHOICE=local \
    VAILA_TEST_APPLICATIONS_DIR="$ITER_HOME/Applications_system" \
    VAILA_HOME="$ITER_HOME/vaila_home" PROJECT_DIR="$PROJECT_DIR_REAL" \
    bash install_vaila_mac.sh
  test -d "$ITER_HOME/Applications/vaila.app"                      # must exist
  test ! -e "$ITER_HOME/Applications_system/vaila.app"             # must NOT exist, no sudo attempted
  # applications case
  HOME="$ITER_HOME" VAILA_TEST_APP_BUNDLE_ONLY=1 VAILA_INSTALL_PATH_CHOICE=applications \
    VAILA_TEST_APPLICATIONS_DIR="$ITER_HOME/Applications_system" \
    VAILA_HOME="$ITER_HOME/vaila_home" PROJECT_DIR="$PROJECT_DIR_REAL" \
    bash install_vaila_mac.sh
  test -d "$ITER_HOME/Applications/vaila.app"
  test -L "$ITER_HOME/Applications_system/vaila.app"                # symlink exists, no sudo (VAILA_TEST_APPLICATIONS_DIR set -> SUDO_CMD="")
  # default case: no VAILA_INSTALL_PATH_CHOICE, stdin closed (non-tty) -> must resolve to local
  rm -rf "$ITER_HOME"; mkdir -p "$ITER_HOME"
  HOME="$ITER_HOME" VAILA_TEST_APP_BUNDLE_ONLY=1 VAILA_TEST_APPLICATIONS_DIR="$ITER_HOME/Applications_system" \
    VAILA_HOME="$ITER_HOME/vaila_home" PROJECT_DIR="$PROJECT_DIR_REAL" \
    bash install_vaila_mac.sh < /dev/null
  test -d "$ITER_HOME/Applications/vaila.app"
  test ! -e "$ITER_HOME/Applications_system/vaila.app"
  # registration commands, run against the sandboxed bundle (never real /Applications)
  "$LSREGISTER" -f "$ITER_HOME/Applications/vaila.app"; echo "lsregister exit: $?"
  mdimport "$ITER_HOME/Applications/vaila.app"; echo "mdimport exit: $?"   # NOT `mdimport -d1` — that requires -t (test-import, doesn't index)
  ```
- **Evidence:** `bash -n` result, full stdout/stderr of all three installer runs (local, applications, default/non-tty), the six `test` exit codes, `lsregister`/`mdimport` exit codes — all appended verbatim to the state file each iteration.
- **Completion criterion:** `bash -n` passes; all six `test` assertions pass; `lsregister` and `mdimport` both exit 0; AND the default run (no `VAILA_INSTALL_PATH_CHOICE`, stdin closed) resolves to `local` (verified by the same two `test` assertions as the local case).
- **Verifier protection:** the check script itself (this file's Verification block) is frozen — the maker (surgical-patch) may edit `install_vaila_mac.sh` only, never the assertions above. Sandbox `$ITER_HOME` is destroyed and recreated each iteration so no state leaks between attempts and inflates apparent success.
- **Scientific validity:** N/A (no biomechanical data path); software-process validity instead — GUI/CLI parity preserved (installer remains a script, no GUI added), no silent overwrite of a real user's `~/vaila` or `/Applications` during loop iterations (sandboxed HOME only), header/date/version metadata on `install_vaila_mac.sh` updated per repo convention on the accepted change.

## Trigger
Manual. You run this loop by hand after editing `install_vaila_mac.sh`, or before cutting a new DMG release. No cadence; duplicate-run protection is simply not starting a second instance while `/tmp/vaila_loop_test_*` from a prior run still exists unresolved (clean it first).

## Iteration
0. On the first iteration, confirm `install_vaila_mac.sh` and `create_dmg_installer.sh` exist at repo root; freeze the flag names above as the interface contract.
1. Load this spec and `loops/dmg-installer-macos-path-loop-state.json`; confirm iteration budget remains.
2. Snapshot `install_vaila_mac.sh` (git diff baseline) and run the governing check as-is to establish the failing/passing baseline.
3. Rank unresolved targets worst-first: (a) missing `--install-path`/`--non-interactive` flags, (b) unconditional sudo write, (c) missing `lsregister -f`/`mdimport -d1` calls on the local-only path.
4. Invoke `surgical-patch` to make exactly one attributable change addressing the top-ranked target in `install_vaila_mac.sh`.
5. Run the governing check; record raw stdout/stderr and all exit codes into the state file.
6. Keep the change only if the check improves (more assertions pass) without regressing a previously-passing assertion; otherwise `git checkout -- install_vaila_mac.sh` to roll back just this iteration's edit.
7. Curate lessons (e.g. "sudo path needs env-var override for testability") only if evidence from this run supports them; persist state atomically (write to `.tmp` then rename).
8. Evaluate terminal states; otherwise continue to the next iteration.

## Terminal States
- **success:** completion criterion above fully met, evidenced by the last recorded check run.
- **no-op:** governing check already fully passes on the unmodified script at iteration 0 — nothing to change.
- **no-progress/stalled:** 2 consecutive iterations with no new assertion turned from fail→pass.
- **blocked:** sandboxed `lsregister`/`mdimport` unavailable or return non-zero for a reason unrelated to the script (e.g. tool absent in CI/sandbox) — needs a human running the check on an actual macOS session.
- **exhausted:** iteration budget (below) reached without success.

Errors, missing evidence, and budget exhaustion are never success.

## Guardrails
- **Maximum allocation:** 6 iterations, ≈150k tokens.
- **Human approval required:** any run against the real `$HOME`/real `/Applications` (i.e. omitting the `HOME=`/`VAILA_TEST_APPLICATIONS_DIR=` overrides), any real `sudo` invocation, and the final level-5 visual check (open Launchpad, Spotlight-search "vaila", confirm icon) on an actual Mac before shipping a new DMG.
- **Isolation and credentials:** sandbox iterations run with `HOME` and the Applications target redirected under `/tmp`; no network calls needed for the check itself; no credentials touched.
- **Protected verifier:** the Verification block's check commands and completion criterion in this file — the maker edits only `install_vaila_mac.sh`.
- **Rollback:** `git checkout -- install_vaila_mac.sh` after each rejected iteration (repo is on branch `dmg`, working tree was clean at loop start).

## State Memory
- **Path:** `loops/dmg-installer-macos-path-loop-state.json`.
- **Persist:** baseline check result, iteration count, per-iteration diff summary, accepted/rejected changes with reason, raw check evidence (stdout/stderr/exit codes), curated lessons, cost/tokens spent, terminal status.
- **Recovery:** a fresh context reads this file, re-runs the governing check against the current `install_vaila_mac.sh` to confirm it matches the last recorded evidence (detects an interrupted write by evidence mismatch), then resumes from the next unresolved ranked target.

## Skills
- `surgical-patch` — makes the one bounded change to `install_vaila_mac.sh` per iteration (flag parsing, path branch, registration calls), narrowest responsible layer, regression-proof discipline.

## Sub-Loops
None.

## Why It Works
Sandboxed `HOME`/Applications-dir overrides make the check deterministic and repeatable without real sudo or touching the user's actual system each iteration (prevents guardrail violations from ordinary dev iteration). One attributable change per iteration plus git-checkout rollback prevents compounding a bad edit. Freezing the check block stops the maker from "fixing" the test instead of the script. The explicit level-5 human checkpoint for real Launchpad/Spotlight visual confirmation prevents mistaking "the commands exited 0" for "the icon actually shows up" — those are not the same claim.

## How to Trigger
### Context-bound
Run inline in a Claude Code session on branch `dmg`: read this file, then execute Iteration steps 1–8 in order, stopping at any terminal state.

### Fresh-context / Ralph
Not required at this scale (6-iteration budget fits one context window); if resumed later, a fresh context must re-read this file and `loops/dmg-installer-macos-path-loop-state.json` before acting, and must not treat a stale/interrupted state file as a completed iteration.

## Health Metrics
- **Cost per accepted change:** total tokens spent / verified non-regressive changes retained.
- **Assertions passing:** count out of the 6 `test` assertions + 2 registration exit codes + `bash -n` clean, per iteration (progress metric).
- **Regressions:** any previously-passing assertion that fails after a change (must be 0 at acceptance).
