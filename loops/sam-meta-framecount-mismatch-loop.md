---
name: sam-meta-framecount-mismatch
category: Vailá
trigger: manual
verification-level: 1
theory-base: arXiv:2607.00038
---

# SAM Meta Frame-Count Mismatch Loop

## Description
Fix the incomplete-SAM-frame-coverage failure class where OpenCV/container
`nb_frames` over-reports decodable frames, so chunked SAM3 merges every real
frame then `validate_sam_run_complete` still fails and blocks Sapiens2.

## Use When
- `sam3sapiens2` / `vaila_sam` reports
  `incomplete SAM frame coverage: expected=N, present=M, missing=N-M`
  after chunked fallback with `failed_chunks=0` and a contiguous missing suffix.
- Source video shows `CAP_PROP_FRAME_COUNT` (or ffprobe `nb_frames`) **>**
  sequential decode count (or `duration × fps`).
- Not for: true partial chunk OOM (`failed_chunks>0`), empty SAM sessions,
  Sapiens2 install/weight failures, or GPU recovery barrier timeouts.

## Inputs
1. `incident_artifact` — optional path to a failed run dir (example:
   `/home/preto/data/fmoura/processed_sam3sapiens2_20260903_214003/JJ_Kabuto/sam3`).
   Used only to reproduce the symptom signature; not required for unit tests.
2. `source_video` — optional MP4 with lying metadata (example:
   `/home/preto/data/fmoura/JJ_Kabuto.mp4`). GPU re-run is a final human
   checkpoint only; never the per-iteration governing check.

## Goal
After the loop succeeds, any video whose container metadata over-reports frame
count is accepted by SAM3 completion gates using the **decodable** frame count
as `expected_frames`. Chunked split + merge + `validate_sam_run_complete` agree
on that count. `sam3sapiens2` therefore proceeds to the Sapiens2 stage when
every decodable frame is covered. Objectively verifiable via pytest (no GPU).

**Frozen incident baseline (2026-09-03, JJ_Kabuto.mp4):**
- metadata / OpenCV `FRAME_COUNT` = 393; sequential decode = 339; duration×fps ≈ 339.3
- chunked README: `total_chunks=8`, `successful_chunks=8`, `failed_chunks=0`, `chunk_size=48`
- merged `sam_frames_meta.csv`: frames 0–338 only (present=339)
- `FAILED_sam.txt`: missing=54 `[339…392]`; SAM3 subprocess exit=3; Sapiens2 never started
- root mechanism: `_split_video_into_chunks` plans from metadata, shortens/drops
  chunks when `cap.read()` fails, then `_process_video_chunked` validates against
  `_video_frame_count()` still returning 393

## Verification (Governing Check)
- **True level:** 1 (deterministic) per iteration; level 5 human checkpoint only
  for an optional GPU re-run of the incident video.
- **Check:**
  ```bash
  uv run ruff check vaila/vaila_sam.py vaila/sam3sapiens2.py --fix
  uv run ruff format vaila/vaila_sam.py vaila/sam3sapiens2.py
  uv run ty check vaila/vaila_sam.py vaila/sam3sapiens2.py
  uv run pytest tests/test_vaila_sam.py tests/test_sam3sapiens2.py -v \
    -k "frame_count or validate_sam_run_complete or split_video_into_chunks or coverage"
  ```
  Broaden to the full two test modules before declaring success.
- **Evidence:** raw pytest stdout/stderr each iteration; record before/after
  counts for any new tests that assert metadata≠decode acceptance.
- **Completion criterion:**
  1. A helper (or `_video_frame_count` behavior change) returns the sequential
     decode count when it differs from `CAP_PROP_FRAME_COUNT`, with a logged
     warning when they disagree.
  2. `_split_video_into_chunks` and `_process_video_chunked` use that same count
     for planning **and** final `validate_sam_run_complete`.
  3. New regression test builds/stubs a video (or monkeypatches read/count) where
     metadata=N, decode=M&lt;N; after split+validate path, complete is True for M
     and False only when a real mid-range hole exists.
  4. Existing `test_validate_sam_run_complete_*` and chunk-split tests still pass.
  5. Module header date/version + required help/README metadata updated per
     repo rules when `.py` files change.
- **Verifier protection:** do not weaken `validate_sam_run_complete` to ignore
  missing frames; fix the **expected** count source. Do not delete or skip the
  coverage gate. New tests must fail (red) before the production fix (green).
- **Scientific validity:** frame index remains 0-based contiguous over
  **decodable** frames; no silent drop of frames that OpenCV can read; units
  stay pixels/frame indices; track IDs unchanged. Document that container
  `nb_frames` is untrusted when it disagrees with decode.

## Trigger
Manual. Start only when this failure class is confirmed (contiguous missing
suffix + `failed_chunks=0` + metadata&gt;decode) or when implementing the fix
from the frozen incident baseline. Do not run concurrent copies against the
same working tree.

## Iteration
0. On first iteration: re-read this file + state; confirm baseline numbers on
   `source_video` if present (`FRAME_COUNT` vs sequential decode); freeze
   target helper API.
1. Load durable state; confirm budget and approvals.
2. Snapshot baseline (`git diff` / failing new test red).
3. Rank unresolved targets worst-first (wrong expected count &gt; split planning
   drift &gt; docs/metadata sync).
4. Invoke `$investigate-first` then `$surgical-patch` for **exactly one**
   attributable change (prefer shared count helper in `vaila_sam.py`, then
   call sites in chunked path / `sam3sapiens2`).
5. Run the governing check; paste raw evidence into state.
6. Retain only if checks pass without regression; else rollback that iteration.
7. Curate lessons; persist state atomically.
8. Evaluate terminal states; else next iteration.

## Terminal States
- **success:** completion criteria 1–5 all evidenced; full
  `tests/test_vaila_sam.py` + `tests/test_sam3sapiens2.py` green.
- **no-op:** codebase already uses decode-authoritative frame counts and the
  regression test already exists and passes.
- **no-progress/stalled:** two consecutive iterations with no improvement on
  the new mismatch test or with repeated identical failures.
- **blocked:** cannot create a fixture/monkeypatch that reproduces
  metadata≠decode without GPU; or incident video unreadable and no synthetic
  path exists.
- **exhausted:** hard budget reached.

Errors, missing evidence, and budget exhaustion are never success.

## Guardrails
- **Maximum allocation:** 8 maker iterations; ~200k tokens or equivalent; stop
  earlier on stall.
- **Human approval required:** committing/pushing; deleting user run artifacts;
  re-encoding or overwriting `source_video`; full GPU batch re-runs billed to
  workstation time.
- **Isolation and credentials:** unit tests offline; no Hugging Face downloads
  in the governing check; no new network deps.
- **Protected verifier:** `validate_sam_run_complete` must keep rejecting true
  holes; tests that assert rejection of mid-video gaps stay frozen.
- **Rollback:** `git checkout -- <touched files>` for the single iteration
  changeset (or revert the iteration commit if one was made with approval).

## State Memory
- **Path:** `loops/state/sam-meta-framecount-mismatch-state.json`
- **Persist:** baseline metrics (393/339 incident), attempts, accepted/rejected
  diffs, pytest evidence paths/snippets, curated lessons, decision log, cost
  counters.
- **Recovery:** if JSON parse fails, look for `*.bak` sibling; if absent,
  restart from frozen incident baseline in this document and mark
  `recovered_from_corrupt_state=true`.

## Skills
- `$investigate-first` — confirm class membership before editing.
- `$surgical-patch` — one narrow change per iteration in `vaila_sam.py` /
  callers.
- `$sam3sapiens2-pose` — pipeline stage order and CLI/GPU context.
- `$verify-and-stop` — stop when governing check proves done; do not expand scope.
- `$caveman-commit` — only if the human asks for a commit.

## Sub-Loops
None. Do not nest `sam3-batch-auto-resume-loop.md` here; resume logic is out of
scope. Parent × child budget N/A.

## Why It Works
- External check (pytest on decode-authoritative count) changes the next edit.
- Atomic single-change iterations attribute regressions to one diff.
- Verifier stays strict on real holes; only the expected-count source is fixed —
  prevents specification gaming by “accept partial”.
- Terminal states separate install red herrings and true OOM from this class.
- Durable state + frozen incident numbers let a fresh context resume without chat.

## How to Trigger
### Context-bound
```text
/goal Continue loops/sam-meta-framecount-mismatch-loop.md from
loops/state/sam-meta-framecount-mismatch-state.json. Do not skip the governing check.
```

### Fresh-context / Ralph
Re-read this file and the state JSON every turn. Run the governing check before
and after each change. Stop only on a named terminal state. Do not execute a
GPU re-run unless the human approves that checkpoint.

## Health Metrics
- **Cost per accepted change:** `total tokens (or currency) / verified
  non-regressive changes retained`.
- **Mismatch gap closed:** `metadata_count - decode_count` must no longer cause
  false failure when present==decode_count.
- **Regression:** count of previously green `validate_sam_run_complete` /
  chunk tests still green each iteration.

## Logged interview defaults (2026-09-03)
- End state: **code fix** (decode-authoritative `expected_frames`), not
  diagnose-only and not “re-encode the one video” as the primary fix.
- Loop name: `sam-meta-framecount-mismatch`.
- Destination: project-local `./loops/` (existing vailá convention).
- Trigger: manual; no launcher artifact requested.
- Optional data workaround (re-encode MP4) remains documented under Use When /
  human approval, not the governing success path.
