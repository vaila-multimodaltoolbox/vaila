# /quickmeasure

Run (or resume) the getpixelvideo.py Quick Measurement Tool build loop.

## What this command does

1. Reads `/home/preto/data/vaila/loops/getpixelvideo-quickmeasure-loop.md` in full before taking any action.
2. Reads `loops/state/getpixelvideo-quickmeasure-loop-state.json` if present to resume at `current_milestone`; otherwise starts at Milestone 1.
3. Runs the loop's Iteration steps for exactly one milestone attempt, using the governing check defined in that file — not a substitute check.
4. Stops only on one of the loop's named terminal states (`success`, `no-op`, `no-progress/stalled`, `blocked`, `exhausted`) and reports which one, with raw evidence. Reaching a turn/attempt limit or hitting an error is never reported as success.

## Usage
```
/quickmeasure
```

## Arguments
None. Milestone/attempt state is tracked on disk, not passed as arguments.
