# Analyzer Agent

Role: Analyze benchmark results and surface patterns that aggregate stats hide.

## When to Invoke

After running `scripts/aggregate_benchmark.py` to produce `benchmark.json`. Read `benchmark.json` and the transcripts, then produce analyst notes.

## Input

- **benchmark.json** — aggregated stats from `aggregate_benchmark.py`
- **eval_metadata.json** files — assertions for each eval
- **grading.json** files — per-eval grading results
- **transcripts** (optional) — full subagent outputs if available

## Output

Write a list of analyst notes to `benchmark.json` under `analyst_notes[]`. Also surface findings verbally.

## What to Look For

### 1. Non-Discriminating Assertions

An assertion that passes on **both** `with_skill` and `without_skill` configurations. These don't differentiate skill quality — they measure things the model does anyway.

```
Assertion 'output_is_valid_json' passes on with_skill and without_skill — non-discriminating.
Assertion 'file_created' passes on both configurations — measures basic capability, not skill value.
```

Action: Flag these. Consider removing them from future evals if they don't add signal.

### 2. High-Variance Evals

An eval whose time or token stddev is large relative to its mean (e.g., stddev > 30% of mean). These may be flaky.

```
Eval 'complex-xlsx-analysis' has time stddev 15.2s vs mean 52.3s (29%) — borderline high variance.
Eval 'multi-file-processing' has token variance of 45% — possibly non-deterministic or unstable.
```

Action: Note for potential rerunning with multiple trials.

### 3. Regression Cases

An eval where `without_skill` outperforms `with_skill` (negative delta where it should be positive).

```
Eval 'basic-csv-read' shows with_skill took 30s vs without_skill 20s — regression. Check if skill adds unnecessary overhead for simple tasks.
```

### 4. Time/Token Tradeoffs

Skilled runs might use more or fewer tokens/time. Surface the tradeoffs.

```
with_skill uses 15% more tokens (82K vs 71K) but completes 50% more evals successfully.
Skilled runs are slower but significantly more reliable — good tradeoff for production use.
```

### 5. Pass Rate by Eval Complexity

Group evals by complexity (from eval names or assertion counts) and see if skill helps more on harder tasks.

```
Simple evals (1-2 assertions): with_skill 100%, without_skill 90% — skill adds little
Complex evals (5+ assertions): with_skill 75%, without_skill 30% — skill adds major value
```

### 6. Feedback Correlation

If `feedback.json` from previous iterations is available, correlate specific complaints with assertion failures.

```
Eval 'eval-2-with_skill' had feedback "chart missing labels" — assertion 'output_has_axis_labels' also failed.
```

## Output Format

Append analyst notes to `benchmark.json`:

```json
{
  "analyst_notes": [
    "Assertion 'output_contains_timestamp' passes on both with_skill and without_skill — non-discriminating.",
    "Eval 'complex-xlsx-analysis' has high time variance (stddev 15.2s) — consider running multiple trials."
  ]
}
```

Write these back to `benchmark.json` in the workspace directory.
