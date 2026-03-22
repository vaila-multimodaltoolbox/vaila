# Grader Agent

Role: Evaluate skill outputs against assertions and produce `grading.json`.

## When to Invoke

Spawn as a subagent (or run inline) after a test run completes. Read this file to understand how to grade.

## Input

- **Eval metadata**: `eval_metadata.json` — contains assertions to check
- **Output directory**: The directory where the run saved its outputs
- **Assertions**: List of `{id, text, check, type}` from `eval_metadata.json`

## Output

Write `grading.json` to the run directory:

```json
{
  "expectations": [
    {
      "text": "Assertion description",
      "passed": true,
      "evidence": "Why it passed"
    }
  ]
}
```

**Must use `text`, `passed`, `evidence`** — not `name`, `met`, `details`, etc. The eval-viewer depends on these exact fields.

## Grading Logic by Type

### `file_exists`

PASS if the file at `check` path exists within the output directory (or absolute path if absolute).

Evidence examples:
- PASS: `"File exists at outputs/result.csv"`
- FAIL: `"File outputs/result.csv not found"`

### `string_contains`

PASS if `check` substring is found in any text file in the output directory.

Read output files, search for the substring. Case-insensitive match.

Evidence examples:
- PASS: `"Found 'profit_margin' in outputs/result.csv: 'profit_margin,0.23'"`
- FAIL: `"Substring 'profit_margin' not found in any output file"`

### `output_format`

PASS if the output conforms to the format described in `check`.

Read `check` (which describes the expected format) and verify output. Common patterns:
- `check` = `"csv_with_columns: time_s, x, y"` — verify CSV has those columns
- `check` = `"json_with_keys: results, summary"` — verify JSON has those top-level keys
- `check` = `"html_with_plot"` — verify HTML contains `<img>` or `<svg>` tags

Evidence examples:
- PASS: `"CSV contains required columns: time_s, x, y"`
- FAIL: `"CSV missing column 'z'. Found: time_s, x, y"`

### `programmatic`

Execute the script at `check` with the output directory as `$1`.

```bash
python3 /path/to/check_script.py outputs/
```

Exit code 0 = PASS, non-zero = FAIL. Capture stdout/stderr as evidence.

Evidence examples:
- PASS: `"Script exited 0. Output: all 10 rows valid"`
- FAIL: `"Script exited 1. stderr: AssertionError: expected 10 rows, got 9"`

## How to Grade

1. Read `eval_metadata.json` to get the assertions list
2. Explore the output directory — list all files, read text files
3. For each assertion:
   a. Determine the type
   b. Apply the grading logic above
   c. Record `text`, `passed`, and `evidence`
4. Write `grading.json`

Be thorough — read the actual file contents. Don't just check file existence; verify the content is reasonable.

## Edge Cases

- **No output files at all**: All assertions FAIL with evidence `"No output files found in directory"`.
- **Assertion type unknown**: FAIL with evidence `"Unknown assertion type: X"`.
- **Multiple matching files**: Search all files, report what you found.
- **Binary files**: Skip binary files for `string_contains` checks.

## Output Format

Write directly to the run directory as `grading.json`. Do not print to stdout — just save the file.
