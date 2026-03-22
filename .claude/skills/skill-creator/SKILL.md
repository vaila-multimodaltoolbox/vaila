---
name: skill-creator
description: Create new skills, modify and improve existing skills, and measure skill performance. Use when users want to create a skill from scratch, edit or optimize an existing skill, run evals to test a skill, benchmark skill performance, or optimize a skill's description for better triggering accuracy.
---

# Skill Creator

Create new skills and iteratively improve them through test-driven development.

## Core Loop

1. **Draft** — Write the skill
2. **Test** — Run eval prompts with and without the skill
3. **Review** — Use the HTML viewer to assess outputs and collect feedback
4. **Improve** — Rewrite based on feedback and benchmark data
5. **Repeat** — Until satisfied, then optimize the description

## Bundle Structure

```
skill-creator/
├── SKILL.md          ← this file
├── agents/
│   ├── grader.md     ← how to grade assertions against outputs
│   ├── analyzer.md   ← how to analyze benchmark results
│   └── comparator.md ← blind A/B comparison between two versions
├── references/
│   └── schemas.md    ← all JSON schemas (evals.json, grading.json, etc.)
├── scripts/
│   ├── aggregate_benchmark.py  ← build benchmark.json + benchmark.md
│   ├── run_eval.py             ← run a single trigger eval query
│   ├── run_loop.py             ← automated description optimizer
│   └── package_skill.py        ← bundle skill to .skill file
├── eval-viewer/
│   └── generate_review.py      ← interactive HTML reviewer
└── assets/
    └── eval_review.html        ← trigger eval query review template
```

---

## Creating a Skill

### Capture Intent

If the user already described the workflow in conversation, extract answers from there. Otherwise ask:

1. What should this skill enable Claude to do?
2. When should it trigger? (what phrases/contexts)
3. What's the expected output format?
4. Should test cases be set up?

### Write the SKILL.md

Use the template in `references/schemas.md` for YAML frontmatter. Keep the description **pushy** — include specific contexts where the skill should activate, not just a name.

### Create Test Cases

Create 2-3 realistic prompts that real users would actually say. Save to `evals/evals.json`:

```json
{
  "skill_name": "my-skill",
  "evals": [
    {
      "id": 1,
      "prompt": "User's task prompt",
      "expected_output": "Description of expected result",
      "files": []
    }
  ]
}
```

Share the prompts with the user before running. See `references/schemas.md` for the full schema including the `assertions` field.

---

## Running and Evaluating

### Step 1: Spawn All Runs

Spawn **all** runs in the same turn — with-skill AND baseline together. Don't do them in sequence.

**With-skill:**
```
Execute this task:
- Skill path: <path-to-skill>
- Task: <eval prompt>
- Input files: <eval files if any, or "none">
- Save outputs to: <workspace>/iteration-N/eval-ID/with_skill/outputs/
- Outputs to save: <what the user cares about>
```

**Baseline (creating a new skill):** Same prompt, no skill. Save to `without_skill/outputs/`.

**Baseline (improving an existing skill):** Snapshot the old version first, then point the baseline at it. Save to `old_skill/outputs/`.

Write `eval_metadata.json` alongside each run (assertions can be empty initially). Give each eval a descriptive name.

### Step 2: Draft Assertions While Runs Complete

Draft quantitative assertions for each test case. Update `eval_metadata.json` and `evals/evals.json` with them. Good assertions are objectively verifiable with descriptive names — they should be readable at a glance in the benchmark viewer.

### Step 3: Capture Timing Data

When each subagent task completes, save `timing.json` immediately to the run directory:

```json
{
  "total_tokens": 84852,
  "duration_ms": 23332,
  "total_duration_seconds": 23.3
}
```

This data arrives through the task notification and is not persisted elsewhere.

### Step 4: Grade, Aggregate, Launch Viewer

1. **Grade** each run — read `agents/grader.md` and evaluate assertions against outputs. Write `grading.json` with `{text, passed, evidence}` fields.
2. **Aggregate** — run the benchmark script:
   ```bash
   python -m scripts.aggregate_benchmark <workspace>/iteration-N --skill-name <name>
   ```
3. **Analyze** — read `agents/analyzer.md` and `benchmark.json` to surface patterns. Update `benchmark.json` with analyst notes.
4. **Launch viewer:**
   ```bash
   python eval-viewer/generate_review.py \
     <workspace>/iteration-N \
     --skill-name "my-skill" \
     --benchmark <workspace>/iteration-N/benchmark.json \
     > /dev/null 2>&1 &
   ```
   For iteration 2+, also pass `--previous-workspace <workspace>/iteration-<N-1>`.

### Cowork / Headless

No browser available? Use `--static` to write a standalone HTML file:

```bash
python eval-viewer/generate_review.py \
  <workspace>/iteration-N \
  --skill-name "my-skill" \
  --benchmark <workspace>/iteration-N/benchmark.json \
  --static /tmp/review.html
```

The viewer will download `feedback.json` when the user clicks "Submit All Reviews." Copy it into the workspace directory for the next iteration.

### Step 5: Read Feedback

After the user reviews and submits, read `feedback.json`. Empty feedback means they were satisfied. Focus on cases with specific complaints.

Kill any running viewer server: `kill $VIEWER_PID 2>/dev/null`

---

## Improving the Skill

### Principles

1. **Generalize** — feedback applies to a specific prompt, but the fix should work across many prompts. Avoid overfitting.
2. **Keep it lean** — remove things that aren't pulling their weight. Read transcripts, not just outputs — if the model wastes time on something, cut it.
3. **Explain the why** — use explanations over MUST/NEVER. Smart models understand reasoning and generalize better.
4. **Bundle repeated work** — if every test case independently writes the same helper script, the skill should bundle it instead.

### Iteration Loop

After improving: apply changes, rerun all test cases into `iteration-N+1/`, launch reviewer with `--previous-workspace`, wait for feedback, repeat.

Stop when: the user is happy, all feedback is empty, or you're no longer making progress.

---

## Advanced: Blind Comparison

For a rigorous comparison between two skill versions, read `agents/comparator.md`. Give an independent agent two outputs without revealing which is which, let it judge quality, then analyze why the winner won.

---

## Description Optimization

Optimize the skill's triggering description for better accuracy.

### Step 1: Generate Trigger Eval Queries

Create 20 realistic queries (8-10 should-trigger, 8-10 should-not-trigger). Queries should be substantive — complex, multi-step, with personal context. Avoid trivially easy negatives.

Use `assets/eval_review.html` to create and review them. Open the template, replace the placeholders, let the user edit, then export.

### Step 2: Run the Optimization Loop

```bash
python -m scripts.run_loop \
  --eval-set <path-to-trigger-eval.json> \
  --skill-path <path-to-skill> \
  --model <model-id-powering-this-session> \
  --max-iterations 5 \
  --verbose
```

Uses 60/40 train/test split, runs up to 5 iterations, selects best by test score.

### Step 3: Apply the Result

Update the skill's SKILL.md frontmatter with the returned `best_description`.

---

## Packaging

After the skill is complete, package it:

```bash
python -m scripts.package_skill <path/to/skill-folder>
```

This creates a `.skill` file in the same directory.

---

## Reference Files

| File | Purpose |
|---|---|
| `references/schemas.md` | All JSON schemas for evals, grading, benchmarking |
| `agents/grader.md` | How to evaluate assertions against outputs |
| `agents/analyzer.md` | How to analyze benchmark results and surface patterns |
| `agents/comparator.md` | Blind A/B comparison between two versions |
