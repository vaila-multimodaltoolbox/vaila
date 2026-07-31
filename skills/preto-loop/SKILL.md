---
name: preto-loop
description: Write and harden reusable agent-loop specification documents for biomechanics, data science, and vailá coding work. Use when the user says `/preto-loop`, asks to write/create/forge/specify/document an agent loop, turn an iterative vailá task into an autonomous loop, make an overnight loop, or design a biomechanics, data-pipeline, markerless-tracking, testing, refactoring, or performance loop. Include external verification, scientific validity, GUI/CLI parity, reproducibility, memory, guardrails, and named terminal states. Do not use merely to repeat a fixed scheduled prompt, execute an existing loop, or create unrelated documents.
---

# Preto Loop

Write the specification of an agent loop for the user's biomechanics, data-science, and vailá development work. Do not execute the resulting loop. Produce a bounded, readable artifact that a person can version, inspect, and hand to an agent harness.

## Domain scope: vailá and biomechanics

Classify the work before choosing the loop's check:

- **Biomechanical analysis:** kinematics, kinetics, EMG, force plates, IMU, GNSS, MoCap, DLT/reconstruction, filtering, gait/jump/TUG/turn metrics, uncertainty, or clinical/sports reports.
- **Data science:** ingestion, schema/header detection, cleaning, interpolation, smoothing, feature engineering, statistics, visualization, model training/evaluation, leakage control, or dataset curation.
- **Markerless/video/AI:** YOLO, SAM3, Sapiens2, tracking, re-identification, keypoints, contours, masks, camera calibration, and downstream CSV/2D/3D reconstruction.
- **vailá software:** a Python module, GUI button, CLI, test, docs/help page, installer, dependency, or performance/reliability fix.

If a request crosses categories, state the interfaces and invariants between them. Keep scientific acceptance separate from code acceptance: a green test suite does not prove a biomechanical conclusion, and a plausible plot does not prove a correct implementation.

## Preserve the distinction

- Treat a loop specification as an external pilot for an existing agent harness.
- Do not confuse it with a programming `for`/`while` loop or the harness's internal perceive-act-observe cycle.
- Keep prompt, context, harness, and loop engineering complementary; a loop still contains prompts.
- Center the design on the governing check, not on persuasive wording.

## Phase 0: Triage

Ask whether the outcome and verification evidence from each iteration can change the next action.

- If no, explain that this is a scheduled one-shot rather than a loop. Return a concise scheduled prompt containing the task, cadence, reporting destination, and failure-notification rule, then stop.
- If the goal is pure taste with no reproducible acceptance rule, ambiguous greenfield work with no known direction, or the iteration cost cannot be justified, recommend a human-led or one-shot workflow and stop.
- If yes, continue.

## Phase 1: Interview one question at a time

Do not dump a questionnaire. Ask only the next unresolved question and reuse facts already supplied by the user.

1. Define the concrete end state and a short hyphen-case loop name.
2. Identify runtime inputs. Omit the Inputs section if none are required.
3. Classify success using the verification ladder below and identify the strongest honest check available.
4. Freeze the governing command, measurement, rule, rubric, or field signal and its exact completion criterion.
5. Choose a manual, scheduled, or event-driven trigger and its cadence/event.
6. Name terminal states beyond success: clean `no-op`, `no-progress` or `stalled`, `blocked`, and `exhausted` as applicable.
7. Name the existing skills the loop composes. Avoid wrapping raw unguided generation in `while true`.
8. Identify any nested loop by exact filename. Forbid circular references.
9. Choose file-backed state and define what is curated there: baseline, attempts, accepted changes, rejected lessons, evidence, costs, and decisions.
10. Set hard turn/token/cost limits and the exact actions requiring human approval.
11. Ask where to save the specification: project-local `./loops/` or a user-specified global loops directory.
12. Ask whether to create an optional launcher/command artifact, but only for a harness and command location the user identifies. Never invent support for a slash-command format.

For vailá work, also identify the input modality, coordinate system, units, sampling rate/frame rate, expected output schema, reference dataset or fixture, and whether the target is CPU, CUDA, or another hardware matrix.

When safe defaults would not materially alter behavior, propose them and log them rather than prolonging the interview. Never invent a verifier, budget, irreversible-action policy, or destination whose choice changes the loop materially.

## Verification ladder

State the verifier's true level; never present assisted judgment as deterministic proof.

1. **Deterministic:** assertion, exit code, golden output, exact metric.
2. **Rule/constraint:** linter, schema, policy, static analysis.
3. **Delayed field truth:** deployment result, experiment, customer or production signal.
4. **Model judge:** score against a frozen rubric; opinion, not field truth.
5. **Human checkpoint:** supervision, not automated verification.

Prefer levels 1–2 for unattended operation. Levels 1–3 are objective evidence. For level 4, use a distinct evaluator instance/model with isolated context, a frozen rubric, explicit evidence requirements, and preferably independent voting. Place consequential human approval at the irreversible action, not at a routine preparatory step.

## Phase 2: Harden the design

Before drafting, enforce every applicable rule:

- Require external feedback. Never stop because the agent says the result “looks good.”
- Require each iteration to run the check and put raw stdout/stderr or the complete evaluation evidence into the transcript and state file.
- Separate maker and checker when judgment is involved; do not share the maker's reasoning history with the checker.
- Snapshot a baseline, rank targets worst-first, and make exactly one attributable change per iteration.
- Keep a change only when the governing check passes without regression; otherwise revert only that iteration's change using a previously declared recoverable method.
- Prove new or repaired verifiers with red-before/green-after evidence when feasible. Do not let the agent weaken, skip, hard-code around, or silently edit its governing check.
- Use a holdout the maker cannot edit when specification gaming is plausible.
- Define terminal states explicitly. Errors, timeouts, exhausted budgets, blocked checks, and missing evidence are never success.
- Detect stagnation from measurable evidence, normally after two consecutive iterations without improvement.
- Multiply parent and child maximum iterations to expose the real worst-case budget; reject circular nesting.
- Curate memory: retain lessons only when later evidence supports them; do not append speculation indefinitely.
- Gate destructive filesystem changes, production deployments, paid/external actions, credential use, and security-sensitive network access behind explicit human approval.
- For unattended execution, require isolation, scoped credentials, restricted network access where feasible, and a hard budget ceiling.
- Track `cost per accepted change = total tokens or currency spent / verified non-regressive changes retained`.

Read [references/loop-engineering.md](references/loop-engineering.md) when explaining the theory, reviewing an existing loop, or resolving a verification/architecture tradeoff.

Read [references/vaila-biomechanics.md](references/vaila-biomechanics.md) for domain checks, vailá repository conventions, and scientific failure modes.

## Vailá and scientific hardening

Apply the relevant gates before accepting an iteration:

- Preserve raw inputs and a reproducible, timestamped output; never overwrite source data silently.
- Record units, coordinate frames, axis directions, sign conventions, sampling/frame rate, synchronization assumptions, filtering parameters, and missing-data policy.
- Validate transformations with dimensional analysis and synthetic/known-value fixtures. Check NaN/Inf handling, boundary conditions, frame indexing, and subject/track identity continuity.
- Prevent subject/video/session leakage between train, validation, and test data. Report sample counts and exclusions, not only a score or plot.
- Treat visual plausibility as level-4/5 evidence unless paired with deterministic assertions or an external reference measurement.
- For measurements, report uncertainty or sensitivity where the task calls for it; do not convert a statistical association into a clinical or causal claim.
- For markerless pipelines, verify keypoint ordering, bbox/contour/mask coordinate conversion, ID persistence, confidence thresholds, occlusion behavior, and downstream CSV schema.
- For every changed Python script in vailá, update its header date/version and the required README/help/index metadata according to the repository instructions.
- Keep vailá GUI and CLI paths behaviorally equivalent where both exist. GUI actions must print a copy-pasteable `>>` CLI command; headless runs must not accidentally open a hidden Tk root.
- Prefer the repository's Python 3.12, `uv run`, pytest, Ruff, and Ty checks. Run the narrowest relevant checks after the change, then the broader regression set proportional to risk.
- Keep lazy imports and Tkinter's single-root model; do not introduce a second `tk.Tk()` or a blocking GUI loop inside a module launched by vailá.
- For CUDA/video work, state hardware requirements, VRAM behavior, checkpoint paths, deterministic mode, and whether a full inference run was actually performed.

## Phase 3: Select actuation

- For a short or medium run that fits one context window, specify a harness-native goal invocation if the target harness supports it.
- For a long run, specify a fresh-context external runner (Ralph pattern) that re-reads the loop document and disk state each iteration.
- Do not write an executable runner unless the user explicitly requests one. A shell skeleton in the specification must remain illustrative and must not bypass approvals or budgets.

## Phase 4: Write artifacts

Write `<name>-loop.md` to the agreed location. Create parent directories only after the user has selected the location. Use the following skeleton, omitting only sections explicitly marked optional.

```markdown
---
name: <loop-name>
category: <Biomechanics | Data Science | Vailá | Evaluation | Coverage | Refactoring | Multi-Agent | other>
trigger: <manual | scheduled | event>
verification-level: <1 | 2 | 3 | 4 | 5>
theory-base: arXiv:2607.00038
---

# <Human-readable loop name>

## Description
<Purpose in one or two sentences.>

## Use When
<Concrete operational scenarios and important exclusions.>

## Inputs
<!-- Optional: omit if autonomous. -->
1. `<input>` — <meaning, validation, default if safe>.

## Goal
<Concrete end state and whether it is objectively verifiable.>

## Verification (Governing Check)
- **True level:** <1–5 and why>.
- **Check:** `<command>` or <measurement/frozen rubric/field signal>.
- **Evidence:** <raw output or result that must be recorded each iteration>.
- **Completion criterion:** <condition parsed directly from the evidence>.
- **Verifier protection:** <frozen files, holdout, red/green proof, isolated judge>.
- **Scientific validity:** <units, coordinate frame, reference fixture, uncertainty/leakage checks>.

## Trigger
<Who/what starts the loop, cadence/event, and duplicate-run protection.>

## Iteration
0. On the first iteration, validate and freeze runtime inputs.
1. Load the specification and durable state; confirm budget and approvals.
2. Snapshot the baseline and run the governing check.
3. Rank unresolved targets and select the worst/highest-priority one.
4. Invoke `<named-skill>` or `<sub-loop.md>` to make exactly one change.
5. Run the governing check and record raw evidence.
6. Retain the change only if accepted without regression; otherwise use the declared rollback.
7. Curate lessons and atomically persist state, evidence, counters, and cost.
8. Evaluate terminal states; otherwise begin the next iteration.

## Terminal States
- **success:** <exact evidence-backed condition>.
- **no-op:** <nothing actionable; clean outcome, if applicable>.
- **no-progress/stalled:** <measurable stagnation rule>.
- **blocked:** <specific condition requiring external intervention>.
- **exhausted:** <hard budget/turn ceiling reached>.

Errors, missing evidence, and budget exhaustion are never success.

## Guardrails
- **Maximum allocation:** <turns, tokens and/or currency>.
- **Human approval required:** <irreversible/external actions>.
- **Isolation and credentials:** <sandbox, network and least privilege>.
- **Protected verifier:** <what the maker cannot modify or silence>.
- **Rollback:** <recoverable method scoped to one iteration>.

## State Memory
- **Path:** `<state-file.json or markdown>`.
- **Persist:** baseline, terminal status, attempts, accepted/rejected changes, evidence, curated lessons, decisions, and cost.
- **Recovery:** <how a fresh context resumes safely and detects an interrupted write>.

## Skills
- `$<skill-name>` — <role in one iteration>.
- `$vaila-<module-or-domain>` — <use an existing vailá-specific skill when available>.

## Sub-Loops
<!-- Optional: omit if none. -->
- `<sub-name>-loop.md` — <call site and terminal-state contract>.

Parent and child limits yield at most `<parent × child>` child iterations. Circular calls are prohibited.

## Why It Works
<Map the check, architecture, atomic change, memory, stop states, and guardrails to the failure modes they prevent.>

## How to Trigger
### Context-bound
<Exact harness-native invocation, only if supported.>

### Fresh-context / Ralph
<External runner contract: re-read this file and state every turn; stop only on a named terminal state.>

## Health Metrics
- **Cost per accepted change:** `total cost / verified changes retained`.
- <Task-specific progress, regression, latency, and failure metrics>.
```

If requested and supported, write one launcher artifact that points to the absolute loop-document path, requires reading it before action, passes runtime arguments, and repeats that errors or exhaustion are not success.

## Final quality gate

Before reporting completion, verify:

- Feedback genuinely changes the next action.
- The artifact exists at the agreed path and follows the skeleton.
- Goal, check, evidence, verifier level, and completion criterion agree.
- Every terminal state is distinguishable from success.
- Iteration and nested-loop budgets are finite and internally consistent.
- State can recover across fresh contexts without relying on chat history.
- Maker cannot approve itself or trivially game the verifier.
- Irreversible actions have explicit approval gates.
- The trigger syntax matches a real declared harness.
- The health metric can be computed from persisted data.
- Biomechanical units, frames, rates, identity, uncertainty, and leakage assumptions are explicit where relevant.
- vailá Python metadata, GUI/CLI parity, tests, lint, typing, and help documentation are included for software changes.

Report the created paths, the verifier's true level, the hard limits, and any unresolved assumptions. Do not start the loop.
