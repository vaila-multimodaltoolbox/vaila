# Loop Engineering Reference

Use this reference to explain design choices or audit an existing loop. It summarizes Sandeco Macedo, *Stop Hand-Holding Your Coding Agent: Engineering the Loops that Replace Step-by-Step Prompting*, arXiv:2607.00038v1 (28 June 2026).

## Core model

A loop specification is a bounded reusable artifact layered above an agent harness. It contains a trigger, goal, execution phase, real verification, stopping rule, and durable memory. It is distinct from ordinary program control flow and from the harness's internal perceive-act-observe cycle.

The decisive design object is the check. Use a loop only when feedback from one iteration changes the next action. Otherwise use a scheduled one-shot.

## Evidence reported in the paper

The paper manually codes 50 Loop Library artifacts; it is a descriptive corpus study, not a controlled experiment. Within that corpus:

- 70% use level-1 or level-2 verification, the autonomous zone.
- 76% use objective levels 1–3.
- 74% name terminal states.
- 66% set a verifiable goal.
- 78% use manual triggers and 78% use solo-agent architectures.
- Only 20% call named reusable skills and 32% develop persistent memory.

Treat these figures as characteristics of that corpus, not causal proof or population estimates.

## Design families

### A. Define done

Freeze the yardstick, require reproducible evidence, consider consecutive successes, name terminal states, and stop on goal, stagnation, or budget. Unassisted self-correction and self-scoring are unreliable.

### B. Act without regression

Snapshot the baseline, change one thing, address the worst target first, rerun checks, retain only non-regressive changes, and begin from clean recoverable state. Atomicity makes attribution possible.

### C. Earn trust

Separate maker and checker, use fresh holdouts, ground claims in evidence, and prove verifiers red-before/green-after. Shared generator/judge context invites reward hacking; deterministic or rule-based checks are preferable.

### D. Sustain operation

Persist and curate state on disk, enumerate the full work surface, protect irreversible actions with human approval, isolate unattended execution, scope credentials, restrict network access when feasible, and cap budget.

## Anti-patterns

- **While-true around a stranger:** raw unguided retries without skills or external checks.
- **Self-approving loop:** the maker grades itself and inflates its score.
- **Specification gaming:** the agent weakens tests, hard-codes answers, or manipulates the environment.
- **Pretending level 4 is level 1:** model opinion is mislabeled as deterministic truth.
- **Unattended runaway:** no task-related stop, stagnation detector, budget ceiling, or approval gates.

## Honest limits

Loops automate repeated action, not human judgment. Account for verification burden, comprehension debt, cognitive surrender, security exposure, and cost. Avoid loops for pure taste, directionless greenfield work, non-adaptive scheduled tasks, or situations where expected value does not justify the loop.

## Primary source

- Sandeco Macedo, arXiv:2607.00038v1, https://arxiv.org/abs/2607.00038
