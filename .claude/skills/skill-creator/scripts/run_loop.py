#!/usr/bin/env python3
"""Automated description optimization loop for skills.

Splits the eval set into train/test, runs multiple iterations of evaluation
and description improvement, and returns the best description.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
import re
from pathlib import Path

try:
    import markdown

    _HAS_MARKDOWN = True
except ImportError:
    _HAS_MARKDOWN = False


def load_eval_set(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def split_train_test(evals: list[dict], train_pct: float = 0.6) -> tuple[list[dict], list[dict]]:
    shuffled = evals.copy()
    random.shuffle(shuffled)
    n_train = int(len(shuffled) * train_pct)
    return shuffled[:n_train], shuffled[n_train:]


def get_current_description(skill_path: Path) -> str:
    sk = skill_path / "SKILL.md"
    text = sk.read_text()
    m = re.search(r"^description:\s*(.+)$", text, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return ""


def set_description(skill_path: Path, description: str) -> None:
    sk = skill_path / "SKILL.md"
    text = sk.read_text()
    text = re.sub(
        r"^description:\s*.+$",
        f"description: {description}",
        text,
        flags=re.MULTILINE,
    )
    sk.write_text(text)


def run_eval(query: str, skill_path: Path, model: str, n_trials: int = 3) -> dict:
    results = []
    for i in range(n_trials):
        try:
            result = subprocess.run(
                [
                    "claude",
                    "-p",
                    model,
                    f"--system-prompt",
                    f"Skill path: {skill_path}\nQuery: {query}",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            triggered = False
            output_lower = (result.stdout + result.stderr).lower()
            skill_name = skill_path.name
            indicators = [f"skill/{skill_name}", "invoking skill", "using skill", "loading skill"]
            triggered = any(ind.lower() in output_lower for ind in indicators)
            results.append({"trial": i + 1, "triggered": triggered})
        except Exception as e:
            results.append({"trial": i + 1, "triggered": False, "error": str(e)})
    trigger_rate = sum(1 for r in results if r.get("triggered")) / len(results) if results else 0
    return {"query": query, "trigger_rate": trigger_rate, "trials": results}


def evaluate_description(
    skill_path: Path,
    eval_set: list[dict],
    model: str,
) -> dict:
    results = []
    for item in eval_set:
        result = run_eval(item["query"], skill_path, model, n_trials=3)
        expected = item.get("should_trigger", True)
        correct = (result["trigger_rate"] > 0.5) == expected
        results.append({**item, **result, "correct": correct})
    return {
        "accuracy": sum(1 for r in results if r["correct"]) / len(results) if results else 0,
        "results": results,
    }


def propose_improvement(
    skill_path: Path,
    eval_set: list[dict],
    eval_result: dict,
    model: str,
) -> str:
    prompt = f"""The current skill description is:
{get_current_description(skill_path)}

Eval results:
{json.dumps(eval_result["results"], indent=2)}

Based on these results, propose an improved skill description.
Return ONLY the new description text (one sentence to a short paragraph).
Do not include any explanation or preamble."""
    try:
        result = subprocess.run(
            ["claude", "-p", model, f"--system-prompt", prompt],
            capture_output=True,
            text=True,
            timeout=30,
        )
        desc = result.stdout.strip()
        return desc if desc else get_current_description(skill_path)
    except Exception:
        return get_current_description(skill_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize skill description")
    parser.add_argument("--eval-set", type=Path, required=True, help="Path to trigger eval JSON")
    parser.add_argument("--skill-path", type=Path, required=True, help="Path to skill directory")
    parser.add_argument("--model", default="claude-opus-4-5", help="Model ID")
    parser.add_argument("--max-iterations", type=int, default=5, help="Max optimization iterations")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    args = parser.parse_args()

    evals = load_eval_set(args.eval_set)
    train_set, test_set = split_train_test(evals)

    best_desc = get_current_description(args.skill_path)
    best_score = 0.0
    best_test_score = 0.0
    history = []

    skill_backup = args.skill_path / "SKILL.md.bak"
    sk_orig = (args.skill_path / "SKILL.md").read_text()
    skill_backup.write_text(sk_orig)

    try:
        for i in range(1, args.max_iterations + 1):
            if args.verbose:
                print(f"\n--- Iteration {i} ---")

            result = evaluate_description(args.skill_path, train_set, args.model)
            train_score = result["accuracy"]

            test_result = evaluate_description(args.skill_path, test_set, args.model)
            test_score = test_result["accuracy"]

            history.append(
                {
                    "iteration": i,
                    "train_score": train_score,
                    "test_score": test_score,
                    "description": get_current_description(args.skill_path),
                }
            )

            if args.verbose:
                print(f"  Train accuracy: {train_score:.1%}")
                print(f"  Test accuracy:  {test_score:.1%}")

            if test_score > best_test_score:
                best_test_score = test_score
                best_desc = get_current_description(args.skill_path)
                if args.verbose:
                    print(f"  ** New best description (test: {test_score:.1%})")

            new_desc = propose_improvement(args.skill_path, train_set, result, args.model)
            if new_desc and new_desc != get_current_description(args.skill_path):
                set_description(args.skill_path, new_desc)
                if args.verbose:
                    print(f"  Proposed: {new_desc[:80]}...")

    finally:
        skill_backup.write_text(sk_orig)
        skill_backup.unlink()

    output = {
        "best_description": best_desc,
        "best_test_score": best_test_score,
        "best_train_score": best_score,
        "history": history,
    }
    print(json.dumps(output, indent=2))

    report = f"""# Description Optimization Report

## Best Description
{best_desc}

## Scores
- Test accuracy: {best_test_score:.1%}

## Iteration History
| Iter | Train | Test |
|------|-------|------|
"""
    for h in history:
        report += f"| {h['iteration']} | {h['train_score']:.1%} | {h['test_score']:.1%} |\n"

    print(report)


if __name__ == "__main__":
    main()
