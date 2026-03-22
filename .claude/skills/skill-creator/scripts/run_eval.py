#!/usr/bin/env python3
"""Run a single trigger eval query against a skill description.

Evaluates whether the skill would be triggered for a given query by
calling the model multiple times and checking if it consults the skill.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


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
            triggered = _check_trigger(result.stdout + result.stderr, query, skill_path)
            results.append(
                {
                    "trial": i + 1,
                    "triggered": triggered,
                    "output_preview": (result.stdout + result.stderr)[:500],
                }
            )
        except Exception as e:
            results.append({"trial": i + 1, "triggered": False, "error": str(e)})
        time.sleep(0.5)
    trigger_rate = sum(1 for r in results if r.get("triggered")) / len(results) if results else 0
    return {"query": query, "trigger_rate": trigger_rate, "trials": results}


def _check_trigger(output: str, query: str, skill_path: Path) -> bool:
    skill_name = skill_path.name
    indicators = [
        f"skill/{skill_name}",
        f"invoking skill",
        f"using skill: {skill_name}",
        f"skill path: {skill_path}",
        f"loading skill",
    ]
    output_lower = output.lower()
    for indicator in indicators:
        if indicator.lower() in output_lower:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single trigger eval")
    parser.add_argument("query", help="Query string to test")
    parser.add_argument("--skill-path", type=Path, required=True, help="Path to skill directory")
    parser.add_argument("--model", default="claude-opus-4-5", help="Model ID")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials")
    parser.add_argument("--output", type=Path, help="Output JSON file")
    args = parser.parse_args()

    result = run_eval(args.query, args.skill_path, args.model, args.trials)
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
