"""Persistence and summary utilities for HLE team solver runs."""

import dataclasses
import json
from pathlib import Path
from typing import Any

from model_diversity.agents import AgentResult, TeamMember


def write_execution_log(
    log_dir: Path,
    member: TeamMember,
    result: AgentResult,
) -> None:
    log = {
        "provider": member.provider,
        "persona": member.persona,
        "metrics": dataclasses.asdict(result.metrics),
        "entries": result.log_entries,
    }
    with (log_dir / "execution_log.json").open("w") as f:
        json.dump(log, f, indent=2, default=str)


def save_run_result(base_workdir: Path, result_data: dict[str, Any]) -> None:
    """Persist the current state of a run to disk so partial results survive crashes."""
    with (base_workdir / "result.json").open("w") as f:
        json.dump(result_data, f, indent=2, default=str)


def is_run_complete(base_workdir: Path, team: list[TeamMember]) -> bool:
    """Check if a previous run completed successfully (all outputs present and graded)."""
    result_file = base_workdir / "result.json"
    if not result_file.exists():
        return False
    with result_file.open() as f:
        result = json.load(f)
    if "grade" not in result:
        return False

    answer_file = base_workdir / "my-project" / "answer.txt"
    if not answer_file.exists() or not answer_file.read_text().strip():
        return False

    for i in range(1, len(team) + 1):
        analysis = base_workdir / f"explorer_{i}" / "problem_exploration" / "analysis.md"
        if not analysis.exists() or not analysis.read_text().strip():
            return False

    return True


def compute_summary(results_root: Path) -> dict[str, Any]:
    """Compute aggregate statistics from all completed runs on disk.

    Groups by config_name (solo, diverse, homogeneous) and aggregates:
    - correctness (correct / total)
    - per-provider token usage and cost
    - total token usage and cost across all agents per problem
    """
    configs: dict[str, dict[str, Any]] = {}

    for result_file in sorted(results_root.rglob("result.json")):
        with result_file.open() as f:
            result = json.load(f)

        config_name = result["config_name"]
        if config_name not in configs:
            configs[config_name] = {
                "correct": 0,
                "total": 0,
                "per_provider": {},
                "totals": {
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                    "total_cost_usd": 0.0,
                    "total_duration_ms": 0,
                },
            }

        cfg = configs[config_name]
        cfg["total"] += 1
        if result.get("grade", {}).get("correct"):
            cfg["correct"] += 1
        cfg["totals"]["total_duration_ms"] += result.get("total_duration_ms", 0)

        config_dir = result_file.parent
        for log_file in sorted(config_dir.rglob("execution_log.json")):
            with log_file.open() as f:
                log = json.load(f)

            provider = log["provider"]
            metrics = log["metrics"]

            if provider not in cfg["per_provider"]:
                cfg["per_provider"][provider] = {
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                    "total_cost_usd": 0.0,
                }

            for key in ["input_tokens", "cached_input_tokens", "output_tokens", "total_cost_usd"]:
                value = metrics.get(key) or 0
                cfg["per_provider"][provider][key] += value
                cfg["totals"][key] += value

    return configs


def print_summary(summary: dict[str, Any]) -> None:
    """Print a human-readable summary table."""
    for config_name, cfg in summary.items():
        print(f"\n{'=' * 60}")
        print(f"  {config_name.upper()} -- {cfg['correct']}/{cfg['total']} correct")
        print(f"{'=' * 60}")

        print("\n  Per-provider token usage:")
        for provider, tokens in sorted(cfg["per_provider"].items()):
            print(f"    {provider}:")
            print(f"      input:  {tokens['input_tokens']:>12,}")
            print(f"      cached: {tokens['cached_input_tokens']:>12,}")
            print(f"      output: {tokens['output_tokens']:>12,}")
            print(f"      cost:   ${tokens['total_cost_usd']:>11.4f}")

        totals = cfg["totals"]
        print("\n  Totals across all agents and samples:")
        print(f"    input:  {totals['input_tokens']:>12,}")
        print(f"    cached: {totals['cached_input_tokens']:>12,}")
        print(f"    output: {totals['output_tokens']:>12,}")
        print(f"    cost:   ${totals['total_cost_usd']:>11.4f}")
        total_secs = totals["total_duration_ms"] / 1000
        mins, secs = divmod(int(total_secs), 60)
        print(f"    time:   {mins}m {secs}s")

    print()
