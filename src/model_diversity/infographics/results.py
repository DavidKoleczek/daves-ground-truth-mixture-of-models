"""Persistence and skip-logic utilities for the infographic experiment."""

import json
from pathlib import Path
from typing import Any


def save_run_result(scenario_dir: Path, result_data: dict[str, Any]) -> None:
    """Persist the current state of a scenario run to disk."""
    with (scenario_dir / "result.json").open("w") as f:
        json.dump(result_data, f, indent=2, default=str)


def is_generation_complete(scenario_dir: Path) -> bool:
    """Check if the initial image has already been generated for this scenario."""
    if not (scenario_dir / "initial_image.png").exists():
        return False
    result_file = scenario_dir / "result.json"
    if not result_file.exists():
        return False
    with result_file.open() as f:
        result = json.load(f)
    return "generation_cost_usd" in result


def is_refinement_complete(scenario_dir: Path, config_name: str) -> bool:
    """Check if refinement has already completed for this config."""
    if not (scenario_dir / f"after_{config_name}_step_3_visual_design.png").exists():
        return False
    result_file = scenario_dir / "result.json"
    if not result_file.exists():
        return False
    with result_file.open() as f:
        result = json.load(f)
    return config_name in result
