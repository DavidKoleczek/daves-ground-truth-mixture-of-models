"""Infographic experiment: generate and iteratively refine infographics.

Step 1 -- Generate an initial infographic for each scenario using Gemini CLI + nanobanana.
Step 2 -- Run sequential evaluation/refinement through three criteria (same_model and diverse configs).
"""

import argparse
import asyncio
import dataclasses
from dataclasses import dataclass
import json
from pathlib import Path
import random
import shutil
import time
from typing import Any

from liquid import render as render_liquid
import yaml

from model_diversity.agents import AgentResult, TeamMember, run_agent, run_gemini_agent
from model_diversity.hle.results import write_execution_log
from model_diversity.infographics.prompts import CRITERIA, EVALUATION_PROMPT_TEMPLATE, INITIAL_IMAGE_TEMPLATE
from model_diversity.infographics.results import is_generation_complete, is_refinement_complete, save_run_result

_DEFAULT_OUTPUT_DIR = Path(__file__).parents[3] / "output" / "infographics"
_SCENARIOS_PATH = Path(__file__).parent / "scenarios.yaml"

_MAX_GENERATION_ATTEMPTS = 3

_CRITERIA_ORDER = ["information_architecture", "visual_communication", "visual_design"]

# Neutral directory names so agents can't infer the experiment design
_CONFIG_WORKDIRS: dict[str, str] = {
    "same_model": "image_working_dir",
    "diverse": "working_dir_image",
}


@dataclass(frozen=True)
class Scenario:
    name: str
    prompt: str


def load_scenarios() -> list[Scenario]:
    """Load infographic scenarios from the YAML file."""
    with _SCENARIOS_PATH.open() as f:
        data = yaml.safe_load(f)
    return [Scenario(name=s["name"], prompt=s["prompt"].strip()) for s in data["scenarios"]]


def _find_generated_image(workdir: Path, expected_stem: str) -> Path | None:
    """Find the generated image matching the expected filename (any image extension)."""
    output_dir = workdir / "nanobanana-output"
    if not output_dir.exists():
        return None
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = output_dir / f"{expected_stem}{ext}"
        if candidate.exists() and candidate.stat().st_size > 1000:
            return candidate
    return None


async def _generate_initial_image(scenario: Scenario, scenario_dir: Path) -> AgentResult:
    """Generate the initial infographic for a scenario using Gemini CLI + nanobanana.

    Retries up to _MAX_GENERATION_ATTEMPTS times if nanobanana produces an image
    with an unexpected filename (indicates something went wrong).
    """
    generation_dir = scenario_dir / "generation"
    image_abspath = (generation_dir / "nanobanana-output" / f"{scenario.name}.png").resolve()

    prompt = render_liquid(
        INITIAL_IMAGE_TEMPLATE,
        infographic_prompt=scenario.prompt,
        image_abspath=str(image_abspath),
    )

    member = TeamMember(provider="gemini")

    for attempt in range(1, _MAX_GENERATION_ATTEMPTS + 1):
        if generation_dir.exists():
            shutil.rmtree(generation_dir)
        generation_dir.mkdir(parents=True, exist_ok=True)

        result = await run_gemini_agent(prompt, generation_dir)
        write_execution_log(generation_dir, member, result)

        generated = _find_generated_image(generation_dir, expected_stem=scenario.name)
        if generated is not None:
            dest = scenario_dir / "initial_image.png"
            shutil.copy2(generated, dest)
            print(f"  Saved initial image: {dest}")
            return result

        if attempt < _MAX_GENERATION_ATTEMPTS:
            print(f"  Attempt {attempt} failed for {scenario.name}, retrying...")
        else:
            print(f"  WARNING: All {_MAX_GENERATION_ATTEMPTS} attempts failed for {scenario.name}.")
            print(f"  Agent response: {result.text[:300]}")

    return result


def _assign_models(scenario_name: str) -> list[TeamMember]:
    """Deterministically assign one model to each criterion for the diverse config."""
    members = [TeamMember(provider="claude"), TeamMember(provider="codex"), TeamMember(provider="gemini")]
    rng = random.Random(scenario_name)
    rng.shuffle(members)
    return members


async def _run_refinement(
    scenario: Scenario,
    scenario_dir: Path,
    config_name: str,
    step_assignments: list[tuple[str, TeamMember]],
) -> dict[str, Any]:
    """Run sequential evaluation/refinement steps for one config.

    The agent works in a neutrally-named directory and edits {scenario.name}.png in place.
    After each step, we snapshot the image to scenario_dir with a readable name.
    """
    workdir_name = _CONFIG_WORKDIRS[config_name]
    config_dir = scenario_dir / workdir_name
    if config_dir.exists():
        shutil.rmtree(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    working_image = config_dir / f"{scenario.name}.png"
    shutil.copy2(scenario_dir / "initial_image.png", working_image)

    steps: list[dict[str, Any]] = []
    config_start = time.monotonic()

    for i, (criterion_key, member) in enumerate(step_assignments, 1):
        step_label = f"step_{i}_{criterion_key}"
        print(f"    [{config_name}] Step {i}/3: {criterion_key} ({member.provider})")

        image_abspath = str(working_image.resolve())
        prompt = render_liquid(
            EVALUATION_PROMPT_TEMPLATE,
            original_prompt=scenario.prompt,
            image_path=image_abspath,
            image_abspath=image_abspath,
            criterion_prompt=CRITERIA[criterion_key],
        )

        step_start = time.monotonic()
        result = await run_agent(member, prompt, config_dir, network_access=True)
        step_duration_ms = int((time.monotonic() - step_start) * 1000)

        log_dir = scenario_dir / f"{config_name}_{step_label}"
        log_dir.mkdir(parents=True, exist_ok=True)
        write_execution_log(log_dir, member, result)

        snapshot = scenario_dir / f"after_{config_name}_{step_label}.png"
        shutil.copy2(working_image, snapshot)

        nanobanana_dir = config_dir / "nanobanana-output"
        if nanobanana_dir.exists():
            shutil.rmtree(nanobanana_dir)

        steps.append(
            {
                "criterion": criterion_key,
                "provider": member.provider,
                "cost_usd": result.metrics.total_cost_usd,
                "duration_ms": step_duration_ms,
                "metrics": dataclasses.asdict(result.metrics),
                "agent_response": result.text,
            }
        )

    total_duration_ms = int((time.monotonic() - config_start) * 1000)
    total_cost = sum(s["cost_usd"] or 0.0 for s in steps)

    return {
        "steps": steps,
        "total_cost_usd": total_cost,
        "total_duration_ms": total_duration_ms,
    }


async def _process_scenario(
    scenario: Scenario,
    semaphore: asyncio.Semaphore,
    output_dir: Path,
) -> None:
    """Run all phases for a single scenario: generation + refinement configs."""
    async with semaphore:
        scenario_dir = output_dir / scenario.name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Generate initial image
        if is_generation_complete(scenario_dir):
            print(f"[{scenario.name}] Generation already complete")
        else:
            print(f"[{scenario.name}] Generating initial image...")
            start = time.monotonic()

            result = await _generate_initial_image(scenario, scenario_dir)
            elapsed_ms = int((time.monotonic() - start) * 1000)

            result_data: dict[str, Any] = {
                "scenario_name": scenario.name,
                "scenario_prompt": scenario.prompt,
                "generation_cost_usd": result.metrics.total_cost_usd,
                "generation_duration_ms": elapsed_ms,
                "generation_metrics": dataclasses.asdict(result.metrics),
                "initial_image_path": "initial_image.png",
            }
            save_run_result(scenario_dir, result_data)

            cost = result.metrics.total_cost_usd or 0.0
            print(f"[{scenario.name}] Generation done in {elapsed_ms / 1000:.1f}s (${cost:.4f})")

        if not (scenario_dir / "initial_image.png").exists():
            print(f"[{scenario.name}] No initial image, skipping refinement")
            return

        result_data = _load_result(scenario_dir)

        # Phase 2a: same_model refinement (all Claude)
        if is_refinement_complete(scenario_dir, "same_model"):
            print(f"[{scenario.name}] same_model refinement already complete")
        else:
            print(f"[{scenario.name}] Running same_model refinement...")
            same_model_assignments = [(k, TeamMember(provider="claude")) for k in _CRITERIA_ORDER]
            config_result = await _run_refinement(scenario, scenario_dir, "same_model", same_model_assignments)
            result_data["same_model"] = config_result
            save_run_result(scenario_dir, result_data)
            print(f"[{scenario.name}] same_model done (${config_result['total_cost_usd']:.4f})")

        # Phase 2b: diverse refinement (random model per criterion)
        if is_refinement_complete(scenario_dir, "diverse"):
            print(f"[{scenario.name}] diverse refinement already complete")
        else:
            print(f"[{scenario.name}] Running diverse refinement...")
            diverse_members = _assign_models(scenario.name)
            diverse_assignments = list(zip(_CRITERIA_ORDER, diverse_members, strict=True))
            config_result = await _run_refinement(scenario, scenario_dir, "diverse", diverse_assignments)
            result_data["diverse"] = config_result
            save_run_result(scenario_dir, result_data)
            assignment_str = ", ".join(f"{k}={m.provider}" for k, m in diverse_assignments)
            print(f"[{scenario.name}] diverse done (${config_result['total_cost_usd']:.4f}) [{assignment_str}]")


def _load_result(scenario_dir: Path) -> dict[str, Any]:
    """Load existing result.json or return empty dict."""
    result_file = scenario_dir / "result.json"
    if result_file.exists():
        with result_file.open() as f:
            return json.load(f)
    return {}


async def async_main(concurrent: int, output_dir: Path, scenario_filter: str | None = None) -> None:
    scenarios = load_scenarios()
    if scenario_filter:
        scenarios = [s for s in scenarios if s.name == scenario_filter]
        if not scenarios:
            print(f"No scenario found matching '{scenario_filter}'")
            return
    print(f"Running {len(scenarios)} scenario(s)")

    semaphore = asyncio.Semaphore(concurrent)
    overall_start = time.monotonic()

    await asyncio.gather(*[_process_scenario(s, semaphore, output_dir) for s in scenarios])

    total_secs = time.monotonic() - overall_start
    mins, secs = divmod(int(total_secs), 60)
    print(f"\nAll scenarios complete in {mins}m {secs}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Infographic experiment")
    parser.add_argument("--concurrent", type=int, default=3, help="max scenarios to run in parallel")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT_DIR, help="output directory for results")
    parser.add_argument("--scenario", type=str, default=None, help="run a single scenario by name")
    args = parser.parse_args()

    asyncio.run(async_main(concurrent=args.concurrent, output_dir=args.output, scenario_filter=args.scenario))


if __name__ == "__main__":
    main()
