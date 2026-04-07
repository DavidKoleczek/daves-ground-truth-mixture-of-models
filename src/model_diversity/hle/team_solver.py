"""HLE team solver experiment.

We need three experiments
1. Each agent on its own for each HLE task

2. Task each agent with search for the hard parts of each problem, web search and web fetch disabled.
Then Claude Code writes a final answer to the problem to a file
  - In one case each agent is three different Claudes
  - The other case its a Claude, Codex, and Gemini
"""

import argparse
import asyncio
import base64
import dataclasses
import json
from pathlib import Path
import random
import re
import shutil
import time
from typing import Any

from anthropic import AsyncAnthropic
from interop_router.router import Router
from liquid import render as render_liquid

from model_diversity.agents import AgentResult, TeamMember, run_agent
from model_diversity.hle.dataset import HLESample, grade_hle_response, load_hle_dataset
from model_diversity.hle.prompts import (
    EXPLORATION_PROMPT_TEMPLATE,
    PERSONAS,
    SOLO_PROMPT_TEMPLATE,
    SOLVER_PROMPT_TEMPLATE,
)
from model_diversity.hle.results import (
    compute_summary,
    is_run_complete,
    print_summary,
    save_run_result,
    write_execution_log,
)

_DEFAULT_OUTPUT_DIR = Path(__file__).parents[3] / "output" / "hle"

SOLO_TEAM: list[TeamMember] = []

HOMOGENEOUS_TEAM: list[TeamMember] = [
    TeamMember(provider="claude"),
    TeamMember(provider="claude", persona="contrarian"),
    TeamMember(provider="claude", persona="professor"),
]

DIVERSE_TEAM: list[TeamMember] = [
    TeamMember(provider="claude"),
    TeamMember(provider="codex"),
    TeamMember(provider="gemini"),
]


def get_workdir(output_dir: Path, sample_id: str, config_name: str) -> Path:
    workdir = output_dir / sample_id / config_name
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


def save_sample_image(sample: HLESample, workdir: Path) -> Path | None:
    if not sample.image:
        return None
    match = re.match(r"data:image/(\w+);base64,(.+)", sample.image, re.DOTALL)
    if match is None:
        raise ValueError(f"Unexpected image data URI format for sample {sample.id}")
    ext = match.group(1)
    data = base64.b64decode(match.group(2))
    image_path = workdir / f"question_image.{ext}"
    image_path.write_bytes(data)
    return image_path


def _assign_personas(team: list[TeamMember], sample_id: str) -> list[TeamMember]:
    """Randomly assign personas to two of three team members, deterministic per sample."""
    personas: list[str | None] = ["professor", "contrarian", None]
    rng = random.Random(sample_id)
    rng.shuffle(personas)
    return [TeamMember(provider=m.provider, persona=p) for m, p in zip(team, personas, strict=True)]


def _image_note(image_path: Path | None) -> str:
    if image_path is None:
        return ""
    return (
        f"\nAn image is provided for this question. "
        f"It is saved at `{image_path.name}` in the current directory. "
        f"Examine it as part of your work.\n"
    )


async def _run_team_exploration(
    sample: HLESample,
    team: list[TeamMember],
    config_name: str,
    output_dir: Path,
) -> tuple[Path, list[AgentResult]]:
    """Run the exploration phase where each agent independently analyzes the question.

    Returns the base working directory and the list of explorer results.
    """
    base_workdir = get_workdir(output_dir, sample.id, config_name)

    # Clean previous run data so we don't mix old and new results
    if base_workdir.exists():
        shutil.rmtree(base_workdir)
    base_workdir.mkdir(parents=True, exist_ok=True)

    # Each agent works in its own isolated subdirectory so they can't
    # see each other's work during the exploration phase
    tasks = []
    for i, member in enumerate(team):
        agent_workdir = base_workdir / f"explorer_{i + 1}" / "problem_exploration"
        agent_workdir.mkdir(parents=True, exist_ok=True)
        image_path = save_sample_image(sample, agent_workdir)
        prompt = render_liquid(
            EXPLORATION_PROMPT_TEMPLATE,
            output_filename="analysis.md",
            question=sample.question,
            image_note=_image_note(image_path),
        )
        if member.persona:
            prompt = PERSONAS[member.persona] + "\n\n" + prompt
        tasks.append(run_agent(member, prompt, agent_workdir))

    # All agents explore the problem concurrently
    results: list[AgentResult] = list(await asyncio.gather(*tasks))

    for i, (member, result) in enumerate(zip(team, results, strict=True)):
        write_execution_log(base_workdir / f"explorer_{i + 1}", member, result)

    analysis_files = sorted(base_workdir.rglob("analysis.md"))
    print(f"  Created {len(analysis_files)} analysis files in {base_workdir}")
    for f in analysis_files:
        print(f"    - {f.relative_to(base_workdir)}")

    return base_workdir, results


async def _run_solver(
    sample: HLESample,
    base_workdir: Path,
    seed: int | None = None,
) -> AgentResult:
    """Synthesize explorer analyses into a final answer using Claude."""
    solver_workdir = base_workdir / "my-project"
    solver_workdir.mkdir(parents=True, exist_ok=True)

    # Collect analysis files and randomize order so position doesn't
    # leak which model produced which analysis
    analysis_files = [f for f in sorted(base_workdir.rglob("analysis.md")) if f.read_text().strip()]
    rng = random.Random(seed)
    rng.shuffle(analysis_files)

    # Copy to solver directory with anonymized sequential names
    analysis_filenames: list[str] = []
    shuffle_mapping: dict[str, str] = {}
    for i, src in enumerate(analysis_files):
        dest_name = f"analysis_{i + 1}.md"
        shutil.copy(src, solver_workdir / dest_name)
        analysis_filenames.append(dest_name)
        shuffle_mapping[dest_name] = str(src.relative_to(base_workdir))

    # Record the mapping for debugging/reproducibility
    with (base_workdir / "shuffle_mapping.json").open("w") as f:
        json.dump(shuffle_mapping, f, indent=2)

    image_path = save_sample_image(sample, solver_workdir)
    answer_path = str(solver_workdir / "answer.txt")
    prompt = render_liquid(
        SOLVER_PROMPT_TEMPLATE,
        question=sample.question,
        image_note=_image_note(image_path),
        analysis_files=analysis_filenames,
        answer_path=answer_path,
    )

    solver_member = TeamMember(provider="claude")
    result = await run_agent(solver_member, prompt, solver_workdir)
    write_execution_log(base_workdir / "my-project", solver_member, result)

    answer_file = solver_workdir / "answer.txt"
    answer_text = answer_file.read_text() if answer_file.exists() else ""
    return AgentResult(
        text=answer_text,
        log_entries=result.log_entries,
        metrics=result.metrics,
    )


async def _run_solo(sample: HLESample, config_name: str, output_dir: Path) -> tuple[Path, AgentResult]:
    """Run a single Claude agent directly on the problem (baseline)."""
    base_workdir = get_workdir(output_dir, sample.id, config_name)

    if base_workdir.exists():
        shutil.rmtree(base_workdir)
    base_workdir.mkdir(parents=True, exist_ok=True)

    solver_workdir = base_workdir / "my-project"
    solver_workdir.mkdir(parents=True, exist_ok=True)

    image_path = save_sample_image(sample, solver_workdir)
    answer_path = str(solver_workdir / "answer.txt")
    prompt = render_liquid(
        SOLO_PROMPT_TEMPLATE,
        question=sample.question,
        image_note=_image_note(image_path),
        answer_path=answer_path,
    )

    solver_member = TeamMember(provider="claude")
    result = await run_agent(solver_member, prompt, solver_workdir)
    write_execution_log(base_workdir / "my-project", solver_member, result)

    answer_file = solver_workdir / "answer.txt"
    answer_text = answer_file.read_text() if answer_file.exists() else ""
    return base_workdir, AgentResult(
        text=answer_text,
        log_entries=result.log_entries,
        metrics=result.metrics,
    )


async def _process_sample(sample: HLESample, router: Router, semaphore: asyncio.Semaphore, output_dir: Path) -> None:
    """Run all configs (solo, diverse, homogeneous) for a single sample."""
    async with semaphore:
        print(f"Sample {sample.id} ({sample.answer_type})")
        print(f"  Q: {sample.question[:120]}...")
        print()

        for config_name, base_team in [
            ("solo", SOLO_TEAM),
            ("diverse", DIVERSE_TEAM),
            ("homogeneous", HOMOGENEOUS_TEAM),
        ]:
            # For the diverse team, randomly assign personas per sample
            team = _assign_personas(base_team, sample.id) if config_name == "diverse" else base_team

            base_workdir = get_workdir(output_dir, sample.id, config_name)
            if is_run_complete(base_workdir, team):
                print(f"  [{sample.id[:8]}] Skipping {config_name} (already complete)")
                continue

            result_data: dict[str, Any] = {
                "sample_id": sample.id,
                "question": sample.question,
                "correct_answer": sample.answer,
                "answer_type": sample.answer_type,
                "config_name": config_name,
                "team": [{"provider": m.provider, "persona": m.persona} for m in team],
            }

            scenario_start = time.monotonic()

            if team:
                # Phase 1: Exploration
                print(f"  [{sample.id[:8]}] Running {config_name} team exploration...")
                base_workdir, explorer_results = await _run_team_exploration(sample, team, config_name, output_dir)
                save_run_result(base_workdir, result_data)

                # Phase 2: Solver
                print(f"  [{sample.id[:8]}] Running solver for {config_name} team...")
                solver_result = await _run_solver(sample, base_workdir, seed=42)

                explorer_cost = sum(r.metrics.total_cost_usd or 0.0 for r in explorer_results)
                solver_cost = solver_result.metrics.total_cost_usd or 0.0
                result_data["total_cost_usd"] = explorer_cost + solver_cost
            else:
                # Solo baseline: single Claude agent, no exploration phase
                print(f"  [{sample.id[:8]}] Running {config_name} baseline...")
                base_workdir, solver_result = await _run_solo(sample, config_name, output_dir)
                result_data["total_cost_usd"] = solver_result.metrics.total_cost_usd or 0.0

            result_data["solver_answer"] = solver_result.text
            result_data["solver_metrics"] = dataclasses.asdict(solver_result.metrics)
            save_run_result(base_workdir, result_data)
            print(f"  [{sample.id[:8]}] Solver answer: {solver_result.text[:200]}...")

            # Grading
            if solver_result.text:
                print(f"  [{sample.id[:8]}] Grading {config_name} answer...")
                grade = await grade_hle_response(router, sample, solver_result.text)
                result_data["grade"] = {
                    "correct": grade.correct,
                    "judge_response": grade.judge_response,
                }
                save_run_result(base_workdir, result_data)
                print(f"  [{sample.id[:8]}] Grade: {'correct' if grade.correct else 'incorrect'}")
            else:
                print(f"  [{sample.id[:8]}] No solver answer produced, skipping grading.")

            result_data["total_duration_ms"] = int((time.monotonic() - scenario_start) * 1000)
            save_run_result(base_workdir, result_data)
            print()


async def async_main(concurrent: int, output_dir: Path) -> None:
    router = Router()
    router.register("anthropic", AsyncAnthropic())

    samples = load_hle_dataset(n=50, seed=42)

    semaphore = asyncio.Semaphore(concurrent)
    await asyncio.gather(*[_process_sample(s, router, semaphore, output_dir) for s in samples])

    summary = compute_summary(output_dir)
    print_summary(summary)
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="HLE team solver experiment")
    parser.add_argument("--concurrent", type=int, default=3, help="max samples to run in parallel")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT_DIR, help="output directory for results")
    args = parser.parse_args()

    asyncio.run(async_main(concurrent=args.concurrent, output_dir=args.output))


if __name__ == "__main__":
    main()
