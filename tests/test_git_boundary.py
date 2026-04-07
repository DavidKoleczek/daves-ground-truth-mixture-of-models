"""E2E test: Claude agent writes files in its cwd, not the parent git repo root."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

from model_diversity.agents import run_claude_agent

_PROJECT_ROOT = Path(__file__).parents[1]

_SEED_FILES = {
    "scores.csv": "model,accuracy\ngpt-4o,0.72\nclaude-opus,0.85\ngemini-pro,0.91\n",
    "config.yaml": "dataset: mmlu\nsplit: test\nsamples: 500\n",
}

_PROMPT = (
    "Read the files 'scores.csv', 'results.json', and 'config.yaml'. "
    "Then write a file called 'summary.md' with a Markdown heading '# Summary' "
    "followed by the dataset name from config.yaml and the highest accuracy from scores.csv."
)


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
async def test_claude_agent_respects_cwd_boundary() -> None:
    """The agent must read seed files from cwd and write output there, not the repo root."""
    workdir = _PROJECT_ROOT / "tmp" / "git_boundary_test"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    for name, content in _SEED_FILES.items():
        (workdir / name).write_text(content)

    # Verify the workdir is inside the parent git repo before the boundary is set up.
    # Without this, the test could pass trivially (e.g. if workdir were already
    # outside any git repo).
    result = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=workdir, capture_output=True, text=True)
    detected_root = Path(result.stdout.strip())
    assert detected_root == _PROJECT_ROOT, (
        f"Test setup error: workdir should be inside the parent repo "
        f"(expected root {_PROJECT_ROOT}, got {detected_root})"
    )

    try:
        await run_claude_agent(_PROMPT, cwd=workdir)

        summary_path = workdir / "summary.md"
        assert summary_path.exists(), f"Expected summary.md in {workdir}, but it was not created there"
        assert not (_PROJECT_ROOT / "summary.md").exists(), "summary.md was created at the repo root instead of the cwd"

        summary_text = summary_path.read_text()
        assert "mmlu" in summary_text.lower(), (
            f"Agent did not read config.yaml from cwd (expected 'mmlu' in summary). Got: {summary_text}"
        )
        assert "0.91" in summary_text, (
            f"Agent did not read scores.csv from cwd (expected '0.91' in summary). Got: {summary_text}"
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        (_PROJECT_ROOT / "summary.md").unlink(missing_ok=True)
