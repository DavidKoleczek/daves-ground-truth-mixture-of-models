"""E2E tests: Claude Code and Codex can invoke Gemini CLI + nanobanana to edit an infographic."""

import os
from pathlib import Path
import shutil

import pytest

from model_diversity.agents import run_claude_agent, run_codex_agent

_FIXTURES_DIR = Path(__file__).parents[0] / "fixtures"
_OUTPUT_SUBDIR = "nanobanana-output"
_STABLE_DIR = Path("/tmp/nanobanana_edit_test")

_EDIT_VIA_CLI_PROMPT = """\
There is an infographic image at `{image_path}`.

Look at the image. Then improve it by running the following shell command \
(fill in your own edit instructions where indicated):

```
gemini -p 'Use the edit_image tool to edit the file {image_relpath}. \
<describe your edits: e.g. make the title larger, improve color contrast, fix alignment>' \
-m gemini-3.1-pro-preview -o json --approval-mode yolo
```

Make a single concrete improvement to the infographic (e.g., improve the color palette, \
fix text legibility, or enhance the layout). Do NOT open a preview. \
After running the command, write a one-line summary of what you changed to `edit_summary.txt`."""


@pytest.fixture
def seed_infographic(tmp_path: Path) -> Path:
    """Copy the fixture infographic into a nanobanana-output/ layout inside tmp_path."""
    output_dir = tmp_path / _OUTPUT_SUBDIR
    output_dir.mkdir()
    dest = output_dir / "water_cycle.png"
    shutil.copy2(_FIXTURES_DIR / "water_cycle.png", dest)
    return dest


def _build_edit_prompt(image_path: Path, work_dir: Path) -> str:
    return _EDIT_VIA_CLI_PROMPT.format(
        image_path=image_path,
        image_relpath=image_path.relative_to(work_dir),
    )


def _find_edit_result(work_dir: Path, original_path: Path, original_bytes: bytes) -> Path | None:
    """Find the edited image: a new file anywhere under work_dir, or the original with changed content."""
    for img in sorted(work_dir.rglob("*.png")) + sorted(work_dir.rglob("*.jpg")):
        if img != original_path and img.stat().st_size > 1000:
            return img
    if original_path.exists() and original_path.read_bytes() != original_bytes:
        return original_path
    return None


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY") or not os.getenv("ANTHROPIC_API_KEY"),
    reason="GEMINI_API_KEY and ANTHROPIC_API_KEY required",
)
async def test_claude_edits_infographic_via_gemini_cli(seed_infographic: Path) -> None:
    """Claude Code can invoke gemini CLI to edit an infographic via nanobanana."""
    work_dir = seed_infographic.parents[1]

    # Snapshot original before the edit
    stable = _STABLE_DIR / "claude"
    stable.mkdir(parents=True, exist_ok=True)
    original_bytes = seed_infographic.read_bytes()
    shutil.copy2(seed_infographic, stable / f"before{seed_infographic.suffix}")

    prompt = _build_edit_prompt(seed_infographic, work_dir)
    result = await run_claude_agent(prompt, work_dir)

    edited = _find_edit_result(work_dir, seed_infographic, original_bytes)
    assert edited is not None, (
        f"No edited image found under {work_dir} (new file or changed content). Agent response: {result.text[:500]}"
    )

    summary = work_dir / "edit_summary.txt"
    assert summary.exists(), f"Agent did not write edit_summary.txt. Response: {result.text[:500]}"

    shutil.copy2(edited, stable / f"after{edited.suffix}")
    print(f"\nClaude edit summary: {summary.read_text().strip()}")
    print(f"Before/after saved to: {stable}")


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY") or not os.getenv("OPENAI_API_KEY"),
    reason="GEMINI_API_KEY and OPENAI_API_KEY required",
)
async def test_codex_edits_infographic_via_gemini_cli(seed_infographic: Path) -> None:
    """Codex can invoke gemini CLI to edit an infographic via nanobanana."""
    work_dir = seed_infographic.parents[1]

    # Copy the seed to an isolated dir so results don't collide with the Claude test
    codex_dir = work_dir / "codex_workspace"
    codex_dir.mkdir()
    codex_output = codex_dir / _OUTPUT_SUBDIR
    codex_output.mkdir()
    dest_image = codex_output / seed_infographic.name
    shutil.copy2(seed_infographic, dest_image)

    stable = _STABLE_DIR / "codex"
    stable.mkdir(parents=True, exist_ok=True)
    original_bytes = dest_image.read_bytes()
    shutil.copy2(dest_image, stable / f"before{dest_image.suffix}")

    prompt = _build_edit_prompt(dest_image, codex_dir)
    result = await run_codex_agent(prompt, codex_dir, network_access=True)

    edited = _find_edit_result(codex_dir, dest_image, original_bytes)
    assert edited is not None, (
        f"No edited image found in {codex_output} (new file or changed content). Agent response: {result.text[:500]}"
    )

    summary = codex_dir / "edit_summary.txt"
    assert summary.exists(), f"Agent did not write edit_summary.txt. Response: {result.text[:500]}"

    shutil.copy2(edited, stable / f"after{edited.suffix}")
    print(f"\nCodex edit summary: {summary.read_text().strip()}")
    print(f"Before/after saved to: {stable}")
