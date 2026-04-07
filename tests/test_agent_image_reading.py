"""Verify that each coding CLI agent can read and describe an image from its working directory."""

import os
from pathlib import Path
import shutil

import pytest

from model_diversity.agents import run_claude_agent, run_codex_agent, run_gemini_agent

_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_TEST_IMAGE = _FIXTURES_DIR / "cat_windowsill.png"
_PROJECT_DIR = Path(__file__).parents[1]

_PROMPT = (
    "There is an image file at `./test_image.png` in your current working directory. "
    "Read the image and describe what you see in 2-3 sentences. What is depicted?"
)


def _make_workdir(agent_name: str) -> Path:
    workdir = _PROJECT_DIR / "tmp" / f"{agent_name}_image_test"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    shutil.copy2(_TEST_IMAGE, workdir / "test_image.png")
    return workdir


def _assert_describes_image(text: str) -> None:
    lower = text.lower()
    assert any(kw in lower for kw in ("cat", "kitten", "feline")), f"Expected response to mention a cat. Got: {text}"


@pytest.fixture(autouse=True)
def _skip_if_no_image() -> None:
    if not _TEST_IMAGE.exists():
        pytest.skip(f"Test image not found at {_TEST_IMAGE}")


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
async def test_claude_reads_image() -> None:
    workdir = _make_workdir("claude")
    try:
        result = await run_claude_agent(_PROMPT, cwd=workdir)
        print(f"\nClaude response: {result.text}")
        _assert_describes_image(result.text)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_codex_reads_image() -> None:
    workdir = _make_workdir("codex")
    try:
        result = await run_codex_agent(_PROMPT, cwd=workdir)
        print(f"\nCodex response: {result.text}")
        _assert_describes_image(result.text)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
async def test_gemini_reads_image() -> None:
    workdir = _make_workdir("gemini")
    try:
        result = await run_gemini_agent(_PROMPT, cwd=workdir)
        print(f"\nGemini response: {result.text}")
        _assert_describes_image(result.text)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
