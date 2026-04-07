"""Tests verifying token, cost, and timing metrics are captured for each agent."""

from collections.abc import Callable, Generator
import os
from pathlib import Path
import shutil

import pytest

from model_diversity.agents import (
    AgentMetrics,
    AgentResult,
    _compute_cost,
    run_claude_agent,
    run_codex_agent,
    run_gemini_agent,
)

_PROJECT_DIR = Path(__file__).parents[1]

_SIMPLE_PROMPT = "Write the text 'hello' to a file called hello.txt in the current directory."


@pytest.fixture
def create_workdir() -> Generator[Callable[[str], Path], None, None]:
    """Factory fixture that creates agent working directories and cleans up after."""
    created: list[Path] = []

    def _factory(agent_name: str) -> Path:
        workdir = _PROJECT_DIR / "tmp" / f"{agent_name}_metrics_test"
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True)
        created.append(workdir)
        return workdir

    yield _factory

    for d in created:
        shutil.rmtree(d, ignore_errors=True)


def _assert_has_duration(metrics: AgentMetrics) -> None:
    assert metrics.duration_ms is not None, "Expected duration_ms to be set"
    assert metrics.duration_ms > 0, f"Expected positive duration, got {metrics.duration_ms}"


# -- Unit tests for cost computation (no API keys needed) --


def test_compute_cost_gpt54() -> None:
    """GPT-5.4: $2.50/1M input, $0.25/1M cached, $15.00/1M output."""
    cost = _compute_cost("gpt-5.4", input_tokens=10_000, output_tokens=1_000, cached_input_tokens=8_000)
    assert cost is not None
    # non_cached = 10_000 - 8_000 = 2_000
    # 2_000 * 2.50/1M + 8_000 * 0.25/1M + 1_000 * 15.00/1M
    expected = 2_000 * 2.50 / 1_000_000 + 8_000 * 0.25 / 1_000_000 + 1_000 * 15.00 / 1_000_000
    assert cost == pytest.approx(expected)


def test_compute_cost_gemini() -> None:
    """Gemini 3.1 Pro: $2.00/1M input, $0.20/1M cached, $12.00/1M output."""
    cost = _compute_cost("gemini-3.1-pro-preview", input_tokens=5_000, output_tokens=2_000, cached_input_tokens=0)
    assert cost is not None
    expected = 5_000 * 2.00 / 1_000_000 + 2_000 * 12.00 / 1_000_000
    assert cost == pytest.approx(expected)


def test_compute_cost_unknown_model() -> None:
    """Unknown models return None."""
    assert _compute_cost("unknown-model", 100, 100, 0) is None


def test_compute_cost_zero_tokens() -> None:
    """Zero tokens should produce zero cost."""
    cost = _compute_cost("gpt-5.4", input_tokens=0, output_tokens=0, cached_input_tokens=0)
    assert cost == pytest.approx(0.0)


def test_compute_cost_all_cached() -> None:
    """When all input tokens are cached, only cached rate applies."""
    cost = _compute_cost("gpt-5.4", input_tokens=10_000, output_tokens=500, cached_input_tokens=10_000)
    assert cost is not None
    expected = 10_000 * 0.25 / 1_000_000 + 500 * 15.00 / 1_000_000
    assert cost == pytest.approx(expected)


# -- Integration tests (require API keys) --


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
async def test_claude_metrics(create_workdir: Callable[[str], Path]) -> None:
    """Claude Agent SDK provides tokens, cost, and duration."""
    workdir = create_workdir("claude")
    result = await run_claude_agent(_SIMPLE_PROMPT, workdir)

    assert isinstance(result, AgentResult)
    m = result.metrics
    assert isinstance(m, AgentMetrics)
    _assert_has_duration(m)
    assert m.input_tokens is not None
    assert m.input_tokens > 0
    assert m.output_tokens is not None
    assert m.output_tokens > 0
    assert m.total_cost_usd is not None
    assert m.total_cost_usd > 0


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_codex_metrics(create_workdir: Callable[[str], Path]) -> None:
    """Codex provides tokens via raw_events; cost computed from pricing table."""
    workdir = create_workdir("codex")
    result = await run_codex_agent(_SIMPLE_PROMPT, workdir)

    assert isinstance(result, AgentResult)
    m = result.metrics
    assert isinstance(m, AgentMetrics)
    _assert_has_duration(m)
    assert m.input_tokens is not None
    assert m.input_tokens > 0
    assert m.output_tokens is not None
    assert m.output_tokens > 0
    assert m.total_cost_usd is not None
    assert m.total_cost_usd > 0
    # Codex frequently caches across turns within a single session
    assert m.cached_input_tokens is not None


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
async def test_gemini_metrics(create_workdir: Callable[[str], Path]) -> None:
    """Gemini CLI provides tokens via stats; cost computed from pricing table."""
    workdir = create_workdir("gemini")
    result = await run_gemini_agent(_SIMPLE_PROMPT, workdir)

    assert isinstance(result, AgentResult)
    m = result.metrics
    assert isinstance(m, AgentMetrics)
    _assert_has_duration(m)
    assert m.input_tokens is not None
    assert m.input_tokens > 0
    assert m.output_tokens is not None
    assert m.output_tokens > 0
    assert m.total_cost_usd is not None
    assert m.total_cost_usd > 0
