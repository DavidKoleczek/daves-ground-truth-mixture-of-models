"""Agent runner utilities for HLE experiments."""

import dataclasses
from dataclasses import dataclass
from pathlib import Path
import subprocess
import time
from typing import Any, Literal

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, SystemMessage, UserMessage, query
from codex_app_server_sdk import CodexClient, ThreadConfig, TurnOverrides
from codex_app_server_sdk.models import WorkspaceWriteSandboxPolicy

from model_diversity.gemini import query_gemini

# Per-million-token pricing (standard tier).
# Gemini uses the <=200k prompt pricing since our prompts are short.
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-5.4": {
        "input": 2.50,
        "cached_input": 0.25,
        "output": 15.00,
    },
    "gemini-3.1-pro-preview": {
        "input": 2.00,
        "cached_input": 0.20,
        "output": 12.00,
    },
}


def _compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int,
) -> float | None:
    """Compute USD cost from token counts using published pricing.

    Cached input tokens are a subset of input_tokens charged at a lower rate.
    """
    pricing = _MODEL_PRICING.get(model)
    if pricing is None:
        return None
    non_cached = input_tokens - cached_input_tokens
    return (
        non_cached * pricing["input"] / 1_000_000
        + cached_input_tokens * pricing["cached_input"] / 1_000_000
        + output_tokens * pricing["output"] / 1_000_000
    )


@dataclass(frozen=True)
class AgentMetrics:
    """Token, cost, and timing metrics from a single agent execution."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    total_cost_usd: float | None = None
    duration_ms: int | None = None


@dataclass(frozen=True)
class AgentResult:
    text: str
    log_entries: list[dict[str, Any]]
    metrics: AgentMetrics


@dataclass(frozen=True)
class TeamMember:
    provider: Literal["claude", "codex", "gemini"]
    persona: str | None = None


def _claude_message_to_dict(
    message: UserMessage | AssistantMessage | SystemMessage | ResultMessage,
) -> dict[str, Any]:
    entry: dict[str, Any] = {"type": type(message).__name__}
    entry.update(dataclasses.asdict(message))
    return entry


def _ensure_git_boundary(cwd: Path) -> None:
    """Initialize a git repo so Claude Code treats cwd as the project root
    instead of walking up to a parent repository.
    """
    if (cwd / ".git").exists():
        return
    subprocess.run(["git", "init"], cwd=cwd, check=True, capture_output=True)


async def run_claude_agent(prompt: str, cwd: Path) -> AgentResult:
    _ensure_git_boundary(cwd)
    stderr_lines: list[str] = []
    options = ClaudeAgentOptions(
        model="claude-opus-4-6",
        effort="medium",
        disallowed_tools=["WebSearch", "WebFetch"],
        permission_mode="bypassPermissions",
        cwd=str(cwd),
        stderr=stderr_lines.append,
    )
    log_entries: list[dict[str, Any]] = []
    result: ResultMessage | None = None
    input_tokens = 0
    output_tokens = 0
    cached_input_tokens = 0
    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, UserMessage | AssistantMessage | SystemMessage | ResultMessage):
                log_entries.append(_claude_message_to_dict(message))
            if isinstance(message, AssistantMessage) and message.usage:
                # Claude API reports input_tokens as non-cached only;
                # total input = input_tokens + cache_read + cache_creation
                cached_read = message.usage.get("cache_read_input_tokens", 0)
                input_tokens += (
                    message.usage.get("input_tokens", 0)
                    + cached_read
                    + message.usage.get("cache_creation_input_tokens", 0)
                )
                output_tokens += message.usage.get("output_tokens", 0)
                cached_input_tokens += cached_read
            if isinstance(message, ResultMessage):
                result = message
    except Exception as exc:
        stderr_detail = "\n".join(stderr_lines) if stderr_lines else "no stderr captured"
        raise RuntimeError(f"Claude Code failed: {exc}\nstderr: {stderr_detail}") from exc

    if result is None:
        stderr_detail = "\n".join(stderr_lines) if stderr_lines else "no stderr captured"
        raise RuntimeError(f"No ResultMessage received from Claude Code\nstderr: {stderr_detail}")
    if result.is_error:
        stderr_detail = "\n".join(stderr_lines) if stderr_lines else "no stderr captured"
        raise RuntimeError(f"Claude Code error: {result.errors}\nstderr: {stderr_detail}")

    metrics = AgentMetrics(
        input_tokens=input_tokens or None,
        output_tokens=output_tokens or None,
        cached_input_tokens=cached_input_tokens or None,
        total_cost_usd=result.total_cost_usd,
        duration_ms=result.duration_ms,
    )
    return AgentResult(text=result.result or "", log_entries=log_entries, metrics=metrics)


_CODEX_MAX_RETRIES = 3


async def run_codex_agent(prompt: str, cwd: Path, *, network_access: bool = False) -> AgentResult:
    start = time.monotonic()
    last_exc: Exception | None = None
    for attempt in range(1, _CODEX_MAX_RETRIES + 1):
        try:
            async with CodexClient.connect_stdio() as client:
                thread = await client.start_thread(
                    ThreadConfig(
                        model="gpt-5.4",
                        approval_policy="never",
                        sandbox="workspace-write",
                        cwd=str(cwd),
                    )
                )
                chat_result = await client.chat_once(
                    text=prompt,
                    thread_id=thread.thread_id,
                    turn_overrides=TurnOverrides(
                        effort="medium",
                        sandbox_policy=WorkspaceWriteSandboxPolicy(
                            type="workspaceWrite",
                            networkAccess=network_access,
                        ),
                    ),
                )
            break
        except Exception as exc:
            last_exc = exc
            if attempt < _CODEX_MAX_RETRIES:
                print(f"Codex attempt {attempt} failed ({exc}), retrying...")
            else:
                raise last_exc from None
    elapsed_ms = int((time.monotonic() - start) * 1000)

    # Extract cumulative token counts from the last tokenUsage event
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    for event in chat_result.raw_events:
        if event.get("method") == "thread/tokenUsage/updated":
            totals = event.get("params", {}).get("tokenUsage", {}).get("total", {})
            input_tokens = totals.get("inputTokens")
            output_tokens = totals.get("outputTokens")
            cached_input_tokens = totals.get("cachedInputTokens")

    cost = _compute_cost("gpt-5.4", input_tokens or 0, output_tokens or 0, cached_input_tokens or 0)

    metrics = AgentMetrics(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        total_cost_usd=cost,
        duration_ms=elapsed_ms,
    )
    return AgentResult(text=chat_result.final_text, log_entries=chat_result.raw_events, metrics=metrics)


async def run_gemini_agent(prompt: str, cwd: Path) -> AgentResult:
    start = time.monotonic()
    result = await query_gemini(
        prompt=prompt,
        model="gemini-3.1-pro-preview",
        cwd=str(cwd),
        approval_mode="yolo",
    )
    elapsed_ms = int((time.monotonic() - start) * 1000)

    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    if result.stats:
        for model_stats in result.stats.get("models", {}).values():
            tokens = model_stats.get("tokens", {})
            # "prompt" is total input (non-cached + cached); "candidates" + "thoughts" is total output
            input_tokens = (input_tokens or 0) + tokens.get("prompt", 0)
            output_tokens = (output_tokens or 0) + tokens.get("candidates", 0) + tokens.get("thoughts", 0)
            cached_input_tokens = (cached_input_tokens or 0) + tokens.get("cached", 0)

    cost = _compute_cost("gemini-3.1-pro-preview", input_tokens or 0, output_tokens or 0, cached_input_tokens or 0)

    metrics = AgentMetrics(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        total_cost_usd=cost,
        duration_ms=elapsed_ms,
    )
    return AgentResult(
        text=result.text,
        log_entries=[{"type": "GeminiResult", "text": result.text, "stats": result.stats}],
        metrics=metrics,
    )


async def run_agent(member: TeamMember, prompt: str, cwd: Path, *, network_access: bool = False) -> AgentResult:
    try:
        if member.provider == "claude":
            return await run_claude_agent(prompt, cwd)
        if member.provider == "codex":
            return await run_codex_agent(prompt, cwd, network_access=network_access)
        if member.provider == "gemini":
            return await run_gemini_agent(prompt, cwd)
        raise ValueError(f"Unknown provider: {member.provider}")
    except Exception as exc:
        print(f"Agent ({member.provider}) failed: {exc}")
        return AgentResult(text="", log_entries=[{"type": "error", "error": str(exc)}], metrics=AgentMetrics())
