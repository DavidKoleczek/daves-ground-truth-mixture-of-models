"""Python wrapper for the Gemini CLI (installed via npm install -g @google/gemini-cli)."""

import asyncio
from dataclasses import dataclass
import json
import shutil
from typing import Any, Literal


@dataclass(frozen=True)
class GeminiResult:
    """Result from a Gemini CLI agent query."""

    text: str
    stats: dict[str, Any] | None = None


async def query_gemini(
    *,
    prompt: str,
    model: str = "gemini-3.1-pro-preview",
    cwd: str | None = None,
    approval_mode: Literal["default", "auto_edit", "yolo", "plan"] = "plan",
) -> GeminiResult:
    """Query Gemini CLI in headless mode.

    Args:
        prompt: The prompt to send to Gemini.
        model: Gemini model identifier.
        cwd: Working directory for the agent.
        approval_mode: Tool approval mode. "plan" is read-only (no tool execution),
            "yolo" auto-approves all tool use.

    Returns:
        GeminiResult with the agent's text response.

    Raises:
        RuntimeError: If the gemini CLI is not found or returns an error.
    """
    gemini_bin = shutil.which("gemini")
    if gemini_bin is None:
        raise RuntimeError("gemini CLI not found on PATH. Install via: npm install -g @google/gemini-cli")

    # Gemini CLI flags reference: `gemini --help` or
    # https://geminicli.com/docs/cli/cli-reference
    cmd = [
        gemini_bin,
        "-p",
        prompt,  # non-interactive (headless) mode with the given prompt
        "-m",
        model,  # model identifier
        "-o",
        "json",  # structured JSON output with response and stats fields
        "--approval-mode",
        approval_mode,  # controls tool execution permissions
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"gemini CLI failed (exit {proc.returncode}): {stderr.decode()}")

    if not stdout:
        raise RuntimeError(f"gemini CLI produced no output. stderr: {stderr.decode()}")

    result: dict[str, object] = json.loads(stdout.decode())
    response = result.get("response")
    if not isinstance(response, str):
        raise RuntimeError(f"Unexpected gemini CLI output: {result}")

    stats = result.get("stats")
    return GeminiResult(text=response, stats=stats if isinstance(stats, dict) else None)
