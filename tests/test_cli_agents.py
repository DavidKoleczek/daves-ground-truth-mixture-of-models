"""Tests for scripting Claude Code, Codex CLI, and Gemini CLI agents."""

from collections.abc import Callable
import os
from pathlib import Path
import shutil

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
from codex_app_server_sdk import CodexClient, ThreadConfig, TurnOverrides
from codex_app_server_sdk.models import WorkspaceWriteSandboxPolicy
import pytest

from model_diversity.gemini import query_gemini

_FIXTURES_DIR = Path(__file__).parent / "fixtures"

_SEED_FILES = {
    "alpha.txt": "First test file.",
    "beta.txt": "Second test file.",
    "gamma.txt": "Third test file.",
}


_TASK_PROMPT = (
    "Run the bash script at './list_files.sh' to list the files in the current directory. "
    "Then write a file called 'summary.md' in the current directory. "
    "The summary.md file should contain a Markdown heading '# File Summary' "
    "followed by a bullet list of every filename reported by the script."
)


def _setup_workdir(workdir: Path) -> None:
    workdir.mkdir(parents=True)
    for name, content in _SEED_FILES.items():
        (workdir / name).write_text(content)
    shutil.copy2(_FIXTURES_DIR / "list_files.sh", workdir / "list_files.sh")


@pytest.fixture
def create_workdir(tmp_path: Path) -> Callable[[str], Path]:
    """Factory fixture that creates seeded agent working directories outside the git repo."""

    def _factory(agent_name: str) -> Path:
        workdir = tmp_path / f"{agent_name}_work"
        _setup_workdir(workdir)
        return workdir

    return _factory


def _assert_summary_references_seed_files(workdir: Path) -> None:
    summary_path = workdir / "summary.md"
    assert summary_path.exists(), f"Expected summary.md in {workdir}"
    summary_text = summary_path.read_text()
    for filename in _SEED_FILES:
        assert filename in summary_text, f"Expected '{filename}' in summary.md, got: {summary_text}"


def _claude_options(cwd: str) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        model="claude-opus-4-6",
        effort="high",
        disallowed_tools=["WebSearch", "WebFetch"],
        permission_mode="bypassPermissions",
        cwd=cwd,
    )


def _codex_thread_config(cwd: str) -> ThreadConfig:
    return ThreadConfig(
        model="gpt-5.4",
        approval_policy="never",
        sandbox="workspace-write",
        cwd=cwd,
    )


def _codex_turn_overrides() -> TurnOverrides:
    return TurnOverrides(
        effort="xhigh",
        sandbox_policy=WorkspaceWriteSandboxPolicy(
            type="workspaceWrite",
            networkAccess=False,
        ),
    )


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
async def test_claude_code_agent(create_workdir: Callable[[str], Path]) -> None:
    """Verify Claude Code can run a bash script and write a summary file."""
    workdir = create_workdir("claude")
    options = _claude_options(cwd=str(workdir))
    result: ResultMessage | None = None

    async for message in query(prompt=_TASK_PROMPT, options=options):
        if isinstance(message, ResultMessage):
            result = message

    assert result is not None, "Expected a ResultMessage from Claude Code"
    assert not result.is_error, f"Claude Code returned an error: {result.errors}"
    _assert_summary_references_seed_files(workdir)


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_codex_cli_agent(create_workdir: Callable[[str], Path]) -> None:
    """Verify Codex CLI can run a bash script and write a summary file."""
    workdir = create_workdir("codex")

    async with CodexClient.connect_stdio() as client:
        thread = await client.start_thread(_codex_thread_config(cwd=str(workdir)))
        await client.chat_once(
            text=_TASK_PROMPT,
            thread_id=thread.thread_id,
            turn_overrides=_codex_turn_overrides(),
        )

    _assert_summary_references_seed_files(workdir)


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
async def test_gemini_cli_agent(create_workdir: Callable[[str], Path]) -> None:
    """Verify Gemini CLI can run a bash script and write a summary file."""
    workdir = create_workdir("gemini")

    await query_gemini(
        prompt=_TASK_PROMPT,
        model="gemini-3.1-pro-preview",
        cwd=str(workdir),
        approval_mode="yolo",
    )

    _assert_summary_references_seed_files(workdir)
