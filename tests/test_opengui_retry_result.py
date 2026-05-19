from __future__ import annotations

from pathlib import Path

import pytest

from opengui.agent import AgentResult, GuiAgent
from opengui.backends.dry_run import DryRunBackend
from opengui.interfaces import LLMResponse
from opengui.trajectory.recorder import TrajectoryRecorder


class _UnusedLLM:
    async def chat(self, *args, **kwargs) -> LLMResponse:  # pragma: no cover
        raise AssertionError("retry aggregation test should not call the LLM")


class _RetryScriptAgent(GuiAgent):
    def __init__(self, *args, scripted_results: list[AgentResult], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._scripted_results = list(scripted_results)

    async def _retrieve_memory(self, task: str) -> str | None:
        return None

    async def _run_once(self, *args, **kwargs) -> AgentResult:
        if not self._scripted_results:
            raise AssertionError("No scripted AgentResult left")
        return self._scripted_results.pop(0)


@pytest.mark.asyncio
async def test_successful_retry_does_not_preserve_previous_error(tmp_path: Path) -> None:
    agent = _RetryScriptAgent(
        _UnusedLLM(),
        DryRunBackend(),
        trajectory_recorder=TrajectoryRecorder(output_dir=tmp_path / "traj", task="retry task"),
        artifacts_root=tmp_path / "runs",
        scripted_results=[
            AgentResult(
                success=False,
                summary="first attempt hit max steps",
                trace_path=str(tmp_path / "runs" / "attempt0"),
                steps_taken=5,
                error="max_steps_exceeded",
            ),
            AgentResult(
                success=True,
                summary="completed on retry",
                model_summary="done",
                trace_path=str(tmp_path / "runs" / "attempt1"),
                steps_taken=2,
                error=None,
            ),
        ],
    )

    result = await agent.run("retry task", max_retries=2)

    assert result.success is True
    assert result.error is None
    assert result.steps_taken == 2
    assert result.trace_path == str(tmp_path / "runs" / "attempt1")
