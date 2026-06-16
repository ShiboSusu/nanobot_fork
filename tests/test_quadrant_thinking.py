from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from opengui.action import Action
from opengui.agent import GuiAgent
from opengui.interfaces import LLMResponse
from opengui.observation import Observation
from opengui.trajectory.recorder import TrajectoryRecorder


class _NoopLLM:
    async def chat(self, *args: Any, **kwargs: Any) -> LLMResponse:
        del args, kwargs
        return LLMResponse(content="")


class _StaticBackend:
    platform = "dry-run"

    async def preflight(self) -> None:
        return None

    async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
        del timeout
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color=(90, 90, 90)).save(screenshot_path, format="PNG")
        return Observation(
            screenshot_path=str(screenshot_path),
            screen_width=64,
            screen_height=64,
            foreground_app="DryRun",
            platform=self.platform,
        )

    async def execute(self, action: Action, timeout: float = 5.0) -> str:
        del action, timeout
        return "ok"

    async def list_apps(self) -> list[str]:
        return []


@pytest.mark.parametrize(
    ("s1_effort", "s2_effort", "expect_no_think"),
    [
        ("none", "none", True),
        ("none", "high", False),
        ("high", "none", True),
        ("high", "high", False),
    ],
)
def test_quadrant_reasoning_effort_controls_actors_independently(
    tmp_path: Path,
    s1_effort: str,
    s2_effort: str,
    expect_no_think: bool,
) -> None:
    agent = GuiAgent(
        _NoopLLM(),
        _StaticBackend(),
        trajectory_recorder=TrajectoryRecorder(output_dir=tmp_path / "trajectory", task="quadrant"),
        artifacts_root=tmp_path / "runs",
        s1_reasoning_effort=s1_effort,
        s2_reasoning_effort=s2_effort,
    )

    text = agent._build_s2_takeover_context(
        task="recover",
        trigger_reason="quadrant check",
        stagnation_streak=0,
        stagnation_limit=2,
        recent_history=[],
        s2_guidance_notes=[],
    )

    assert agent._s1_reasoning_effort == s1_effort
    assert agent._s2_reasoning_effort == s2_effort
    assert ("/no_think" in text) is expect_no_think
