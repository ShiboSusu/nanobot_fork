from __future__ import annotations

import json
from pathlib import Path

import pytest

from opengui.action import Action
from opengui.agent import GuiAgent
from opengui.autonomy_monitor import (
    AutonomyDecisionKind,
    AutonomyMonitor,
    StepMonitorInput,
)
from opengui.interfaces import LLMResponse, ToolCall
from opengui.observation import Observation
from opengui.trajectory.recorder import TrajectoryRecorder


def _observation(path: Path, *, app: str = "Settings", data: bytes = b"screen") -> Observation:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return Observation(
        screenshot_path=str(path),
        screen_width=390,
        screen_height=844,
        foreground_app=app,
        platform="ios",
    )


def test_monitor_accumulates_unchanged_repeated_action_risk(tmp_path: Path) -> None:
    monitor = AutonomyMonitor(horizon_threshold=0.50)
    action = Action(action_type="tap", x=100, y=200)
    first = _observation(tmp_path / "first.png")
    second = _observation(tmp_path / "second.png")
    third = _observation(tmp_path / "third.png")

    first_decision = monitor.assess_step(
        StepMonitorInput(
            task="打开设置",
            step_index=1,
            max_steps=5,
            action=action,
            current_observation=first,
            next_observation=second,
            tool_result="ok",
            action_summary="tap Settings",
        )
    )
    second_decision = monitor.assess_step(
        StepMonitorInput(
            task="打开设置",
            step_index=2,
            max_steps=5,
            action=action,
            current_observation=second,
            next_observation=third,
            tool_result="ok",
            action_summary="tap Settings again",
        )
    )

    assert first_decision.decision == AutonomyDecisionKind.S1_EXECUTE
    assert first_decision.risk > 0
    assert "screen_unchanged" in first_decision.signal_keys
    assert second_decision.decision == AutonomyDecisionKind.HALT
    assert second_decision.cumulative_risk >= 0.50
    assert "repeated_action" in second_decision.signal_keys


def test_monitor_failed_observation_is_red_halt(tmp_path: Path) -> None:
    monitor = AutonomyMonitor()
    current = _observation(tmp_path / "current.png")

    decision = monitor.assess_step(
        StepMonitorInput(
            task="打开设置",
            step_index=1,
            max_steps=5,
            action=Action(action_type="tap", x=100, y=200),
            current_observation=current,
            next_observation=None,
            tool_result="ok (observation failed: timeout)",
            action_summary="tap Settings",
        )
    )

    assert decision.decision == AutonomyDecisionKind.HALT
    assert decision.tier == "red"
    assert "observe_failed" in decision.signal_keys


class _RecordingLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)

    async def chat(self, messages, tools=None, tool_choice=None) -> LLMResponse:
        del messages, tools, tool_choice
        if not self._responses:
            raise AssertionError("No scripted responses left")
        return self._responses.pop(0)


class _StaticScreenBackend:
    platform = "ios"

    def __init__(self) -> None:
        self.execute_calls: list[Action] = []

    async def preflight(self) -> None:
        return None

    async def list_apps(self) -> list[str]:
        return []

    async def execute(self, action: Action, timeout: float = 5.0) -> str:
        del timeout
        self.execute_calls.append(action)
        return "ok"

    async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
        del timeout
        return _observation(screenshot_path, data=b"unchanged-screen")


@pytest.mark.asyncio
async def test_gui_agent_records_monitor_decision_and_stops_on_red(tmp_path: Path) -> None:
    monitor = AutonomyMonitor(horizon_threshold=0.20)
    backend = _StaticScreenBackend()
    llm = _RecordingLLM([
        LLMResponse(
            content="Action: tap Settings",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name="computer_use",
                    arguments={
                        "action_type": "tap",
                        "x": 120,
                        "y": 220,
                        "intent": "tap Settings",
                        "summary": "Settings is visible",
                    },
                )
            ],
        )
    ])
    task = "点击屏幕上的设置图标"
    recorder = TrajectoryRecorder(output_dir=tmp_path / "traj", task=task, platform="ios")
    agent = GuiAgent(
        llm,
        backend,
        trajectory_recorder=recorder,
        artifacts_root=tmp_path / "runs",
        max_steps=3,
        include_date_context=False,
        autonomy_monitor=monitor,
    )

    result = await agent.run(task, max_retries=1)

    assert result.success is False
    assert result.error == "autonomy_monitor_halt"
    assert len(backend.execute_calls) == 1

    trace_events = [
        json.loads(line)
        for line in (Path(result.trace_path) / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    step_event = next(event for event in trace_events if event["event"] == "step")
    assert step_event["autonomy_monitor"]["decision"] == AutonomyDecisionKind.HALT.value
    assert "screen_unchanged" in step_event["autonomy_monitor"]["signal_keys"]

    trajectory_events = [
        json.loads(line)
        for line in recorder.path.read_text(encoding="utf-8").splitlines()
    ]
    monitor_events = [event for event in trajectory_events if event["type"] == "autonomy_monitor"]
    assert monitor_events[-1]["decision"] == AutonomyDecisionKind.HALT.value
