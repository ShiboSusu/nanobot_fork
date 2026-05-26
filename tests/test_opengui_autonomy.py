from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from opengui.action import Action
from opengui.agent import GuiAgent
from opengui.autonomy import (
    AutonomyDecisionType,
    AutonomyMonitor,
    AutonomyMonitorConfig,
    AutonomySignal,
)
from opengui.backends.dry_run import DryRunBackend
from opengui.interfaces import LLMResponse, ToolCall
from opengui.observation import Observation
from opengui.policy import PolicyAction, PolicyDecision
from opengui.trajectory.recorder import TrajectoryRecorder


def _make_recorder(tmp_path: Path, task: str = "autonomy test") -> TrajectoryRecorder:
    return TrajectoryRecorder(output_dir=tmp_path / "traj", task=task)


class _ScriptedLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    async def chat(self, messages, tools=None, tool_choice=None) -> LLMResponse:
        del tools, tool_choice
        self.calls.append(copy.deepcopy(messages))
        if not self._responses:
            raise AssertionError("No scripted responses left.")
        return self._responses.pop(0)


class _StaticScreenshotBackend(DryRunBackend):
    async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
        del timeout
        from PIL import Image

        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color=(90, 90, 90)).save(
            screenshot_path,
            format="PNG",
        )
        return Observation(
            screenshot_path=str(screenshot_path),
            screen_width=64,
            screen_height=64,
            foreground_app="DryRun",
            platform=self.platform,
        )


class _ChangingScreenshotBackend(DryRunBackend):
    def __init__(self) -> None:
        super().__init__()
        self._observe_calls = 0

    async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
        del timeout
        from PIL import Image

        self._observe_calls += 1
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        color = (30 * self._observe_calls) % 255
        Image.new("RGB", (64, 64), color=(color, 90, 180)).save(
            screenshot_path,
            format="PNG",
        )
        return Observation(
            screenshot_path=str(screenshot_path),
            screen_width=64,
            screen_height=64,
            foreground_app="DryRun",
            platform=self.platform,
        )


def _wait_response(index: int) -> LLMResponse:
    return LLMResponse(
        content=f"wait {index}",
        tool_calls=[ToolCall(
            id=f"call-{index}",
            name="computer_use",
            arguments={
                "action_type": "wait",
                "duration_ms": 1,
                "intent": "wait for visible progress",
                "summary": "screen has not visibly progressed",
            },
        )],
    )


def test_autonomy_monitor_executes_low_risk_step() -> None:
    monitor = AutonomyMonitor()

    decision = monitor.evaluate(
        AutonomySignal(
            task="open settings",
            step_index=1,
            max_steps=10,
            action=Action(action_type="tap", x=100, y=200),
        )
    )

    assert decision.decision == AutonomyDecisionType.S1_EXECUTE
    assert decision.risk_score < monitor.config.low_threshold
    assert decision.cumulative_risk < monitor.config.cumulative_risk_budget
    assert decision.signals == ()


def test_autonomy_monitor_escalates_action_and_observe_failures() -> None:
    monitor = AutonomyMonitor()

    decision = monitor.evaluate(
        AutonomySignal(
            task="open settings",
            step_index=1,
            max_steps=10,
            action=Action(action_type="tap", x=100, y=200),
            action_failed=True,
            observe_failed=True,
        )
    )

    assert decision.decision == AutonomyDecisionType.S2_TAKEOVER
    assert decision.risk_score >= monitor.config.high_threshold
    assert "action_failed" in decision.signals
    assert "observe_failed" in decision.signals


def test_autonomy_monitor_tracks_cumulative_risk_budget() -> None:
    monitor = AutonomyMonitor(
        AutonomyMonitorConfig(
            low_threshold=0.30,
            mid_threshold=0.55,
            high_threshold=0.80,
            cumulative_risk_budget=0.45,
        )
    )
    action = Action(action_type="wait", duration_ms=1)

    decisions = [
        monitor.evaluate(
            AutonomySignal(
                task="wait on unchanged screen",
                step_index=index,
                max_steps=10,
                action=action,
                screen_unchanged=True,
                repeated_action=True,
            )
        )
        for index in range(1, 5)
    ]

    assert decisions[-1].decision == AutonomyDecisionType.S2_HINT
    assert "cumulative_risk_over_budget" in decisions[-1].signals
    assert decisions[-1].cumulative_risk > monitor.config.cumulative_risk_budget


def test_autonomy_monitor_respects_safety_policy_signal() -> None:
    monitor = AutonomyMonitor()

    decision = monitor.evaluate(
        AutonomySignal(
            task="approve payment",
            step_index=1,
            max_steps=5,
            action=Action(action_type="tap", x=500, y=500),
            policy_decision=PolicyDecision(
                action=PolicyAction.ASK_HUMAN_CONFIRM,
                categories=("payment_or_purchase",),
                reason="Payment requires confirmation.",
            ),
        )
    )

    assert decision.decision == AutonomyDecisionType.HUMAN_CONFIRM
    assert "safety_risk" in decision.signals
    assert decision.reason == "Payment requires confirmation."


def test_autonomy_monitor_treats_screen_change_as_progress() -> None:
    monitor = AutonomyMonitor()

    decision = monitor.evaluate(
        AutonomySignal(
            task="wait for loading",
            step_index=2,
            max_steps=10,
            action=Action(action_type="wait", duration_ms=1),
            repeated_action=True,
            progress_observed=True,
        )
    )

    assert decision.decision == AutonomyDecisionType.S1_EXECUTE
    assert "repeated_action" not in decision.signals


@pytest.mark.asyncio
async def test_gui_agent_autonomy_monitor_stops_repeated_unchanged_steps(tmp_path: Path) -> None:
    llm = _ScriptedLLM([_wait_response(index) for index in range(1, 8)])
    agent = GuiAgent(
        llm,
        _StaticScreenshotBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "monitor repeated unchanged"),
        artifacts_root=tmp_path / "runs",
        max_steps=7,
        include_date_context=False,
    )

    result = await agent.run("wait until the screen changes", max_retries=1)

    assert not result.success
    assert result.error == "autonomy_monitor_intervention"
    assert result.steps_taken < agent.max_steps
    assert "Autonomy monitor" in result.summary

    assert result.trace_path is not None
    trace_path = Path(result.trace_path) / "trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    autonomy_steps = [
        event["execution"]["autonomy_monitor"]
        for event in events
        if event.get("event") == "step" and "autonomy_monitor" in event["execution"]
    ]
    assert autonomy_steps
    assert autonomy_steps[-1]["decision"] == AutonomyDecisionType.S2_HINT.value
    assert "cumulative_risk_over_budget" in autonomy_steps[-1]["signals"]


@pytest.mark.asyncio
async def test_gui_agent_autonomy_monitor_allows_changed_screens(tmp_path: Path) -> None:
    llm = _ScriptedLLM([
        _wait_response(1),
        _wait_response(2),
        LLMResponse(
            content="done",
            tool_calls=[ToolCall(
                id="call-3",
                name="computer_use",
                arguments={
                    "action_type": "done",
                    "status": "success",
                    "intent": "finish after visible progress",
                    "summary": "task is complete",
                },
            )],
        ),
    ])
    agent = GuiAgent(
        llm,
        _ChangingScreenshotBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "monitor changed screens"),
        artifacts_root=tmp_path / "runs",
        max_steps=5,
        include_date_context=False,
    )

    result = await agent.run("wait through changing screens", max_retries=1)

    assert result.success
    assert result.error is None
