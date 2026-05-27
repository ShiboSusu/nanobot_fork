from __future__ import annotations

import json
from pathlib import Path

import pytest

from opengui.action import Action
from opengui.agent import GuiAgent
from opengui.autonomy_monitor import (
    AutonomyDecisionKind,
    AutonomyMonitor,
    PreActionMonitorInput,
    StepMonitorInput,
)
from opengui.interfaces import InterventionResolution, LLMResponse, ToolCall
from opengui.observation import Observation
from opengui.policy import PolicyStore
from opengui.trajectory.recorder import TrajectoryRecorder


def _observation(
    path: Path,
    *,
    app: str = "Settings",
    data: bytes = b"screen",
    extra: dict[str, object] | None = None,
) -> Observation:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return Observation(
        screenshot_path=str(path),
        screen_width=390,
        screen_height=844,
        foreground_app=app,
        platform="ios",
        extra=extra or {},
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


def test_monitor_flags_safety_keywords_before_action(tmp_path: Path) -> None:
    monitor = AutonomyMonitor()
    current = _observation(
        tmp_path / "current.png",
        extra={"visible_text": ["Delete account", "Cancel"]},
    )

    decision = monitor.assess_pre_action(
        PreActionMonitorInput(
            task="清理账号设置",
            step_index=1,
            max_steps=5,
            action=Action(action_type="tap", x=120, y=240),
            current_observation=current,
            action_summary="tap Delete account",
            state_summary="The page shows a destructive Delete account button.",
        )
    )

    assert decision.decision == AutonomyDecisionKind.HUMAN_CONFIRM
    assert decision.tier == "red"
    assert "safety_keyword_flag" in decision.signal_keys


def test_monitor_can_discount_risk_after_s2_recovery(tmp_path: Path) -> None:
    monitor = AutonomyMonitor(horizon_threshold=0.40)
    action = Action(action_type="tap", x=100, y=200)
    first = _observation(tmp_path / "first.png")
    second = _observation(tmp_path / "second.png")
    third = _observation(tmp_path / "third.png", data=b"changed-screen")

    red_decision = monitor.assess_step(
        StepMonitorInput(
            task="打开设置",
            step_index=1,
            max_steps=5,
            action=action,
            current_observation=first,
            next_observation=None,
            tool_result="ok (observation failed: timeout)",
            action_summary="tap Settings",
        )
    )
    monitor.mark_s2_guidance_issued()
    next_decision = monitor.assess_step(
        StepMonitorInput(
            task="打开设置",
            step_index=2,
            max_steps=5,
            action=Action(action_type="wait"),
            current_observation=second,
            next_observation=third,
            tool_result="ok",
            action_summary="wait for Settings to settle",
        )
    )

    assert red_decision.decision == AutonomyDecisionKind.HALT
    assert next_decision.decision == AutonomyDecisionKind.S1_EXECUTE
    assert next_decision.cumulative_risk < monitor.horizon_threshold


class _RecordingLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, object]]] = []

    async def chat(
        self,
        messages,
        tools=None,
        tool_choice=None,
        model=None,
        max_tokens=None,
    ) -> LLMResponse:
        del tools, tool_choice, model, max_tokens
        self.calls.append(messages)
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


@pytest.mark.asyncio
async def test_gui_agent_uses_s2_hint_before_halting_on_monitor_red(tmp_path: Path) -> None:
    monitor = AutonomyMonitor(horizon_threshold=0.20)
    backend = _StaticScreenBackend()
    s1 = _RecordingLLM([
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
        ),
        LLMResponse(
            content="Action: done",
            tool_calls=[
                ToolCall(
                    id="call-2",
                    name="computer_use",
                    arguments={
                        "action_type": "done",
                        "status": "success",
                        "intent": "confirm task is complete after S2 guidance",
                        "summary": "Settings is open",
                    },
                )
            ],
        ),
    ])
    s2 = _RecordingLLM([
        LLMResponse(
            content=(
                "<think>private chain of thought that must not be passed to S1</think>\n"
                '{"route":"S2_HINT","hint":"Verify Settings is already open, then call done if complete.",'
                '"rationale":"hidden verifier notes"}'
            )
        )
    ])
    recorder = TrajectoryRecorder(output_dir=tmp_path / "traj", task="点击屏幕上的设置图标", platform="ios")
    agent = GuiAgent(
        s1,
        backend,
        trajectory_recorder=recorder,
        artifacts_root=tmp_path / "runs",
        max_steps=3,
        include_date_context=False,
        autonomy_monitor=monitor,
        s2_llm=s2,
        s2_model="qwen3.5-397b-a17b",
        s2_max_hints=1,
    )

    result = await agent.run("点击屏幕上的设置图标", max_retries=1)

    assert result.success is True
    assert len(s2.calls) == 1
    assert len(s1.calls) == 2

    second_prompt = json.dumps(s1.calls[1], ensure_ascii=False)
    assert "System 2 recovery guidance" in second_prompt
    assert "Verify Settings is already open, then call done if complete." in second_prompt
    assert "private chain of thought" not in second_prompt
    assert "hidden verifier notes" not in second_prompt

    s2_prompt = json.dumps(s2.calls[0], ensure_ascii=False)
    assert "Respond only with JSON" in s2_prompt

    trace_events = [
        json.loads(line)
        for line in (Path(result.trace_path) / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    s2_events = [event for event in trace_events if event["event"] == "s2_guidance"]
    assert s2_events
    assert s2_events[0]["hint"] == "Verify Settings is already open, then call done if complete."


@pytest.mark.asyncio
async def test_gui_agent_uses_s2_hint_for_amber_monitor_decision(tmp_path: Path) -> None:
    monitor = AutonomyMonitor(mid_threshold=0.20, red_threshold=0.90, horizon_threshold=0.95)
    backend = _StaticScreenBackend()
    s1 = _RecordingLLM([
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
        ),
        LLMResponse(
            content="Action: done",
            tool_calls=[
                ToolCall(
                    id="call-2",
                    name="computer_use",
                    arguments={
                        "action_type": "done",
                        "status": "success",
                        "intent": "complete after amber guidance",
                        "summary": "Settings is open",
                    },
                )
            ],
        ),
    ])
    s2 = _RecordingLLM([
        LLMResponse(content='{"route":"S2_HINT","hint":"Do not repeat the tap; verify the current page first."}')
    ])
    recorder = TrajectoryRecorder(output_dir=tmp_path / "traj", task="点击屏幕上的设置图标", platform="ios")
    agent = GuiAgent(
        s1,
        backend,
        trajectory_recorder=recorder,
        artifacts_root=tmp_path / "runs",
        max_steps=3,
        include_date_context=False,
        autonomy_monitor=monitor,
        s2_llm=s2,
        s2_model="qwen3.5-397b-a17b",
        s2_max_hints=1,
    )

    result = await agent.run("点击屏幕上的设置图标", max_retries=1)

    assert result.success is True
    assert len(s2.calls) == 1
    assert len(s1.calls) == 2
    assert monitor.cumulative_risk >= 0.20
    second_prompt = json.dumps(s1.calls[1], ensure_ascii=False)
    assert "Do not repeat the tap; verify the current page first." in second_prompt

    trace_events = [
        json.loads(line)
        for line in (Path(result.trace_path) / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    first_step = next(event for event in trace_events if event["event"] == "step")
    assert first_step["autonomy_monitor"]["decision"] == AutonomyDecisionKind.CHEAP_VERIFY.value


@pytest.mark.asyncio
async def test_gui_agent_rejects_unverified_first_step_done(tmp_path: Path) -> None:
    monitor = AutonomyMonitor()
    backend = _StaticScreenBackend()
    s1 = _RecordingLLM([
        LLMResponse(
            content="Action: done",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name="computer_use",
                    arguments={
                        "action_type": "done",
                        "status": "success",
                        "intent": "declare completion immediately",
                        "summary": "The task is complete.",
                    },
                )
            ],
        )
    ])
    task = "打开设置并进入蓝牙"
    recorder = TrajectoryRecorder(output_dir=tmp_path / "traj", task=task, platform="ios")
    agent = GuiAgent(
        s1,
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
    assert backend.execute_calls == []

    trace_events = [
        json.loads(line)
        for line in (Path(result.trace_path) / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    first_step = next(event for event in trace_events if event["event"] == "step")
    assert first_step["autonomy_monitor"]["decision"] == AutonomyDecisionKind.HALT.value
    assert "unverified_done" in first_step["autonomy_monitor"]["signal_keys"]


@pytest.mark.asyncio
async def test_gui_agent_pre_action_monitor_pauses_before_unsafe_action(tmp_path: Path) -> None:
    monitor = AutonomyMonitor()
    backend = _StaticScreenBackend()
    s1 = _RecordingLLM([
        LLMResponse(
            content="Action: tap Delete account",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name="computer_use",
                    arguments={
                        "action_type": "tap",
                        "x": 120,
                        "y": 240,
                        "intent": "tap Delete account",
                        "summary": "The Delete account button is visible.",
                    },
                )
            ],
        ),
        LLMResponse(
            content="Action: done",
            tool_calls=[
                ToolCall(
                    id="call-2",
                    name="computer_use",
                    arguments={
                        "action_type": "done",
                        "status": "success",
                        "intent": "user confirmed the sensitive step manually",
                        "summary": "The task is complete after human confirmation.",
                    },
                )
            ],
        ),
    ])
    requests = []

    class _Handler:
        async def request_intervention(self, request) -> InterventionResolution:
            requests.append(request)
            assert not backend.execute_calls
            return InterventionResolution(resume_confirmed=True, note="confirmed manually")

    task = "清理账号设置"
    recorder = TrajectoryRecorder(output_dir=tmp_path / "traj", task=task, platform="ios")
    agent = GuiAgent(
        s1,
        backend,
        trajectory_recorder=recorder,
        artifacts_root=tmp_path / "runs",
        max_steps=2,
        include_date_context=False,
        autonomy_monitor=monitor,
        intervention_handler=_Handler(),
        policy_store=PolicyStore(rules=()),
    )

    result = await agent.run(task, max_retries=1)

    assert result.success is True
    assert backend.execute_calls == []
    assert len(requests) == 1
    assert "Safety keyword detected" in requests[0].reason

    trace_events = [
        json.loads(line)
        for line in (Path(result.trace_path) / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    first_step = next(event for event in trace_events if event["event"] == "step")
    pre_action = first_step["execution"]["autonomy_monitor_pre_action"]
    assert pre_action["decision"] == AutonomyDecisionKind.HUMAN_CONFIRM.value
    assert "safety_keyword_flag" in pre_action["signal_keys"]
