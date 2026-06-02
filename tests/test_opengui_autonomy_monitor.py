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


def test_monitor_escalates_repeated_wait_on_unchanged_screen(tmp_path: Path) -> None:
    monitor = AutonomyMonitor(horizon_threshold=0.50)
    action = Action(action_type="wait", duration_ms=3000)
    first = _observation(tmp_path / "first.png", app="Bilibili")
    second = _observation(tmp_path / "second.png", app="Bilibili")
    third = _observation(tmp_path / "third.png", app="Bilibili")

    first_decision = monitor.assess_step(
        StepMonitorInput(
            task="在B站播放罗翔的刑法课视频",
            step_index=1,
            max_steps=15,
            action=action,
            current_observation=first,
            next_observation=second,
            tool_result="wait 3000 ms",
            action_summary="wait for app loading launch screen",
            state_summary="Bilibili is still on splash/loading screen",
        )
    )
    second_decision = monitor.assess_step(
        StepMonitorInput(
            task="在B站播放罗翔的刑法课视频",
            step_index=2,
            max_steps=15,
            action=action,
            current_observation=second,
            next_observation=third,
            tool_result="wait 3000 ms",
            action_summary="wait for app loading launch screen again",
            state_summary="Bilibili is still on splash/loading screen",
        )
    )

    assert "wait_no_change" in first_decision.signal_keys
    assert second_decision.decision == AutonomyDecisionKind.HALT
    assert "repeated_wait" in second_decision.signal_keys


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


def test_monitor_expected_app_mismatch_is_red_halt(tmp_path: Path) -> None:
    monitor = AutonomyMonitor()
    current = _observation(tmp_path / "current.png", app="tv.danmaku.bilianime")
    next_observation = _observation(
        tmp_path / "next.png",
        app="com.apple.ScreenshotServicesService",
        data=b"taobao-ad-redirect",
    )

    decision = monitor.assess_step(
        StepMonitorInput(
            task="在B站播放罗翔的刑法课视频",
            step_index=1,
            max_steps=15,
            action=Action(action_type="wait", duration_ms=3000),
            current_observation=current,
            next_observation=next_observation,
            tool_result="wait 3000 ms",
            action_summary="wait for splash ad",
            expected_app="tv.danmaku.bilianime",
        )
    )

    assert decision.decision == AutonomyDecisionKind.HALT
    assert decision.tier == "red"
    assert "app_mismatch" in decision.signal_keys


def test_monitor_does_not_hard_gate_safety_keywords_before_action(tmp_path: Path) -> None:
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

    assert decision.decision == AutonomyDecisionKind.S1_EXECUTE
    assert decision.tier == "green"
    assert "safety_keyword_flag" not in decision.signal_keys


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


class _ChangingScreenBackend:
    platform = "ios"

    def __init__(self) -> None:
        self.execute_calls: list[Action] = []
        self.observe_count = 0

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
        self.observe_count += 1
        return _observation(
            screenshot_path,
            app="tv.danmaku.bilianime",
            data=f"changing-screen-{self.observe_count}".encode(),
        )


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
async def test_gui_agent_requests_s2_once_global_step_budget_reaches_twenty(tmp_path: Path) -> None:
    backend = _ChangingScreenBackend()
    s1_responses = [
        LLMResponse(
            content=f"Action: explore {i}",
            tool_calls=[
                ToolCall(
                    id=f"call-{i}",
                    name="computer_use",
                    arguments={
                        "action_type": "tap",
                        "x": 10 + i,
                        "y": 20 + i,
                        "summary": f"继续查找设置入口 {i}",
                    },
                )
            ],
        )
        for i in range(30)
    ]
    s1_responses.append(
        LLMResponse(
            content="Action: done",
            tool_calls=[
                ToolCall(
                    id="call-done",
                    name="computer_use",
                    arguments={
                        "action_type": "done",
                        "status": "success",
                        "summary": "按S2提示确认任务完成",
                    },
                )
            ],
        )
    )
    s1 = _RecordingLLM(s1_responses)
    s2 = _RecordingLLM([
        LLMResponse(content='{"route":"S2_HINT","hint":"停止继续盲目查找，先进入设置/隐私路线。"}')
    ])
    task = "检查B站里我的关注列表设置是不是‘不公开’。"
    recorder = TrajectoryRecorder(output_dir=tmp_path / "traj", task=task, platform="ios")
    agent = GuiAgent(
        s1,
        backend,
        trajectory_recorder=recorder,
        artifacts_root=tmp_path / "runs",
        max_steps=15,
        include_date_context=False,
        s2_llm=s2,
        s2_model="qwen3.5-397b-a17b",
        s2_max_hints=1,
    )

    await agent.run(task, max_retries=2, app_hint="tv.danmaku.bilianime")

    assert len(s2.calls) == 1
    trace_events = [
        json.loads(line)
        for line in recorder.path.read_text(encoding="utf-8").splitlines()
    ]
    s2_event = next(event for event in trace_events if event["type"] == "s2_guidance")
    assert 20 <= s2_event["at_step"] <= 21
    assert s2_event["monitor"]["signal_keys"] == ["global_step_budget_pressure"]


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
    assert monitor.cumulative_risk < 0.20
    second_prompt = json.dumps(s1.calls[1], ensure_ascii=False)
    assert "Do not repeat the tap; verify the current page first." in second_prompt

    trace_events = [
        json.loads(line)
        for line in (Path(result.trace_path) / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    first_step = next(event for event in trace_events if event["event"] == "step")
    assert first_step["autonomy_monitor"]["decision"] == AutonomyDecisionKind.CHEAP_VERIFY.value


@pytest.mark.asyncio
async def test_gui_agent_resets_stagnation_after_s2_recovery(tmp_path: Path) -> None:
    class _RecoveringBackend:
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
            if screenshot_path.name in {"step_002.png", "step_002_reobserve_1.png"}:
                return _observation(
                    screenshot_path,
                    app="com.apple.springboard",
                    data=b"home-screen",
                )
            return _observation(
                screenshot_path,
                app="com.xingin.discover",
                data=b"collection-list",
            )

    backend = _RecoveringBackend()
    s1 = _RecordingLLM([
        LLMResponse(
            content="Action: tap first note",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name="computer_use",
                    arguments={
                        "action_type": "tap",
                        "x": 500,
                        "y": 756,
                        "summary": "点击收藏列表中的第一篇笔记",
                    },
                )
            ],
        ),
        LLMResponse(
            content="Action: wait",
            tool_calls=[
                ToolCall(
                    id="call-2",
                    name="computer_use",
                    arguments={
                        "action_type": "wait",
                        "summary": "等待页面恢复",
                    },
                )
            ],
        ),
        LLMResponse(
            content="Action: tap Xiaohongshu",
            tool_calls=[
                ToolCall(
                    id="call-3",
                    name="computer_use",
                    arguments={
                        "action_type": "tap",
                        "x": 500,
                        "y": 500,
                        "summary": "重新打开小红书应用",
                    },
                )
            ],
        ),
        LLMResponse(
            content="Action: done",
            tool_calls=[
                ToolCall(
                    id="call-4",
                    name="computer_use",
                    arguments={
                        "action_type": "done",
                        "status": "success",
                        "summary": "已恢复并继续完成任务",
                    },
                )
            ],
        ),
    ])
    s2 = _RecordingLLM([
        LLMResponse(content='{"route":"S2_HINT","hint":"重新打开小红书后继续，不要因为回到收藏列表就停止。"}')
    ])
    task = "去小红书把我收藏的第一篇笔记取消收藏。"
    recorder = TrajectoryRecorder(output_dir=tmp_path / "traj", task=task, platform="ios")
    agent = GuiAgent(
        s1,
        backend,
        trajectory_recorder=recorder,
        artifacts_root=tmp_path / "runs",
        max_steps=5,
        include_date_context=False,
        stagnation_limit=1,
        s2_llm=s2,
        s2_model="qwen3.5-397b-a17b",
        s2_max_hints=1,
    )

    result = await agent.run(task, max_retries=1, app_hint="com.xingin.discover")

    assert result.success is True
    assert len(s2.calls) == 1
    trace_events = [
        json.loads(line)
        for line in (Path(result.trace_path) / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["event"] == "s2_guidance" for event in trace_events)
    assert not any(event["event"] == "stagnation_detected" for event in trace_events)


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
async def test_gui_agent_pre_action_monitor_does_not_pause_on_safety_keywords(tmp_path: Path) -> None:
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
    assert len(backend.execute_calls) == 1
    assert requests == []

    trace_events = [
        json.loads(line)
        for line in (Path(result.trace_path) / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    first_step = next(event for event in trace_events if event["event"] == "step")
    assert "autonomy_monitor_pre_action" not in first_step["execution"]
