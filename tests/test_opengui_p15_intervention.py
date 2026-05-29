from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from opengui.action import ActionError, parse_action
from opengui.agent import _COMPUTER_USE_TOOL, GuiAgent
from opengui.backends.dry_run import DryRunBackend
from opengui.interfaces import LLMResponse, ToolCall
from opengui.observation import Observation
from opengui.policy import PolicyStore
from opengui.prompts.system import build_system_prompt
from opengui.trajectory.recorder import TrajectoryRecorder


def _make_recorder(tmp_path: Path, task: str = "test task") -> TrajectoryRecorder:
    return TrajectoryRecorder(output_dir=tmp_path / "traj", task=task, platform="linux")


class _RecordingLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, object]]] = []

    async def chat(self, messages, tools=None, tool_choice=None) -> LLMResponse:
        self.calls.append(json.loads(json.dumps(messages)))
        if not self._responses:
            raise AssertionError("No scripted responses left.")
        return self._responses.pop(0)


class _BackendDouble:
    platform = "linux"

    def __init__(self, observations: list[dict[str, object]]) -> None:
        self._observations = list(observations)
        self.preflight = AsyncMock()
        self.execute = AsyncMock(return_value="backend execute should not run")
        self.list_apps = AsyncMock(return_value=[])
        self.observe = AsyncMock(side_effect=self._observe)

    async def _observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
        del timeout
        if not self._observations:
            raise AssertionError("No scripted observations left.")
        payload = self._observations.pop(0)
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        screenshot_path.write_bytes(b"png")
        return Observation(
            screenshot_path=str(screenshot_path),
            screen_width=1280,
            screen_height=720,
            foreground_app=payload.get("foreground_app"),
            platform=self.platform,
            extra=dict(payload.get("extra", {})),
        )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_parse_action_accepts_request_intervention() -> None:
    action = parse_action({
        "action_type": "request_intervention",
        "text": "Need the user to complete a login challenge.",
    })

    assert action.action_type == "request_intervention"
    assert action.text == "Need the user to complete a login challenge."


def test_request_intervention_requires_reason_text() -> None:
    with pytest.raises(ActionError, match="requires a non-empty 'text' field"):
        parse_action({"action_type": "request_intervention"})

    with pytest.raises(ActionError, match="requires a non-empty 'text' field"):
        parse_action({
            "action_type": "request_intervention",
            "text": "   ",
        })


def test_system_prompt_lists_request_intervention_action() -> None:
    prompt = build_system_prompt()

    assert '"request_intervention"' in prompt
    assert "sensitive, blocked, or unsafe state" in prompt
    assert 'action_type="request_intervention"' in prompt


def test_agent_tool_schema_lists_request_intervention() -> None:
    action_types = _COMPUTER_USE_TOOL["function"]["parameters"]["properties"]["action_type"]["enum"]

    assert "request_intervention" in action_types


@pytest.mark.asyncio
@pytest.mark.pauses_backend_io
async def test_intervention_request_pauses_backend_execute_and_observe(tmp_path: Path) -> None:
    llm = _RecordingLLM([
        LLMResponse(
            content="Action: request intervention for login",
            tool_calls=[ToolCall(
                id="call-1",
                name="computer_use",
                arguments={
                    "action_type": "request_intervention",
                    "text": "Need the user to finish the OTP login challenge.",
                },
            )],
        ),
        LLMResponse(
            content="Action: done",
            tool_calls=[ToolCall(
                id="call-2",
                name="computer_use",
                arguments={"action_type": "done", "status": "success"},
            )],
        ),
    ])
    backend = _BackendDouble([
        {
            "foreground_app": "Secure Login",
            "extra": {"display_id": "bg-desktop-1", "window_title": "Bank Login"},
        },
        {"foreground_app": "Bank Dashboard"},
    ])
    handler = AsyncMock()

    async def _handle(request) -> SimpleNamespace:
        backend.execute.assert_not_awaited()
        assert backend.observe.await_count == 1
        assert request.platform == "linux"
        assert request.foreground_app == "Secure Login"
        assert request.target["display_id"] == "bg-desktop-1"
        return SimpleNamespace(resume_confirmed=True, note="user finished login")

    handler.request_intervention.side_effect = _handle

    agent = GuiAgent(
        llm,
        backend,
        trajectory_recorder=_make_recorder(tmp_path, "secure login"),
        intervention_handler=handler,
        artifacts_root=tmp_path / "runs",
        max_steps=2,
        include_date_context=False,
    )

    result = await agent.run("Complete the secure login", max_retries=1)

    assert result.success
    backend.execute.assert_not_awaited()
    assert backend.observe.await_count == 2
    handler.request_intervention.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.explicit_resume_confirmation
async def test_intervention_waits_for_explicit_resume_confirmation(tmp_path: Path) -> None:
    llm = _RecordingLLM([
        LLMResponse(
            content="Action: request intervention",
            tool_calls=[ToolCall(
                id="call-1",
                name="computer_use",
                arguments={
                    "action_type": "request_intervention",
                    "text": "The user must approve a payment step.",
                },
            )],
        ),
        LLMResponse(
            content="Action: done",
            tool_calls=[ToolCall(
                id="call-2",
                name="computer_use",
                arguments={"action_type": "done", "status": "success"},
            )],
        ),
    ])
    backend = _BackendDouble([
        {"foreground_app": "Payment Approval"},
        {"foreground_app": "Payment Complete"},
    ])
    entered = asyncio.Event()
    release = asyncio.Event()

    class _Handler:
        async def request_intervention(self, request) -> SimpleNamespace:
            assert request.reason == "The user must approve a payment step."
            entered.set()
            await release.wait()
            return SimpleNamespace(resume_confirmed=True, note="approved")

    agent = GuiAgent(
        llm,
        backend,
        trajectory_recorder=_make_recorder(tmp_path, "payment approval"),
        intervention_handler=_Handler(),
        artifacts_root=tmp_path / "runs",
        max_steps=2,
        include_date_context=False,
    )

    task = asyncio.create_task(agent.run("Approve the payment", max_retries=1))
    await asyncio.wait_for(entered.wait(), timeout=1.0)
    await asyncio.sleep(0)

    backend.execute.assert_not_awaited()
    assert backend.observe.await_count == 1
    assert not task.done()

    release.set()
    result = await asyncio.wait_for(task, timeout=1.0)

    assert result.success
    assert backend.observe.await_count == 2


@pytest.mark.asyncio
@pytest.mark.fresh_observation_after_intervention
async def test_resume_uses_fresh_observation_after_intervention(tmp_path: Path) -> None:
    llm = _RecordingLLM([
        LLMResponse(
            content="Action: request intervention",
            tool_calls=[ToolCall(
                id="call-1",
                name="computer_use",
                arguments={
                    "action_type": "request_intervention",
                    "text": "User must complete MFA before continuing.",
                },
            )],
        ),
        LLMResponse(
            content="Action: done",
            tool_calls=[ToolCall(
                id="call-2",
                name="computer_use",
                arguments={"action_type": "done", "status": "success"},
            )],
        ),
    ])
    backend = _BackendDouble([
        {"foreground_app": "Login Screen"},
        {"foreground_app": "Authenticated Workspace"},
    ])

    class _Handler:
        async def request_intervention(self, request) -> SimpleNamespace:
            assert request.step_index == 1
            return SimpleNamespace(resume_confirmed=True, note="mfa complete")

    agent = GuiAgent(
        llm,
        backend,
        trajectory_recorder=_make_recorder(tmp_path, "mfa resume"),
        intervention_handler=_Handler(),
        artifacts_root=tmp_path / "runs",
        max_steps=2,
        include_date_context=False,
    )

    result = await agent.run("Resume after MFA", max_retries=1)

    assert result.success
    assert Path(backend.observe.await_args_list[1].args[0]).name == "step_001.png"
    current_user_message = llm.calls[1][-1]
    current_text = "\n".join(
        block["text"]
        for block in current_user_message["content"]
        if isinstance(block, dict) and block.get("type") == "text"
    )
    assert "Foreground app: Authenticated Workspace" in current_text


@pytest.mark.asyncio
async def test_gui_action_layer_does_not_hard_gate_sensitive_keywords(tmp_path: Path) -> None:
    llm = _RecordingLLM([
        LLMResponse(
            content="Action: tap payment confirmation",
            tool_calls=[ToolCall(
                id="call-1",
                name="computer_use",
                arguments={
                    "action_type": "tap",
                    "x": 500,
                    "y": 640,
                    "intent": "tap the 确认支付 button",
                    "summary": "payment confirmation screen is visible",
                },
            )],
        ),
        LLMResponse(
            content="Action: done",
            tool_calls=[ToolCall(
                id="call-2",
                name="computer_use",
                arguments={"action_type": "done", "status": "success"},
            )],
        ),
    ])
    backend = _BackendDouble([
        {"foreground_app": "Payment"},
        {"foreground_app": "Payment"},
    ])
    handler = SimpleNamespace(
        request_intervention=AsyncMock(return_value=SimpleNamespace(
            resume_confirmed=False,
            note="hard gate should not run",
        ))
    )

    agent = GuiAgent(
        llm,
        backend,
        trajectory_recorder=_make_recorder(tmp_path, "payment gate"),
        intervention_handler=handler,
        artifacts_root=tmp_path / "runs",
        max_steps=2,
        include_date_context=False,
    )

    result = await agent.run("检查订单状态", max_retries=1)

    assert result.success
    backend.execute.assert_awaited_once()
    handler.request_intervention.assert_not_awaited()


@pytest.mark.asyncio
async def test_policy_allows_benign_search_input_text_with_delete_summary(tmp_path: Path) -> None:
    search_summary = (
        "当前屏幕显示搜索框中已输入“runningmen2013”，但任务目标是搜索“罗翔的刑法课”。"
        "因此，需要删除当前错误关键词，并输入正确的搜索词。键盘已弹出，可直接进行文本输入操作。"
    )
    llm = _RecordingLLM([
        LLMResponse(
            content=f"Action: {search_summary}",
            tool_calls=[ToolCall(
                id="call-1",
                name="computer_use",
                arguments={
                    "action_type": "input_text",
                    "text": "罗翔的刑法课",
                    "intent": search_summary,
                    "summary": "搜索框已聚焦，准备输入查询词。",
                },
            )],
        ),
        LLMResponse(
            content="Action: done",
            tool_calls=[ToolCall(
                id="call-2",
                name="computer_use",
                arguments={"action_type": "done", "status": "success"},
            )],
        ),
    ])
    backend = _BackendDouble([
        {"foreground_app": "哔哩哔哩"},
        {"foreground_app": "哔哩哔哩搜索结果"},
    ])
    handler = SimpleNamespace(request_intervention=AsyncMock())

    agent = GuiAgent(
        llm,
        backend,
        trajectory_recorder=_make_recorder(tmp_path, "bilibili search"),
        intervention_handler=handler,
        artifacts_root=tmp_path / "runs",
        max_steps=2,
        include_date_context=False,
    )

    result = await agent.run("在B站播放罗翔的刑法课视频。", max_retries=1)

    assert result.success
    backend.execute.assert_awaited_once()
    handler.request_intervention.assert_not_awaited()


@pytest.mark.asyncio
async def test_policy_allows_navigation_away_from_comment_area(tmp_path: Path) -> None:
    llm = _RecordingLLM([
        LLMResponse(
            content="Action: leave comment area",
            tool_calls=[ToolCall(
                id="call-1",
                name="computer_use",
                arguments={
                    "action_type": "tap",
                    "x": 75,
                    "y": 376,
                    "intent": "点击左上角的返回按钮，离开评论区。",
                    "summary": "当前屏幕显示的是B站的一个视频评论区，需要返回视频页继续任务。",
                },
            )],
        ),
        LLMResponse(
            content="Action: done",
            tool_calls=[ToolCall(
                id="call-2",
                name="computer_use",
                arguments={"action_type": "done", "status": "success"},
            )],
        ),
    ])
    backend = _BackendDouble([
        {"foreground_app": "哔哩哔哩"},
        {"foreground_app": "哔哩哔哩"},
    ])
    handler = SimpleNamespace(
        request_intervention=AsyncMock(return_value=SimpleNamespace(
            resume_confirmed=False,
            note="navigation away should not request approval",
        ))
    )

    agent = GuiAgent(
        llm,
        backend,
        trajectory_recorder=_make_recorder(tmp_path, "leave comment area"),
        intervention_handler=handler,
        artifacts_root=tmp_path / "runs",
        max_steps=2,
        include_date_context=False,
    )

    result = await agent.run("在B站播放罗翔的刑法课视频。", max_retries=1)

    assert result.success
    backend.execute.assert_awaited_once()
    handler.request_intervention.assert_not_awaited()


@pytest.mark.asyncio
async def test_gui_action_layer_does_not_hard_gate_send_comment_tap(tmp_path: Path) -> None:
    llm = _RecordingLLM([
        LLMResponse(
            content="Action: send comment",
            tool_calls=[ToolCall(
                id="call-1",
                name="computer_use",
                arguments={
                    "action_type": "tap",
                    "x": 1180,
                    "y": 640,
                    "intent": "点击发送评论按钮，发表当前评论。",
                    "summary": "评论输入框旁边的发送按钮可见。",
                },
            )],
        ),
        LLMResponse(
            content="Action: done",
            tool_calls=[ToolCall(
                id="call-2",
                name="computer_use",
                arguments={"action_type": "done", "status": "success"},
            )],
        ),
    ])
    backend = _BackendDouble([
        {"foreground_app": "Video App"},
        {"foreground_app": "Video App"},
    ])
    handler = SimpleNamespace(
        request_intervention=AsyncMock(return_value=SimpleNamespace(
            resume_confirmed=False,
            note="hard gate should not run",
        ))
    )

    agent = GuiAgent(
        llm,
        backend,
        trajectory_recorder=_make_recorder(tmp_path, "send comment gate"),
        intervention_handler=handler,
        artifacts_root=tmp_path / "runs",
        max_steps=2,
        include_date_context=False,
    )

    result = await agent.run("给视频发一条评论", max_retries=1)

    assert result.success
    backend.execute.assert_awaited_once()
    handler.request_intervention.assert_not_awaited()


@pytest.mark.asyncio
async def test_gui_action_layer_does_not_hard_gate_message_input_text(tmp_path: Path) -> None:
    llm = _RecordingLLM([
        LLMResponse(
            content="Action: input comment",
            tool_calls=[ToolCall(
                id="call-1",
                name="computer_use",
                arguments={
                    "action_type": "input_text",
                    "text": "这个视频很不错",
                    "intent": "在评论框输入评论内容，下一步将发送评论。",
                    "summary": "评论输入框已聚焦。",
                },
            )],
        ),
        LLMResponse(
            content="Action: done",
            tool_calls=[ToolCall(
                id="call-2",
                name="computer_use",
                arguments={"action_type": "done", "status": "success"},
            )],
        ),
    ])
    backend = _BackendDouble([
        {"foreground_app": "Video App"},
        {"foreground_app": "Video App"},
    ])
    handler = SimpleNamespace(
        request_intervention=AsyncMock(return_value=SimpleNamespace(
            resume_confirmed=False,
            note="hard gate should not run",
        ))
    )

    agent = GuiAgent(
        llm,
        backend,
        trajectory_recorder=_make_recorder(tmp_path, "comment gate"),
        intervention_handler=handler,
        artifacts_root=tmp_path / "runs",
        max_steps=2,
        include_date_context=False,
    )

    result = await agent.run("给视频发一条评论", max_retries=1)

    assert result.success
    backend.execute.assert_awaited_once()
    handler.request_intervention.assert_not_awaited()


@pytest.mark.asyncio
async def test_gui_action_layer_does_not_hard_gate_profile_update_input_text(tmp_path: Path) -> None:
    llm = _RecordingLLM([
        LLMResponse(
            content="Action: input new nickname",
            tool_calls=[ToolCall(
                id="call-1",
                name="computer_use",
                arguments={
                    "action_type": "input_text",
                    "text": "新的昵称",
                    "intent": "在个人资料页输入新的昵称，准备修改昵称。",
                    "summary": "昵称输入框已聚焦。",
                },
            )],
        ),
        LLMResponse(
            content="Action: done",
            tool_calls=[ToolCall(
                id="call-2",
                name="computer_use",
                arguments={"action_type": "done", "status": "success"},
            )],
        ),
    ])
    backend = _BackendDouble([
        {"foreground_app": "Profile"},
        {"foreground_app": "Profile"},
    ])
    handler = SimpleNamespace(
        request_intervention=AsyncMock(return_value=SimpleNamespace(
            resume_confirmed=False,
            note="hard gate should not run",
        ))
    )

    agent = GuiAgent(
        llm,
        backend,
        trajectory_recorder=_make_recorder(tmp_path, "profile gate"),
        intervention_handler=handler,
        artifacts_root=tmp_path / "runs",
        max_steps=2,
        include_date_context=False,
    )

    result = await agent.run("把我的昵称改成新的昵称", max_retries=1)

    assert result.success
    backend.execute.assert_awaited_once()
    handler.request_intervention.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.scrub_sensitive_trace_fields
async def test_trace_and_trajectory_scrub_sensitive_intervention_fields(tmp_path: Path) -> None:
    reason = "Need the user to enter OTP 123456 for the payroll login."
    llm = _RecordingLLM([
        LLMResponse(
            content="Action: request intervention",
            tool_calls=[ToolCall(
                id="call-1",
                name="computer_use",
                arguments={"action_type": "request_intervention", "text": reason},
            )],
        ),
        LLMResponse(
            content="Action: done",
            tool_calls=[ToolCall(
                id="call-2",
                name="computer_use",
                arguments={"action_type": "done", "status": "success"},
            )],
        ),
    ])
    backend = _BackendDouble([
        {
            "foreground_app": "Payroll Login",
            "extra": {"display_id": "desk-2", "session_token": "secret-session-token"},
        },
        {"foreground_app": "Payroll Home"},
    ])
    recorder = _make_recorder(tmp_path, "scrub intervention")

    class _Handler:
        async def request_intervention(self, request) -> SimpleNamespace:
            assert request.target["session_token"] == "secret-session-token"
            return SimpleNamespace(resume_confirmed=True, note="user completed OTP")

    agent = GuiAgent(
        llm,
        backend,
        trajectory_recorder=recorder,
        intervention_handler=_Handler(),
        artifacts_root=tmp_path / "runs",
        max_steps=2,
        include_date_context=False,
    )

    result = await agent.run("Handle the payroll OTP screen", max_retries=1)

    assert result.success
    trace_text = (Path(result.trace_path) / "trace.jsonl").read_text(encoding="utf-8")
    trajectory_text = recorder.path.read_text(encoding="utf-8")

    assert "<redacted:intervention_reason>" in trace_text
    assert "<redacted:intervention_reason>" in trajectory_text
    assert reason not in trace_text
    assert reason not in trajectory_text


@pytest.mark.asyncio
async def test_input_text_is_preserved_in_trace_artifacts(tmp_path: Path) -> None:
    typed_secret = "Sup3rS3cret OTP 123456"
    recorder = _make_recorder(tmp_path, "type secret")
    agent = GuiAgent(
        _RecordingLLM([
            LLMResponse(
                content="Action: input the code",
                tool_calls=[ToolCall(
                    id="call-1",
                    name="computer_use",
                    arguments={"action_type": "input_text", "text": typed_secret},
                )],
            ),
            LLMResponse(
                content="Action: done",
                tool_calls=[ToolCall(
                    id="call-2",
                    name="computer_use",
                    arguments={"action_type": "done", "status": "success"},
                )],
            ),
        ]),
        DryRunBackend(),
        trajectory_recorder=recorder,
        artifacts_root=tmp_path / "runs",
        max_steps=2,
        include_date_context=False,
        policy_store=PolicyStore(rules=()),
    )

    result = await agent.run("Enter the temporary code", max_retries=1)

    assert result.success
    trace_events = _read_jsonl(Path(result.trace_path) / "trace.jsonl")
    trajectory_events = _read_jsonl(recorder.path)
    serialized_trace = json.dumps(trace_events)
    serialized_trajectory = json.dumps(trajectory_events)

    assert typed_secret in serialized_trace
    assert typed_secret in serialized_trajectory
