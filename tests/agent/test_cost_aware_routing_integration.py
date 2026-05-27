from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.base import Tool
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import Config
from nanobot.providers.base import LLMResponse


class _FakeGuiTaskTool(Tool):
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._payload = payload or {
            "success": True,
            "summary": "system action completed",
            "steps_taken": 1,
            "error": None,
        }

    @property
    def name(self) -> str:
        return "gui_task"

    @property
    def description(self) -> str:
        return "fake gui task"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"task": {"type": "string"}},
            "required": ["task"],
        }

    async def execute(self, **kwargs: Any) -> str:
        self.calls.append(dict(kwargs))
        return json.dumps(self._payload)


def _make_loop(tmp_path: Path) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = SimpleNamespace(max_tokens=4096)
    provider.chat_with_retry = AsyncMock()
    return AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
    )


def _make_loop_with_planner(tmp_path: Path, planner_content: str) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = SimpleNamespace(max_tokens=4096)
    provider.chat_with_retry = AsyncMock()
    planner_provider = MagicMock()
    planner_provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content=planner_content))
    planner_provider.generation = SimpleNamespace(max_tokens=4096)
    return AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        gui_config=Config(gui={
            "backend": "dry-run",
            "plannerEnabled": True,
            "plannerConfidenceThreshold": 0.65,
        }).gui,
        planner_provider=planner_provider,
        planner_model="planner-model",
    )


@pytest.mark.asyncio
async def test_cli_and_telegram_sensitive_tasks_use_same_policy_route(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)

    contents = []
    for channel in ("cli", "telegram"):
        response = await loop._process_message(
            InboundMessage(
                channel=channel,
                sender_id="u1",
                chat_id=f"{channel}-chat",
                content="帮我登录支付宝并完成付款",
            )
        )
        assert response is not None
        contents.append(response.content)

    assert contents[0] == contents[1]
    assert "需要你确认或接管" in contents[0]
    loop.provider.chat_with_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_policy_route_blocks_before_35b_planner(tmp_path: Path) -> None:
    loop = _make_loop_with_planner(
        tmp_path,
        '{"route":"gui_task","confidence":0.99,"reason":"bad","subtasks":[]}',
    )

    response = await loop._process_message(
        InboundMessage(
            channel="cli",
            sender_id="u1",
            chat_id="cli-chat",
            content="帮我登录支付宝并完成付款",
        )
    )

    assert response is not None
    assert "需要你确认或接管" in response.content
    loop._main_planner.provider.chat_with_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_cli_and_telegram_system_actions_use_same_gui_system_route(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    gui_tool = _FakeGuiTaskTool()
    loop.tools.register(gui_tool)

    contents = []
    for channel in ("cli", "telegram"):
        response = await loop._process_message(
            InboundMessage(
                channel=channel,
                sender_id="u1",
                chat_id=f"{channel}-chat",
                content="帮我用手机打开设置",
            )
        )
        assert response is not None
        contents.append(response.content)

    assert contents[0] == contents[1]
    assert "system action completed" in contents[0]
    assert gui_tool.calls == [
        {"task": "帮我用手机打开设置"},
        {"task": "帮我用手机打开设置"},
    ]
    loop.provider.chat_with_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_query_route_hint_is_injected_for_cli_and_telegram(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    loop._run_agent_loop = AsyncMock(return_value=("天气结果", [], [], "stop", False))  # type: ignore[method-assign]

    for channel in ("cli", "telegram"):
        response = await loop._process_message(
            InboundMessage(
                channel=channel,
                sender_id="u1",
                chat_id=f"{channel}-chat",
                content="查一下今天深圳天气",
            )
        )
        assert response is not None

    calls = loop._run_agent_loop.await_args_list  # type: ignore[attr-defined]
    assert len(calls) == 2
    for call in calls:
        initial_messages = call.args[0]
        user_message = initial_messages[-1]["content"]
        assert "Cost-Aware Router" in user_message
        assert "Recommended route: tool_call" in user_message
        assert "Prefer web_search/web_fetch" in user_message


@pytest.mark.asyncio
async def test_35b_planner_gui_route_executes_gui_task_directly(tmp_path: Path) -> None:
    loop = _make_loop_with_planner(
        tmp_path,
        '{"route":"gui_task","confidence":0.92,"reason":"App internal task",'
        '"subtasks":[{"route":"gui_task","task":"在抖音里搜索旅行Vlog"}]}',
    )
    gui_tool = _FakeGuiTaskTool()
    loop.tools.register(gui_tool)

    response = await loop._process_message(
        InboundMessage(
            channel="cli",
            sender_id="u1",
            chat_id="cli-chat",
            content="在抖音里搜索旅行Vlog",
        )
    )

    assert response is not None
    assert "system action completed" in response.content
    assert gui_tool.calls == [{"task": "在抖音里搜索旅行Vlog"}]
    loop.provider.chat_with_retry.assert_not_awaited()
    loop._main_planner.provider.chat_with_retry.assert_awaited_once()


@pytest.mark.asyncio
async def test_35b_planner_subtask_queue_executes_system_then_gui(tmp_path: Path) -> None:
    loop = _make_loop_with_planner(
        tmp_path,
        '{"route":"plan","confidence":0.92,"reason":"App playback",'
        '"subtasks":['
        '{"id":"open_bilibili","route":"system_action","task":"Open Bilibili","system_action":"open_app"},'
        '{"id":"search_video","route":"gui_task","task":"Search Bilibili for 罗翔 刑法课"}'
        "]}",
    )
    assert loop._gui_config is not None
    loop._gui_config.planner_subtasks_enabled = True
    gui_tool = _FakeGuiTaskTool()
    loop.tools.register(gui_tool)

    response = await loop._process_message(
        InboundMessage(
            channel="cli",
            sender_id="u1",
            chat_id="cli-chat",
            content="在B站播放罗翔的刑法课视频",
        )
    )

    assert response is not None
    assert "2/2 subtasks completed" in response.content
    assert gui_tool.calls == [
        {"task": "Open Bilibili"},
        {"task": "Search Bilibili for 罗翔 刑法课"},
    ]


@pytest.mark.asyncio
async def test_planner_subtask_policy_block_stops_before_gui(tmp_path: Path) -> None:
    loop = _make_loop_with_planner(
        tmp_path,
        '{"route":"plan","confidence":0.92,"reason":"financial",'
        '"subtasks":[{"id":"credit","route":"gui_task","task":"查看京东金融白条额度"}]}',
    )
    assert loop._gui_config is not None
    loop._gui_config.planner_subtasks_enabled = True
    gui_tool = _FakeGuiTaskTool()
    loop.tools.register(gui_tool)

    response = await loop._process_message(
        InboundMessage(
            channel="cli",
            sender_id="u1",
            chat_id="cli-chat",
            content="继续执行刚才那个账户信息检查计划",
        )
    )

    assert response is not None
    assert (
        "需要你确认或接管" in response.content
        or "requires human confirmation" in response.content
        or "blocked by policy" in response.content
    )
    assert gui_tool.calls == []


@pytest.mark.asyncio
async def test_planner_subtasks_enabled_tool_plan_uses_route_hint(tmp_path: Path) -> None:
    loop = _make_loop_with_planner(
        tmp_path,
        '{"route":"plan","confidence":0.9,"reason":"Public lookup",'
        '"subtasks":[{"route":"web_search","tool":"web_search","task":"查询深圳天气"}]}',
    )
    assert loop._gui_config is not None
    loop._gui_config.planner_subtasks_enabled = True
    gui_tool = _FakeGuiTaskTool()
    loop.tools.register(gui_tool)
    loop._run_agent_loop = AsyncMock(return_value=("天气结果", [], [], "stop", False))  # type: ignore[method-assign]

    response = await loop._process_message(
        InboundMessage(
            channel="cli",
            sender_id="u1",
            chat_id="cli-chat",
            content="查询一下今天深圳天气",
        )
    )

    assert response is not None
    assert response.content == "天气结果"
    initial_messages = loop._run_agent_loop.await_args.args[0]  # type: ignore[attr-defined]
    user_message = initial_messages[-1]["content"]
    assert "Cost-Aware Router" in user_message
    assert "Recommended route: tool_call" in user_message
    assert gui_tool.calls == []
    loop._main_planner.provider.chat_with_retry.assert_awaited_once()


@pytest.mark.asyncio
async def test_planner_subtask_gui_failure_blocks_plan(tmp_path: Path) -> None:
    loop = _make_loop_with_planner(
        tmp_path,
        '{"route":"plan","confidence":0.92,"reason":"App playback",'
        '"subtasks":[{"id":"open_bilibili","route":"gui_task","task":"Open Bilibili"}]}',
    )
    assert loop._gui_config is not None
    loop._gui_config.planner_subtasks_enabled = True
    gui_tool = _FakeGuiTaskTool(payload={
        "success": False,
        "summary": "",
        "steps_taken": 0,
        "error": "wda down",
    })
    loop.tools.register(gui_tool)

    response = await loop._process_message(
        InboundMessage(
            channel="cli",
            sender_id="u1",
            chat_id="cli-chat",
            content="在B站播放罗翔的刑法课视频",
        )
    )

    assert response is not None
    assert "subtask_failed" in response.content
    assert "1/1 subtasks completed" not in response.content
    assert gui_tool.calls == [{"task": "Open Bilibili"}]


@pytest.mark.asyncio
async def test_35b_planner_tool_route_injects_route_hint(tmp_path: Path) -> None:
    loop = _make_loop_with_planner(
        tmp_path,
        '{"route":"tool_call","confidence":0.9,"reason":"Public lookup",'
        '"subtasks":[{"route":"web_search","tool":"web_search","task":"查询深圳天气"}]}',
    )
    loop._run_agent_loop = AsyncMock(return_value=("天气结果", [], [], "stop", False))  # type: ignore[method-assign]

    response = await loop._process_message(
        InboundMessage(
            channel="cli",
            sender_id="u1",
            chat_id="cli-chat",
            content="查询一下今天深圳天气",
        )
    )

    assert response is not None
    initial_messages = loop._run_agent_loop.await_args.args[0]  # type: ignore[attr-defined]
    user_message = initial_messages[-1]["content"]
    assert "Cost-Aware Router" in user_message
    assert "Recommended route: tool_call" in user_message
    loop._main_planner.provider.chat_with_retry.assert_awaited_once()


@pytest.mark.asyncio
async def test_malformed_35b_planner_output_falls_back_to_deterministic_system_route(
    tmp_path: Path,
) -> None:
    loop = _make_loop_with_planner(tmp_path, "not json")
    gui_tool = _FakeGuiTaskTool()
    loop.tools.register(gui_tool)

    response = await loop._process_message(
        InboundMessage(
            channel="cli",
            sender_id="u1",
            chat_id="cli-chat",
            content="打开设置",
        )
    )

    assert response is not None
    assert "system action completed" in response.content
    assert gui_tool.calls == [{"task": "打开设置"}]
    loop._main_planner.provider.chat_with_retry.assert_awaited_once()


@pytest.mark.asyncio
async def test_low_confidence_35b_planner_output_falls_back_to_deterministic_tool_route(
    tmp_path: Path,
) -> None:
    loop = _make_loop_with_planner(
        tmp_path,
        '{"route":"gui_task","confidence":0.1,"reason":"not sure","subtasks":[]}',
    )
    loop._run_agent_loop = AsyncMock(return_value=("天气结果", [], [], "stop", False))  # type: ignore[method-assign]

    response = await loop._process_message(
        InboundMessage(
            channel="cli",
            sender_id="u1",
            chat_id="cli-chat",
            content="查询一下今天深圳天气",
        )
    )

    assert response is not None
    initial_messages = loop._run_agent_loop.await_args.args[0]  # type: ignore[attr-defined]
    user_message = initial_messages[-1]["content"]
    assert "Cost-Aware Router" in user_message
    assert "Recommended route: tool_call" in user_message
    loop._main_planner.provider.chat_with_retry.assert_awaited_once()
