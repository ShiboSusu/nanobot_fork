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


class _FakeGuiTaskTool(Tool):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

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
        return json.dumps(
            {
                "success": True,
                "summary": "system action completed",
                "steps_taken": 1,
                "error": None,
            }
        )


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
