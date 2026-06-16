from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from nanobot.agent.runner import AgentRunSpec, AgentRunner
from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.schema import IntegerSchema, StringSchema, tool_parameters_schema
from nanobot.config.schema import AgentDefaults
from nanobot.providers.base import LLMResponse, ToolCallRequest


@tool_parameters(
    tool_parameters_schema(
        url=StringSchema("URL"),
        required=["url"],
    )
)
class _FakeWebFetchTool(Tool):
    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "fake web fetch"

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, url: str, **_: Any) -> str:
        self.calls.append(url)
        return json.dumps(
            {
                "url": url,
                "source": url,
                "raw_length": 5000,
                "returned_length": 1800,
                "title": "Fetched page",
                "relevant_points": ["useful point"],
            },
            ensure_ascii=False,
        )


@tool_parameters(
    tool_parameters_schema(
        query=StringSchema("Search query"),
        count=IntegerSchema(1, description="Results", minimum=1, maximum=10),
        required=["query"],
    )
)
class _FakeWebSearchTool(Tool):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "fake web search"

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, query: str, count: int | None = None, **_: Any) -> str:
        self.calls.append({"query": query, "count": count})
        return f"Results for {query}\n1. Example result\n   useful snippet"


@pytest.mark.asyncio
async def test_runner_allows_one_web_fetch_per_task_and_blocks_second_fetch():
    provider = MagicMock()
    captured_second_call: list[dict[str, Any]] = []
    call_count = {"n": 0}

    async def chat_with_retry(*, messages, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="fetch_1",
                        name="web_fetch",
                        arguments={"url": "https://example.com/one"},
                    ),
                    ToolCallRequest(
                        id="fetch_2",
                        name="web_fetch",
                        arguments={"url": "https://example.com/two"},
                    ),
                ],
            )
        captured_second_call[:] = messages
        return LLMResponse(content="done", tool_calls=[])

    provider.chat_with_retry = chat_with_retry
    tools = ToolRegistry()
    fetch_tool = _FakeWebFetchTool()
    tools.register(fetch_tool)

    result = await AgentRunner(provider).run(
        AgentRunSpec(
            initial_messages=[{"role": "user", "content": "find places"}],
            tools=tools,
            model="test-model",
            max_iterations=2,
            max_tool_result_chars=AgentDefaults().max_tool_result_chars,
            concurrent_tools=True,
        )
    )

    assert result.final_content == "done"
    assert fetch_tool.calls == ["https://example.com/one"]

    tool_messages = {
        msg["tool_call_id"]: json.loads(msg["content"])
        for msg in captured_second_call
        if msg.get("role") == "tool"
    }
    assert tool_messages["fetch_1"]["web_budget"]["web_fetch_count"] == 1
    assert tool_messages["fetch_2"]["web_budget_exceeded"] is True
    assert tool_messages["fetch_2"]["web_budget"]["web_fetch_count"] == 1


@pytest.mark.asyncio
async def test_runner_blocks_web_search_after_total_search_budget():
    provider = MagicMock()
    captured_final_call: list[dict[str, Any]] = []
    call_count = {"n": 0}

    async def chat_with_retry(*, messages, **kwargs):
        call_count["n"] += 1
        if call_count["n"] <= 4:
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id=f"search_{call_count['n']}",
                        name="web_search",
                        arguments={
                            "query": f"上海南京路酒店 第{call_count['n']}次",
                            "count": 10,
                        },
                    )
                ],
            )
        captured_final_call[:] = messages
        return LLMResponse(content="done", tool_calls=[])

    provider.chat_with_retry = chat_with_retry
    tools = ToolRegistry()
    search_tool = _FakeWebSearchTool()
    tools.register(search_tool)

    result = await AgentRunner(provider).run(
        AgentRunSpec(
            initial_messages=[{"role": "user", "content": "find a hotel"}],
            tools=tools,
            model="test-model",
            max_iterations=5,
            max_tool_result_chars=AgentDefaults().max_tool_result_chars,
            concurrent_tools=False,
        )
    )

    assert result.final_content == "done"
    assert len(search_tool.calls) == 3
    assert {call["count"] for call in search_tool.calls} == {3}

    tool_messages = {
        msg["tool_call_id"]: json.loads(msg["content"])
        for msg in captured_final_call
        if msg.get("role") == "tool" and str(msg.get("content", "")).startswith("{")
    }
    assert tool_messages["search_4"]["error"] == "web_search_budget_exceeded"
    assert tool_messages["search_4"]["web_budget_exceeded"] is True
    assert tool_messages["search_4"]["web_budget"]["web_search_count"] == 3
