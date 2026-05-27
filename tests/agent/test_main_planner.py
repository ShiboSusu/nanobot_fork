from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.cost_aware_router import RouteKind
from nanobot.agent.main_planner import MainPlanner, PlannerConfig
from nanobot.providers.base import LLMResponse


def test_parse_gui_plan_to_route_decision() -> None:
    planner = MainPlanner(provider=None, model=None, config=PlannerConfig(enabled=True))

    decision = planner.parse_decision(
        '{"route":"gui_task","confidence":0.91,"reason":"App internal task",'
        '"subtasks":[{"route":"gui_task","task":"在抖音里搜索旅行Vlog"}]}',
        original_task="在抖音里搜索旅行Vlog",
    )

    assert decision.route == RouteKind.GUI
    assert decision.requires_gui is True
    assert decision.reason == "App internal task"
    assert decision.system_action is None
    assert decision.routed_task == "在抖音里搜索旅行Vlog"
    assert decision.suggested_tools == ()


def test_parse_tool_plan_to_route_decision() -> None:
    planner = MainPlanner(provider=None, model=None, config=PlannerConfig(enabled=True))

    decision = planner.parse_decision(
        "```json\n"
        '{"route":"tool_call","confidence":0.88,"reason":"Public lookup",'
        '"subtasks":[{"route":"web_search","tool":"web_search","task":"查询深圳天气"}]}'
        "\n```",
        original_task="查询深圳天气",
    )

    assert decision.route == RouteKind.TOOL_CALL
    assert decision.requires_gui is False
    assert decision.suggested_tools == ("web_search", "web_fetch")


def test_parse_system_action_plan_to_route_decision() -> None:
    planner = MainPlanner(provider=None, model=None, config=PlannerConfig(enabled=True))

    decision = planner.parse_decision(
        '{"route":"system_action","confidence":0.95,"reason":"Safe direct action",'
        '"subtasks":[{"route":"system_action","task":"打开设置","system_action":"open_settings"}]}',
        original_task="打开设置",
    )

    assert decision.route == RouteKind.SYSTEM_ACTION
    assert decision.system_action == {
        "backend": None,
        "task": "打开设置",
        "intent": "open_settings",
    }


def test_parse_rejects_low_confidence() -> None:
    planner = MainPlanner(
        provider=None,
        model=None,
        config=PlannerConfig(enabled=True, confidence_threshold=0.65),
    )

    with pytest.raises(ValueError, match="low confidence"):
        planner.parse_decision(
            '{"route":"gui_task","confidence":0.2,"reason":"not sure","subtasks":[]}',
            original_task="打开设置",
        )


def test_parse_rejects_unknown_route() -> None:
    planner = MainPlanner(provider=None, model=None, config=PlannerConfig(enabled=True))

    with pytest.raises(ValueError, match="unsupported route"):
        planner.parse_decision(
            '{"route":"shell","confidence":0.9,"reason":"bad","subtasks":[]}',
            original_task="运行命令",
        )


def test_parse_full_plan_with_multiple_subtasks() -> None:
    planner = MainPlanner(provider=None, model=None, config=PlannerConfig(enabled=True))

    plan = planner.parse_plan(
        """
        {
          "route": "plan",
          "confidence": 0.92,
          "reason": "App playback needs app-local actions.",
          "subtasks": [
            {
              "id": "open_bilibili",
              "route": "system_action",
              "task": "Open Bilibili",
              "system_action": "open_app",
              "success_condition": "Bilibili is in foreground",
              "risk_level": "low"
            },
            {
              "id": "search_video",
              "route": "gui_task",
              "task": "Search Bilibili for 罗翔 刑法课",
              "success_condition": "Search results are visible",
              "risk_level": "low"
            },
            {
              "id": "play_video",
              "route": "gui_task",
              "task": "Open a matching video and start playback",
              "success_condition": "Video is playing",
              "risk_level": "low"
            }
          ],
          "risk_notes": ["public video playback"]
        }
        """,
        original_task="在B站播放罗翔的刑法课视频。",
    )

    assert plan.original_task == "在B站播放罗翔的刑法课视频。"
    assert plan.confidence == 0.92
    assert plan.reason == "App playback needs app-local actions."
    assert plan.risk_notes == ("public video playback",)
    assert [subtask.id for subtask in plan.subtasks] == [
        "open_bilibili",
        "search_video",
        "play_video",
    ]
    assert plan.subtasks[0].route == RouteKind.SYSTEM_ACTION
    assert plan.subtasks[0].system_action == "open_app"
    assert plan.subtasks[1].route == RouteKind.GUI
    assert plan.subtasks[1].success_condition == "Search results are visible"


def test_parse_full_plan_synthesizes_subtask_for_legacy_route() -> None:
    planner = MainPlanner(provider=None, model=None, config=PlannerConfig(enabled=True))

    plan = planner.parse_plan(
        '{"route":"gui_task","confidence":0.9,"reason":"App task","subtasks":[]}',
        original_task="在抖音里搜索旅行Vlog",
    )

    assert plan.route == RouteKind.GUI
    assert len(plan.subtasks) == 1
    assert plan.subtasks[0].id == "subtask_1"
    assert plan.subtasks[0].route == RouteKind.GUI
    assert plan.subtasks[0].task == "在抖音里搜索旅行Vlog"


def test_parse_full_plan_rejects_unsupported_subtask_route() -> None:
    planner = MainPlanner(provider=None, model=None, config=PlannerConfig(enabled=True))

    with pytest.raises(ValueError, match="unsupported subtask route"):
        planner.parse_plan(
            '{"route":"plan","confidence":0.9,"reason":"bad",'
            '"subtasks":[{"route":"shell","task":"run rm"}]}',
            original_task="bad",
        )


@pytest.mark.asyncio
async def test_planner_uses_content_json_without_native_tool_calls() -> None:
    provider = SimpleNamespace(
        chat_with_retry=AsyncMock(return_value=LLMResponse(
            content='{"route":"tool_call","confidence":0.9,"reason":"Public lookup","subtasks":[]}'
        ))
    )
    planner = MainPlanner(
        provider=provider,
        model="qwen3.6-35b-a3b",
        config=PlannerConfig(enabled=True),
    )

    decision = await planner.plan("查询深圳天气", available_tools={"web_search", "gui_task"})

    assert decision.route == RouteKind.TOOL_CALL
    call = provider.chat_with_retry.await_args
    assert call.kwargs["tools"] is None
    assert call.kwargs["tool_choice"] is None
    assert call.kwargs["model"] == "qwen3.6-35b-a3b"
    assert "Return only JSON" in call.args[0][0]["content"]


@pytest.mark.asyncio
async def test_plan_full_uses_content_json_without_native_tool_calls() -> None:
    provider = SimpleNamespace(
        chat_with_retry=AsyncMock(return_value=LLMResponse(
            content='{"route":"plan","confidence":0.9,"reason":"ok",'
            '"subtasks":[{"route":"web_search","tool":"web_search","task":"查询深圳天气"}]}'
        ))
    )
    planner = MainPlanner(
        provider=provider,
        model="qwen3.6-35b-a3b",
        config=PlannerConfig(enabled=True),
    )

    plan = await planner.plan_full("查询深圳天气", available_tools={"web_search", "gui_task"})

    assert plan.subtasks[0].route == RouteKind.TOOL_CALL
    call = provider.chat_with_retry.await_args
    assert call.kwargs["tools"] is None
    assert call.kwargs["tool_choice"] is None
    assert call.kwargs["model"] == "qwen3.6-35b-a3b"
