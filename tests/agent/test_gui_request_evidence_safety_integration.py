from __future__ import annotations

import json
from typing import Any

import pytest

from nanobot.agent.gui_task_schema import normalize_gui_task_request
from nanobot.agent.tools.gui import GuiWorkflowPlan, GuiWorkflowRunner, GuiWorkflowSubtask
from opengui.evidence import apply_evidence_contract
from opengui.interfaces import LLMResponse


class _FakeBackend:
    platform = "dry-run"


class _ExtractLLM:
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        del messages, tools, tool_choice, model, max_tokens
        return LLMResponse(content=json.dumps({"hot_rank_3": "测试事件A"}))


def test_single_information_query_without_evidence_is_partial() -> None:
    request = normalize_gui_task_request({"task": "打开微博,看看今天热搜榜的第三名是什么"})
    payload = {
        "success": True,
        "summary": "Status: completed Done: 已打开微博",
        "model_summary": "已打开微博页面",
    }

    out = apply_evidence_contract(request=request, payload=payload)

    assert out["success"] is False
    assert out["status"] == "partial"
    assert out["error"] == "missing_required_answer_evidence"


def test_single_information_query_model_summary_without_candidate_is_partial() -> None:
    request = normalize_gui_task_request({"task": "打开微博,看看今天热搜榜的第三名是什么"})
    payload = {
        "success": True,
        "model_summary": "微博热搜榜第三名是测试事件A",
    }

    out = apply_evidence_contract(request=request, payload=payload)

    assert out["success"] is False
    assert out["error"] == "missing_required_answer_evidence"
    assert out["answer_candidates"] == []


@pytest.mark.asyncio
async def test_multi_app_stops_before_external_send_with_blackboard() -> None:
    async def run_task(active_backend: Any, task: str, **kwargs: Any) -> str:
        del active_backend, task, kwargs
        return json.dumps(
            {
                "success": True,
                "summary": "Status: completed Done: 已找到第三名",
                "model_summary": "微博热搜榜第三名是测试事件A",
                "trace_path": "/tmp/trace-placeholder",
                "steps_taken": 1,
            }
        )

    runner = GuiWorkflowRunner(
        llm=_ExtractLLM(),
        run_task=run_task,
        load_latest_step_event=lambda _path: {"visible_text": ["微博热搜榜第三名是测试事件A"]},
    )
    plan = GuiWorkflowPlan(
        mode="multi_app",
        subtasks=(
            GuiWorkflowSubtask(
                app_hint="微博",
                task="在微博查看热搜第三名",
                outputs=("hot_rank_3",),
            ),
            GuiWorkflowSubtask(
                app_hint="微信",
                task="在微信把 hot_rank_3 发给张三",
                inputs=("hot_rank_3",),
            ),
        ),
    )
    request = normalize_gui_task_request({"task": "打开微博查热搜第三名，然后发给微信里的张三"})

    payload = json.loads(
        await runner._run_multi_app(_FakeBackend(), request.task, plan, task_request=request)
    )

    assert payload["success"] is False
    assert payload["status"] == "needs_human_confirm"
    assert payload["error"] == "needs_human_confirm"
    assert payload["blackboard"]["hot_rank_3"] == "测试事件A"
    assert payload["pending_action"]["known_values"]["hot_rank_3"] == "测试事件A"
