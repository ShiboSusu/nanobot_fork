from __future__ import annotations

import json
from typing import Any

import pytest

from nanobot.agent.tools.gui import GuiWorkflowPlan, GuiWorkflowRunner, GuiWorkflowSubtask
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
        return LLMResponse(content=json.dumps({"hot_rank_3": "X"}))


@pytest.mark.asyncio
async def test_multi_app_payload_keeps_blackboard_and_adds_sidecars() -> None:
    calls: list[str] = []

    async def run_task(active_backend: Any, task: str, **kwargs: Any) -> str:
        del active_backend, kwargs
        calls.append(task)
        return json.dumps(
            {
                "success": True,
                "summary": "found X",
                "model_summary": "hot_rank_3 is X",
                "trace_path": "/tmp/trace-placeholder",
                "steps_taken": 1,
                "s2_usage": {
                    "enabled": True,
                    "hints_used": 1,
                    "takeover_used": False,
                    "takeover_steps": 0,
                    "s1_steps": 1,
                    "s2_steps": 0,
                    "triggers": [
                        {
                            "step_index": 1,
                            "trigger": "stagnation",
                            "mode": "hint",
                            "reason": "same screen",
                        }
                    ],
                    "token_usage": {"input_tokens": 3},
                },
            }
        )

    runner = GuiWorkflowRunner(
        llm=_ExtractLLM(),
        run_task=run_task,
        load_latest_step_event=lambda _path: {"visible_text": ["X"]},
    )
    plan = GuiWorkflowPlan(
        mode="multi_app",
        subtasks=(
            GuiWorkflowSubtask(
                app_hint="Weibo",
                task="Find hot rank 3",
                outputs=("hot_rank_3",),
            ),
            GuiWorkflowSubtask(
                app_hint="Notes",
                task="Write the item",
                inputs=("hot_rank_3",),
            ),
        ),
    )

    payload = json.loads(await runner._run_multi_app(_FakeBackend(), "copy hot rank 3", plan))

    assert payload["schema_version"] == "gui_task_result.v1"
    assert payload["blackboard"] == {"hot_rank_3": "X"}
    assert payload["blackboard_meta"]["hot_rank_3"]["value"] == "X"
    assert payload["blackboard_meta"]["hot_rank_3"]["source_subtask"] == "Find hot rank 3"
    assert payload["blackboard_meta"]["hot_rank_3"]["source_app_hint"] == "Weibo"
    assert payload["answer_candidates"] == [
        {
            "key": "hot_rank_3",
            "text": "X",
            "source": "blackboard",
            "confidence": None,
            "source_subtask": "Find hot rank 3",
            "evidence_refs": ["/tmp/trace-placeholder"],
        }
    ]
    assert payload["s2_usage"]["enabled"] is True
    assert payload["s2_usage"]["hints_used"] == 2
    assert payload["s2_usage"]["s1_steps"] == 2
    assert len(payload["s2_usage"]["triggers"]) == 2
    assert "hot_rank_3=X" in calls[1]
