from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from nanobot.agent.context import ContextBuilder
from nanobot.agent.gui_task_schema import normalize_gui_task_request
from nanobot.agent.tools.gui import GuiSubagentTool, GuiWorkflowRunner
from nanobot.config.schema import GuiConfig
from opengui.agent import GuiAgent
from opengui.interfaces import LLMResponse
from opengui.observation import Observation
from opengui.trajectory.recorder import TrajectoryRecorder


class _CaptureLLM:
    def __init__(self, content: str = '{"mode":"single","subtasks":[]}') -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        return LLMResponse(content=self.content)


class _NoopBackend:
    platform = "dry-run"


class _NoopProvider:
    api_key = "test-key"
    api_base = None
    extra_headers = None
    _client = None

    def get_default_model(self) -> str:
        return "gui-default"


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


def test_gui_app_task_policy_mentions_gui_task() -> None:
    from nanobot.agent import gui_experiment_policy as p

    assert "Phone/app GUI tasks must use gui_task" in p.GUI_APP_TASK_POLICY
    assert "Do not answer app-visible information from memory" in p.GUI_APP_TASK_POLICY


def test_gui_task_description_policy_mentions_evidence_and_confirm() -> None:
    from nanobot.agent import gui_experiment_policy as p

    assert "answer_candidates" in p.GUI_TASK_DESCRIPTION_POLICY
    assert "evidence" in p.GUI_TASK_DESCRIPTION_POLICY
    assert "needs_human_confirm" in p.GUI_TASK_DESCRIPTION_POLICY


def test_workflow_policy_splits_sensitive_followup() -> None:
    from nanobot.agent import gui_experiment_policy as p

    text = p.GUI_WORKFLOW_PLANNER_POLICY
    assert "information gathering" in text
    assert "sensitive follow-up action" in text
    assert "split them into separate subtasks" in text
    assert "outputs" in text
    assert "inputs" in text


def test_information_query_policy_requires_answer_extraction() -> None:
    from nanobot.agent import gui_experiment_policy as p

    text = p.GUI_INFORMATION_QUERY_POLICY
    assert "do not mark done" in text
    assert "explicitly extracted" in text
    assert "merely opening the target page or app" in text


def test_bounded_list_collection_policy_limits_open_ended_scrolling() -> None:
    from nanobot.agent import gui_experiment_policy as p

    text = p.GUI_BOUNDED_LIST_COLLECTION_POLICY
    assert "scrollable list" in text
    assert "max_scrolls" in text
    assert "max_items" in text
    assert "do not keep scrolling to exhaust an open-ended list" in text
    assert "not exhaustive" in text


def test_final_answer_policy_blocks_internal_leaks() -> None:
    from nanobot.agent import gui_experiment_policy as p

    text = p.GUI_FINAL_ANSWER_POLICY
    assert "Do not expose raw JSON" in text
    assert "trace_path" in text
    assert "Status: completed" in text


def test_policy_constants_are_importable() -> None:
    from nanobot.agent import gui_experiment_policy as p

    assert p.S2_NO_THINKING_POLICY.startswith("/no_think")


def test_main_system_prompt_includes_gui_policy(tmp_path: Path) -> None:
    prompt = ContextBuilder(_workspace(tmp_path)).build_system_prompt()

    assert "Phone/app GUI tasks must use gui_task" in prompt
    assert "Do not answer app-visible information from memory or general knowledge" in prompt
    assert "Do not expose raw JSON" in prompt
    assert "Do not expose trace_path" in prompt
    assert "Do not expose \"Status: completed\"" in prompt


def test_gui_task_description_includes_policy(tmp_path: Path) -> None:
    tool = GuiSubagentTool(
        gui_config=GuiConfig(backend="dry-run"),
        provider=_NoopProvider(),
        model="gui-default",
        workspace=tmp_path,
    )

    assert "used for both GUI operations and GUI information queries" in tool.description
    assert "answer_candidates" in tool.description
    assert "evidence" in tool.description
    assert "needs_human_confirm" in tool.description


@pytest.mark.asyncio
async def test_workflow_planner_prompt_includes_experiment_policy() -> None:
    llm = _CaptureLLM()
    runner = GuiWorkflowRunner(
        llm=llm,  # type: ignore[arg-type]
        run_task=lambda *args, **kwargs: "",  # type: ignore[arg-type]
        load_latest_step_event=lambda path: {},
        router_memory=None,
    )

    await runner._plan_workflow("打开微博查热搜第三名，然后发给微信里的张三")

    system_prompt = llm.calls[0]["messages"][0]["content"]
    assert "If a task contains both information gathering and a sensitive follow-up action" in system_prompt
    assert "split them into separate subtasks" in system_prompt
    assert "The information-gathering subtask must declare outputs" in system_prompt
    assert "The sensitive subtask must consume those outputs through inputs" in system_prompt
    assert "Do not merge a sensitive action with an information-gathering subtask" in system_prompt
    assert "Do not generate low-level UI action scripts" in system_prompt


@pytest.mark.asyncio
async def test_information_query_policy_is_added_to_gui_task_prompt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}
    tool = GuiSubagentTool(
        gui_config=GuiConfig(backend="dry-run"),
        provider=_NoopProvider(),
        model="gui-default",
        workspace=tmp_path,
    )

    class FakeAgent:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        async def run(self, *, task: str, **kwargs: Any) -> Any:
            del kwargs
            seen["task"] = task
            return type(
                "Result",
                (),
                {
                    "success": True,
                    "summary": "done",
                    "model_summary": "微博热搜榜第三名是测试事件A",
                    "trace_path": None,
                    "steps_taken": 1,
                    "error": None,
                    "token_usage": {},
                    "s2_usage": {},
                    "answer_candidates": [{"key": "hot_rank_3", "text": "测试事件A"}],
                    "evidence": {},
                },
            )()

    monkeypatch.setattr("nanobot.agent.tools.gui.GuiAgent", FakeAgent)
    monkeypatch.setattr(tool, "_load_policy_context_and_memory_store", lambda: ("", None))
    monkeypatch.setattr(tool, "_make_run_dir", lambda: tmp_path / "run")
    monkeypatch.setattr(tool, "_load_gui_metrics", lambda path: {})
    monkeypatch.setattr(tool._postprocessor, "schedule", lambda *args, **kwargs: None)

    request = normalize_gui_task_request({"task": "打开微博,看看今天热搜榜的第三名是什么"})
    await tool._run_task(_NoopBackend(), request.task, task_request=request)

    assert "For information_query tasks, do not mark done unless the requested information is explicitly extracted." in seen["task"]
    assert "do not treat merely opening the target page or app as task completion" in seen["task"]
    assert "For tasks that collect items from a scrollable list" in seen["task"]
    assert "do not keep scrolling to exhaust an open-ended list" in seen["task"]


@pytest.mark.asyncio
async def test_s2_hint_prompt_is_no_thinking_json_only(tmp_path: Path) -> None:
    llm = _CaptureLLM('{"diagnosis":"ok"}')
    agent = GuiAgent(
        llm,
        _NoopBackend(),
        trajectory_recorder=TrajectoryRecorder(output_dir=tmp_path / "trajectory", task="s2"),
        artifacts_root=tmp_path / "runs",
        max_steps=1,
        s2_llm=llm,
        s2_enabled=True,
    )

    await agent._request_s2_hint(
        task="recover",
        step_index=1,
        current_observation=Observation(
            screenshot_path=None,
            screen_width=64,
            screen_height=64,
            foreground_app="DryRun",
            platform="dry-run",
        ),
        history=[],
        last_action_summary=None,
        reason="same screen",
    )

    system_prompt = llm.calls[0]["messages"][0]["content"]
    assert "/no_think" in system_prompt
    assert "Return ONLY one minified JSON object" in system_prompt
    assert "Do not output thinking" in system_prompt
    assert "Do not output analysis" in system_prompt
    assert "Do not output markdown" in system_prompt
    assert "Do not output code fences" in system_prompt
    assert "Keep the JSON under 120 tokens" in system_prompt
    assert llm.calls[0]["max_tokens"] <= 180
