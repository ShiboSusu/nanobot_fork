from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from nanobot.agent.gui_task_schema import normalize_gui_task_request
from nanobot.agent.tools.gui import GuiWorkflowRunner
from nanobot.config.schema import GuiConfig


def _provider() -> MagicMock:
    provider = MagicMock()
    provider.api_key = "test-key"
    provider.api_base = None
    provider.extra_headers = None
    provider._client = None
    provider.get_default_model.return_value = "gui-default"
    return provider


def _tool(tmp_path: Path):
    from nanobot.agent.tools.gui import GuiSubagentTool

    return GuiSubagentTool(
        gui_config=GuiConfig(backend="dry-run"),
        provider=_provider(),
        model="gui-default",
        workspace=tmp_path,
    )


class _FakeBackend:
    platform = "android"


class _FakeLLM:
    async def chat(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("planner LLM should not be required in this test")


@pytest.mark.asyncio
async def test_direct_payment_is_blocked_before_backend_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = _tool(tmp_path)

    def fail_select_backend(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("_select_backend should not be called for direct sensitive GUI task")

    monkeypatch.setattr(tool, "_select_backend", fail_select_backend)

    payload = json.loads(await tool.execute("打开支付宝付款给张三"))

    assert payload["success"] is False
    assert payload["status"] == "needs_human_confirm"
    assert payload["error"] == "needs_human_confirm"
    assert payload["steps_taken"] == 0
    assert payload["workflow_mode"] == "blocked_before_gui_task"
    assert payload["safety"]["requires_human_confirm"] is True
    assert payload["safety"]["risk"] == "payment"


@pytest.mark.asyncio
async def test_direct_external_send_is_blocked_before_workflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = _tool(tmp_path)

    async def fail_run_workflow(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("_run_workflow_or_task should not be called for direct external send")

    monkeypatch.setattr(tool, "_run_workflow_or_task", fail_run_workflow)

    payload = json.loads(await tool.execute("打开微信发给张三：测试消息"))

    assert payload["success"] is False
    assert payload["status"] == "needs_human_confirm"
    assert payload["error"] == "needs_human_confirm"
    assert payload["steps_taken"] == 0
    assert payload["safety"]["risk"] == "external_send"
    assert "pending_action" in payload


@pytest.mark.asyncio
async def test_direct_prepare_before_send_is_allowed_to_enter_workflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = _tool(tmp_path)
    called = {"run": False}

    monkeypatch.setattr(tool, "_select_backend", lambda backend=None: _FakeBackend())

    async def fake_run_workflow(active_backend: Any, task: str, **kwargs: Any) -> str:
        del active_backend, task, kwargs
        called["run"] = True
        return json.dumps(
            {
                "schema_version": "gui_task_result.v1",
                "success": True,
                "summary": "Prepared message and stopped before sending.",
                "model_summary": "Prepared message and stopped before sending.",
                "trace_path": None,
                "steps_taken": 1,
                "error": None,
                "post_run_state": None,
                "duration_s": 0,
                "token_usage": {},
                "s2_usage": {},
                "answer_candidates": [],
                "evidence": {},
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(tool, "_run_workflow_or_task", fake_run_workflow)

    payload = json.loads(await tool.execute("打开微信准备发给张三，但停在发送前，不要真的发送"))

    assert called["run"] is True
    assert payload["success"] is True


@pytest.mark.asyncio
async def test_mixed_query_action_single_fallback_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    request = normalize_gui_task_request({"task": "打开微博查热搜第三名，然后发给微信里的张三"})

    async def fail_run_task(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("_run_task should not be called when mixed task cannot be safely decomposed")

    runner = GuiWorkflowRunner(
        llm=_FakeLLM(),
        run_task=fail_run_task,
        load_latest_step_event=lambda _path: {},
        router_memory=None,
    )

    async def fake_safe_plan_workflow(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        return None

    monkeypatch.setattr(runner, "_safe_plan_workflow", fake_safe_plan_workflow)

    payload = json.loads(await runner.run(_FakeBackend(), request.task, task_request=request))

    assert payload["success"] is False
    assert payload["status"] == "needs_human_confirm"
    assert payload["error"] == "needs_human_confirm"
    assert payload["workflow_mode"] == "blocked_mixed_single_fallback"
    assert payload["steps_taken"] == 0

