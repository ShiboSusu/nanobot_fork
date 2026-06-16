from __future__ import annotations

import json

from nanobot.agent.runner import AgentRunSpec, AgentRunner
from nanobot.config.schema import AgentDefaults


def test_gui_task_result_is_sanitized_before_model_context(tmp_path) -> None:
    runner = AgentRunner(provider=object())
    spec = AgentRunSpec(
        initial_messages=[],
        tools=object(),
        model="test-model",
        max_iterations=1,
        max_tool_result_chars=AgentDefaults().max_tool_result_chars,
        workspace=tmp_path,
    )
    raw = json.dumps(
        {
            "schema_version": "gui_task_result.v1",
            "success": True,
            "summary": "done",
            "model_summary": "answer",
            "answer_candidates": [{"key": "answer", "text": "42"}],
            "evidence": {"sources": ["latest_step"]},
            "error": None,
            "trace_path": "/tmp/trace.jsonl",
            "metrics_path": "/tmp/gui_metrics.json",
            "token_usage": {"prompt_tokens": 10},
            "schema_version_extra": "x",
            "s2_usage": {"enabled": True},
            "native_launch": {"status": "ok"},
        }
    )

    content = runner._normalize_tool_result(spec, "tc_1", "gui_task", raw)
    payload = json.loads(content)

    assert payload == {
        "success": True,
        "summary": "done",
        "model_summary": "answer",
        "answer_candidates": [{"key": "answer", "text": "42"}],
        "evidence": {"sources": ["latest_step"]},
        "error": None,
    }
    forbidden = {"trace_path", "metrics_path", "token_usage", "schema_version", "s2_usage", "native_launch"}
    assert forbidden.isdisjoint(payload)
