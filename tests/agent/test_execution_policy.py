"""Tests for rule-based execution policy and trace writer."""

from __future__ import annotations

import json

from nanobot.agent.execution_policy import (
    PolicyNodeContext,
    PolicyTraceWriter,
    RuleExecutionPolicy,
)


def test_rule_policy_disabled_by_default(monkeypatch):
    monkeypatch.delenv("NANOBOT_EXEC_POLICY_ENABLED", raising=False)
    policy = RuleExecutionPolicy.from_env()
    assert policy.enabled is False


def test_rule_policy_route_hybrid_for_gui_task_when_tools_available(monkeypatch):
    monkeypatch.setenv("NANOBOT_EXEC_POLICY_ENABLED", "1")
    policy = RuleExecutionPolicy.from_env()
    node = PolicyNodeContext(
        task="请在页面点击提交按钮并确认结果",
        has_gui=True,
        available_tools=("web_search", "read_file"),
        history_len=3,
        retry_count=0,
    )

    decision = policy.decide(node)

    assert decision.route == "hybrid"
    assert decision.mode in {"fast", "verify", "slow", "replan"}
    assert decision.source == "rule-v1"


def test_rule_policy_replan_after_multiple_failures():
    policy = RuleExecutionPolicy(enabled=True)
    node = PolicyNodeContext(
        task="update record fields",
        has_gui=False,
        available_tools=("web_fetch",),
        history_len=2,
        retry_count=2,
        tool_error_signal=True,
    )

    decision = policy.decide(node)

    assert decision.mode == "replan"


def test_policy_trace_writer_appends_jsonl(tmp_path):
    writer = PolicyTraceWriter(tmp_path)
    node = PolicyNodeContext(
        task="query users",
        has_gui=False,
        available_tools=("web_search",),
        history_len=1,
        retry_count=0,
    )
    decision = RuleExecutionPolicy(enabled=True).decide(node)

    writer.append(
        session_key="cli:direct",
        channel="cli",
        chat_id="direct",
        node_context=node,
        decision=decision,
    )

    lines = writer.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["decision"]["route"] == decision.route
    assert payload["decision"]["mode"] == decision.mode
    assert payload["node_context"]["task"] == "query users"

