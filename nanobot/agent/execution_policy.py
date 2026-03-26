"""Execution policy for node-level route selection and deliberation mode.

This module provides a small, rule-based controller that can be enabled at
runtime to experiment with execution-policy decisions without changing the core
agent architecture.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.environ.get(name)
    if not value:
        return default
    parts = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    return parts or default


@dataclass(frozen=True)
class PolicyNodeContext:
    """Compact node context used by execution-policy decisions."""

    task: str
    has_gui: bool
    available_tools: tuple[str, ...]
    history_len: int
    retry_count: int
    tool_error_signal: bool = False
    mismatch_signal: bool = False


@dataclass(frozen=True)
class ExecutionDecision:
    """Decision output for one workflow node."""

    route: str  # tool | gui | hybrid
    mode: str  # fast | slow | verify | replan
    risk_level: str  # low | medium | high
    reason: str
    source: str = "rule-v1"


@dataclass(frozen=True)
class PolicyTraceRecord:
    """Persisted trace row for offline policy analysis."""

    timestamp: str
    session_key: str
    channel: str
    chat_id: str
    node_context: PolicyNodeContext
    decision: ExecutionDecision

    def to_jsonl(self) -> str:
        payload = {
            "timestamp": self.timestamp,
            "session_key": self.session_key,
            "channel": self.channel,
            "chat_id": self.chat_id,
            "node_context": asdict(self.node_context),
            "decision": asdict(self.decision),
        }
        return json.dumps(payload, ensure_ascii=False)


@dataclass
class RuleExecutionPolicy:
    """Rule-based policy baseline for route + deliberation selection."""

    enabled: bool = False
    gui_keywords: tuple[str, ...] = field(default_factory=tuple)
    high_risk_keywords: tuple[str, ...] = field(default_factory=tuple)
    long_horizon_keywords: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_env(cls) -> "RuleExecutionPolicy":
        return cls(
            enabled=_env_bool("NANOBOT_EXEC_POLICY_ENABLED", default=False),
            gui_keywords=_env_csv(
                "NANOBOT_EXEC_POLICY_GUI_KEYWORDS",
                (
                    "click",
                    "button",
                    "screen",
                    "ui",
                    "网页",
                    "页面",
                    "界面",
                    "点击",
                    "拖拽",
                    "图标",
                ),
            ),
            high_risk_keywords=_env_csv(
                "NANOBOT_EXEC_POLICY_HIGH_RISK_KEYWORDS",
                (
                    "delete",
                    "payment",
                    "transfer",
                    "production",
                    "权限",
                    "删除",
                    "转账",
                    "发布",
                ),
            ),
            long_horizon_keywords=_env_csv(
                "NANOBOT_EXEC_POLICY_LONG_HORIZON_KEYWORDS",
                (
                    "monitor",
                    "retry",
                    "schedule",
                    "long-running",
                    "持续",
                    "定时",
                    "轮询",
                    "等待",
                    "重试",
                ),
            ),
        )

    @staticmethod
    def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in text for keyword in keywords)

    def decide(self, node: PolicyNodeContext) -> ExecutionDecision:
        """Return route + mode decision for one node context."""
        text = node.task.lower()
        has_tools = bool(node.available_tools)
        needs_gui = node.has_gui and self._contains_any(text, self.gui_keywords)
        high_risk = (
            self._contains_any(text, self.high_risk_keywords)
            or node.tool_error_signal
            or node.mismatch_signal
        )
        long_horizon = self._contains_any(text, self.long_horizon_keywords)

        if needs_gui and has_tools:
            route = "hybrid"
            route_reason = "task indicates GUI grounding while tools are available"
        elif needs_gui:
            route = "gui"
            route_reason = "task indicates GUI grounding"
        else:
            route = "tool"
            route_reason = "task can be executed with tool/API path"

        if node.retry_count >= 2 or (node.tool_error_signal and node.mismatch_signal):
            mode = "replan"
            mode_reason = "multiple failures/mismatch detected, replan is safer"
        elif high_risk and long_horizon:
            mode = "slow"
            mode_reason = "high-risk and long-horizon node requires slower reasoning"
        elif high_risk or node.tool_error_signal:
            mode = "verify"
            mode_reason = "risk/error signals require verification before finalizing"
        else:
            mode = "fast"
            mode_reason = "no elevated risk signals"

        risk_level = "high" if high_risk else ("medium" if long_horizon else "low")
        reason = f"route={route_reason}; mode={mode_reason}"

        return ExecutionDecision(
            route=route,
            mode=mode,
            risk_level=risk_level,
            reason=reason,
        )


class PolicyTraceWriter:
    """Append policy decisions to JSONL for offline analysis."""

    def __init__(self, workspace: Path) -> None:
        output_dir = workspace / "policy_traces"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = output_dir / f"execution_policy_{timestamp}.jsonl"

    def append(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        node_context: PolicyNodeContext,
        decision: ExecutionDecision,
    ) -> None:
        record = PolicyTraceRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            node_context=node_context,
            decision=decision,
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.to_jsonl())
            handle.write("\n")

