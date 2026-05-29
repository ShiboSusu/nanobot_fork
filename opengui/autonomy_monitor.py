"""
opengui.autonomy_monitor
========================
Rule-based V0 monitor for deciding whether S1 should keep acting autonomously.

The monitor is deliberately small and observable: it consumes cheap execution
signals after each GUI step, updates a cumulative risk budget, and returns a
decision payload that can be logged or used by the agent loop.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from opengui.action import Action
from opengui.observation import Observation


class AutonomyDecisionKind(str, Enum):
    """Step-level autonomy routing decision."""

    S1_EXECUTE = "S1_EXECUTE"
    S1_RESAMPLE = "S1_RESAMPLE"
    CHEAP_VERIFY = "CHEAP_VERIFY"
    S2_HINT = "S2_HINT"
    S2_TAKEOVER = "S2_TAKEOVER"
    HUMAN_CONFIRM = "HUMAN_CONFIRM"
    HALT = "HALT"


@dataclass(frozen=True)
class RiskSignal:
    """One cheap signal contributing to the step risk score."""

    key: str
    category: str
    value: float
    weight: float
    reason: str

    @property
    def contribution(self) -> float:
        return max(0.0, self.value) * max(0.0, self.weight)

    def to_trace(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "category": self.category,
            "value": round(self.value, 4),
            "weight": round(self.weight, 4),
            "contribution": round(self.contribution, 4),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PreActionMonitorInput:
    """Cheap, observable state for one proposed GUI action before execution."""

    task: str
    step_index: int
    max_steps: int
    action: Action
    current_observation: Observation
    action_summary: str | None = None
    state_summary: str | None = None


@dataclass(frozen=True)
class StepMonitorInput:
    """Cheap, observable state for one completed GUI step."""

    task: str
    step_index: int
    max_steps: int
    action: Action
    current_observation: Observation
    next_observation: Observation | None
    tool_result: str
    action_summary: str | None = None
    state_summary: str | None = None
    expected_app: str | None = None


@dataclass(frozen=True)
class MonitorDecision:
    """Autonomy monitor output for one GUI step."""

    decision: AutonomyDecisionKind
    tier: str
    risk: float
    cumulative_risk: float
    signals: tuple[RiskSignal, ...] = ()
    reason: str = ""

    @property
    def signal_keys(self) -> tuple[str, ...]:
        return tuple(signal.key for signal in self.signals)

    @property
    def allows_s1(self) -> bool:
        return self.decision in {
            AutonomyDecisionKind.S1_EXECUTE,
            AutonomyDecisionKind.S1_RESAMPLE,
            AutonomyDecisionKind.CHEAP_VERIFY,
        }

    def to_trace(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "tier": self.tier,
            "risk": round(self.risk, 4),
            "cumulative_risk": round(self.cumulative_risk, 4),
            "signal_keys": list(self.signal_keys),
            "signals": [signal.to_trace() for signal in self.signals],
            "reason": self.reason,
        }


@dataclass
class AutonomyMonitor:
    """Risk-budgeted V0 autonomy monitor.

    The monitor intentionally uses only cheap signals available in the existing
    loop: backend failures, screenshots, action repetition, budget pressure,
    and app drift. More expensive Amber/Red verifiers can hang off this
    decision object later without changing the agent loop contract.
    """

    low_threshold: float = 0.35
    mid_threshold: float = 0.50
    red_threshold: float = 0.70
    horizon_threshold: float = 0.65
    recovery_risk_discount: float = 0.25
    safety_keywords: tuple[str, ...] = (
        "支付",
        "付款",
        "转账",
        "删除",
        "清空",
        "授权",
        "允许",
        "发送",
        "提交",
        "confirm payment",
        "pay",
        "delete",
        "authorize",
        "allow",
        "send",
        "submit",
    )
    cumulative_risk: float = field(default=0.0, init=False)
    _last_action_signature: tuple[Any, ...] | None = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        self.cumulative_risk = 0.0
        self._last_action_signature = None

    def mark_prior_attempt_progress(self) -> None:
        """Record that a previous retry attempt already produced execution context."""
        self._last_action_signature = ("prior_attempt",)

    def mark_s2_guidance_issued(self) -> None:
        """Discount accumulated risk after S2 has inspected the trajectory."""
        discount = min(1.0, max(0.0, self.recovery_risk_discount))
        self.cumulative_risk *= discount

    def assess_pre_action(self, step: PreActionMonitorInput) -> MonitorDecision:
        signals = tuple(self._signals_for_pre_action(step))
        risk = min(1.0, sum(signal.contribution for signal in signals))

        return MonitorDecision(
            decision=AutonomyDecisionKind.S1_EXECUTE,
            tier="green",
            risk=risk,
            cumulative_risk=self.cumulative_risk,
            signals=signals,
            reason="Pre-action risk allows S1 to execute.",
        )

    def assess_step(self, step: StepMonitorInput) -> MonitorDecision:
        signals = tuple(self._signals_for_step(step))
        risk = min(1.0, sum(signal.contribution for signal in signals))
        self.cumulative_risk = 1.0 - (1.0 - self.cumulative_risk) * (1.0 - risk)

        cumulative_over_budget = (
            self.cumulative_risk >= self.horizon_threshold
            and step.action.action_type != "done"
        )
        red = risk >= self.red_threshold or cumulative_over_budget
        amber = risk >= self.mid_threshold

        if red:
            decision = AutonomyDecisionKind.HALT
            tier = "red"
            if cumulative_over_budget:
                reason = "Cumulative autonomy risk exceeded the task risk budget."
            else:
                reason = "Step risk exceeded the red intervention threshold."
        elif amber:
            decision = AutonomyDecisionKind.CHEAP_VERIFY
            tier = "amber"
            reason = "Step looks suspicious; log as cheap-verifier candidate."
        else:
            decision = AutonomyDecisionKind.S1_EXECUTE
            tier = "green"
            reason = "Risk budget allows S1 to continue."

        self._last_action_signature = _action_signature(step.action)
        return MonitorDecision(
            decision=decision,
            tier=tier,
            risk=risk,
            cumulative_risk=self.cumulative_risk,
            signals=signals,
            reason=reason,
        )

    def _signals_for_pre_action(self, step: PreActionMonitorInput) -> list[RiskSignal]:
        del step
        signals: list[RiskSignal] = []
        return signals

    def _signals_for_step(self, step: StepMonitorInput) -> list[RiskSignal]:
        signals: list[RiskSignal] = []
        tool_result = (step.tool_result or "").casefold()

        if "action failed" in tool_result:
            signals.append(RiskSignal(
                key="action_failed",
                category="transition",
                value=1.0,
                weight=0.70,
                reason="Backend reported action execution failure.",
            ))

        if step.next_observation is None and step.action.action_type not in {"done", "request_intervention"}:
            signals.append(RiskSignal(
                key="observe_failed",
                category="transition",
                value=1.0,
                weight=0.75,
                reason="No post-action observation is available.",
            ))
        elif "observation failed" in tool_result:
            signals.append(RiskSignal(
                key="observe_failed",
                category="transition",
                value=1.0,
                weight=0.75,
                reason="Backend reported post-action observation failure.",
            ))

        unchanged_screen = _screen_unchanged(step.current_observation, step.next_observation)
        no_progress_action = step.action.action_type not in {"done", "request_intervention", "wait"}
        if no_progress_action and unchanged_screen:
            signals.append(RiskSignal(
                key="screen_unchanged",
                category="transition",
                value=1.0,
                weight=0.28,
                reason="Post-action screenshot and foreground app match the pre-action state.",
            ))

        wait_loading_loop = step.action.action_type == "wait" and _wait_looks_like_loading_loop(step)
        if wait_loading_loop and unchanged_screen:
            signals.append(RiskSignal(
                key="wait_no_change",
                category="transition",
                value=1.0,
                weight=0.25,
                reason="Waiting did not change the visible screen or foreground app.",
            ))

        if no_progress_action and self._last_action_signature == _action_signature(step.action):
            signals.append(RiskSignal(
                key="repeated_action",
                category="progress",
                value=1.0,
                weight=0.30,
                reason="The same GUI action was proposed on consecutive steps.",
            ))
        elif (
            step.action.action_type == "wait"
            and wait_loading_loop
            and unchanged_screen
            and self._last_action_signature == _action_signature(step.action)
        ):
            signals.append(RiskSignal(
                key="repeated_wait",
                category="progress",
                value=1.0,
                weight=0.35,
                reason="The model is repeatedly waiting on an unchanged screen instead of recovering or choosing another action.",
            ))

        if step.max_steps > 0 and step.action.action_type not in {"done", "request_intervention"}:
            budget_ratio = step.step_index / step.max_steps
            if budget_ratio >= 0.90:
                signals.append(RiskSignal(
                    key="step_budget_pressure",
                    category="progress",
                    value=1.0,
                    weight=0.25,
                    reason="The task is near its configured step limit.",
                ))
            elif budget_ratio >= 0.80:
                signals.append(RiskSignal(
                    key="step_budget_pressure",
                    category="progress",
                    value=1.0,
                    weight=0.15,
                    reason="The task is approaching its configured step limit.",
                ))

        if _expected_app_mismatch(step.expected_app, step.next_observation or step.current_observation):
            signals.append(RiskSignal(
                key="app_mismatch",
                category="capability",
                value=1.0,
                weight=0.75,
                reason="The foreground app does not match the expected app.",
            ))

        if _premature_done(step):
            signals.append(RiskSignal(
                key="premature_done",
                category="progress",
                value=1.0,
                weight=0.80,
                reason="The model declared success while its own text suggests failure or incompletion.",
            ))

        if _unverified_done(step, self._last_action_signature):
            requires_gui_progress = _task_requires_observable_gui_progress(step.task)
            signals.append(RiskSignal(
                key="unverified_done",
                category="progress",
                value=1.0,
                weight=0.80 if requires_gui_progress else 0.45,
                reason=(
                    "The model declared success before any verified GUI progress was recorded."
                    if requires_gui_progress
                    else "The model declared success before any verified GUI progress was recorded; treat as a review signal rather than a hard stop."
                ),
            ))

        return signals

    def _has_safety_keyword(
        self,
        *,
        action: Action,
        task: str,
        action_summary: str | None = None,
        state_summary: str | None = None,
        tool_result: str | None = None,
        current_observation: Observation | None = None,
        next_observation: Observation | None = None,
    ) -> bool:
        if action.action_type in {"done", "wait", "request_intervention"}:
            return False
        text = " ".join(
            part
            for part in (
                task,
                action_summary or "",
                state_summary or "",
                tool_result or "",
                _observation_text(current_observation),
                _observation_text(next_observation),
            )
            if part
        ).casefold()
        return any(keyword.casefold() in text for keyword in self.safety_keywords)


def _action_signature(action: Action) -> tuple[Any, ...]:
    return (
        action.action_type,
        action.x,
        action.y,
        action.x2,
        action.y2,
        action.text,
        tuple(action.key or ()),
        action.pixels,
        action.duration_ms,
        action.relative,
    )


def _screen_unchanged(
    current: Observation,
    next_observation: Observation | None,
) -> bool:
    if next_observation is None:
        return False
    if (current.foreground_app or "") != (next_observation.foreground_app or ""):
        return False
    current_digest = _screenshot_digest(current)
    next_digest = _screenshot_digest(next_observation)
    return current_digest is not None and current_digest == next_digest


def _screenshot_digest(observation: Observation | None) -> str | None:
    if observation is None or not observation.screenshot_path:
        return None
    path = Path(observation.screenshot_path)
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest()


def _expected_app_mismatch(expected_app: str | None, observation: Observation) -> bool:
    if not expected_app:
        return False
    actual = (observation.foreground_app or "").casefold()
    expected = expected_app.casefold()
    return bool(actual and expected and expected not in actual)


def _premature_done(step: StepMonitorInput) -> bool:
    if step.action.action_type != "done" or step.action.status != "success":
        return False
    text = " ".join(
        part
        for part in (step.action_summary or "", step.state_summary or "", step.tool_result or "")
        if part
    ).casefold()
    failure_hints = (
        "failed",
        "unable",
        "cannot",
        "not completed",
        "incomplete",
        "失败",
        "无法",
        "不能",
        "未完成",
    )
    return any(hint in text for hint in failure_hints)


def _unverified_done(step: StepMonitorInput, last_action_signature: tuple[Any, ...] | None) -> bool:
    return (
        step.action.action_type == "done"
        and step.action.status == "success"
        and last_action_signature is None
    )


def _task_requires_observable_gui_progress(task: str) -> bool:
    text = (task or "").casefold()
    cjk_terms = (
        "打开", "进入", "点击", "搜索", "播放", "查看", "检查", "购买", "预订",
        "打车", "发送", "发一条", "改成", "取消", "领取", "筛选", "排序",
        "输入", "设置", "切换", "调到",
    )
    if any(term in text for term in cjk_terms):
        return True
    english_terms = (
        "open", "enter", "tap", "click", "search", "play", "view", "check",
        "buy", "book", "send", "change", "cancel", "claim", "filter", "sort",
        "type", "set", "switch",
    )
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text)
        for term in english_terms
    )


def _wait_looks_like_loading_loop(step: StepMonitorInput) -> bool:
    text = " ".join(
        part
        for part in (
            step.action_summary or "",
            step.state_summary or "",
            step.tool_result or "",
        )
        if part
    ).casefold()
    cjk_terms = (
        "加载",
        "启动页",
        "开屏",
        "闪屏",
        "主界面",
        "启动中",
        "等待页面",
    )
    if any(term in text for term in cjk_terms):
        return True
    english_terms = (
        "loading",
        "launch screen",
        "splash",
        "startup",
        "wait for app",
        "main screen",
        "home screen to load",
    )
    return any(term in text for term in english_terms)


def _observation_text(observation: Observation | None) -> str:
    if observation is None:
        return ""
    parts: list[str] = []
    if observation.foreground_app:
        parts.append(observation.foreground_app)
    _flatten_text(observation.extra, parts, budget=80)
    return " ".join(parts)


def _flatten_text(value: Any, out: list[str], *, budget: int) -> None:
    if len(out) >= budget:
        return
    if isinstance(value, str):
        out.append(value)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if len(out) >= budget:
                return
            out.append(str(key))
            _flatten_text(item, out, budget=budget)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            if len(out) >= budget:
                return
            _flatten_text(item, out, budget=budget)
