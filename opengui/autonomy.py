"""Rule-based autonomy monitor for GUI agent step execution.

The monitor answers a narrow question: is it still cheap and safe to let the
fast GUI actor continue, or should the host escalate before more actions are
sent to the device?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from opengui.policy import PolicyAction, PolicyDecision


class AutonomyDecisionType(str, Enum):
    S1_EXECUTE = "s1_execute"
    S1_RESAMPLE = "s1_resample"
    CHEAP_VERIFY = "cheap_verify"
    S2_HINT = "s2_hint"
    S2_TAKEOVER = "s2_takeover"
    HUMAN_CONFIRM = "human_confirm"
    HALT = "halt"


@dataclass(frozen=True)
class AutonomyMonitorConfig:
    low_threshold: float = 0.25
    mid_threshold: float = 0.45
    high_threshold: float = 0.70
    cumulative_risk_budget: float = 0.65
    step_budget_pressure_ratio: float = 0.80
    signal_weights: dict[str, float] = field(default_factory=lambda: {
        "action_failed": 0.45,
        "observe_failed": 0.45,
        "screen_unchanged": 0.16,
        "repeated_action": 0.16,
        "step_budget_pressure": 0.20,
        "safety_risk": 1.0,
    })


@dataclass(frozen=True)
class AutonomySignal:
    task: str
    step_index: int
    max_steps: int
    action: Any
    action_failed: bool = False
    observe_failed: bool = False
    screen_unchanged: bool = False
    repeated_action: bool = False
    policy_decision: PolicyDecision | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AutonomyDecision:
    decision: AutonomyDecisionType
    reason: str
    risk_score: float
    cumulative_risk: float
    signals: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AutonomyMonitorState:
    cumulative_risk: float = 0.0
    steps_evaluated: int = 0


class AutonomyMonitor:
    """Low-cost step monitor for fast/slow GUI control."""

    def __init__(self, config: AutonomyMonitorConfig | None = None) -> None:
        self.config = config or AutonomyMonitorConfig()
        self.state = AutonomyMonitorState()

    def reset(self) -> None:
        self.state = AutonomyMonitorState()

    def evaluate(self, signal: AutonomySignal) -> AutonomyDecision:
        signals = self._collect_signals(signal)
        risk_score = self._risk_score(signals)
        cumulative_risk = self._update_cumulative_risk(risk_score)

        policy = signal.policy_decision
        if policy is not None and policy.action != PolicyAction.ALLOW:
            return AutonomyDecision(
                decision=AutonomyDecisionType.HUMAN_CONFIRM,
                reason=policy.reason or "Sensitive action requires human confirmation.",
                risk_score=1.0,
                cumulative_risk=cumulative_risk,
                signals=tuple(dict.fromkeys((*signals, "safety_risk"))),
                metadata={
                    "policy_action": policy.action.value,
                    "policy_categories": policy.categories,
                    **signal.metadata,
                },
            )

        if risk_score >= self.config.high_threshold:
            return AutonomyDecision(
                decision=AutonomyDecisionType.S2_TAKEOVER,
                reason="High-risk GUI step detected; slow model should recover or take over.",
                risk_score=risk_score,
                cumulative_risk=cumulative_risk,
                signals=signals,
                metadata=signal.metadata,
            )

        if cumulative_risk > self.config.cumulative_risk_budget:
            signals = tuple(dict.fromkeys((*signals, "cumulative_risk_over_budget")))
            return AutonomyDecision(
                decision=AutonomyDecisionType.S2_HINT,
                reason=(
                    "Cumulative GUI execution risk exceeded the autonomy budget; "
                    "slow model should verify direction or provide a recovery hint."
                ),
                risk_score=risk_score,
                cumulative_risk=cumulative_risk,
                signals=signals,
                metadata=signal.metadata,
            )

        if risk_score >= self.config.mid_threshold:
            return AutonomyDecision(
                decision=AutonomyDecisionType.S2_HINT,
                reason="Suspicious GUI step detected; slow model should verify before continuing.",
                risk_score=risk_score,
                cumulative_risk=cumulative_risk,
                signals=signals,
                metadata=signal.metadata,
            )

        if risk_score >= self.config.low_threshold:
            return AutonomyDecision(
                decision=AutonomyDecisionType.CHEAP_VERIFY,
                reason="Low-to-medium GUI risk detected; run cheap verification if available.",
                risk_score=risk_score,
                cumulative_risk=cumulative_risk,
                signals=signals,
                metadata=signal.metadata,
            )

        return AutonomyDecision(
            decision=AutonomyDecisionType.S1_EXECUTE,
            reason="GUI step is within the small model autonomy budget.",
            risk_score=risk_score,
            cumulative_risk=cumulative_risk,
            signals=signals,
            metadata=signal.metadata,
        )

    def _collect_signals(self, signal: AutonomySignal) -> tuple[str, ...]:
        signals: list[str] = []
        if signal.action_failed:
            signals.append("action_failed")
        if signal.observe_failed:
            signals.append("observe_failed")
        if signal.screen_unchanged:
            signals.append("screen_unchanged")
        if signal.repeated_action:
            signals.append("repeated_action")
        if self._step_budget_pressure(signal.step_index, signal.max_steps):
            signals.append("step_budget_pressure")
        if signal.policy_decision is not None and signal.policy_decision.action != PolicyAction.ALLOW:
            signals.append("safety_risk")
        return tuple(dict.fromkeys(signals))

    def _risk_score(self, signals: tuple[str, ...]) -> float:
        return min(
            1.0,
            sum(self.config.signal_weights.get(signal, 0.0) for signal in signals),
        )

    def _update_cumulative_risk(self, risk_score: float) -> float:
        self.state.steps_evaluated += 1
        self.state.cumulative_risk = 1 - (
            (1 - self.state.cumulative_risk) * (1 - risk_score)
        )
        return self.state.cumulative_risk

    def _step_budget_pressure(self, step_index: int, max_steps: int) -> bool:
        if max_steps <= 0:
            return False
        return (step_index / max_steps) >= self.config.step_budget_pressure_ratio


__all__ = [
    "AutonomyDecision",
    "AutonomyDecisionType",
    "AutonomyMonitor",
    "AutonomyMonitorConfig",
    "AutonomyMonitorState",
    "AutonomySignal",
]
