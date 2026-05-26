from __future__ import annotations

from opengui.action import Action
from opengui.autonomy import (
    AutonomyDecisionType,
    AutonomyMonitor,
    AutonomyMonitorConfig,
    AutonomySignal,
)
from opengui.policy import PolicyAction, PolicyDecision


def test_autonomy_monitor_executes_low_risk_step() -> None:
    monitor = AutonomyMonitor()

    decision = monitor.evaluate(
        AutonomySignal(
            task="open settings",
            step_index=1,
            max_steps=10,
            action=Action(action_type="tap", x=100, y=200),
        )
    )

    assert decision.decision == AutonomyDecisionType.S1_EXECUTE
    assert decision.risk_score < monitor.config.low_threshold
    assert decision.cumulative_risk < monitor.config.cumulative_risk_budget
    assert decision.signals == ()


def test_autonomy_monitor_escalates_action_and_observe_failures() -> None:
    monitor = AutonomyMonitor()

    decision = monitor.evaluate(
        AutonomySignal(
            task="open settings",
            step_index=1,
            max_steps=10,
            action=Action(action_type="tap", x=100, y=200),
            action_failed=True,
            observe_failed=True,
        )
    )

    assert decision.decision == AutonomyDecisionType.S2_TAKEOVER
    assert decision.risk_score >= monitor.config.high_threshold
    assert "action_failed" in decision.signals
    assert "observe_failed" in decision.signals


def test_autonomy_monitor_tracks_cumulative_risk_budget() -> None:
    monitor = AutonomyMonitor(
        AutonomyMonitorConfig(
            low_threshold=0.30,
            mid_threshold=0.55,
            high_threshold=0.80,
            cumulative_risk_budget=0.45,
        )
    )
    action = Action(action_type="wait", duration_ms=1)

    decisions = [
        monitor.evaluate(
            AutonomySignal(
                task="wait on unchanged screen",
                step_index=index,
                max_steps=10,
                action=action,
                screen_unchanged=True,
                repeated_action=True,
            )
        )
        for index in range(1, 5)
    ]

    assert decisions[-1].decision == AutonomyDecisionType.S2_HINT
    assert "cumulative_risk_over_budget" in decisions[-1].signals
    assert decisions[-1].cumulative_risk > monitor.config.cumulative_risk_budget


def test_autonomy_monitor_respects_safety_policy_signal() -> None:
    monitor = AutonomyMonitor()

    decision = monitor.evaluate(
        AutonomySignal(
            task="approve payment",
            step_index=1,
            max_steps=5,
            action=Action(action_type="tap", x=500, y=500),
            policy_decision=PolicyDecision(
                action=PolicyAction.ASK_HUMAN_CONFIRM,
                categories=("payment_or_purchase",),
                reason="Payment requires confirmation.",
            ),
        )
    )

    assert decision.decision == AutonomyDecisionType.HUMAN_CONFIRM
    assert "safety_risk" in decision.signals
    assert decision.reason == "Payment requires confirmation."
