from __future__ import annotations

from opengui.s2_policy import (
    S2Mode,
    S2Trigger,
    S2TriggerEvent,
    S2Usage,
    decide_s2_mode,
)


def test_decide_s2_mode_disabled_returns_off() -> None:
    mode = decide_s2_mode(
        enabled=False,
        trigger=S2Trigger.STAGNATION,
        hints_used=0,
        max_hints=1,
        hint_enabled=True,
        takeover_enabled=True,
        takeover_after_hints=1,
    )

    assert mode == S2Mode.OFF


def test_decide_s2_mode_uses_hint_when_budget_available() -> None:
    mode = decide_s2_mode(
        enabled=True,
        trigger=S2Trigger.STAGNATION,
        hints_used=0,
        max_hints=1,
        hint_enabled=True,
        takeover_enabled=True,
        takeover_after_hints=1,
    )

    assert mode == S2Mode.HINT


def test_decide_s2_mode_takes_over_after_hint_budget() -> None:
    mode = decide_s2_mode(
        enabled=True,
        trigger=S2Trigger.STAGNATION,
        hints_used=1,
        max_hints=1,
        hint_enabled=True,
        takeover_enabled=True,
        takeover_after_hints=1,
    )

    assert mode == S2Mode.TAKEOVER


def test_decide_s2_mode_takeover_disabled_returns_off() -> None:
    mode = decide_s2_mode(
        enabled=True,
        trigger=S2Trigger.STAGNATION,
        hints_used=1,
        max_hints=1,
        hint_enabled=True,
        takeover_enabled=False,
        takeover_after_hints=1,
    )

    assert mode == S2Mode.OFF


def test_decide_s2_mode_zero_hint_budget_can_take_over_immediately() -> None:
    mode = decide_s2_mode(
        enabled=True,
        trigger=S2Trigger.STAGNATION,
        hints_used=0,
        max_hints=0,
        hint_enabled=True,
        takeover_enabled=True,
        takeover_after_hints=0,
    )

    assert mode == S2Mode.TAKEOVER


def test_s2_usage_to_dict_is_stable() -> None:
    usage = S2Usage(enabled=True)
    usage.hints_used = 1
    usage.s1_steps = 3
    usage.s2_steps = 2
    usage.takeover_used = True
    usage.takeover_steps = 2
    usage.token_usage["input_tokens"] = 11
    usage.triggers.append(
        S2TriggerEvent(
            step_index=4,
            trigger=S2Trigger.STAGNATION,
            mode=S2Mode.HINT,
            reason="same screen repeated",
            metadata={"app": "DryRun"},
        )
    )

    assert usage.to_dict() == {
        "enabled": True,
        "hints_used": 1,
        "takeover_used": True,
        "takeover_steps": 2,
        "s1_steps": 3,
        "s2_steps": 2,
        "triggers": [
            {
                "step_index": 4,
                "trigger": "stagnation",
                "mode": "hint",
                "reason": "same screen repeated",
                "actor_before": "s1",
                "metadata": {"app": "DryRun"},
            }
        ],
        "token_usage": {"input_tokens": 11},
    }
