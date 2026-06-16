from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class S2Mode(str, Enum):
    OFF = "off"
    HINT = "hint"
    TAKEOVER = "takeover"


class S2Trigger(str, Enum):
    STAGNATION = "stagnation"
    STEP_ERROR = "step_error"
    LOW_CONFIDENCE = "low_confidence"
    FAKE_DONE = "fake_done"
    MISSING_EVIDENCE = "missing_evidence"
    MAX_STEPS_NEAR = "max_steps_near"
    WRONG_APP = "wrong_app"


@dataclass
class S2TriggerEvent:
    step_index: int
    trigger: S2Trigger
    mode: S2Mode
    reason: str
    actor_before: str = "s1"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class S2Usage:
    enabled: bool = False
    hints_used: int = 0
    takeover_used: bool = False
    takeover_steps: int = 0
    s1_steps: int = 0
    s2_steps: int = 0
    triggers: list[S2TriggerEvent] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "hints_used": self.hints_used,
            "takeover_used": self.takeover_used,
            "takeover_steps": self.takeover_steps,
            "s1_steps": self.s1_steps,
            "s2_steps": self.s2_steps,
            "triggers": [
                {
                    "step_index": item.step_index,
                    "trigger": item.trigger.value,
                    "mode": item.mode.value,
                    "reason": item.reason,
                    "actor_before": item.actor_before,
                    "metadata": item.metadata,
                }
                for item in self.triggers
            ],
            "token_usage": dict(self.token_usage),
        }


def decide_s2_mode(
    *,
    enabled: bool,
    trigger: S2Trigger,
    hints_used: int,
    max_hints: int,
    hint_enabled: bool,
    takeover_enabled: bool,
    takeover_after_hints: int,
    takeover_used: bool = False,
    allowed_triggers: frozenset[S2Trigger] = frozenset({S2Trigger.STAGNATION}),
) -> S2Mode:
    """Decide the minimal V0 S2 mode for configured recovery triggers."""
    if not enabled:
        return S2Mode.OFF

    if trigger not in allowed_triggers:
        return S2Mode.OFF

    if hint_enabled and max_hints > 0 and hints_used < max_hints:
        return S2Mode.HINT

    if takeover_used:
        return S2Mode.OFF

    if takeover_enabled and hints_used >= max(0, takeover_after_hints):
        return S2Mode.TAKEOVER

    return S2Mode.OFF
