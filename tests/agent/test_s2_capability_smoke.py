from __future__ import annotations

import pytest

from nanobot.agent.s2_capability_smoke import (
    S2CapabilityError,
    adapt_s2_action_output,
    extract_json_object,
    safety_filter_passed,
)
from opengui.action import Action


def test_extract_json_object_strips_markdown_wrapper() -> None:
    content = '```json\n{"route":"halt","reason":"blocked"}\n```'

    assert extract_json_object(content) == {"route": "halt", "reason": "blocked"}


def test_adapts_continue_click_to_opengui_action() -> None:
    candidate = adapt_s2_action_output(
        {
            "route": "continue",
            "reason": "tap the target",
            "semantic_target": "visible button",
            "action": {
                "type": "click",
                "arguments": {"x": 500, "y": 300, "relative": True},
            },
            "safety_check": {
                "side_effect": False,
                "requires_human_confirm": False,
            },
        }
    )

    assert candidate.route == "continue"
    assert candidate.action == Action(
        action_type="tap",
        x=500.0,
        y=300.0,
        relative=True,
    )
    assert candidate.reason == "tap the target"
    assert candidate.semantic_target == "visible button"
    assert safety_filter_passed(candidate) is True


def test_adapts_type_action_to_input_text() -> None:
    candidate = adapt_s2_action_output(
        {
            "route": "continue",
            "action": {
                "type": "type",
                "arguments": {"text": "广州", "auto_enter": False},
            },
            "safety_check": {},
        }
    )

    assert candidate.action == Action(
        action_type="input_text",
        text="广州",
        auto_enter=False,
    )


def test_rejects_natural_language_hint_without_json_action() -> None:
    with pytest.raises(S2CapabilityError, match="No JSON object"):
        extract_json_object("Click the visible 5th day.")


def test_safety_filter_blocks_side_effect_action() -> None:
    candidate = adapt_s2_action_output(
        {
            "route": "continue",
            "action": {
                "type": "wait",
                "arguments": {},
            },
            "safety_check": {
                "side_effect": True,
                "requires_human_confirm": True,
            },
        }
    )

    assert safety_filter_passed(candidate) is False


def test_done_route_requires_done_action() -> None:
    candidate = adapt_s2_action_output(
        {
            "route": "done",
            "action": {
                "type": "done",
                "arguments": {"status": "success"},
            },
            "safety_check": {},
        }
    )

    assert candidate.action == Action(action_type="done", status="success")
