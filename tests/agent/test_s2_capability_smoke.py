from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.agent.s2_capability_smoke import (
    S2CapabilityError,
    adapt_s2_action_output,
    build_s2_action_messages,
    extract_json_object,
    safety_filter_passed,
)
from opengui.action import Action


PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4"
    b"\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05"
    b"\xfe\x02\xfeA\xe2`\x82\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_extract_json_object_strips_markdown_wrapper() -> None:
    content = '```json\n{"route":"halt","reason":"blocked"}\n```'

    assert extract_json_object(content) == {"route": "halt", "reason": "blocked"}


def test_build_s2_action_messages_include_image_block(tmp_path: Path) -> None:
    screenshot = tmp_path / "screen.png"
    screenshot.write_bytes(PNG_1X1)

    messages = build_s2_action_messages(
        task="选择 2026-06-05 的出发日期",
        screenshot_path=screenshot,
        case_name="datepicker",
    )

    assert messages[0]["role"] == "system"
    assert "return only json" in messages[0]["content"].lower()
    user_content = messages[1]["content"]
    assert user_content[0]["type"] == "image_url"
    assert user_content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert user_content[1]["type"] == "text"
    assert "datepicker" in user_content[1]["text"]


def test_build_s2_action_messages_rejects_non_image(tmp_path: Path) -> None:
    screenshot = tmp_path / "screen.txt"
    screenshot.write_text("not an image")

    with pytest.raises(S2CapabilityError, match="Unsupported image"):
        build_s2_action_messages(
            task="选择 2026-06-05 的出发日期",
            screenshot_path=screenshot,
            case_name="datepicker",
        )


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


def test_safety_filter_allows_false_string_flags() -> None:
    candidate = adapt_s2_action_output(
        {
            "route": "continue",
            "action": {
                "type": "wait",
                "arguments": {},
            },
            "safety_check": {
                "side_effect": "false",
                "requires_human_confirm": "false",
            },
        }
    )

    assert candidate.side_effect is False
    assert candidate.requires_human_confirm is False
    assert safety_filter_passed(candidate) is True


def test_normalizes_route_case_and_whitespace() -> None:
    candidate = adapt_s2_action_output(
        {
            "route": " Continue ",
            "action": {
                "type": "wait",
                "arguments": {},
            },
            "safety_check": {},
        }
    )

    assert candidate.route == "continue"


def test_raw_is_copied_from_payload() -> None:
    payload = {
        "route": "continue",
        "action": {
            "type": "wait",
            "arguments": {},
        },
        "safety_check": {
            "side_effect": "false",
        },
    }

    candidate = adapt_s2_action_output(payload)
    payload["route"] = "halt"
    payload["action"]["type"] = "back"
    payload["safety_check"]["side_effect"] = "true"

    assert candidate.raw is not payload
    assert candidate.raw["route"] == "continue"
    assert candidate.raw["action"]["type"] == "wait"
    assert candidate.raw["safety_check"]["side_effect"] == "false"


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
