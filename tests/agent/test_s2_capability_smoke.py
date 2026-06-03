from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanobot.agent.s2_capability_smoke import (
    S2CapabilityError,
    S2SmokeCase,
    _amain,
    actions_are_materially_different,
    adapt_s2_action_output,
    build_s2_action_messages,
    extract_json_object,
    parse_args,
    run_s2_capability_smoke,
    safety_filter_passed,
)
from nanobot.providers.base import LLMResponse
from opengui.action import Action

PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4"
    b"\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05"
    b"\xfe\x02\xfeA\xe2`\x82\x00\x00\x00\x00IEND\xaeB`\x82"
)
PNG_1X1_ALT = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4"
    b"\x89\x00\x00\x00\rIDATx\x9cc\xf8\xcf\xc0\xf0\x1f\x00"
    b"\x05\x00\x01\xff\x89\x99=\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
)


class FakeProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = []

    async def chat_with_retry(self, **kwargs) -> LLMResponse:
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        return LLMResponse(content=response, usage={"total_tokens": 10})


class ImageStrippingProvider:
    async def chat_with_retry(self, **kwargs) -> LLMResponse:
        messages = kwargs["messages"]
        messages[1]["content"] = [
            block for block in messages[1]["content"] if block["type"] != "image_url"
        ]
        return LLMResponse(
            content="""{
                "route": "continue",
                "action": {"type": "back", "arguments": {}},
                "safety_check": {}
            }""",
            usage={"total_tokens": 7},
        )


def test_parse_args_accepts_required_smoke_inputs(tmp_path: Path) -> None:
    screenshot_a = tmp_path / "a.png"
    screenshot_b = tmp_path / "b.png"
    output = tmp_path / "report.json"

    args = parse_args(
        [
            "--task",
            "选择 2026-06-05 的出发日期",
            "--screenshot-a",
            str(screenshot_a),
            "--screenshot-b",
            str(screenshot_b),
            "--output",
            str(output),
            "--max-tokens",
            "256",
        ]
    )

    assert args.task == "选择 2026-06-05 的出发日期"
    assert args.screenshot_a == screenshot_a
    assert args.screenshot_b == screenshot_b
    assert args.output == output
    assert args.max_tokens == 256


async def test_amain_rejects_explicit_missing_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_config = tmp_path / "missing.json"

    def fail_load_config(_path: Path) -> None:
        raise AssertionError("load_config should not be called")

    monkeypatch.setattr("nanobot.config.loader.load_config", fail_load_config)

    with pytest.raises(SystemExit, match=str(missing_config)):
        await _amain(
            [
                "--task",
                "选择 2026-06-05 的出发日期",
                "--screenshot-a",
                str(tmp_path / "a.png"),
                "--config",
                str(missing_config),
            ]
        )


async def test_amain_converts_provider_snapshot_value_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    config = object()
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda _path: config)
    monkeypatch.setattr(
        "nanobot.config.loader.resolve_config_env_vars",
        lambda value: value,
    )

    def raise_invalid_config(_config: object) -> None:
        raise ValueError("unknown provider")

    monkeypatch.setattr(
        "nanobot.providers.factory.build_gui_s2_provider_snapshot",
        raise_invalid_config,
    )

    with pytest.raises(
        SystemExit,
        match="Invalid GUI S2 provider configuration: unknown provider",
    ):
        await _amain(
            [
                "--task",
                "选择 2026-06-05 的出发日期",
                "--screenshot-a",
                str(tmp_path / "a.png"),
                "--config",
                str(config_path),
            ]
        )


async def test_amain_writes_report_json_and_returns_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "reports" / "smoke.json"
    config = object()
    snapshot = SimpleNamespace(provider=object(), model="s2-model")
    report_json = {
        "model": "s2-model",
        "task": "选择 2026-06-05 的出发日期",
        "schema_parse_success": False,
    }
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda _path: config)
    monkeypatch.setattr(
        "nanobot.config.loader.resolve_config_env_vars",
        lambda value: value,
    )
    monkeypatch.setattr(
        "nanobot.providers.factory.build_gui_s2_provider_snapshot",
        lambda _config: snapshot,
    )

    async def fake_run_s2_capability_smoke(**_kwargs):
        return SimpleNamespace(to_json_dict=lambda: report_json)

    monkeypatch.setattr(
        "nanobot.agent.s2_capability_smoke.run_s2_capability_smoke",
        fake_run_s2_capability_smoke,
    )

    result = await _amain(
        [
            "--task",
            "选择 2026-06-05 的出发日期",
            "--screenshot-a",
            str(tmp_path / "a.png"),
            "--output",
            str(output_path),
        ]
    )

    assert result == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == report_json
    assert output_path.read_text(encoding="utf-8").endswith("\n")


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


def test_build_s2_action_messages_specify_canonical_executable_arguments(
    tmp_path: Path,
) -> None:
    screenshot = tmp_path / "screen.png"
    screenshot.write_bytes(PNG_1X1)

    messages = build_s2_action_messages(
        task="选择 2026-06-05 的出发日期",
        screenshot_path=screenshot,
        case_name="datepicker",
    )

    prompt = messages[0]["content"].lower()
    assert "click arguments must contain x, y, relative" in prompt
    assert "swipe arguments must contain x, y, x2, y2, relative" in prompt
    assert "type arguments must contain text, auto_enter" in prompt
    assert "wait arguments must contain duration_ms" in prompt
    assert "done arguments must contain status" in prompt
    assert "back and home arguments must be empty objects" in prompt
    assert "chosen from the screenshot as relative integers in [0, 999]" in prompt
    assert "relative=true" in prompt
    assert "point, coordinate, direction, or distance" in prompt
    assert "do not use aliases" in prompt
    assert "500" not in prompt
    assert "300" not in prompt
    assert "700" not in prompt
    assert '"text to enter"' not in prompt
    assert "1000" not in prompt


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


def test_actions_are_materially_different() -> None:
    first = adapt_s2_action_output(
        {
            "route": "continue",
            "action": {
                "type": "click",
                "arguments": {"x": 100, "y": 200},
            },
            "semantic_target": "select_date",
            "safety_check": {},
        }
    )
    second = adapt_s2_action_output(
        {
            "route": "continue",
            "action": {
                "type": "back",
                "arguments": {},
            },
            "semantic_target": "recover_page",
            "safety_check": {},
        }
    )

    assert actions_are_materially_different(first, second) is True


def test_actions_are_not_materially_different_for_same_action_with_different_prose() -> None:
    first = adapt_s2_action_output(
        {
            "route": "continue",
            "action": {
                "type": "click",
                "arguments": {"x": 100, "y": 200},
            },
            "semantic_target": "select_date",
            "safety_check": {},
        }
    )
    second = adapt_s2_action_output(
        {
            "route": "continue",
            "action": {
                "type": "click",
                "arguments": {"x": 100, "y": 200},
            },
            "semantic_target": "different_description",
            "safety_check": {},
        }
    )

    assert actions_are_materially_different(first, second) is False


async def test_run_s2_capability_smoke_passes_with_contrasting_actions(
    tmp_path: Path,
) -> None:
    screen_a = tmp_path / "date_picker.png"
    screen_b = tmp_path / "wrong_page.png"
    screen_a.write_bytes(PNG_1X1)
    screen_b.write_bytes(PNG_1X1_ALT)
    provider = FakeProvider(
        [
            """{
                "route": "continue",
                "action": {"type": "click", "arguments": {"x": 100, "y": 200}},
                "reason": "select the date",
                "semantic_target": "select_date",
                "safety_check": {"side_effect": false, "requires_human_confirm": false}
            }""",
            """{
                "route": "continue",
                "action": {"type": "back", "arguments": {}},
                "reason": "recover from wrong page",
                "semantic_target": "recover_page",
                "safety_check": {"side_effect": false, "requires_human_confirm": false}
            }""",
        ]
    )

    report = await run_s2_capability_smoke(
        provider=provider,
        model="qwen3.5-397b-a17b",
        task="选择 2026-06-05 的出发日期",
        case_a=S2SmokeCase(name="date_picker", screenshot_path=screen_a),
        case_b=S2SmokeCase(name="wrong_page", screenshot_path=screen_b),
    )

    assert report.model == "qwen3.5-397b-a17b"
    assert report.cases[0].finish_reason == "stop"
    assert report.cases[1].finish_reason == "stop"
    assert report.schema_parse_success is True
    assert report.action_adapter_success is True
    assert report.unsafe_action_filter_pass is True
    assert report.image_use_contrast_pass is True
    assert len(provider.calls) == 2


async def test_run_s2_capability_smoke_fails_contrast_for_identical_screenshot_bytes(
    tmp_path: Path,
) -> None:
    screen_a = tmp_path / "date_picker.png"
    screen_b = tmp_path / "wrong_page.png"
    screen_a.write_bytes(PNG_1X1)
    screen_b.write_bytes(PNG_1X1)
    provider = FakeProvider(
        [
            '{"route":"continue","action":{"type":"back","arguments":{}},"safety_check":{}}',
            '{"route":"continue","action":{"type":"home","arguments":{}},"safety_check":{}}',
        ]
    )

    report = await run_s2_capability_smoke(
        provider=provider,
        model="qwen3.5-397b-a17b",
        task="选择 2026-06-05 的出发日期",
        case_a=S2SmokeCase(name="date_picker", screenshot_path=screen_a),
        case_b=S2SmokeCase(name="wrong_page", screenshot_path=screen_b),
    )

    assert report.image_use_contrast_pass is False


async def test_run_s2_capability_smoke_uses_identical_text_for_contrast_cases(
    tmp_path: Path,
) -> None:
    screen_a = tmp_path / "date_picker.png"
    screen_b = tmp_path / "wrong_page.png"
    screen_a.write_bytes(PNG_1X1)
    screen_b.write_bytes(PNG_1X1)
    provider = FakeProvider(
        [
            '{"route":"continue","action":{"type":"back","arguments":{}},"safety_check":{}}',
            '{"route":"continue","action":{"type":"home","arguments":{}},"safety_check":{}}',
        ]
    )

    await run_s2_capability_smoke(
        provider=provider,
        model="qwen3.5-397b-a17b",
        task="选择 2026-06-05 的出发日期",
        case_a=S2SmokeCase(name="date_picker", screenshot_path=screen_a),
        case_b=S2SmokeCase(name="wrong_page", screenshot_path=screen_b),
    )

    first_text = provider.calls[0]["messages"][1]["content"][1]["text"]
    second_text = provider.calls[1]["messages"][1]["content"][1]["text"]
    assert first_text == second_text


async def test_run_s2_capability_smoke_fails_contrast_when_actions_match(
    tmp_path: Path,
) -> None:
    screen_a = tmp_path / "date_picker.png"
    screen_b = tmp_path / "wrong_page.png"
    screen_a.write_bytes(PNG_1X1)
    screen_b.write_bytes(PNG_1X1)
    response = """{
        "route": "continue",
        "action": {"type": "click", "arguments": {"x": 100, "y": 200}},
        "reason": "select the date",
        "semantic_target": "select_date",
        "safety_check": {"side_effect": false, "requires_human_confirm": false}
    }"""
    provider = FakeProvider([response, response])

    report = await run_s2_capability_smoke(
        provider=provider,
        model="qwen3.5-397b-a17b",
        task="选择 2026-06-05 的出发日期",
        case_a=S2SmokeCase(name="date_picker", screenshot_path=screen_a),
        case_b=S2SmokeCase(name="wrong_page", screenshot_path=screen_b),
    )

    assert report.image_use_contrast_pass is False


async def test_run_s2_capability_smoke_rejects_image_stripped_fallback(
    tmp_path: Path,
) -> None:
    screen = tmp_path / "screen.png"
    screen.write_bytes(PNG_1X1)

    report = await run_s2_capability_smoke(
        provider=ImageStrippingProvider(),
        model="qwen3.5-397b-a17b",
        task="选择 2026-06-05 的出发日期",
        case_a=S2SmokeCase(name="date_picker", screenshot_path=screen),
    )

    result = report.cases[0]
    assert result.schema_parse_success is False
    assert result.action_adapter_success is False
    assert result.error is not None
    assert "without image content" in result.error


async def test_run_s2_capability_smoke_fails_contrast_for_unsafe_candidate(
    tmp_path: Path,
) -> None:
    screen_a = tmp_path / "date_picker.png"
    screen_b = tmp_path / "wrong_page.png"
    screen_a.write_bytes(PNG_1X1)
    screen_b.write_bytes(PNG_1X1)
    provider = FakeProvider(
        [
            '{"route":"continue","action":{"type":"click","arguments":{"x":100,"y":200}},'
            '"safety_check":{"side_effect":true}}',
            '{"route":"continue","action":{"type":"back","arguments":{}},"safety_check":{}}',
        ]
    )

    report = await run_s2_capability_smoke(
        provider=provider,
        model="qwen3.5-397b-a17b",
        task="选择 2026-06-05 的出发日期",
        case_a=S2SmokeCase(name="date_picker", screenshot_path=screen_a),
        case_b=S2SmokeCase(name="wrong_page", screenshot_path=screen_b),
    )

    assert report.cases[0].schema_parse_success is True
    assert report.cases[0].action_adapter_success is True
    assert report.cases[0].unsafe_action_filter_pass is False
    assert report.image_use_contrast_pass is False


async def test_run_s2_capability_smoke_preserves_parse_success_and_usage_on_invalid_action(
    tmp_path: Path,
) -> None:
    screen = tmp_path / "screen.png"
    screen.write_bytes(PNG_1X1)
    provider = FakeProvider(
        [
            '{"route":"continue","action":{"type":"click","arguments":{}},'
            '"safety_check":{}}'
        ]
    )

    report = await run_s2_capability_smoke(
        provider=provider,
        model="qwen3.5-397b-a17b",
        task="选择 2026-06-05 的出发日期",
        case_a=S2SmokeCase(name="date_picker", screenshot_path=screen),
    )

    result = report.cases[0]
    assert result.schema_parse_success is True
    assert result.action_adapter_success is False
    assert result.usage == {"total_tokens": 10}
    assert report.usage == {"total_tokens": 10}


@pytest.mark.parametrize("finish_reason", ["error", "content_filter"])
async def test_run_s2_capability_smoke_treats_unsuccessful_finish_reason_as_provider_failure(
    tmp_path: Path,
    finish_reason: str,
) -> None:
    screen = tmp_path / "screen.png"
    screen.write_bytes(PNG_1X1)

    class ErrorProvider:
        async def chat_with_retry(self, **kwargs) -> LLMResponse:
            return LLMResponse(
                content='{"route":"continue","action":{"type":"back","arguments":{}}}',
                finish_reason=finish_reason,
                usage={"total_tokens": 3},
            )

    report = await run_s2_capability_smoke(
        provider=ErrorProvider(),
        model="qwen3.5-397b-a17b",
        task="选择 2026-06-05 的出发日期",
        case_a=S2SmokeCase(name="date_picker", screenshot_path=screen),
    )

    result = report.cases[0]
    assert result.schema_parse_success is False
    assert result.action_adapter_success is False
    assert result.usage == {"total_tokens": 3}
    assert result.finish_reason == finish_reason
    assert result.error is not None
    assert finish_reason in result.error


async def test_run_s2_capability_smoke_reports_no_finish_reason_without_response(
    tmp_path: Path,
) -> None:
    screen = tmp_path / "screen.txt"
    screen.write_text("not an image")
    provider = FakeProvider([])

    report = await run_s2_capability_smoke(
        provider=provider,
        model="qwen3.5-397b-a17b",
        task="选择 2026-06-05 的出发日期",
        case_a=S2SmokeCase(name="date_picker", screenshot_path=screen),
    )

    assert report.cases[0].finish_reason is None
