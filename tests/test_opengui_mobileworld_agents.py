from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from opengui.action import parse_action
from opengui.agent import GuiAgent
from opengui.agent_profiles import (
    build_mobileworld_messages,
    canonicalize_agent_profile,
    normalize_profile_response_for_screen,
)
from opengui.backends.dry_run import DryRunBackend
from opengui.interfaces import LLMResponse
from opengui.observation import Observation
from opengui.trajectory.recorder import TrajectoryRecorder


class _RecordingLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    async def chat(self, messages, tools=None, tool_choice=None, **kwargs):  # noqa: ANN001
        self.calls.append(messages)
        assert tools is None
        assert tool_choice is None
        if not self._responses:
            raise AssertionError("No scripted response left")
        return self._responses.pop(0)


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (1, 1), (255, 255, 255, 255)).save(path)


def _observation(path: Path) -> Observation:
    _write_png(path)
    return Observation(
        screenshot_path=str(path),
        screen_width=1080,
        screen_height=1920,
        foreground_app="Settings",
        platform="android",
    )


def test_default_profile_aliases_mobileworld_general_e2e() -> None:
    assert canonicalize_agent_profile(None) == "general_e2e"
    assert canonicalize_agent_profile("default") == "general_e2e"
    assert canonicalize_agent_profile("mobileworld_general_e2e") == "mobileworld_general_e2e"
    assert canonicalize_agent_profile("mobileworld-general-e2e") == "mobileworld_general_e2e"
    assert (
        canonicalize_agent_profile("mobileworld_general_e2e_compact_skill")
        == "mobileworld_general_e2e_compact_skill"
    )
    assert (
        canonicalize_agent_profile("mw-general-e2e-compact-skill")
        == "mobileworld_general_e2e_compact_skill"
    )
    assert canonicalize_agent_profile("planner_executor") == "planner_executor"


def test_general_e2e_messages_use_mobileworld_prompt(tmp_path: Path) -> None:
    messages = build_mobileworld_messages(
        "general_e2e",
        task="Open Settings",
        current_observation=_observation(tmp_path / "screen.png"),
        history=[],
        model_name="qwen",
        history_image_window=3,
    )

    assert messages[0]["role"] == "system"
    assert "# Role: Android Phone Operator AI" in messages[0]["content"]
    assert "Thought:" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"][0]["text"] == "Open Settings"
    assert messages[1]["content"][1]["type"] == "image_url"


def test_general_e2e_history_uses_compact_summary_not_raw_response(tmp_path: Path) -> None:
    history = [
        SimpleNamespace(
            step_index=1,
            observation=_observation(tmp_path / "previous.png"),
            action_intent="tap Settings",
            action_summary="tap Settings",
            state_summary="Settings opened",
            tool_result_message={"content": "executed:tap"},
            assistant_message={"role": "assistant", "content": "assistant fallback"},
            raw_response_content="RAW_SHOULD_NOT_APPEAR " * 100,
        )
    ]

    messages = build_mobileworld_messages(
        "general_e2e",
        task="Open Wi-Fi",
        current_observation=_observation(tmp_path / "current.png"),
        history=history,
        model_name="qwen",
        history_image_window=1,
    )
    assistant_text = messages[2]["content"][0]["text"]

    assert "RAW_SHOULD_NOT_APPEAR" not in assistant_text
    assert "Step 1: tap Settings" in assistant_text
    assert "State summary: Settings opened" in assistant_text
    assert "Tool result: executed:tap" in assistant_text


def test_prompt_stats_counts_messages_text_and_images() -> None:
    stats = GuiAgent._prompt_stats({
        "messages": [
            {"role": "system", "content": "abc"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
                ],
            },
        ]
    })

    assert stats == {"message_count": 2, "text_chars": 8, "image_count": 1}


def test_general_e2e_parse_uses_real_screen_dimensions() -> None:
    response = LLMResponse(
        content='Thought: tap it\nAction: {"action_type":"click","coordinate":[500,250]}',
        tool_calls=None,
    )

    normalized = normalize_profile_response_for_screen(
        "general_e2e",
        response,
        screen_width=1080,
        screen_height=1920,
    )

    assert normalized.tool_calls is not None
    assert normalized.tool_calls[0].arguments["action_type"] == "tap"
    assert normalized.tool_calls[0].arguments["x"] == 540
    assert normalized.tool_calls[0].arguments["y"] == 480
    assert "relative" not in normalized.tool_calls[0].arguments


def test_mobileworld_general_e2e_compact_skill_uses_general_e2e_parser() -> None:
    response = LLMResponse(
        content='Thought: tap it\nAction: {"action_type":"click","coordinate":[500,250]}',
        tool_calls=None,
    )

    normalized = normalize_profile_response_for_screen(
        "mobileworld_general_e2e_compact_skill",
        response,
        screen_width=1080,
        screen_height=1920,
    )

    assert normalized.tool_calls is not None
    assert normalized.tool_calls[0].arguments["action_type"] == "tap"
    assert normalized.tool_calls[0].arguments["x"] == 540
    assert normalized.tool_calls[0].arguments["y"] == 480


def test_general_e2e_parse_uses_first_action_when_model_outputs_multiple_actions() -> None:
    response = LLMResponse(
        content=(
            'Thought: tap the search box\nAction: {"action_type":"click","coordinate":[500,250]}\n'
            'Thought: maybe done now\nAction: {"action_type":"status","goal_status":"complete"}'
        ),
        tool_calls=None,
    )

    normalized = normalize_profile_response_for_screen(
        "general_e2e",
        response,
        screen_width=1080,
        screen_height=1920,
    )

    assert normalized.tool_calls is not None
    assert normalized.tool_calls[0].arguments["action_type"] == "tap"
    assert normalized.tool_calls[0].arguments["x"] == 540
    assert normalized.tool_calls[0].arguments["y"] == 480


def test_general_e2e_parse_accepts_bare_json_list_action_target() -> None:
    response = LLMResponse(
        content='```json\n[{"action":"tap","target":[146,905]}]\n```',
        tool_calls=None,
    )

    normalized = normalize_profile_response_for_screen(
        "general_e2e",
        response,
        screen_width=496,
        screen_height=1080,
    )

    assert normalized.tool_calls is not None
    arguments = normalized.tool_calls[0].arguments
    assert arguments["action_type"] == "tap"
    assert arguments["x"] == 72
    assert arguments["y"] == 977


def test_general_e2e_parse_extracts_json_after_slow_thinking() -> None:
    response = LLMResponse(
        content=(
            "The model wrote a long chain of text before the action.\n"
            "Action: click the password field first.\n"
            "After reconsidering, output the actual action:\n"
            '{"action_type":"click","coordinate":[250,400],"confidence":0.7}'
        ),
        tool_calls=None,
    )

    normalized = normalize_profile_response_for_screen(
        "general_e2e",
        response,
        screen_width=1000,
        screen_height=1000,
    )

    assert normalized.tool_calls is not None
    arguments = normalized.tool_calls[0].arguments
    assert arguments["action_type"] == "tap"
    assert arguments["x"] == 250
    assert arguments["y"] == 400


def test_general_e2e_scroll_adds_opengui_default_pixels() -> None:
    response = LLMResponse(
        content='Thought: scroll\nAction: {"action_type":"scroll","direction":"up"}',
        tool_calls=None,
    )

    normalized = normalize_profile_response_for_screen(
        "general_e2e",
        response,
        screen_width=1080,
        screen_height=1920,
    )

    assert normalized.tool_calls is not None
    arguments = normalized.tool_calls[0].arguments
    assert arguments["action_type"] == "scroll"
    assert arguments["direction"] == "up"
    assert arguments["pixels"] == 400
    action = parse_action(arguments)
    assert action.action_type == "scroll"
    assert action.text == "up"
    assert action.pixels == 400


def test_qwen3vl_parse_uses_real_screen_dimensions() -> None:
    response = LLMResponse(
        content=(
            'Thought: tap\nAction: "Tap target"\n'
            '<tool_call>{"name":"mobile_use","arguments":{"action":"click","coordinate":[500,250]}}</tool_call>'
        ),
        tool_calls=None,
    )

    normalized = normalize_profile_response_for_screen(
        "qwen3vl",
        response,
        screen_width=1080,
        screen_height=1920,
    )

    assert normalized.tool_calls is not None
    assert normalized.tool_calls[0].arguments["action_type"] == "tap"
    assert normalized.tool_calls[0].arguments["x"] == 541
    assert normalized.tool_calls[0].arguments["y"] == 480
    assert "relative" not in normalized.tool_calls[0].arguments


@pytest.mark.asyncio
async def test_gui_agent_uses_mobileworld_messages_and_compact_history(tmp_path: Path) -> None:
    first_response = 'Thought: wait\nAction: {"action_type":"wait"}'
    second_response = 'Thought: done\nAction: {"action_type":"status","goal_status":"complete"}'
    llm = _RecordingLLM(
        [
            LLMResponse(content=first_response, tool_calls=None),
            LLMResponse(content=second_response, tool_calls=None),
        ]
    )
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        TrajectoryRecorder(output_dir=tmp_path / "traj", task="mobileworld agent"),
        artifacts_root=tmp_path / "runs",
        max_steps=2,
        include_date_context=False,
        agent_profile="default",
    )

    result = await agent.run("Wait once and finish", max_retries=1)

    assert result.success is True
    assert len(llm.calls) == 2
    assert "# Role: Android Phone Operator AI" in llm.calls[0][0]["content"]
    assert llm.calls[1][2]["role"] == "assistant"
    history_text = llm.calls[1][2]["content"][0]["text"]
    assert history_text != first_response
    assert "Step 1:" in history_text
    assert 'Action: {"action_type":"wait"}' in history_text
    assert "Tool result: [dry-run] wait" in history_text
    assert llm.calls[1][3]["content"][0]["text"].startswith("Tool call result:")
    assert llm.calls[1][3]["content"][1]["type"] == "image_url"
