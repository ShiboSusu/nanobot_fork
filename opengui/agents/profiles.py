"""MobileWorld-aligned agent profile registry for OpenGUI.

The prompt, parser, action-space, and history behavior in this module mirrors
the vendored MobileWorld agents under :mod:`opengui.agents.implementations`.
GeneralE2E is copied from MobileWorld itself; several other MobileWorld agents
retain their upstream-origin comments in the vendored implementation files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from opengui.agents.implementations import gelab_agent
from opengui.agents.implementations import general_e2e_agent
from opengui.agents.implementations import gui_owl_1_5
from opengui.agents.implementations import mai_ui_agent
from opengui.agents.implementations import planner_executor
from opengui.agents.implementations import qwen3vl
from opengui.agents.implementations import seed_agent
from opengui.agents.implementations import ui_venus_agent
from opengui.agents.runtime.models import (
    ANSWER,
    ASK_USER,
    CLICK,
    DOUBLE_TAP,
    DRAG,
    ENV_FAIL,
    FINISHED,
    INPUT_TEXT,
    KEYBOARD_ENTER,
    LONG_PRESS,
    MCP,
    NAVIGATE_BACK,
    NAVIGATE_HOME,
    OPEN_APP,
    SCROLL,
    UNKNOWN,
    WAIT,
)
from opengui.agents.utils.helpers import pil_adaptive_resize, pil_to_base64, reverse_swipe_direction
from opengui.agents.utils.prompts import (
    GELAB_INSTRUCTION_SUFFIX,
    GELAB_SYSTEM_PROMPT,
    GELAB_USER_PROMPT_TEMPLATE,
    GENERAL_E2E_PROMPT_TEMPLATE,
    GUI_OWL_1_5_SYSTEM_PROMPT_TEMPLATE,
    GUI_OWL_1_5_USER_PROMPT_TEMPLATE,
    GUI_OWL_1_5_USER_PROMPT_WITH_HISTSTEPS_TEMPLATE,
    MAI_MOBILE_SYS_PROMPT_ASK_USER_MCP,
    MOBILE_QWEN3VL_PROMPT_WITH_ASK_USER,
    MOBILE_QWEN3VL_USER_TEMPLATE,
    PLANNER_EXECUTOR_PROMPT_TEMPLATE,
    SEED_PROMPT,
)
from opengui.interfaces import LLMResponse, ToolCall
from opengui.observation import Observation

SUPPORTED_AGENT_PROFILES: tuple[str, ...] = (
    "general_e2e",
    "mobileworld_general_e2e",
    "mobileworld_general_e2e_compact_skill",
    "planner_executor",
    "qwen3vl",
    "mai_ui",
    "gelab",
    "seed",
    "gui_owl_1_5",
    "ui_venus",
)

_PROFILE_ALIASES: dict[str | None, str] = {
    None: "general_e2e",
    "": "general_e2e",
    "default": "general_e2e",
    "general": "general_e2e",
    "general_e2e_compact_skill": "general_e2e",
    "mobileworld-general-e2e": "mobileworld_general_e2e",
    "mw_general_e2e": "mobileworld_general_e2e",
    "mw-general-e2e": "mobileworld_general_e2e",
    "mobileworld-general-e2e-compact-skill": "mobileworld_general_e2e_compact_skill",
    "mw_general_e2e_compact_skill": "mobileworld_general_e2e_compact_skill",
    "mw-general-e2e-compact-skill": "mobileworld_general_e2e_compact_skill",
    "gui-owl-1.5": "gui_owl_1_5",
    "gui_owl": "gui_owl_1_5",
    "venus": "ui_venus",
}

_CLAUDE_IMAGE_SIZE = (1280, 720)
_CLAUDE_OPUS_MAX_DIMENSION = 1280
DEFAULT_SCROLL_PIXELS = 400

_MOBILEWORLD_TOOL = {
    "type": "function",
    "function": {
        "name": "mobile_use",
        "description": "MobileWorld textual profile tool schema; OpenGUI parses this from assistant text.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


def canonicalize_agent_profile(profile_name: str | None) -> str:
    key = (profile_name or "").strip().lower()
    key = _PROFILE_ALIASES.get(key, key or "general_e2e")
    if key not in SUPPORTED_AGENT_PROFILES:
        raise ValueError(
            f"Unsupported agent profile {profile_name!r}. "
            f"Expected one of: {', '.join(SUPPORTED_AGENT_PROFILES)}."
        )
    return key


def profile_uses_native_tools(profile_name: str | None) -> bool:
    del profile_name
    return False


def coordinate_mode_for_profile(profile_name: str | None, model_name: str = "") -> str:
    del profile_name, model_name
    return "absolute"


def profile_tool_definition(profile_name: str | None) -> dict[str, Any]:
    del profile_name
    return _MOBILEWORLD_TOOL


def _is_general_e2e_profile(profile_name: str | None) -> bool:
    return canonicalize_agent_profile(profile_name) in {
        "general_e2e",
        "mobileworld_general_e2e",
        "mobileworld_general_e2e_compact_skill",
    }


def prompt_contract_for_profile(profile_name: str | None) -> dict[str, tuple[str, ...]]:
    profile = canonicalize_agent_profile(profile_name)
    return {
        "environment": (f"- MobileWorld agent profile: {profile}.",),
        "format": ("Use the exact MobileWorld response format for this profile.",),
        "rules": ("Do not use provider-native tool calling for MobileWorld profiles.",),
    }


def build_mobileworld_messages(
    profile_name: str | None,
    *,
    task: str,
    current_observation: Observation,
    history: list[Any],
    model_name: str,
    history_image_window: int,
    compact_prompt_parts: Any | None = None,
) -> list[dict[str, Any]]:
    profile = canonicalize_agent_profile(profile_name)
    if _is_general_e2e_profile(profile):
        return _build_general_e2e_messages(
            task=task,
            current_observation=current_observation,
            history=history,
            model_name=model_name,
            history_image_window=history_image_window,
            prompt_template=GENERAL_E2E_PROMPT_TEMPLATE,
            compact_prompt_parts=compact_prompt_parts,
        )
    if profile == "planner_executor":
        return _build_planner_executor_messages(
            task=task,
            current_observation=current_observation,
            history=history,
            history_image_window=history_image_window,
        )
    if profile == "qwen3vl":
        return _build_qwen3vl_messages(
            task=task, current_observation=current_observation, history=history
        )
    if profile == "mai_ui":
        return _build_mai_ui_messages(
            task=task,
            current_observation=current_observation,
            history=history,
            history_image_window=history_image_window,
        )
    if profile == "gelab":
        return _build_gelab_messages(
            task=task, current_observation=current_observation, history=history
        )
    if profile == "seed":
        return _build_seed_messages(
            task=task,
            current_observation=current_observation,
            history=history,
            history_image_window=history_image_window,
        )
    if profile == "gui_owl_1_5":
        return _build_gui_owl_messages(
            task=task,
            current_observation=current_observation,
            history=history,
            history_image_window=history_image_window,
        )
    if profile == "ui_venus":
        return _build_ui_venus_messages(
            task=task,
            current_observation=current_observation,
            history=history,
            history_image_window=history_image_window,
        )
    raise ValueError(f"Unsupported MobileWorld profile: {profile}")


def normalize_profile_response(profile_name: str | None, response: LLMResponse) -> LLMResponse:
    """Parse a MobileWorld textual response without screen context.

    This compatibility path is used by older helper call sites. The main
    GuiAgent loop uses ``normalize_profile_response_for_observation`` so
    MobileWorld parsers can convert coordinates against the actual screenshot.
    """
    return normalize_profile_response_for_screen(
        profile_name,
        response,
        screen_width=999,
        screen_height=999,
        model_name="",
        fallback_relative=True,
    )


def normalize_profile_response_for_observation(
    profile_name: str | None,
    response: LLMResponse,
    observation: Observation,
    *,
    model_name: str = "",
) -> LLMResponse:
    return normalize_profile_response_for_screen(
        profile_name,
        response,
        screen_width=int(observation.screen_width or 999),
        screen_height=int(observation.screen_height or 999),
        model_name=model_name,
    )


def normalize_profile_response_for_screen(
    profile_name: str | None,
    response: LLMResponse,
    *,
    screen_width: int,
    screen_height: int,
    model_name: str = "",
    fallback_relative: bool = False,
) -> LLMResponse:
    profile = canonicalize_agent_profile(profile_name)
    content = response.content or ""
    if not content.strip() and response.tool_calls:
        return response
    try:
        payload = parse_mobileworld_action(
            profile,
            content,
            screen_width=screen_width,
            screen_height=screen_height,
            model_name=model_name,
        )
    except Exception as exc:
        raise ValueError(f"Failed to parse {profile} response: {exc}") from exc
    if fallback_relative and payload.get("action_type") in {
        "tap",
        "long_press",
        "double_tap",
        "drag",
        "swipe",
        "scroll",
    }:
        payload.setdefault("relative", True)
    return LLMResponse(
        content=response.content,
        tool_calls=[ToolCall(id="content-tool-call-0", name="computer_use", arguments=payload)],
        raw=response.raw,
        usage=response.usage,
        ttft_s=response.ttft_s,
        latency_s=response.latency_s,
    )


def parse_mobileworld_action(
    profile_name: str | None,
    content: str,
    *,
    screen_width: int,
    screen_height: int,
    model_name: str = "",
) -> dict[str, Any]:
    profile = canonicalize_agent_profile(profile_name)
    if _is_general_e2e_profile(profile):
        action_str = _general_e2e_action_text(content)
        action = general_e2e_agent.parse_response_to_action(
            action_str,
            screen_width,
            screen_height,
            scale_factor=_general_e2e_scale_factor(model_name, screen_width, screen_height),
        )
        return _to_opengui_payload(action, summary=content)
    if profile == "planner_executor":
        _thought, action_str = planner_executor.parse_action(content)
        action = planner_executor.parsing_planner_response_to_android_world_env_action(action_str)
        if action.get("action_type") in {"click", "double_tap", "long_press", "drag"}:
            action = {
                "action_type": UNKNOWN,
                "text": "planner_executor target grounding requires a MobileWorld executor agent.",
            }
        return _to_opengui_payload(action, summary=content)
    if profile == "qwen3vl":
        structured = qwen3vl.parse_action_to_structure_output(content)
        action = qwen3vl.parsing_response_to_andoid_world_env_action(
            structured,
            image_height=screen_height,
            image_width=screen_width,
        )
        return _to_opengui_payload(action, summary=structured.get("conclusion") or content)
    if profile == "mai_ui":
        structured = mai_ui_agent.parse_action_to_structure_output(content)
        action = _mai_ui_to_action(
            structured, screen_width=screen_width, screen_height=screen_height
        )
        return _to_opengui_payload(action, summary=structured.get("thinking") or content)
    if profile == "gelab":
        action = gelab_agent.transform_gelab_action(
            gelab_agent.parse_gelab_response(content),
            width=screen_width,
            height=screen_height,
        )
        return _to_opengui_payload(action, summary=content)
    if profile == "seed":
        parsed = seed_agent.parse_seed_xml_action(content)
        if not parsed:
            raise ValueError("No Seed action parsed from response.")
        action = _seed_to_action(parsed[0], screen_width=screen_width, screen_height=screen_height)
        return _to_opengui_payload(action, summary=content)
    if profile == "gui_owl_1_5":
        structured = gui_owl_1_5.parse_action_to_structure_output(content)
        action = gui_owl_1_5.parsing_response_to_andoid_world_env_action(
            structured,
            image_height=screen_height,
            image_width=screen_width,
        )
        return _to_opengui_payload(action, summary=structured.get("conclusion") or content)
    if profile == "ui_venus":
        action_text = _between(content, "<action>", "</action>")
        action_name, action_params = ui_venus_agent.parse_answer(action_text)
        action = ui_venus_agent.convert_venus_action_to_json_action(
            action_name,
            action_params,
            origin_h=screen_height,
            origin_w=screen_width,
        )
        return _to_opengui_payload(
            action, summary=_between(content, "<conclusion>", "</conclusion>") or content
        )
    raise ValueError(f"Unsupported MobileWorld profile: {profile}")


def _build_general_e2e_messages(
    *,
    task: str,
    current_observation: Observation,
    history: list[Any],
    model_name: str,
    history_image_window: int,
    prompt_template: Any,
    compact_prompt_parts: Any | None = None,
) -> list[dict[str, Any]]:
    observations = [turn.observation for turn in history] + [current_observation]
    tool_results = [turn.tool_result_message.get("content") for turn in history]
    responses = [_history_compact_response(turn) for turn in history]
    scale_factor = _general_e2e_scale_factor(
        model_name,
        int(current_observation.screen_width or 999),
        int(current_observation.screen_height or 999),
    )
    messages = [
        {
            "role": "system",
            "content": prompt_template.render(
                tools="",
                scale_factor=scale_factor,
                extra_action_rows=getattr(compact_prompt_parts, "action_rows", "") or "",
                decision_rules=getattr(compact_prompt_parts, "decision_rules", "") or "",
                compact_skill_instructions=getattr(
                    compact_prompt_parts,
                    "compact_skill_instructions",
                    "",
                )
                or "",
            ),
        },
        _general_user_message(
            observations[0],
            tool_result=None,
            ask_user_response=None,
            instruction=task,
            model_name=model_name,
        ),
    ]
    for index, response in enumerate(responses):
        messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        messages.append(
            _general_user_message(
                observations[index + 1],
                tool_result=tool_results[index],
                ask_user_response=None,
                instruction=None,
                model_name=model_name,
            )
        )
    return _hide_history_images_like_general(messages, history_image_window)


def _build_planner_executor_messages(
    *,
    task: str,
    current_observation: Observation,
    history: list[Any],
    history_image_window: int,
) -> list[dict[str, Any]]:
    observations = [turn.observation for turn in history] + [current_observation]
    tool_results = [turn.tool_result_message.get("content") for turn in history]
    responses = [_history_raw_response(turn) for turn in history]
    messages = [
        {
            "role": "system",
            "content": PLANNER_EXECUTOR_PROMPT_TEMPLATE.render(goal=task, tools=""),
        },
        _planner_user_message(observations[0], tool_result=None, ask_user_response=None),
    ]
    for index, response in enumerate(responses):
        messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        messages.append(
            _planner_user_message(
                observations[index + 1],
                tool_result=tool_results[index],
                ask_user_response=None,
            )
        )
    return _hide_history_images_like_general(messages, history_image_window)


def _build_qwen3vl_messages(
    *, task: str, current_observation: Observation, history: list[Any]
) -> list[dict[str, Any]]:
    steps = ""
    for idx, turn in enumerate(history):
        conclusion = turn.action_summary.replace("\n", "").replace('"', "")
        tool_result = turn.tool_result_message.get("content")
        if tool_result:
            conclusion += f"; Tool call result: <tool_response>{tool_result}</tool_response>"
        steps += f"Step {idx + 1}: {conclusion}; "
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": MOBILE_QWEN3VL_PROMPT_WITH_ASK_USER.render(tools="")}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": MOBILE_QWEN3VL_USER_TEMPLATE.format(instruction=task, steps=steps),
                },
                _image_content(current_observation),
            ],
        },
    ]


def _build_mai_ui_messages(
    *,
    task: str,
    current_observation: Observation,
    history: list[Any],
    history_image_window: int,
) -> list[dict[str, Any]]:
    observations = [turn.observation for turn in history] + [current_observation]
    messages = [
        {"role": "system", "content": MAI_MOBILE_SYS_PROMPT_ASK_USER_MCP.render(tools=None)},
        {"role": "user", "content": [{"type": "text", "text": task}]},
        _mai_user_message(observations[0], None),
    ]
    for index, turn in enumerate(history):
        messages.append({"role": "assistant", "content": _history_raw_response(turn)})
        messages.append(
            _mai_user_message(observations[index + 1], turn.tool_result_message.get("content"))
        )
    return _drop_old_image_messages(messages, history_image_window)


def _build_gelab_messages(
    *, task: str, current_observation: Observation, history: list[Any]
) -> list[dict[str, Any]]:
    summary_history = history[-1].action_summary if history else ""
    user_prompt = GELAB_USER_PROMPT_TEMPLATE.render(
        task=task,
        history_display=summary_history if summary_history else "暂无历史操作",
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": GELAB_SYSTEM_PROMPT},
                {"type": "text", "text": user_prompt},
                _image_content(current_observation),
                {"type": "text", "text": GELAB_INSTRUCTION_SUFFIX},
            ],
        }
    ]


def _build_seed_messages(
    *,
    task: str,
    current_observation: Observation,
    history: list[Any],
    history_image_window: int,
) -> list[dict[str, Any]]:
    observations = [turn.observation for turn in history] + [current_observation]
    messages = [
        {
            "role": "system",
            "content": "You are provided with a task description, a history of previous actions, and corresponding screenshots. Your goal is to perform the next action to complete the task. Please note that if performing the same action multiple times results in a static screen with no changes, you should attempt a modified or alternative action.",
        },
        {"role": "system", "content": SEED_PROMPT.render(tools=[])},
        {"role": "user", "content": task},
        _seed_user_message(observations[0], None),
    ]
    for index, turn in enumerate(history):
        messages.append({"role": "assistant", "content": _history_raw_response(turn)})
        messages.append(
            _seed_user_message(observations[index + 1], turn.tool_result_message.get("content"))
        )
    return _drop_old_tool_images(messages, history_image_window)


def _build_gui_owl_messages(
    *,
    task: str,
    current_observation: Observation,
    history: list[Any],
    history_image_window: int,
) -> list[dict[str, Any]]:
    observations = [turn.observation for turn in history] + [current_observation]
    total_history_count = len(history)
    keep_as_messages = min(max(0, history_image_window - 1), total_history_count)
    text_history_count = total_history_count - keep_as_messages
    first_user_content: list[dict[str, Any]] = []
    if text_history_count > 0:
        previous_steps = "\n".join(
            f"Step{i + 1}: {history[i].action_summary}. Tool response: {history[i].tool_result_message.get('content') or 'None'}"
            for i in range(text_history_count)
        )
        first_user_content.append(
            {
                "type": "text",
                "text": GUI_OWL_1_5_USER_PROMPT_WITH_HISTSTEPS_TEMPLATE.format(
                    instruction=task,
                    previous_steps=previous_steps,
                ),
            }
        )
    else:
        first_user_content.append(
            {"type": "text", "text": GUI_OWL_1_5_USER_PROMPT_TEMPLATE.format(instruction=task)}
        )
    first_user_content.append(_image_content(observations[text_history_count]))
    messages = [
        {"role": "system", "content": GUI_OWL_1_5_SYSTEM_PROMPT_TEMPLATE.render(tools="")},
        {"role": "user", "content": first_user_content},
    ]
    for index in range(text_history_count, total_history_count):
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": _history_raw_response(history[index]).strip()}
                ],
            }
        )
        messages.append(
            _gui_owl_user_message(
                observations[index + 1], history[index].tool_result_message.get("content")
            )
        )
    return messages


def _build_ui_venus_messages(
    *,
    task: str,
    current_observation: Observation,
    history: list[Any],
    history_image_window: int,
) -> list[dict[str, Any]]:
    recent = history[-history_image_window:] if history_image_window > 0 else []
    previous_actions = "\n".join(
        f"Step {idx}: <think>{turn.state_summary or ''}</think><action>{turn.action_summary}</action>"
        for idx, turn in enumerate(recent)
    )
    query = ui_venus_agent.UI_VENUS_15_PROMPT.format(
        user_task=task, previous_actions=previous_actions
    )
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [{"type": "text", "text": query}, _image_content(current_observation)],
        },
    ]


def _history_raw_response(turn: Any) -> str:
    raw = getattr(turn, "raw_response_content", None)
    if isinstance(raw, str) and raw.strip():
        return raw
    content = (
        turn.assistant_message.get("content") if isinstance(turn.assistant_message, dict) else None
    )
    return str(content or turn.action_summary or "")


def _history_compact_response(turn: Any) -> str:
    step = getattr(turn, "step_index", "?")
    summary = getattr(turn, "action_intent", None) or getattr(turn, "action_summary", None)
    lines = [f"Step {step}: {summary or _history_raw_response(turn)}"]
    state = getattr(turn, "state_summary", None)
    if isinstance(state, str) and state.strip():
        lines.append(f"State summary: {state.strip()}")
    tool_result_message = getattr(turn, "tool_result_message", {}) or {}
    tool_result = (
        tool_result_message.get("content")
        if isinstance(tool_result_message, dict) else None
    )
    if isinstance(tool_result, str) and tool_result.strip():
        lines.append(f"Tool result: {tool_result.strip()}")
    return "\n".join(lines)


def _general_user_message(
    observation: Observation,
    *,
    tool_result: Any,
    ask_user_response: Any,
    instruction: str | None,
    model_name: str,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if instruction is not None:
        content.append({"type": "text", "text": instruction})
    if tool_result is not None:
        content.append({"type": "text", "text": f"Tool call result: {tool_result}"})
    elif ask_user_response is not None:
        content.append({"type": "text", "text": str(ask_user_response)})
    content.append(_general_e2e_image_content(observation, model_name=model_name))
    return {"role": "user", "content": content}


def _planner_user_message(
    observation: Observation,
    *,
    tool_result: Any,
    ask_user_response: Any,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if tool_result is not None:
        content.append({"type": "text", "text": f"Tool call result: {tool_result}"})
    elif ask_user_response is not None:
        content.append({"type": "text", "text": str(ask_user_response)})
    content.append(_image_content_raw(observation))
    return {"role": "user", "content": content}


def _mai_user_message(observation: Observation, tool_result: Any) -> dict[str, Any]:
    if tool_result is not None:
        return {
            "role": "user",
            "content": [{"type": "text", "text": f"Tool call result: {tool_result}"}],
        }
    return {"role": "user", "content": [_image_content(observation)]}


def _seed_user_message(observation: Observation, tool_result: Any) -> dict[str, Any]:
    if tool_result is not None:
        return {
            "role": "user",
            "content": [{"type": "text", "text": f"Tool call result: {tool_result}"}],
        }
    return {"role": "tool", "content": [_image_content(observation)], "tool_call_id": "1"}


def _gui_owl_user_message(observation: Observation, tool_result: Any) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "<tool_response>\n"},
            {"type": "text", "text": str(tool_result) if tool_result is not None else "None"},
            _image_content(observation),
            {"type": "text", "text": "\n</tool_response>"},
        ],
    }


def _image_content(observation: Observation) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{_observation_base64(observation)}"},
    }


def _image_content_raw(observation: Observation) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{_observation_base64(observation)}"},
    }


def _general_e2e_image_content(observation: Observation, *, model_name: str) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{_general_e2e_observation_base64(observation, model_name=model_name)}"
        },
    }


def _observation_base64(observation: Observation) -> str:
    if not observation.screenshot_path:
        raise ValueError("MobileWorld profiles require screenshots.")
    path = Path(observation.screenshot_path)
    with Image.open(path) as image:
        return pil_to_base64(image.convert("RGB"))


def _general_e2e_observation_base64(observation: Observation, *, model_name: str) -> str:
    if not observation.screenshot_path:
        raise ValueError("MobileWorld profiles require screenshots.")
    path = Path(observation.screenshot_path)
    with Image.open(path) as image:
        image = image.convert("RGB")
        model = model_name.lower()
        if "opus-4" in model or "opus_4" in model:
            image, _, _ = pil_adaptive_resize(image, _CLAUDE_OPUS_MAX_DIMENSION)
        elif "claude" in model:
            image = image.resize(_CLAUDE_IMAGE_SIZE)
        return pil_to_base64(image)


def _general_e2e_scale_factor(
    model_name: str,
    screen_width: int,
    screen_height: int,
) -> int | tuple[int, int]:
    model = model_name.lower()
    if "opus-4" in model or "opus_4" in model:
        largest = max(screen_width, screen_height)
        if largest <= _CLAUDE_OPUS_MAX_DIMENSION:
            return (screen_width, screen_height)
        scale = _CLAUDE_OPUS_MAX_DIMENSION / largest
        return (max(1, round(screen_width * scale)), max(1, round(screen_height * scale)))
    if "claude" in model:
        return _CLAUDE_IMAGE_SIZE
    if "kimi-k" in model:
        return 1
    return 1000


def _general_e2e_action_text(content: str) -> str:
    if "Action:" not in content:
        action_str = content.strip()
    else:
        try:
            _thought, action_str = general_e2e_agent.parse_action(content)
        except ValueError:
            action_str = content.strip()
    try:
        parsed = general_e2e_agent.parse_json_markdown(action_str)
    except Exception:
        parsed = _last_json_object(action_str) or _last_json_object(content)
    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        parsed = parsed[0]
    if isinstance(parsed, dict):
        action = dict(parsed)
        if "action_type" not in action and "action" in action:
            action["action_type"] = action.pop("action")
        if "coordinate" not in action and "target" in action:
            action["coordinate"] = action.pop("target")
        return json.dumps(action, ensure_ascii=False)
    return action_str


def _last_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    found: dict[str, Any] | None = None
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            found = parsed
    return found


def _hide_history_images_like_general(
    messages: list[dict[str, Any]], history_image_window: int
) -> list[dict[str, Any]]:
    used = 0
    for idx in range(len(messages) - 1, -1, -1):
        message = messages[idx]
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        image_idx = next(
            (i for i, item in enumerate(content) if item.get("type") == "image_url"), None
        )
        if image_idx is None:
            continue
        if used < history_image_window:
            used += 1
        else:
            content[image_idx] = {"type": "text", "text": "(Previous turn, screen not shown)"}
    return messages


def _drop_old_image_messages(
    messages: list[dict[str, Any]], history_image_window: int
) -> list[dict[str, Any]]:
    image_positions = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user"
        and isinstance(message.get("content"), list)
        and any(item.get("type") == "image_url" for item in message["content"])
    ]
    remove = (
        set(image_positions[:-history_image_window])
        if history_image_window > 0
        else set(image_positions)
    )
    return [message for index, message in enumerate(messages) if index not in remove]


def _drop_old_tool_images(
    messages: list[dict[str, Any]], history_image_window: int
) -> list[dict[str, Any]]:
    image_positions = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "tool"
        and isinstance(message.get("content"), list)
        and message["content"]
        and message["content"][0].get("type") == "image_url"
    ]
    remove = (
        set(image_positions[:-history_image_window])
        if history_image_window > 0
        else set(image_positions)
    )
    return [message for index, message in enumerate(messages) if index not in remove]


def _mai_ui_to_action(
    structured: dict[str, Any], *, screen_width: int, screen_height: int
) -> dict[str, Any]:
    tool_name = structured.get("tool_name", "mobile_use")
    action_json = structured["action_json"]
    if tool_name != "mobile_use":
        return {"action_type": MCP, "action_name": tool_name, "action_json": action_json}
    action_type = action_json.get("action", UNKNOWN)
    if action_type in {"click", "long_press", "double_click"}:
        x, y = _coord_to_pixel(action_json.get("coordinate"), screen_width, screen_height)
        return {
            "action_type": {"click": CLICK, "long_press": LONG_PRESS, "double_click": DOUBLE_TAP}[
                action_type
            ],
            "x": x,
            "y": y,
        }
    if action_type == "swipe":
        direction = reverse_swipe_direction(action_json.get("direction", "up"))
        payload: dict[str, Any] = {"action_type": SCROLL, "direction": direction}
        if action_json.get("coordinate"):
            x, y = _coord_to_pixel(action_json["coordinate"], screen_width, screen_height)
            payload.update({"x": x, "y": y})
        return payload
    if action_type == "drag":
        sx, sy = _coord_to_pixel(
            action_json.get("start_coordinate", [0, 0]), screen_width, screen_height
        )
        ex, ey = _coord_to_pixel(
            action_json.get("end_coordinate", [0, 0]), screen_width, screen_height
        )
        return {"action_type": DRAG, "start_x": sx, "start_y": sy, "end_x": ex, "end_y": ey}
    if action_type == "system_button":
        button = str(action_json.get("button", "")).lower()
        return {
            "action_type": {
                "back": NAVIGATE_BACK,
                "home": NAVIGATE_HOME,
                "enter": KEYBOARD_ENTER,
            }.get(button, UNKNOWN)
        }
    if action_type == "type":
        return {"action_type": INPUT_TEXT, "text": action_json.get("text", "")}
    if action_type == "open":
        return {"action_type": OPEN_APP, "app_name": action_json.get("text", "")}
    if action_type == "terminate":
        return {"action_type": FINISHED, "text": action_json.get("status", "success")}
    if action_type == "answer":
        return {"action_type": ANSWER, "text": action_json.get("text", "")}
    if action_type == "ask_user":
        return {"action_type": ASK_USER, "text": action_json.get("text", "")}
    if action_type == "wait":
        return {"action_type": WAIT}
    return {"action_type": UNKNOWN, "text": f"Unknown action: {action_type}"}


def _seed_to_action(
    parsed_action: dict[str, Any], *, screen_width: int, screen_height: int
) -> dict[str, Any]:
    func_name = parsed_action["function"]
    params = parsed_action["parameters"]
    if func_name == seed_agent.FINISH_WORD:
        return {"action_type": ANSWER, "text": params.get("content", "success")}
    if func_name == seed_agent.WAIT_WORD:
        return {"action_type": WAIT}
    if func_name == seed_agent.CALL_USER:
        return {"action_type": ASK_USER, "text": params.get("content", "")}
    if func_name in {"click", "left_double", "long_press"}:
        x, y = seed_agent.parse_point_string(params.get("point", "0 0"))
        action_type = {"click": CLICK, "left_double": DOUBLE_TAP, "long_press": LONG_PRESS}[
            func_name
        ]
        return {
            "action_type": action_type,
            "x": int(x * screen_width / 1000),
            "y": int(y * screen_height / 1000),
        }
    if func_name == "drag":
        sx, sy = seed_agent.parse_point_string(params.get("start_point", "0 0"))
        ex, ey = seed_agent.parse_point_string(params.get("end_point", "0 0"))
        return {
            "action_type": DRAG,
            "start_x": int(sx * screen_width / 1000),
            "start_y": int(sy * screen_height / 1000),
            "end_x": int(ex * screen_width / 1000),
            "end_y": int(ey * screen_height / 1000),
        }
    if func_name == "scroll":
        x, y = seed_agent.parse_point_string(params.get("point", "500 500"))
        return {
            "action_type": SCROLL,
            "direction": params.get("direction", "down"),
            "x": int(x * screen_width / 1000),
            "y": int(y * screen_height / 1000),
        }
    if func_name == "type":
        return {"action_type": INPUT_TEXT, "text": params.get("content", "")}
    if func_name == "press_home":
        return {"action_type": NAVIGATE_HOME}
    if func_name == "press_back":
        return {"action_type": NAVIGATE_BACK}
    return {"action_type": UNKNOWN, "text": f"Unknown action: {func_name}"}


def _to_opengui_payload(action: dict[str, Any], *, summary: str) -> dict[str, Any]:
    action_type = action.get("action_type")
    payload: dict[str, Any] = {"summary": summary, "intent": summary}
    if action_type in {CLICK, "click"}:
        payload.update({"action_type": "tap", "x": action.get("x"), "y": action.get("y")})
    elif action_type in {LONG_PRESS, "long_press"}:
        payload.update({"action_type": "long_press", "x": action.get("x"), "y": action.get("y")})
    elif action_type in {DOUBLE_TAP, "double_tap"}:
        payload.update({"action_type": "double_tap", "x": action.get("x"), "y": action.get("y")})
    elif action_type in {DRAG, "drag"}:
        payload.update(
            {
                "action_type": "drag",
                "x": action.get("start_x", action.get("x")),
                "y": action.get("start_y", action.get("y")),
                "x2": action.get("end_x", action.get("x2")),
                "y2": action.get("end_y", action.get("y2")),
            }
        )
    elif action_type in {SCROLL, "scroll"}:
        payload.update(
            {
                "action_type": "scroll",
                "direction": action.get("direction", action.get("text", "down")),
                "pixels": action.get("pixels", DEFAULT_SCROLL_PIXELS),
                "x": action.get("x"),
                "y": action.get("y"),
            }
        )
    elif action_type in {INPUT_TEXT, "input_text"}:
        payload.update({"action_type": "input_text", "text": action.get("text", "")})
    elif action_type in {OPEN_APP, "open_app"}:
        payload.update(
            {"action_type": "open_app", "text": action.get("app_name") or action.get("text", "")}
        )
    elif action_type in {NAVIGATE_BACK, "navigate_back"}:
        payload.update({"action_type": "back"})
    elif action_type in {NAVIGATE_HOME, "navigate_home"}:
        payload.update({"action_type": "home"})
    elif action_type in {KEYBOARD_ENTER, "keyboard_enter"}:
        payload.update({"action_type": "enter"})
    elif action_type in {WAIT, "wait"}:
        payload.update({"action_type": "wait"})
    elif action_type in {ANSWER, FINISHED, "answer", "finished"}:
        payload.update(
            {"action_type": "done", "status": _done_status(action), "text": action.get("text", "")}
        )
    elif action_type in {
        ASK_USER,
        MCP,
        UNKNOWN,
        ENV_FAIL,
        "ask_user",
        "mcp",
        "unknown",
        "env_fail",
    }:
        payload.update(
            {
                "action_type": "request_intervention",
                "text": action.get("text") or f"Unsupported MobileWorld action: {action_type}",
            }
        )
    else:
        payload.update(
            {
                "action_type": str(action_type or "request_intervention"),
                **{k: v for k, v in action.items() if k != "action_type"},
            }
        )
    return {key: value for key, value in payload.items() if value is not None}


def _done_status(action: dict[str, Any]) -> str:
    text = str(
        action.get("text") or action.get("status") or action.get("goal_status") or "success"
    ).lower()
    return "failure" if "fail" in text or "infeasible" in text or "abort" in text else "success"


def _coord_to_pixel(coord: Any, screen_width: int, screen_height: int) -> tuple[int, int]:
    if not isinstance(coord, (list, tuple)) or len(coord) < 2:
        raise ValueError(f"Invalid coordinate: {coord!r}")
    return int(float(coord[0]) * screen_width), int(float(coord[1]) * screen_height)


def _between(text: str, start: str, end: str) -> str:
    if start not in text or end not in text:
        return ""
    return text.split(start, 1)[1].split(end, 1)[0].strip()
