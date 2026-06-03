"""Offline S2 capability smoke utilities.

This module adapts S2 action-schema output for tests only; it does not execute
GUI actions, device commands, live endpoints, or backend calls.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

from nanobot.utils.helpers import build_image_content_blocks, detect_image_mime
from opengui.action import Action, ActionError, parse_action


ALLOWED_ROUTES = {"continue", "done", "halt", "human_confirm"}
ACTION_TYPE_ALIASES = {
    "click": "tap",
    "type": "input_text",
    "swipe": "swipe",
    "wait": "wait",
    "back": "back",
    "home": "home",
    "done": "done",
}
S2_ACTION_SYSTEM_PROMPT = """You are an offline S2 GUI action smoke planner.
Inspect the screenshot and task, then return only JSON.

The JSON schema is:
{
  "route": "continue" | "done" | "halt" | "human_confirm",
  "action": {
    "type": "click" | "type" | "swipe" | "wait" | "back" | "home" | "done",
    "arguments": {}
  },
  "reason": "brief user-visible reason",
  "semantic_target": "visible target or state this action addresses",
  "safety_check": {
    "side_effect": false,
    "requires_human_confirm": false
  }
}

Do not include markdown, prose, hidden reasoning, chain-of-thought, or fields
outside the schema. Halt or request human_confirm for any action involving
payment, purchase, send, submit, delete, account modification, privacy toggles,
or sensitive permission grants. Never propose those side-effecting actions.
"""


class S2CapabilityError(ValueError):
    """Raised when S2 smoke output cannot be adapted safely."""


@dataclass(frozen=True)
class S2ActionCandidate:
    route: str
    action: Action | None
    reason: str
    semantic_target: str
    side_effect: bool
    requires_human_confirm: bool
    raw: Mapping[str, Any]


def build_s2_action_messages(
    *, task: str, screenshot_path: Path, case_name: str
) -> list[dict[str, Any]]:
    raw = screenshot_path.read_bytes()
    mime = detect_image_mime(raw)
    if mime is None:
        raise S2CapabilityError(f"Unsupported image file: {screenshot_path}")

    label = (
        f"Case: {case_name}\n"
        f"Task: {task}\n"
        "Return the next GUI action JSON for this exact screenshot."
    )
    return [
        {"role": "system", "content": S2_ACTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_image_content_blocks(
                raw, mime, str(screenshot_path), label
            ),
        },
    ]


def extract_json_object(content: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.strip()
    fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise S2CapabilityError("No JSON object found in S2 output.")

    try:
        value = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise S2CapabilityError(f"Invalid JSON object in S2 output: {exc.msg}.") from exc

    if not isinstance(value, dict):
        raise S2CapabilityError("S2 output JSON must be an object.")
    return value


def adapt_s2_action_output(payload: Mapping[str, Any]) -> S2ActionCandidate:
    route = str(payload.get("route", "")).strip().lower()
    if route not in ALLOWED_ROUTES:
        raise S2CapabilityError(f"Unsupported S2 route: {route!r}.")

    safety_check = payload.get("safety_check", {})
    if not isinstance(safety_check, Mapping):
        raise S2CapabilityError("S2 safety_check must be an object.")

    action_payload = payload.get("action")
    action: Action | None = None
    if route in {"continue", "done"}:
        if not isinstance(action_payload, Mapping):
            raise S2CapabilityError(f"S2 route {route!r} requires an action object.")
        action = _adapt_action_payload(action_payload)
        if route == "done" and action.action_type != "done":
            raise S2CapabilityError("S2 route 'done' requires a done action.")
    elif action_payload is not None:
        if not isinstance(action_payload, Mapping):
            raise S2CapabilityError("S2 action must be an object when present.")
        action = _adapt_action_payload(action_payload)

    return S2ActionCandidate(
        route=route,
        action=action,
        reason=_optional_string(payload.get("reason")),
        semantic_target=_optional_string(payload.get("semantic_target")),
        side_effect=_coerce_bool(safety_check.get("side_effect", False)),
        requires_human_confirm=_coerce_bool(
            safety_check.get("requires_human_confirm", False)
        ),
        raw=copy.deepcopy(payload),
    )


def safety_filter_passed(candidate: S2ActionCandidate) -> bool:
    return not (candidate.side_effect or candidate.requires_human_confirm)


def _adapt_action_payload(payload: Mapping[str, Any]) -> Action:
    raw_type = payload.get("action_type", payload.get("type", payload.get("action", "")))
    action_type = ACTION_TYPE_ALIASES.get(str(raw_type).strip().lower(), str(raw_type).strip().lower())
    arguments = payload.get("arguments", {})
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, Mapping):
        raise S2CapabilityError("S2 action arguments must be an object.")

    normalized = dict(arguments)
    normalized["action_type"] = action_type
    for key, value in payload.items():
        if key not in {"type", "action_type", "action", "arguments"} and key not in normalized:
            normalized[key] = value

    try:
        return parse_action(normalized)
    except ActionError as exc:
        raise S2CapabilityError(f"Invalid S2 action: {exc}") from exc


def _optional_string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no", ""}:
            return False
    return bool(value)
