"""Saved-screenshot S2 capability smoke utilities.

This module adapts S2 action-schema output without executing GUI actions,
device commands, or backend commands. Its CLI may call the configured S2
endpoint using saved screenshots only.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from nanobot.providers.base import LLMProvider
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

Use these exact canonical action argument fields:
- click arguments must contain x, y, relative.
- swipe arguments must contain x, y, x2, y2, relative.
- type arguments must contain text, auto_enter; text is a string and auto_enter
  is a boolean.
- wait arguments must contain duration_ms as an integer.
- done arguments must contain status as a string.
- back and home arguments must be empty objects.
All x, y, x2, and y2 values must be
chosen from the screenshot as relative integers in [0, 999], and relative=true.
Do not use aliases such as point, coordinate, direction, or distance in action
arguments.

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


@dataclass(frozen=True)
class S2SmokeCase:
    name: str
    screenshot_path: Path


@dataclass(frozen=True)
class S2SmokeCaseResult:
    name: str
    screenshot_path: str
    raw_content: str
    route: str | None
    action_type: str | None
    reason: str
    semantic_target: str
    schema_parse_success: bool
    action_adapter_success: bool
    unsafe_action_filter_pass: bool
    finish_reason: str | None
    error: str | None
    latency_s: float
    usage: dict[str, int]


@dataclass(frozen=True)
class S2SmokeReport:
    model: str
    task: str
    cases: list[S2SmokeCaseResult]
    schema_parse_success: bool
    action_adapter_success: bool
    unsafe_action_filter_pass: bool
    image_use_contrast_pass: bool
    duration_s: float
    usage: dict[str, int]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run saved-screenshot S2 visual GUI executor capability smoke."
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--screenshot-a", type=Path, required=True)
    parser.add_argument("--screenshot-b", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args(argv)


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


def actions_are_materially_different(
    first: S2ActionCandidate,
    second: S2ActionCandidate,
) -> bool:
    if first.route != second.route:
        return True
    if (first.action is None) != (second.action is None):
        return True
    if first.action is None and second.action is None:
        return False

    assert first.action is not None
    assert second.action is not None
    return first.action != second.action


def _messages_contain_image_url(messages: Sequence[Mapping[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if any(
            isinstance(block, Mapping) and block.get("type") == "image_url"
            for block in content
        ):
            return True
    return False


def _screenshots_are_byte_distinct(first: Path, second: Path) -> bool:
    try:
        return first.read_bytes() != second.read_bytes()
    except OSError:
        return False


async def _run_case(
    *,
    provider: LLMProvider,
    model: str,
    task: str,
    case: S2SmokeCase,
    max_tokens: int,
) -> tuple[S2SmokeCaseResult, S2ActionCandidate | None]:
    started = time.perf_counter()
    raw_content = ""
    usage: dict[str, int] = {}
    finish_reason: str | None = None
    schema_parse_success = False
    action_adapter_success = False
    try:
        messages = build_s2_action_messages(
            task=task,
            screenshot_path=case.screenshot_path,
            case_name="image_contrast",
        )
        response = await provider.chat_with_retry(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=0,
        )
        raw_content = response.content or ""
        usage = dict(response.usage or {})
        finish_reason = response.finish_reason
        if finish_reason != "stop":
            raise S2CapabilityError(
                f"Provider returned unsuccessful finish_reason={finish_reason!r}."
            )
        if not _messages_contain_image_url(messages):
            raise S2CapabilityError(
                "Provider retry/fallback completed without image content."
            )
        payload = extract_json_object(raw_content)
        schema_parse_success = True
        candidate = adapt_s2_action_output(payload)
        action_adapter_success = True
        filter_pass = safety_filter_passed(candidate)
        result = S2SmokeCaseResult(
            name=case.name,
            screenshot_path=str(case.screenshot_path),
            raw_content=raw_content,
            route=candidate.route,
            action_type=candidate.action.action_type if candidate.action else None,
            reason=candidate.reason,
            semantic_target=candidate.semantic_target,
            schema_parse_success=schema_parse_success,
            action_adapter_success=action_adapter_success,
            unsafe_action_filter_pass=filter_pass,
            finish_reason=finish_reason,
            error=None,
            latency_s=time.perf_counter() - started,
            usage=usage,
        )
        return result, candidate
    except Exception as exc:
        result = S2SmokeCaseResult(
            name=case.name,
            screenshot_path=str(case.screenshot_path),
            raw_content=raw_content,
            route=None,
            action_type=None,
            reason="",
            semantic_target="",
            schema_parse_success=schema_parse_success,
            action_adapter_success=action_adapter_success,
            unsafe_action_filter_pass=False,
            finish_reason=finish_reason,
            error=f"{type(exc).__name__}: {exc}",
            latency_s=time.perf_counter() - started,
            usage=usage,
        )
        return result, None


def _merge_usage(results: Sequence[S2SmokeCaseResult]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for result in results:
        for key, value in result.usage.items():
            if isinstance(value, int):
                merged[key] = merged.get(key, 0) + value
    return merged


async def run_s2_capability_smoke(
    *,
    provider: LLMProvider,
    model: str,
    task: str,
    case_a: S2SmokeCase,
    case_b: S2SmokeCase | None = None,
    max_tokens: int = 512,
) -> S2SmokeReport:
    started = time.perf_counter()
    case_results: list[S2SmokeCaseResult] = []
    candidates: list[S2ActionCandidate | None] = []

    result_a, candidate_a = await _run_case(
        provider=provider,
        model=model,
        task=task,
        case=case_a,
        max_tokens=max_tokens,
    )
    case_results.append(result_a)
    candidates.append(candidate_a)

    if case_b is not None:
        result_b, candidate_b = await _run_case(
            provider=provider,
            model=model,
            task=task,
            case=case_b,
            max_tokens=max_tokens,
        )
        case_results.append(result_b)
        candidates.append(candidate_b)

    image_use_contrast_pass = (
        len(candidates) == 2
        and candidates[0] is not None
        and candidates[1] is not None
        and case_b is not None
        and _screenshots_are_byte_distinct(
            case_a.screenshot_path,
            case_b.screenshot_path,
        )
        and case_results[0].unsafe_action_filter_pass
        and case_results[1].unsafe_action_filter_pass
        and actions_are_materially_different(candidates[0], candidates[1])
    )

    return S2SmokeReport(
        model=model,
        task=task,
        cases=case_results,
        schema_parse_success=all(result.schema_parse_success for result in case_results),
        action_adapter_success=all(
            result.action_adapter_success for result in case_results
        ),
        unsafe_action_filter_pass=all(
            result.unsafe_action_filter_pass for result in case_results
        ),
        image_use_contrast_pass=image_use_contrast_pass,
        duration_s=time.perf_counter() - started,
        usage=_merge_usage(case_results),
    )


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


async def _amain(argv: Sequence[str] | None = None) -> int:
    from nanobot.config.loader import load_config, resolve_config_env_vars
    from nanobot.providers.factory import build_gui_s2_provider_snapshot

    args = parse_args(argv)
    if args.config is not None and not args.config.is_file():
        raise SystemExit(f"Config path is not a file: {args.config}")

    config = resolve_config_env_vars(load_config(args.config))
    try:
        snapshot = build_gui_s2_provider_snapshot(config)
    except ValueError as exc:
        raise SystemExit(
            f"Invalid GUI S2 provider configuration: {exc}"
        ) from None
    if snapshot is None:
        raise SystemExit(
            "No GUI S2 model configured. Set gui.s2Model and gui.s2Provider."
        )

    report = await run_s2_capability_smoke(
        provider=snapshot.provider,
        model=snapshot.model,
        task=args.task,
        case_a=S2SmokeCase(
            name="screenshot_a",
            screenshot_path=args.screenshot_a,
        ),
        case_b=(
            S2SmokeCase(
                name="screenshot_b",
                screenshot_path=args.screenshot_b,
            )
            if args.screenshot_b is not None
            else None
        ),
        max_tokens=args.max_tokens,
    )
    output = json.dumps(report.to_json_dict(), ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_amain(argv))


if __name__ == "__main__":
    raise SystemExit(main())
