"""Offline S2 handoff packet dry-run utilities.

This module is intentionally offline-only: it reads saved traces/screenshots,
calls the configured S2 model with saved evidence, validates the returned action,
and writes audit rows. It does not import or call GUI backends, controller logic,
monitor logic, ADB, WDA, HDC, iOS, or Android code.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from nanobot.agent.s2_capability_smoke import (
    S2CapabilityError,
    adapt_s2_action_output,
    extract_json_object,
    safety_filter_passed,
)
from nanobot.providers.base import LLMProvider
from nanobot.utils.helpers import build_image_content_blocks, detect_image_mime
from opengui.action import Action, describe_action, parse_action

MAX_TRACES = 3
MAX_RECENT_ACTIONS = 5
ALLOWED_RISK_LEVELS = {"U0", "U1"}
FORBIDDEN_ACTIONS = [
    "send",
    "submit",
    "pay",
    "purchase",
    "delete",
    "account_change",
    "privacy_toggle",
    "sensitive_permission_grant",
]
FORBIDDEN_KEYWORDS = {
    "send": ["send", "发送", "发消息", "发出", "外发"],
    "submit": ["submit", "提交", "确认提交"],
    "pay": ["pay", "payment", "支付", "付款"],
    "purchase": ["purchase", "buy", "购买", "下单"],
    "delete": ["delete", "删除"],
    "account_change": ["account", "账号修改", "账户修改", "改密码"],
    "sensitive_permission_grant": [
        "grant permission",
        "allow access",
        "授权",
        "允许访问",
        "开启权限",
        "授予权限",
        "同意权限",
    ],
}
TASK_SIDE_EFFECT_KEYWORDS = {
    category: words
    for category, words in FORBIDDEN_KEYWORDS.items()
    if category != "privacy_toggle"
}
PRIVACY_CONTEXT_KEYWORDS = [
    "privacy",
    "public",
    "follow-list",
    "follower",
    "隐私",
    "公开",
    "不公开",
    "关注列表",
    "粉丝列表",
]
PRIVACY_TOGGLE_VERBS = [
    "toggle",
    "turn on",
    "turn off",
    "switch on",
    "switch off",
    "change",
    "modify",
    "切换",
    "修改",
    "打开",
    "关闭",
    "设置为",
]
PRIVACY_TOGGLE_NEGATIONS = [
    "without changing",
    "without toggling",
    "do not change",
    "不要修改",
    "不修改",
    "不要切换",
    "不切换",
]

S2_HANDOFF_SYSTEM_PROMPT = """You are an offline S2 GUI recovery actor.
Inspect the saved screenshot and the compact handoff packet, then return only
one JSON object.

The JSON schema is:
{
  "route": "continue" | "done" | "halt" | "human_confirm",
  "action": {
    "type": "click" | "type" | "swipe" | "wait" | "back" | "home" | "done",
    "arguments": {}
  },
  "reason": "brief trace-grounded reason",
  "semantic_target": "visible target or state this action advances",
  "safety_check": {
    "side_effect": false,
    "requires_human_confirm": false
  }
}

Use exact canonical action argument fields:
- click arguments must contain x, y, relative.
- swipe arguments must contain x, y, x2, y2, relative.
- type arguments must contain text, auto_enter.
- wait arguments must contain duration_ms.
- done arguments must contain status.
- back and home arguments must be empty objects.
Coordinates must be screenshot-grounded relative integers in [0, 999], with
relative=true. Do not use aliases such as point, coordinate, direction, or
distance.

This is an offline recovery next-action dry-run. Do not claim final task
success. Prefer route=continue only when proposing a safe next recovery action.
Use halt or human_confirm if safe recovery is not possible. Never propose send,
submit, payment, purchase, delete, account modification, privacy toggles, or
sensitive permission grants.
"""


@dataclass(frozen=True)
class HandoffTraceSpec:
    trace_path: Path
    risk_level: str
    failure_mode: str
    success_criteria: str
    recovery_objective: str
    notes: str = ""
    history_quality: str | None = None
    discovery_source: str = "manifest"


@dataclass(frozen=True)
class TraceValidation:
    valid: bool
    error: str | None
    screenshot_path: Path | None
    screenshot_exists: bool
    task_instruction: str
    trace_id: str
    history_quality: str
    events: list[dict[str, Any]]
    step_events: list[dict[str, Any]]


@dataclass(frozen=True)
class S2HandoffDryRunReport:
    summary: dict[str, Any]
    rows: list[dict[str, Any]]

    @property
    def acceptance_ready(self) -> bool:
        return bool(self.summary.get("strong_pass_count", 0) >= 1)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline S2 handoff packet recovery dry-run."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args(argv)


def load_trace_selection_manifest(manifest_path: Path) -> list[HandoffTraceSpec]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    traces = data.get("traces") if isinstance(data, dict) else None
    if not isinstance(traces, list):
        raise ValueError("S2-2 manifest must contain a 'traces' list.")

    specs: list[HandoffTraceSpec] = []
    for row in traces[:MAX_TRACES]:
        if not isinstance(row, Mapping):
            raise ValueError("Each S2-2 manifest trace entry must be an object.")
        raw_trace_path = row.get("trace_path")
        if not isinstance(raw_trace_path, str) or not raw_trace_path:
            raise ValueError("Each S2-2 manifest trace entry requires trace_path.")
        specs.append(
            HandoffTraceSpec(
                trace_path=_normalize_trace_path(Path(raw_trace_path)),
                risk_level=_required_string(row, "risk_level").upper(),
                failure_mode=_required_string(row, "failure_mode"),
                success_criteria=_required_string(row, "success_criteria"),
                recovery_objective=_required_string(row, "recovery_objective"),
                notes=_optional_string(row.get("notes")),
                history_quality=_optional_string_or_none(row.get("history_quality")),
                discovery_source="manifest",
            )
        )
    return specs


def build_handoff_packet(
    spec: HandoffTraceSpec,
) -> tuple[dict[str, Any], TraceValidation]:
    validation = _validate_trace_spec(spec)
    if not validation.valid:
        return {}, validation

    assert validation.screenshot_path is not None
    recent_actions = _recent_actions(validation.step_events)
    known_bad_actions = [
        {
            "action": item["action"],
            "reason": "Recent S1 action from failed or stuck context; do not repeat unless it "
            "clearly advances the recovery objective.",
        }
        for item in recent_actions
    ]
    page_summary = _latest_string(validation.step_events, "state_summary")
    visible_text = _latest_string(validation.step_events, "visible_text")
    foreground_app = _extract_latest_foreground_app(validation.step_events)

    packet = {
        "packet_version": "s2_handoff_v1",
        "trace_id": validation.trace_id,
        "history_quality": validation.history_quality,
        "task": {
            "instruction": validation.task_instruction,
            "success_criteria": spec.success_criteria,
            "risk_level": spec.risk_level,
        },
        "current_state": {
            "screenshot_path": str(validation.screenshot_path),
            "foreground_app": foreground_app or "unknown",
            "page_summary": page_summary[:600],
            "visible_text": visible_text[:800],
        },
        "s1_history_summary": {
            "history_quality": validation.history_quality,
            "recent_actions": recent_actions,
            "failure_mode": spec.failure_mode,
            "s1_failure_hypothesis": _failure_hypotheses(spec),
            "do_not_repeat": _do_not_repeat_patterns(spec),
            "known_bad_actions": known_bad_actions,
        },
        "recovery": {
            "recovery_objective": spec.recovery_objective,
            "remaining_budget": {
                "steps": 1,
                "tokens": 4096,
                "wall_time_s": 60,
            },
            "forbidden_actions": FORBIDDEN_ACTIONS,
        },
        "required_output": {
            "schema": "s2_action_json",
            "allowed_routes": ["continue", "done", "halt", "human_confirm"],
        },
    }
    return packet, validation


async def run_s2_handoff_offline_dry_run(
    *,
    provider: LLMProvider,
    model: str,
    manifest_path: Path,
    max_tokens: int = 512,
) -> S2HandoffDryRunReport:
    specs = load_trace_selection_manifest(manifest_path)
    rows: list[dict[str, Any]] = []
    invalid_reasons: Counter[str] = Counter()
    valid_attempts = 0
    strong_pass_count = 0

    for spec in specs:
        packet, validation = build_handoff_packet(spec)
        if not validation.valid:
            assert validation.error is not None
            invalid_reasons[validation.error] += 1
            rows.append(_invalid_row(spec, validation, model=model))
            continue

        valid_attempts += 1
        row = await _run_valid_trace(
            provider=provider,
            model=model,
            spec=spec,
            packet=packet,
            validation=validation,
            max_tokens=max_tokens,
        )
        rows.append(row)
        if _is_strong_pass(row):
            strong_pass_count += 1

    summary = {
        "row_type": "summary",
        "candidate_traces_found": len(specs),
        "valid_traces_attempted": valid_attempts,
        "invalid_traces_skipped": sum(invalid_reasons.values()),
        "invalid_reason_counts": dict(invalid_reasons),
        "strong_pass_count": strong_pass_count,
        "task_success_claimed": False,
        "device_action_executed": False,
        "live_controller_or_monitor_called": False,
        "u2_u3_attempted": False,
    }
    return S2HandoffDryRunReport(summary=summary, rows=rows)


def write_jsonl_report(report: S2HandoffDryRunReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [report.summary, *report.rows]
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_s2_handoff_messages(
    *, packet: Mapping[str, Any], screenshot_path: Path
) -> list[dict[str, Any]]:
    raw = screenshot_path.read_bytes()
    mime = detect_image_mime(raw)
    if mime is None:
        raise S2CapabilityError(f"Unsupported image file: {screenshot_path}")

    label = (
        "S2 handoff packet follows. Return exactly one recovery next-action JSON "
        "for this saved screenshot and handoff packet.\n\n"
        f"Handoff packet:\n{json.dumps(packet, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": S2_HANDOFF_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_image_content_blocks(
                raw, mime, str(screenshot_path), label
            ),
        },
    ]


async def _run_valid_trace(
    *,
    provider: LLMProvider,
    model: str,
    spec: HandoffTraceSpec,
    packet: Mapping[str, Any],
    validation: TraceValidation,
    max_tokens: int,
) -> dict[str, Any]:
    assert validation.screenshot_path is not None
    started = time.perf_counter()
    row = _base_row(spec, validation, model=model)
    raw_content = ""
    usage: dict[str, int] = {}
    finish_reason: str | None = None
    try:
        messages = build_s2_handoff_messages(
            packet=packet,
            screenshot_path=validation.screenshot_path,
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

        payload = extract_json_object(raw_content)
        row["s2_output_parse_success"] = True
        candidate = adapt_s2_action_output(payload)
        row["route"] = candidate.route
        row["action_summary"] = describe_action(candidate.action) if candidate.action else None
        row["action_adapter_success"] = True
        row["safety_filter_pass"] = safety_filter_passed(candidate)
        row["forbidden_action_hit"] = _forbidden_action_hit(
            candidate=candidate,
            spec=spec,
            validation=validation,
        )
        row["known_bad_action_repeated"] = _known_bad_action_repeated(
            candidate.action,
            packet,
        )
        row["restart_like_action"] = _is_restart_like(candidate.action)
        rejection_reason = _automatic_rejection_reason(
            candidate=candidate,
            row=row,
        )
        row["rejection_reason"] = rejection_reason
        row["automatic_plausibility_checks_pass"] = rejection_reason is None
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        row["s2_output_raw"] = raw_content
        row["finish_reason"] = finish_reason
        row["usage"] = usage
        row["latency_s"] = time.perf_counter() - started
    return row


def _automatic_rejection_reason(
    *,
    candidate: Any,
    row: Mapping[str, Any],
) -> str | None:
    if candidate.route == "done":
        return "done_not_accepted_for_recovery_dry_run"
    if not row["safety_filter_pass"]:
        return "safety_filter_failed"
    if row["forbidden_action_hit"]:
        return "forbidden_action_hit"
    if row["known_bad_action_repeated"]:
        return "known_bad_action_repeated"
    if candidate.route != "continue":
        return "route_not_recovery_action"
    if candidate.action is None:
        return "missing_action"
    if not candidate.reason.strip() or not candidate.semantic_target.strip():
        return "missing_trace_grounded_reason_or_target"
    return None


def _invalid_row(
    spec: HandoffTraceSpec,
    validation: TraceValidation,
    *,
    model: str,
) -> dict[str, Any]:
    row = _base_row(spec, validation, model=model)
    row["handoff_packet_built"] = False
    row["error"] = validation.error
    row["rejection_reason"] = validation.error
    return row


def _base_row(
    spec: HandoffTraceSpec,
    validation: TraceValidation,
    *,
    model: str,
) -> dict[str, Any]:
    return {
        "row_type": "trace",
        "trace_id": validation.trace_id,
        "trace_path": str(spec.trace_path),
        "discovery_source": spec.discovery_source,
        "task_instruction": validation.task_instruction,
        "risk_level": spec.risk_level,
        "failure_mode": spec.failure_mode,
        "history_quality": validation.history_quality,
        "handoff_packet_built": validation.valid,
        "screenshot_exists": validation.screenshot_exists,
        "screenshot_path": str(validation.screenshot_path) if validation.screenshot_path else None,
        "s2_model": model,
        "s2_output_raw": "",
        "s2_output_parse_success": False,
        "route": None,
        "action_summary": None,
        "action_adapter_success": False,
        "safety_filter_pass": False,
        "forbidden_action_hit": False,
        "known_bad_action_repeated": False,
        "automatic_plausibility_checks_pass": False,
        "stateful_recovery_plausible": None,
        "recovery_objective_advanced": None,
        "restart_like_action": False,
        "human_audit_required": True,
        "human_audit_plausible": None,
        "human_audit_notes": "",
        "rejection_reason": None,
        "task_success_claimed": False,
        "error": None,
        "finish_reason": None,
        "latency_s": 0.0,
        "usage": {},
    }


def _validate_trace_spec(spec: HandoffTraceSpec) -> TraceValidation:
    events = _load_trace_events(spec.trace_path)
    step_events = _step_events(events)
    trace_id = _extract_trace_id(events, spec.trace_path)
    task_instruction = _extract_task_instruction(events)
    screenshot_path = _select_current_screenshot(spec.trace_path, step_events)
    screenshot_exists = screenshot_path is not None and screenshot_path.is_file()
    history_quality = _history_quality(spec, step_events)

    error: str | None = None
    if spec.risk_level not in ALLOWED_RISK_LEVELS:
        error = "invalid_risk_level"
    elif _task_mentions_forbidden(task_instruction):
        error = "forbidden_task_side_effect"
    elif not spec.trace_path.is_file():
        error = "missing_trace"
    elif not task_instruction:
        error = "missing_task_instruction"
    elif not screenshot_exists:
        error = "missing_screenshot"
    elif history_quality == "insufficient" and spec.history_quality != "insufficient":
        error = "insufficient_history"

    return TraceValidation(
        valid=error is None,
        error=error,
        screenshot_path=screenshot_path,
        screenshot_exists=screenshot_exists,
        task_instruction=task_instruction,
        trace_id=trace_id,
        history_quality=history_quality,
        events=events,
        step_events=step_events,
    )


def _normalize_trace_path(path: Path) -> Path:
    return path / "trace.jsonl" if path.is_dir() else path


def _load_trace_events(trace_path: Path) -> list[dict[str, Any]]:
    if not trace_path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for raw_line in trace_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _step_events(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(event)
        for event in events
        if event.get("event") == "step" or event.get("type") == "step"
    ]


def _history_quality(spec: HandoffTraceSpec, step_events: Sequence[Mapping[str, Any]]) -> str:
    if len(step_events) >= 3:
        return "sufficient"
    if spec.history_quality == "insufficient":
        return "insufficient"
    return "insufficient"


def _extract_trace_id(events: Sequence[Mapping[str, Any]], trace_path: Path) -> str:
    for event in events:
        value = event.get("trace_id") or event.get("run_id")
        if isinstance(value, str) and value:
            return value
    return trace_path.parent.name or trace_path.stem


def _extract_task_instruction(events: Sequence[Mapping[str, Any]]) -> str:
    keys = ("task_instruction", "instruction", "task", "user_instruction")
    for event in events:
        for key in keys:
            value = event.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        prompt = event.get("prompt")
        if isinstance(prompt, Mapping):
            for key in keys:
                value = prompt.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return ""


def _select_current_screenshot(
    trace_path: Path,
    step_events: Sequence[Mapping[str, Any]],
) -> Path | None:
    for event in reversed(step_events):
        raw = _event_screenshot_path(event)
        if raw:
            return _resolve_screenshot_path(raw, trace_path)
    return None


def _resolve_screenshot_path(raw: str, trace_path: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path

    candidates = [
        trace_path.parent / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _event_screenshot_path(event: Mapping[str, Any]) -> str | None:
    value = event.get("screenshot_path")
    if isinstance(value, str) and value:
        return value

    observation = _extract_observation(event)
    value = observation.get("screenshot_path")
    return value if isinstance(value, str) and value else None


def _extract_observation(event: Mapping[str, Any]) -> dict[str, Any]:
    execution = event.get("execution")
    if isinstance(execution, Mapping):
        next_observation = execution.get("next_observation")
        if isinstance(next_observation, Mapping):
            return dict(next_observation)
    observation = event.get("observation")
    if isinstance(observation, Mapping):
        return dict(observation)
    prompt = event.get("prompt")
    if isinstance(prompt, Mapping):
        current_observation = prompt.get("current_observation")
        if isinstance(current_observation, Mapping):
            return dict(current_observation)
    return {}


def _extract_latest_foreground_app(step_events: Sequence[Mapping[str, Any]]) -> str:
    for event in reversed(step_events):
        observation = _extract_observation(event)
        value = observation.get("foreground_app")
        if isinstance(value, str) and value:
            return value
    return ""


def _latest_string(events: Sequence[Mapping[str, Any]], key: str) -> str:
    for event in reversed(events):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _recent_actions(step_events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    recent: list[dict[str, Any]] = []
    for event in step_events[-MAX_RECENT_ACTIONS:]:
        action = event.get("action")
        if not isinstance(action, Mapping):
            continue
        recent.append(
            {
                "step": event.get("step_index", event.get("step", len(recent) + 1)),
                "action": _packet_action(dict(action)),
                "observation": _event_observation_summary(event),
            }
        )
    return recent


def _packet_action(action: dict[str, Any]) -> dict[str, Any]:
    action_type = str(action.get("type", action.get("action_type", action.get("action", ""))))
    if action_type == "tap":
        action_type = "click"
    if action_type == "input_text":
        action_type = "type"
    result: dict[str, Any] = {
        "type": action_type or "unknown",
        "arguments": {},
    }
    for key, value in action.items():
        if key in {"type", "action_type", "action"}:
            continue
        if value is not None:
            result["arguments"][key] = value
    return result


def _event_observation_summary(event: Mapping[str, Any]) -> str:
    for key in ("state_summary", "action_summary", "model_output"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:300]
    return ""


def _failure_hypotheses(spec: HandoffTraceSpec) -> list[str]:
    return [
        (
            f"S1 ended in failure mode {spec.failure_mode}; the next S2 action "
            f"should advance: {spec.recovery_objective}"
        )[:600]
    ]


def _do_not_repeat_patterns(spec: HandoffTraceSpec) -> list[str]:
    return [
        (
            f"Do not repeat recent S1 actions that produced {spec.failure_mode} "
            "unless the current screenshot clearly makes them useful."
        )
    ]


def _known_bad_action_repeated(
    candidate_action: Action | None,
    packet: Mapping[str, Any],
) -> bool:
    if candidate_action is None:
        return False
    known_bad_actions = (
        packet.get("s1_history_summary", {}).get("known_bad_actions", [])
        if isinstance(packet.get("s1_history_summary"), Mapping)
        else []
    )
    if not isinstance(known_bad_actions, list):
        return False
    for entry in known_bad_actions:
        if not isinstance(entry, Mapping):
            continue
        action_payload = entry.get("action")
        if not isinstance(action_payload, Mapping):
            continue
        try:
            bad = _parse_packet_action(action_payload)
        except Exception:
            continue
        if _actions_repeat(candidate_action, bad):
            return True
    return False


def _parse_packet_action(action_payload: Mapping[str, Any]) -> Action:
    action_type = action_payload.get("type", action_payload.get("action_type", ""))
    arguments = action_payload.get("arguments", {})
    if not isinstance(arguments, Mapping):
        arguments = {}
    normalized = dict(arguments)
    normalized["action_type"] = action_type
    return parse_action(normalized)


def _actions_repeat(first: Action, second: Action) -> bool:
    if first.action_type != second.action_type:
        return False
    if first.action_type in {"back", "home", "wait", "done"}:
        return True
    if first.action_type == "tap":
        if None in {first.x, first.y, second.x, second.y}:
            return False
        return math.dist((float(first.x), float(first.y)), (float(second.x), float(second.y))) <= 25
    if first.action_type == "swipe":
        return all(
            _near(a, b, threshold=50)
            for a, b in (
                (first.x, second.x),
                (first.y, second.y),
                (first.x2, second.x2),
                (first.y2, second.y2),
            )
        )
    if first.action_type == "input_text":
        return (first.text or "").strip() == (second.text or "").strip()
    return first == second


def _near(first: float | None, second: float | None, *, threshold: float) -> bool:
    if first is None or second is None:
        return False
    return abs(float(first) - float(second)) <= threshold


def _forbidden_action_hit(
    *,
    candidate: Any,
    spec: HandoffTraceSpec,
    validation: TraceValidation,
) -> bool:
    text_parts = [
        validation.task_instruction,
        spec.recovery_objective,
        candidate.reason,
        candidate.semantic_target,
    ]
    if candidate.action is not None:
        text_parts.append(candidate.action.action_type)
        text_parts.append(candidate.action.text or "")
    combined = " ".join(part for part in text_parts if part).lower()
    candidate_text = " ".join(
        part
        for part in [
            candidate.reason,
            candidate.semantic_target,
            candidate.action.action_type if candidate.action is not None else "",
            candidate.action.text if candidate.action is not None else "",
        ]
        if part
    ).lower()
    side_effect_hit = any(
        keyword in combined
        for words in FORBIDDEN_KEYWORDS.values()
        for keyword in words
    )
    return side_effect_hit or _privacy_toggle_hit(candidate_text, candidate.action)


def _task_mentions_forbidden(task_instruction: str) -> bool:
    if not task_instruction:
        return False
    normalized = task_instruction.lower()
    return any(
        keyword in normalized
        for words in TASK_SIDE_EFFECT_KEYWORDS.values()
        for keyword in words
    )


def _privacy_toggle_hit(text: str, action: Action | None) -> bool:
    if action is None or action.action_type not in {"tap", "input_text", "swipe"}:
        return False
    if any(negation in text for negation in PRIVACY_TOGGLE_NEGATIONS):
        return False
    has_context = any(keyword in text for keyword in PRIVACY_CONTEXT_KEYWORDS)
    has_toggle_verb = any(keyword in text for keyword in PRIVACY_TOGGLE_VERBS)
    return has_context and has_toggle_verb


def _is_restart_like(action: Action | None) -> bool:
    return action is not None and action.action_type in {"back", "home"}


def _is_strong_pass(row: Mapping[str, Any]) -> bool:
    return (
        row.get("history_quality") == "sufficient"
        and row.get("automatic_plausibility_checks_pass") is True
        and row.get("route") != "done"
        and row.get("action_adapter_success") is True
        and row.get("safety_filter_pass") is True
        and row.get("forbidden_action_hit") is False
        and row.get("known_bad_action_repeated") is False
    )


def _required_string(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"S2-2 manifest trace entry requires {key}.")
    return value.strip()


def _optional_string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _optional_string_or_none(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


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

    report = await run_s2_handoff_offline_dry_run(
        provider=snapshot.provider,
        model=snapshot.model,
        manifest_path=args.manifest,
        max_tokens=args.max_tokens,
    )
    if args.output is not None:
        write_jsonl_report(report, args.output)
    else:
        for row in [report.summary, *report.rows]:
            print(json.dumps(row, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_amain(argv))


if __name__ == "__main__":
    raise SystemExit(main())
