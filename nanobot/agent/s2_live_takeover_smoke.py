"""Manual-gated S2 U0 live takeover smoke harness.

This module is intentionally separate from production controller and monitor
logic. It may execute live GUI actions only through an injected OpenGUI
DeviceBackend after explicit preflight has passed.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nanobot.agent.s2_capability_smoke import S2CapabilityError
from nanobot.agent.s2_handoff_offline_dry_run import FORBIDDEN_ACTIONS
from nanobot.utils.helpers import build_image_content_blocks, detect_image_mime
from opengui.action import Action, describe_action, parse_action
from opengui.interfaces import DeviceBackend
from opengui.observation import Observation

TAKEOVER_START_MODE = "MANUAL_TAKEOVER_FROM_CURRENT_STATE"
TASK_FAMILY = "date_travel_search"
S2_MODEL_DEFAULT = "qwen3.5-397b-a17b"
MAX_S2_STEPS = 3
MAX_S2_CALLS = 3
MAX_S2_TOTAL_TOKENS = 20000
MAX_S2_WALL_TIME_S = 180
SENSITIVE_FLOW_KEYWORDS = (
    "order form",
    "submit order",
    "checkout",
    "payment",
    "pay",
    "passenger",
    "coupon purchase",
    "login modification",
    "account",
    "permission",
    "订单填写",
    "提交订单",
    "确认订单",
    "结算",
    "支付",
    "付款",
    "乘机人",
    "旅客信息",
    "购买优惠券",
    "登录修改",
    "账号",
    "账户",
    "权限",
)
S2_LIVE_SYSTEM_PROMPT = """You are the S2 live GUI takeover actor for a U0 smoke test.
Return only one JSON object per step. Do not include markdown or prose.

The JSON schema is:
{
  "route": "continue | done | halt | human_confirm",
  "action": {
    "type": "click | type | swipe | wait | back | home | done",
    "arguments": {}
  },
  "reason": "short current-screen-grounded reason",
  "semantic_target": "constraint or subgoal advanced by this action",
  "final_answer": {
    "required": false,
    "text": "",
    "evidence": []
  },
  "safety_check": {
    "side_effect": false,
    "requires_human_confirm": false
  }
}

Use route=halt or human_confirm for any send, submit, payment, purchase,
delete, account modification, privacy toggle, sensitive permission grant, or
ambiguous confirmation flow. Never propose those side-effecting actions.
"""
ROUTE_ORIGIN_KEYWORDS = ("上海", "shanghai", "pvg", "虹桥", "浦东")
ROUTE_DESTINATION_KEYWORDS = ("广州", "guangzhou", "白云")
TARGET_DATE_KEYWORDS = ("2026-06-05", "06-05", "6月5")
TARGET_DATE_PARTIAL_KEYWORDS = ("2026年6月", "2026/06", "2026-06", "6月")
FLIGHT_RESULT_KEYWORDS = (
    "航班",
    "flight",
    "价格",
    "票价",
    "起飞",
    "到达",
    "departure",
    "arrival",
    "price",
    "¥",
    "￥",
)
LIVE_SENSITIVE_PAGE_KEYWORDS = SENSITIVE_FLOW_KEYWORDS + (
    "order",
    "order form",
    "booking form",
    "passenger info",
    "checkout",
    "submit order",
    "订单",
    "订单页",
    "填写乘机人",
    "去支付",
)
LIVE_FORBIDDEN_ACTION_KEYWORDS = (
    "booking",
    "book ",
    "submit",
    "order",
    "checkout",
    "payment",
    "pay",
    "passenger",
    "purchase",
    "下单",
    "预订",
    "提交",
    "订单",
    "支付",
    "付款",
    "乘机人",
    "旅客",
    "购买",
)


@dataclass(frozen=True)
class S2LiveSmokeConfig:
    task_instruction: str
    success_criteria: str
    recovery_objective: str
    setup_description: str
    run_dir: Path
    live_smoke_id: str = field(default_factory=lambda: _live_smoke_id())
    model: str = S2_MODEL_DEFAULT
    task_family: str = TASK_FAMILY
    max_s2_steps: int = MAX_S2_STEPS
    max_s2_calls: int = MAX_S2_CALLS
    max_s2_total_tokens: int = MAX_S2_TOTAL_TOKENS
    max_s2_wall_time_s: int = MAX_S2_WALL_TIME_S


@dataclass(frozen=True)
class PreflightResult:
    passed: bool
    failure_reason: str | None
    screenshot_path: Path | None
    observation: Observation | None
    takeover_start_screen_audited: bool
    sensitive_flow_keywords_visible: list[str]
    backend_preflight_passed: bool
    live_smoke_id: str


@dataclass(frozen=True)
class CtripVerifierResult:
    verified_success: bool
    evidence: list[str]
    failure_reason: str | None = None
    final_answer_required: bool = False


@dataclass(frozen=True)
class ProgressEvidence:
    screenshot_changed: bool = False
    target_date_more_visible: bool = False
    result_list_visible: bool = False
    moved_away_from_known_bad_state: bool = False
    verifier_evidence_improved: bool = False
    evidence: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def has_progress(self) -> bool:
        return bool(self.evidence) or any(
            (
                self.screenshot_changed,
                self.target_date_more_visible,
                self.result_list_visible,
                self.moved_away_from_known_bad_state,
                self.verifier_evidence_improved,
            )
        )


def _live_smoke_id() -> str:
    return f"s2-live-{uuid.uuid4().hex[:12]}"


def _observation_text(observation: Observation) -> str:
    parts: list[str] = []
    if observation.foreground_app:
        parts.append(observation.foreground_app)
    for key in (
        "visible_text",
        "state_summary",
        "ocr_text",
        "screen_summary",
        "page_summary",
    ):
        value = observation.extra.get(key)
        if isinstance(value, str):
            parts.append(value)
    ui_tree = observation.extra.get("ui_tree")
    if isinstance(ui_tree, list):
        for node in ui_tree[:40]:
            if not isinstance(node, Mapping):
                continue
            for key in ("text", "label", "name", "description"):
                value = node.get(key)
                if isinstance(value, str) and value:
                    parts.append(value)
    return "\n".join(parts)


def audit_start_screen(observation: Observation) -> list[str]:
    text = _observation_text(observation).casefold()
    return [
        keyword
        for keyword in SENSITIVE_FLOW_KEYWORDS
        if keyword.casefold() in text
    ]


async def preflight_s2_live_smoke(
    *,
    backend: DeviceBackend,
    config: S2LiveSmokeConfig,
) -> PreflightResult:
    try:
        await backend.preflight()
    except Exception as exc:
        return PreflightResult(
            passed=False,
            failure_reason=f"backend_preflight_failed:{type(exc).__name__}",
            screenshot_path=None,
            observation=None,
            takeover_start_screen_audited=False,
            sensitive_flow_keywords_visible=[],
            backend_preflight_passed=False,
            live_smoke_id=config.live_smoke_id,
        )

    screenshot_path = config.run_dir / "preflight_start.png"
    try:
        observation = await backend.observe(screenshot_path=screenshot_path)
    except Exception as exc:
        return PreflightResult(
            passed=False,
            failure_reason=f"observation_failed:{type(exc).__name__}",
            screenshot_path=screenshot_path,
            observation=None,
            takeover_start_screen_audited=False,
            sensitive_flow_keywords_visible=[],
            backend_preflight_passed=True,
            live_smoke_id=config.live_smoke_id,
        )

    sensitive_hits = audit_start_screen(observation)
    if sensitive_hits:
        return PreflightResult(
            passed=False,
            failure_reason="sensitive_flow_start_screen",
            screenshot_path=screenshot_path,
            observation=observation,
            takeover_start_screen_audited=True,
            sensitive_flow_keywords_visible=sensitive_hits,
            backend_preflight_passed=True,
            live_smoke_id=config.live_smoke_id,
        )

    return PreflightResult(
        passed=True,
        failure_reason=None,
        screenshot_path=screenshot_path,
        observation=observation,
        takeover_start_screen_audited=True,
        sensitive_flow_keywords_visible=[],
        backend_preflight_passed=True,
        live_smoke_id=config.live_smoke_id,
    )


def build_live_handoff_packet(
    *,
    config: S2LiveSmokeConfig,
    observation: Observation,
    screenshot_path: Path,
    recent_actions: Sequence[Action | Mapping[str, Any]],
    known_bad_actions: Sequence[Action | Mapping[str, Any]],
    s2_step_index: int,
) -> dict[str, Any]:
    visible_text = _observation_text(observation)
    remaining_steps = max(0, config.max_s2_steps - s2_step_index)

    return {
        "packet_version": "s2_live_handoff_v1",
        "live_smoke_id": config.live_smoke_id,
        "takeover_start_mode": TAKEOVER_START_MODE,
        "manual_setup_excluded_from_metrics": True,
        "setup_description": config.setup_description,
        "takeover_start_screen_audited": True,
        "task": {
            "instruction": config.task_instruction,
            "task_family": config.task_family,
            "success_criteria": config.success_criteria,
            "risk_level": "U0",
        },
        "current_state": {
            "screenshot_path": str(screenshot_path),
            "foreground_app": observation.foreground_app or "unknown",
            "page_summary": _extra_string(observation, "state_summary")[:600],
            "visible_text": visible_text[:800],
        },
        "s1_history_summary": {
            "recent_actions": [
                _packet_action(action) for action in list(recent_actions)[-5:]
            ],
            "failure_mode": "manual_seeded_state",
            "s1_failure_hypothesis": [
                "The current state was manually selected for S2 live recovery smoke."
            ],
            "do_not_repeat": [
                "Do not enter order, passenger, payment, account, permission, or privacy-toggle flows."
            ],
            "known_bad_actions": [
                _packet_action(action) for action in known_bad_actions
            ],
        },
        "recovery": {
            "recovery_objective": config.recovery_objective,
        },
        "controller": {
            "route": "S2_TAKEOVER",
            "takeover_reason": "manual_live_smoke",
            "remaining_budget": {
                "steps": remaining_steps,
                "calls": max(0, config.max_s2_calls - s2_step_index),
                "tokens": config.max_s2_total_tokens,
                "wall_time_s": config.max_s2_wall_time_s,
            },
            "forbidden_actions": list(FORBIDDEN_ACTIONS),
        },
        "required_output": {
            "schema": "s2_live_action_json",
            "allowed_routes": [
                "continue",
                "done",
                "halt",
                "human_confirm",
            ],
        },
    }


def build_s2_live_messages(
    packet: Mapping[str, Any],
    screenshot_path: Path,
) -> list[dict[str, Any]]:
    raw = screenshot_path.read_bytes()
    mime = detect_image_mime(raw)
    if mime is None:
        raise S2CapabilityError(f"Unsupported image file: {screenshot_path}")

    packet_json = json.dumps(packet, ensure_ascii=False, sort_keys=True, indent=2)
    if len(packet_json) > 4800:
        packet_json = f"{packet_json[:4800]}\n...<packet truncated for prompt bound>"

    label = (
        "S2 U0 Ctrip live takeover smoke.\n"
        "Task family: Ctrip/携程 date travel search recovery only.\n"
        "Risk: U0 read/search smoke; do not book, order, pay, submit, or "
        "enter passenger/account/permission flows.\n"
        "Use the screenshot and compact handoff packet below. Return exactly "
        "one action JSON matching the system schema.\n\n"
        f"{packet_json}"
    )
    return [
        {"role": "system", "content": S2_LIVE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_image_content_blocks(
                raw,
                mime,
                str(screenshot_path),
                label,
            ),
        },
    ]


def ctrip_date_search_verifier(observation: Observation) -> CtripVerifierResult:
    text = _observation_text(observation)
    if _has_any_keyword(text, LIVE_SENSITIVE_PAGE_KEYWORDS):
        return CtripVerifierResult(
            verified_success=False,
            evidence=[],
            failure_reason="sensitive_flow_page",
        )

    evidence: list[str] = []
    if _has_route_evidence(text):
        evidence.append("route_shanghai_guangzhou")
    if _has_target_date_evidence(text):
        evidence.append("target_date_visible")
    if _has_flight_result_evidence(text):
        evidence.append("flight_result_list_visible")

    verified_success = {
        "route_shanghai_guangzhou",
        "target_date_visible",
        "flight_result_list_visible",
    }.issubset(evidence)
    return CtripVerifierResult(
        verified_success=verified_success,
        evidence=evidence,
        failure_reason=None if verified_success else "verifier_unknown",
    )


def progress_evidence(
    *,
    before_observation: Observation,
    after_observation: Observation,
    before_screenshot: Path,
    after_screenshot: Path,
) -> ProgressEvidence:
    evidence: list[str] = []
    screenshot_changed = _file_sha256(before_screenshot) != _file_sha256(
        after_screenshot
    )
    if screenshot_changed:
        evidence.append("screenshot_hash_changed")

    before_text = _observation_text(before_observation)
    after_text = _observation_text(after_observation)
    target_date_more_visible = _date_progress_score(after_text) > _date_progress_score(
        before_text
    )
    if target_date_more_visible:
        evidence.append("target_date_more_visible")

    result_list_visible = _has_flight_result_evidence(after_text)
    if result_list_visible:
        evidence.append("flight_result_list_visible")

    moved_away_from_known_bad_state = _moved_away_from_known_bad_state(
        before_text,
        after_text,
    )
    if moved_away_from_known_bad_state:
        evidence.append("moved_away_from_known_bad_state")

    before_verifier = ctrip_date_search_verifier(before_observation)
    after_verifier = ctrip_date_search_verifier(after_observation)
    verifier_evidence_improved = len(after_verifier.evidence) > len(
        before_verifier.evidence
    )
    if verifier_evidence_improved:
        evidence.append("verifier_evidence_improved")

    return ProgressEvidence(
        screenshot_changed=screenshot_changed,
        target_date_more_visible=target_date_more_visible,
        result_list_visible=result_list_visible,
        moved_away_from_known_bad_state=moved_away_from_known_bad_state,
        verifier_evidence_improved=verifier_evidence_improved,
        evidence=tuple(evidence),
    )


def forbidden_action_hit_for_live_smoke(candidate: Any) -> bool:
    action = getattr(candidate, "action", None)
    text_parts = [
        getattr(candidate, "route", ""),
        getattr(candidate, "reason", ""),
        getattr(candidate, "semantic_target", ""),
    ]
    if action is not None:
        text_parts.extend([
            action.action_type,
            action.text or "",
            action.status or "",
        ])
    raw = getattr(candidate, "raw", None)
    if isinstance(raw, Mapping):
        text_parts.append(json.dumps(raw, ensure_ascii=False, sort_keys=True))
    combined = " ".join(part for part in text_parts if part)
    return _has_any_keyword(combined, LIVE_FORBIDDEN_ACTION_KEYWORDS)


def known_bad_action_repeated_for_live_smoke(
    candidate_action: Action | None,
    known_bad_actions: Sequence[Action | Mapping[str, Any]],
) -> bool:
    if candidate_action is None:
        return False
    for entry in known_bad_actions:
        bad_action = _coerce_known_bad_action(entry)
        if bad_action is not None and _actions_repeat(candidate_action, bad_action):
            return True
    return False


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _date_progress_score(text: str) -> int:
    if _has_any_keyword(text, TARGET_DATE_KEYWORDS):
        return 3
    normalized = text.casefold()
    score = 0
    if "2026" in normalized:
        score += 1
    if _has_any_keyword(normalized, TARGET_DATE_PARTIAL_KEYWORDS):
        score += 1
    return score


def _extra_string(observation: Observation, key: str) -> str:
    value = observation.extra.get(key)
    return value if isinstance(value, str) else ""


def _packet_action(action: Action | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(action, Action):
        return {
            "action_type": action.action_type,
            "summary": describe_action(action),
            "x": action.x,
            "y": action.y,
            "x2": action.x2,
            "y2": action.y2,
            "text": action.text,
            "duration_ms": action.duration_ms,
            "relative": action.relative,
        }
    return dict(action)


def _has_any_keyword(text: str, keywords: Sequence[str]) -> bool:
    normalized = text.casefold()
    return any(keyword.casefold() in normalized for keyword in keywords)


def _has_route_evidence(text: str) -> bool:
    return _has_any_keyword(text, ROUTE_ORIGIN_KEYWORDS) and _has_any_keyword(
        text,
        ROUTE_DESTINATION_KEYWORDS,
    )


def _has_target_date_evidence(text: str) -> bool:
    return _has_any_keyword(text, TARGET_DATE_KEYWORDS)


def _has_flight_result_evidence(text: str) -> bool:
    return _has_any_keyword(text, FLIGHT_RESULT_KEYWORDS)


def _moved_away_from_known_bad_state(before_text: str, after_text: str) -> bool:
    before = before_text.casefold()
    after = after_text.casefold()
    was_bad_calendar = "2027" in before or "wrong date" in before
    if not was_bad_calendar:
        return False
    return "2027" not in after or _has_target_date_evidence(after_text)


def _coerce_known_bad_action(value: Action | Mapping[str, Any]) -> Action | None:
    if isinstance(value, Action):
        return value
    if not isinstance(value, Mapping):
        return None

    if isinstance(value.get("action"), Mapping):
        return _coerce_known_bad_action(value["action"])

    action_type = value.get("action_type", value.get("type", value.get("action")))
    if action_type is None:
        return None

    arguments = value.get("arguments", {})
    if not isinstance(arguments, Mapping):
        arguments = {}
    payload = dict(arguments)
    payload["action_type"] = action_type
    for key in (
        "x",
        "y",
        "x2",
        "y2",
        "text",
        "duration_ms",
        "relative",
        "status",
        "auto_enter",
    ):
        if key in value and key not in payload:
            payload[key] = value[key]

    try:
        return parse_action(payload)
    except Exception:
        return None


def _actions_repeat(first: Action, second: Action) -> bool:
    if first.action_type != second.action_type:
        return False
    if first.action_type in {"back", "home", "wait", "done"}:
        return True
    if first.action_type == "tap":
        if None in {first.x, first.y, second.x, second.y}:
            return False
        return (
            math.dist(
                (float(first.x), float(first.y)),
                (float(second.x), float(second.y)),
            )
            <= 25
        )
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
