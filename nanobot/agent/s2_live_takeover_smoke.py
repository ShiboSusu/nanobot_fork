"""Manual-gated S2 U0 live takeover smoke harness.

This module is intentionally separate from production controller and monitor
logic. It may execute live GUI actions only through an injected OpenGUI
DeviceBackend after explicit preflight has passed.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nanobot.agent.s2_capability_smoke import (
    S2CapabilityError,
    adapt_s2_action_output,
    extract_json_object,
    safety_filter_passed,
)
from nanobot.agent.s2_handoff_offline_dry_run import FORBIDDEN_ACTIONS
from nanobot.providers.base import LLMProvider
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
MAX_LIVE_PACKET_PROMPT_CHARS = 4800
REQUIRED_LIVE_PACKET_FIELDS = (
    "packet_version",
    "task",
    "current_state",
    "controller",
    "required_output",
    "manual_setup_excluded_from_metrics",
    "setup_description",
    "takeover_start_screen_audited",
)
SENSITIVE_FLOW_KEYWORDS = (
    "order form",
    "submit order",
    "checkout",
    "payment",
    "passenger",
    "coupon purchase",
    "login modification",
    "account modification",
    "account settings",
    "grant permission",
    "allow permission",
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
LIVE_SENSITIVE_PAGE_KEYWORDS = (
    "order form",
    "booking form",
    "passenger info",
    "checkout",
    "payment",
    "pay now",
    "submit order",
    "confirm order",
    "passenger form",
    "coupon purchase",
    "login modification",
    "account modification",
    "account settings",
    "grant permission",
    "allow permission",
    "订单页",
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
    "账号设置",
    "账户设置",
    "授权权限",
    "填写乘机人",
    "去支付",
)
LIVE_FORBIDDEN_INTENT_PHRASES = (
    "submit order",
    "submit the order",
    "confirm order",
    "place order",
    "complete order",
    "tap submit",
    "click submit",
    "press submit",
    "checkout",
    "pay now",
    "make payment",
    "proceed to payment",
    "tap pay",
    "click pay",
    "book ticket",
    "book flight",
    "booking button",
    "tap booking",
    "click booking",
    "passenger form",
    "passenger info",
    "fill passenger",
    "enter passenger",
    "purchase",
    "下单",
    "提交订单",
    "确认订单",
    "订单填写",
    "去支付",
    "支付",
    "付款",
    "点支付",
    "点击支付",
    "乘机人",
    "旅客信息",
    "填写乘机人",
    "购买",
    "点击购买",
    "点击预订",
    "点预订",
    "预订按钮",
)
FORBIDDEN_NEGATION_CUES = (
    "avoid",
    "without",
    "do not",
    "don't",
    "not ",
    "never",
    "skip",
    "不要",
    "避免",
    "不进入",
    "不点击",
    "不提交",
    "不支付",
    "不下单",
    "跳过",
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

    packet_json = _compact_live_packet_json(packet)

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


def _compact_live_packet_json(packet: Mapping[str, Any]) -> str:
    for string_limit, list_limit in ((700, 3), (320, 2), (140, 1)):
        compacted = _compact_live_packet(
            packet,
            string_limit=string_limit,
            list_limit=list_limit,
            include_optional=True,
        )
        rendered = json.dumps(compacted, ensure_ascii=False, indent=2)
        if len(rendered) <= MAX_LIVE_PACKET_PROMPT_CHARS:
            return rendered

    compacted = _compact_live_packet(
        packet,
        string_limit=120,
        list_limit=0,
        include_optional=False,
    )
    return json.dumps(compacted, ensure_ascii=False, indent=2)


def _compact_live_packet(
    packet: Mapping[str, Any],
    *,
    string_limit: int,
    list_limit: int,
    include_optional: bool,
) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    for key in REQUIRED_LIVE_PACKET_FIELDS:
        if key in packet:
            field_list_limit = (
                999 if key in {"controller", "required_output"} else list_limit
            )
            compacted[key] = _compact_packet_value(
                packet[key],
                string_limit=string_limit,
                list_limit=field_list_limit,
            )

    if include_optional:
        for key in ("takeover_start_mode", "recovery", "s1_history_summary"):
            if key in packet and key not in compacted:
                compacted[key] = _compact_packet_value(
                    packet[key],
                    string_limit=string_limit,
                    list_limit=list_limit,
                )
    return compacted


def _compact_packet_value(
    value: Any,
    *,
    string_limit: int,
    list_limit: int,
) -> Any:
    if isinstance(value, str):
        return _truncate_packet_string(value, string_limit)
    if isinstance(value, Mapping):
        return {
            str(key): _compact_packet_value(
                item,
                string_limit=string_limit,
                list_limit=list_limit,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        shown = value[-list_limit:] if list_limit > 0 else []
        compacted = [
            _compact_packet_value(
                item,
                string_limit=string_limit,
                list_limit=list_limit,
            )
            for item in shown
        ]
        if len(value) > len(shown):
            compacted.insert(0, f"<{len(value) - len(shown)} entries omitted>")
        return compacted
    if isinstance(value, tuple):
        return _compact_packet_value(
            list(value),
            string_limit=string_limit,
            list_limit=list_limit,
        )
    return value


def _truncate_packet_string(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...<truncated {len(value) - limit} chars>"


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
    if bool(getattr(candidate, "side_effect", False)) or bool(
        getattr(candidate, "requires_human_confirm", False)
    ):
        return True

    action = getattr(candidate, "action", None)
    text_parts = [
        getattr(candidate, "reason", ""),
        getattr(candidate, "semantic_target", ""),
    ]
    if action is not None:
        text_parts.extend([
            action.action_type,
            action.text or "",
            action.status or "",
        ])
    combined = " ".join(part for part in text_parts if part)
    return _has_forbidden_action_intent(combined)


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a manual-gated U0 S2 live takeover smoke."
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--success-criteria", required=True)
    parser.add_argument("--recovery-objective", required=True)
    parser.add_argument("--setup-description", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--backend", choices=["ios"], default="ios")
    parser.add_argument("--model", default=S2_MODEL_DEFAULT)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-s2-steps", type=int, default=MAX_S2_STEPS)
    return parser.parse_args(argv)


async def run_s2_live_takeover_smoke(
    *,
    backend: DeviceBackend,
    provider: LLMProvider,
    model: str,
    config: S2LiveSmokeConfig,
    known_bad_actions: Sequence[Action | Mapping[str, Any]] = (),
    max_tokens: int = 512,
) -> dict[str, Any]:
    """Run a short manual-gated S2 takeover smoke through injected fakes/live IO."""

    started = time.perf_counter()
    run_dir = config.run_dir
    screenshots_dir = run_dir / "screenshots"
    trace_path = run_dir / "trace.jsonl"
    report_path = run_dir / "report.json"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    trace_rows: list[dict[str, Any]] = []
    executed_actions: list[Action] = []
    progress_labels: list[str] = []
    usage = {"s1": 0, "s2": 0, "total": 0}
    model_reported_success = False
    verified_success = False

    preflight = await preflight_s2_live_smoke(backend=backend, config=config)
    if not preflight.passed:
        report = _summary_report(
            config=config,
            result="failed",
            verified_success=False,
            failure_reason=preflight.failure_reason or "preflight_failed",
            s2_steps=0,
            usage=usage,
            started=started,
            model_reported_success=False,
            trace_path=trace_path,
            progress_evidence=progress_labels,
        )
        trace_rows.append(
            {
                "event": "s2_takeover_end",
                "phase": "s2_takeover",
                "result": report["result"],
                "failure_reason": report["failure_reason"],
            }
        )
        _write_jsonl(trace_path, trace_rows)
        _write_json_file(report_path, report)
        return report

    if preflight.observation is None or preflight.screenshot_path is None:
        report = _summary_report(
            config=config,
            result="failed",
            verified_success=False,
            failure_reason="preflight_failed",
            s2_steps=0,
            usage=usage,
            started=started,
            model_reported_success=False,
            trace_path=trace_path,
            progress_evidence=progress_labels,
        )
        _write_jsonl(trace_path, [{"event": "s2_takeover_end", **report}])
        _write_json_file(report_path, report)
        return report

    current_observation = preflight.observation
    current_screenshot = preflight.screenshot_path
    trace_rows.append(
        {
            "event": "s2_takeover_start",
            "phase": "s2_takeover",
            "live_smoke_id": config.live_smoke_id,
            "actor_model": model,
            "task_family": config.task_family,
            "takeover_start_mode": TAKEOVER_START_MODE,
            "takeover_reason": "manual_live_smoke",
            "manual_setup_excluded_from_metrics": True,
            "setup_description": config.setup_description,
            "takeover_start_screen_audited": (
                preflight.takeover_start_screen_audited
            ),
            "screenshot_path": str(current_screenshot),
            "foreground_app": current_observation.foreground_app,
            "budget": {
                "max_s2_steps": config.max_s2_steps,
                "max_s2_total_tokens": config.max_s2_total_tokens,
                "max_s2_wall_time_s": config.max_s2_wall_time_s,
            },
        }
    )

    result = "failed"
    failure_reason: str | None = None
    s2_steps = 0

    for step_index in range(1, config.max_s2_steps + 1):
        step_started = time.perf_counter()
        route: str | None = None
        action_summary: str | None = None
        raw_content = ""
        finish_reason: str | None = None
        step_usage: dict[str, Any] = {}
        executed = False
        screenshot_after: Path | None = None
        step_progress = ProgressEvidence()

        try:
            packet = build_live_handoff_packet(
                config=config,
                observation=current_observation,
                screenshot_path=current_screenshot,
                recent_actions=executed_actions,
                known_bad_actions=known_bad_actions,
                s2_step_index=step_index - 1,
            )
            messages = build_s2_live_messages(packet, current_screenshot)
            response = await provider.chat_with_retry(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=0,
            )
            s2_steps += 1
            raw_content = response.content or ""
            finish_reason = response.finish_reason
            step_usage = dict(response.usage or {})
            usage["s2"] += _usage_total_tokens(step_usage)
            usage["total"] = usage["s1"] + usage["s2"]
            if finish_reason != "stop":
                raise S2CapabilityError(
                    f"Provider returned finish_reason={finish_reason!r}."
                )

            payload = _normalize_live_s2_payload(extract_json_object(raw_content))
            candidate = adapt_s2_action_output(payload)
            route = candidate.route
            action_summary = (
                describe_action(candidate.action) if candidate.action else None
            )

            if route in {"halt", "human_confirm"}:
                result = "halted"
                failure_reason = route
                break
            if (
                not safety_filter_passed(candidate)
                or forbidden_action_hit_for_live_smoke(candidate)
            ):
                failure_reason = "safety_blocked_action"
                break
            if known_bad_action_repeated_for_live_smoke(
                candidate.action,
                known_bad_actions,
            ):
                failure_reason = "repeated_action"
                break
            if route == "done":
                model_reported_success = True
                verdict = ctrip_date_search_verifier(current_observation)
                verified_success = verdict.verified_success
                progress_labels.extend(verdict.evidence)
                if verified_success:
                    result = "verified_success"
                    failure_reason = None
                else:
                    failure_reason = verdict.failure_reason or "false_done"
                break
            if route != "continue" or candidate.action is None:
                failure_reason = "execution_error"
                break

            await backend.execute(candidate.action)
            executed = True
            executed_actions.append(candidate.action)
            screenshot_after = screenshots_dir / f"step_{step_index:03d}_after.png"
            next_observation = await backend.observe(
                screenshot_path=screenshot_after,
            )
            step_progress = progress_evidence(
                before_observation=current_observation,
                after_observation=next_observation,
                before_screenshot=current_screenshot,
                after_screenshot=screenshot_after,
            )
            progress_labels.extend(step_progress.evidence)
            current_observation = next_observation
            current_screenshot = screenshot_after
            if not step_progress.has_progress:
                failure_reason = "no_progress"
                break
        except Exception as exc:
            failure_reason = "execution_error"
            if not raw_content:
                raw_content = f"{type(exc).__name__}: {exc}"
            break
        finally:
            trace_rows.append(
                {
                    "event": "s2_takeover_step",
                    "phase": "s2_takeover",
                    "s2_step_index": step_index,
                    "actor_model": model,
                    "route": route,
                    "action_summary": action_summary,
                    "executed": executed,
                    "screenshot_after": (
                        str(screenshot_after) if screenshot_after else None
                    ),
                    "finish_reason": finish_reason,
                    "step_latency_s": time.perf_counter() - step_started,
                    "usage": step_usage,
                    "progress_evidence": list(step_progress.evidence),
                    "s2_output_raw_preview": _scrub_raw_output(raw_content),
                    "budget_remaining": {
                        "steps": max(0, config.max_s2_steps - step_index),
                    },
                }
            )

    if failure_reason is None and not verified_success:
        failure_reason = "s2_budget_exhausted"

    report = _summary_report(
        config=config,
        result=result if verified_success else result,
        verified_success=verified_success,
        failure_reason=failure_reason,
        s2_steps=s2_steps,
        usage=usage,
        started=started,
        model_reported_success=model_reported_success,
        trace_path=trace_path,
        progress_evidence=progress_labels,
    )
    trace_rows.append(
        {
            "event": "s2_takeover_end",
            "phase": "s2_takeover",
            "result": report["result"],
            "failure_reason": report["failure_reason"],
            "model_reported_success": model_reported_success,
            "controller_verified_success": verified_success,
            "trace_path": str(trace_path),
        }
    )
    _write_jsonl(trace_path, trace_rows)
    _write_json_file(report_path, report)
    return report


def _summary_report(
    *,
    config: S2LiveSmokeConfig,
    result: str,
    verified_success: bool,
    failure_reason: str | None,
    s2_steps: int,
    usage: Mapping[str, int],
    started: float,
    model_reported_success: bool,
    trace_path: Path,
    progress_evidence: Sequence[str],
) -> dict[str, Any]:
    return {
        "task": config.task_instruction,
        "task_family": config.task_family,
        "result": result,
        "verified_success": verified_success,
        "failure_reason": failure_reason,
        "s2_steps": s2_steps,
        "total_steps": s2_steps,
        "tokens": dict(usage),
        "progress_evidence": list(dict.fromkeys(progress_evidence)),
        "model_reported_success": model_reported_success,
        "manual_setup_excluded_from_metrics": True,
        "setup_description": config.setup_description,
        "takeover_start_screen_audited": True,
        "trace_path": str(trace_path),
        "latency": {
            "wall_clock_s": time.perf_counter() - started,
        },
    }


def _usage_total_tokens(usage: Mapping[str, Any]) -> int:
    value = usage.get("total_tokens", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str))
            handle.write("\n")


def _write_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _scrub_raw_output(content: str) -> str:
    return content[:800]


def _normalize_live_s2_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    action = normalized.get("action")
    if not isinstance(action, Mapping):
        return normalized

    action_type = str(action.get("type", action.get("action_type", ""))).casefold()
    arguments = action.get("arguments")
    if not isinstance(arguments, Mapping):
        return normalized

    if action_type == "swipe" and "direction" in arguments and "x" not in arguments:
        normalized_action = dict(action)
        normalized_action["arguments"] = _swipe_direction_arguments(arguments)
        normalized["action"] = normalized_action
    if action_type in {"click", "tap"} and "point" in arguments and "x" not in arguments:
        normalized_action = dict(action)
        normalized_action["arguments"] = _point_click_arguments(arguments)
        normalized["action"] = normalized_action
    return normalized


def _point_click_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    point = arguments.get("point")
    if (
        isinstance(point, Sequence)
        and not isinstance(point, (str, bytes))
        and len(point) >= 2
    ):
        return {"x": point[0], "y": point[1], "relative": True}
    return dict(arguments)


def _swipe_direction_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    direction = str(arguments.get("direction", "")).strip().casefold()
    distance = str(arguments.get("distance", "medium")).strip().casefold()
    span = 520 if distance in {"long", "large", "far", "大", "长"} else 320
    center_x = 500
    center_y = 500
    half = span // 2
    if direction in {"up", "向上", "上"}:
        return {
            "x": center_x,
            "y": min(900, center_y + half),
            "x2": center_x,
            "y2": max(100, center_y - half),
            "relative": True,
        }
    if direction in {"down", "向下", "下"}:
        return {
            "x": center_x,
            "y": max(100, center_y - half),
            "x2": center_x,
            "y2": min(900, center_y + half),
            "relative": True,
        }
    if direction in {"left", "向左", "左"}:
        return {
            "x": min(900, center_x + half),
            "y": center_y,
            "x2": max(100, center_x - half),
            "y2": center_y,
            "relative": True,
        }
    if direction in {"right", "向右", "右"}:
        return {
            "x": max(100, center_x - half),
            "y": center_y,
            "x2": min(900, center_x + half),
            "y2": center_y,
            "relative": True,
        }
    return dict(arguments)


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


def _has_forbidden_action_intent(text: str) -> bool:
    normalized = text.casefold()
    for phrase in LIVE_FORBIDDEN_INTENT_PHRASES:
        phrase_norm = phrase.casefold()
        start = normalized.find(phrase_norm)
        while start != -1:
            if not _is_negated_forbidden_phrase(normalized, start):
                return True
            start = normalized.find(phrase_norm, start + len(phrase_norm))
    return False


def _is_negated_forbidden_phrase(normalized_text: str, phrase_start: int) -> bool:
    prefix = normalized_text[max(0, phrase_start - 32) : phrase_start]
    return any(cue.casefold() in prefix for cue in FORBIDDEN_NEGATION_CUES)


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


async def _amain(argv: Sequence[str] | None = None) -> int:
    from nanobot.config.loader import load_config, resolve_config_env_vars
    from nanobot.providers.factory import build_gui_s2_provider_snapshot
    from opengui.backends.ios_wda import WdaBackend

    args = parse_args(argv)
    config_obj = resolve_config_env_vars(load_config(args.config))
    snapshot = build_gui_s2_provider_snapshot(config_obj)
    if snapshot is None:
        raise SystemExit(
            "No GUI S2 model configured. Set gui.s2Model and gui.s2Provider."
        )

    gui_config = getattr(config_obj, "gui")
    backend = WdaBackend(wda_url=gui_config.ios.wda_url)
    smoke_config = S2LiveSmokeConfig(
        task_instruction=args.task,
        success_criteria=args.success_criteria,
        recovery_objective=args.recovery_objective,
        setup_description=args.setup_description,
        run_dir=args.run_dir,
        max_s2_steps=args.max_s2_steps,
    )
    report = await run_s2_live_takeover_smoke(
        backend=backend,
        provider=snapshot.provider,
        model=args.model or snapshot.model,
        config=smoke_config,
        max_tokens=args.max_tokens,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["result"] in {"verified_success", "failed", "halted"} else 1


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_amain(argv))


if __name__ == "__main__":
    raise SystemExit(main())
