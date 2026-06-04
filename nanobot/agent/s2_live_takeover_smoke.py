"""Manual-gated S2 U0 live takeover smoke harness.

This module is intentionally separate from production controller and monitor
logic. It may execute live GUI actions only through an injected OpenGUI
DeviceBackend after explicit preflight has passed.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nanobot.agent.s2_handoff_offline_dry_run import FORBIDDEN_ACTIONS
from opengui.action import Action, describe_action
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
    notes: tuple[str, ...] = ()

    @property
    def has_progress(self) -> bool:
        return any(
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
