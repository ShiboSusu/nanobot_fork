# S2 U0 Live Takeover Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a manual-gated S2-3 live smoke harness that can execute at most three safe S2 actions on one U0 Ctrip date/search recovery state, with traceable preflight, safety, verifier, progress, and artifact gates.

**Architecture:** Add one isolated harness module under `nanobot/agent/` that reuses S2-1/S2-2 action parsing, safety filtering, image prompting, and provider loading. The harness depends only on an injected `DeviceBackend` and injected `LLMProvider`, so unit tests use fake backends and no production controller, monitor, WDA, ADB, HDC, iOS, or Android implementation is modified.

**Tech Stack:** Python 3, asyncio, dataclasses, `opengui.interfaces.DeviceBackend`, `opengui.action.Action`, existing `nanobot.providers.base.LLMProvider`, existing `nanobot.agent.s2_capability_smoke` adapters, pytest.

---

## Scope Lock

Allowed files:

- Create: `nanobot/agent/s2_live_takeover_smoke.py`
- Create: `tests/agent/test_s2_live_takeover_smoke.py`
- Modify: `nanobot/agent/s2_handoff_offline_dry_run.py` only if a helper must be shared instead of copied

Forbidden files for this implementation:

- `opengui/agent.py`
- `opengui/agent_profiles.py`
- `opengui/policy.py`
- `opengui/prompts/system.py`
- production controller routing files
- broad monitor files
- WDA, ADB, HDC, iOS, Android, or desktop backend implementations

The current worktree contains unrelated WIP in the forbidden files. Do not stage or edit those files. Every commit in this plan must show only S2-3 files in `git diff --cached --name-status`.

## File Structure

`nanobot/agent/s2_live_takeover_smoke.py`

- CLI entry point for the manual-gated S2-3 smoke.
- Pure preflight, packet, verifier, progress, and report helpers.
- `run_s2_live_takeover_smoke()` live loop using injected backend/provider.
- No production controller or monitor imports.

`tests/agent/test_s2_live_takeover_smoke.py`

- Fake provider and fake backend unit tests.
- No real device calls.
- Tests prove sensitive page preflight blocks execution, manual setup is excluded from metrics, `route=done` requires verifier success, unsafe actions do not execute, repeated known-bad actions do not execute, no-progress halts, max three S2 actions is enforced, and JSONL artifacts are scrubbed enough for repository safety.

## Task 1: Test Scaffolding And Preflight Gates

**Files:**

- Create: `tests/agent/test_s2_live_takeover_smoke.py`
- Create: `nanobot/agent/s2_live_takeover_smoke.py`

- [ ] **Step 1: Write failing tests for sensitive preflight and manual setup metrics**

Add this initial test file:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanobot.agent.s2_live_takeover_smoke import (
    S2LiveSmokeConfig,
    audit_start_screen,
    build_live_handoff_packet,
    preflight_s2_live_smoke,
)
from opengui.action import Action
from opengui.observation import Observation

PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4"
    b"\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05"
    b"\xfe\x02\xfeA\xe2`\x82\x00\x00\x00\x00IEND\xaeB`\x82"
)


class FakeBackend:
    platform = "ios"

    def __init__(self, observations: list[Observation]) -> None:
        self.observations = observations
        self.observe_calls: list[Path] = []
        self.execute_calls: list[Action] = []
        self.preflight_calls = 0

    async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
        self.observe_calls.append(screenshot_path)
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        screenshot_path.write_bytes(PNG_1X1)
        if not self.observations:
            raise AssertionError("FakeBackend has no observation queued.")
        observation = self.observations.pop(0)
        observation.screenshot_path = str(screenshot_path)
        return observation

    async def execute(self, action: Action, timeout: float = 5.0) -> str:
        self.execute_calls.append(action)
        return f"executed {action.action_type}"

    async def preflight(self) -> None:
        self.preflight_calls += 1

    async def list_apps(self) -> list[str]:
        return ["ctrip"]


def _obs(
    *,
    foreground_app: str = "ctrip",
    visible_text: str = "携程 机票 上海 广州 选择日期 2027年6月",
) -> Observation:
    return Observation(
        screenshot_path=None,
        screen_width=402,
        screen_height=874,
        foreground_app=foreground_app,
        platform="ios",
        extra={"visible_text": visible_text, "state_summary": visible_text},
    )


def _config(tmp_path: Path) -> S2LiveSmokeConfig:
    return S2LiveSmokeConfig(
        task_instruction="在携程查询2026年6月5日上海到广州的机票，只看到结果列表即可",
        success_criteria="Must show Shanghai to Guangzhou, 2026-06-05, and flight result cards.",
        recovery_objective="Recover from the current calendar state to the target date and result list.",
        setup_description="operator-created Ctrip calendar state",
        run_dir=tmp_path / "s2_live",
    )


@pytest.mark.asyncio
async def test_preflight_rejects_sensitive_start_screen_without_execution(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            _obs(
                visible_text=(
                    "携程 订单填写 乘机人 支付 优惠券 提交订单"
                )
            )
        ]
    )
    config = _config(tmp_path)

    result = await preflight_s2_live_smoke(backend=backend, config=config)

    assert result.passed is False
    assert result.failure_reason == "sensitive_flow_start_screen"
    assert result.takeover_start_screen_audited is True
    assert backend.execute_calls == []


@pytest.mark.asyncio
async def test_preflight_passes_u0_ctrip_calendar_screen(tmp_path: Path) -> None:
    backend = FakeBackend([_obs()])
    config = _config(tmp_path)

    result = await preflight_s2_live_smoke(backend=backend, config=config)

    assert result.passed is True
    assert result.failure_reason is None
    assert result.screenshot_path is not None
    assert result.screenshot_path.is_file()
    assert result.takeover_start_screen_audited is True


def test_build_live_handoff_packet_excludes_manual_setup_from_metrics(tmp_path: Path) -> None:
    config = _config(tmp_path)
    observation = _obs()
    screenshot_path = tmp_path / "screen.png"
    screenshot_path.write_bytes(PNG_1X1)
    observation.screenshot_path = str(screenshot_path)

    packet = build_live_handoff_packet(
        config=config,
        observation=observation,
        screenshot_path=screenshot_path,
        recent_actions=[],
        known_bad_actions=[],
        s2_step_index=0,
    )

    assert packet["packet_version"] == "s2_live_handoff_v1"
    assert packet["manual_setup_excluded_from_metrics"] is True
    assert packet["setup_description"] == "operator-created Ctrip calendar state"
    assert packet["takeover_start_screen_audited"] is True
    assert packet["controller"]["remaining_budget"]["steps"] == 3
    assert packet["required_output"]["allowed_routes"] == [
        "continue",
        "done",
        "halt",
        "human_confirm",
    ]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
pytest tests/agent/test_s2_live_takeover_smoke.py -q
```

Expected: FAIL during import with `ModuleNotFoundError` or `ImportError` because `nanobot.agent.s2_live_takeover_smoke` does not exist.

- [ ] **Step 3: Add minimal module skeleton and preflight implementation**

Create `nanobot/agent/s2_live_takeover_smoke.py` with:

```python
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
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nanobot.agent.s2_capability_smoke import (
    S2CapabilityError,
    adapt_s2_action_output,
    extract_json_object,
    safety_filter_passed,
)
from nanobot.agent.s2_handoff_offline_dry_run import (
    FORBIDDEN_ACTIONS,
    _actions_repeat,
    _forbidden_action_hit,
)
from nanobot.providers.base import LLMProvider
from nanobot.utils.helpers import build_image_content_blocks, detect_image_mime
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
SENSITIVE_FLOW_KEYWORDS = [
    "order",
    "passenger",
    "payment",
    "checkout",
    "coupon",
    "login modification",
    "permission",
    "订单",
    "乘机人",
    "支付",
    "付款",
    "提交订单",
    "结算",
    "优惠券",
    "登录修改",
    "权限",
]

S2_LIVE_SYSTEM_PROMPT = """You are a manual-gated S2 GUI takeover actor.
Inspect the current screenshot and compact live handoff packet, then return
only one JSON object.

The JSON schema is:
{
  "route": "continue" | "done" | "halt" | "human_confirm",
  "action": {
    "type": "click" | "type" | "swipe" | "wait" | "back" | "home" | "done",
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

This is a bounded U0 Ctrip date/search smoke. Never propose send, submit,
payment, purchase, booking, delete, account modification, privacy toggles,
sensitive permission grants, or entering checkout/passenger/order/payment
flows. route=done is only a candidate terminal state; the controller verifier
decides success.
"""


@dataclass(frozen=True)
class S2LiveSmokeConfig:
    task_instruction: str
    success_criteria: str
    recovery_objective: str
    setup_description: str
    run_dir: Path
    task_family: str = TASK_FAMILY
    live_smoke_id: str = ""
    max_s2_steps: int = MAX_S2_STEPS
    max_s2_calls: int = MAX_S2_CALLS
    max_s2_total_tokens: int = MAX_S2_TOTAL_TOKENS
    max_s2_wall_time_s: int = MAX_S2_WALL_TIME_S


@dataclass(frozen=True)
class PreflightResult:
    passed: bool
    failure_reason: str | None
    observation: Observation | None
    screenshot_path: Path | None
    takeover_start_screen_audited: bool


@dataclass(frozen=True)
class CtripVerifierResult:
    verified_success: bool
    failure_reason: str | None
    evidence: list[str]


@dataclass(frozen=True)
class ProgressEvidence:
    progressed: bool
    evidence: list[str]


def _live_smoke_id(config: S2LiveSmokeConfig) -> str:
    return config.live_smoke_id or f"s2-live-{uuid.uuid4().hex[:10]}"


def _observation_text(observation: Observation) -> str:
    parts = [
        observation.foreground_app or "",
        str(observation.extra.get("visible_text", "")),
        str(observation.extra.get("state_summary", "")),
    ]
    return " ".join(part for part in parts if part).lower()


def audit_start_screen(observation: Observation) -> tuple[bool, str | None]:
    text = _observation_text(observation)
    if any(keyword.lower() in text for keyword in SENSITIVE_FLOW_KEYWORDS):
        return False, "sensitive_flow_start_screen"
    return True, None


async def preflight_s2_live_smoke(
    *,
    backend: DeviceBackend,
    config: S2LiveSmokeConfig,
) -> PreflightResult:
    await backend.preflight()
    config.run_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = config.run_dir / "screenshots" / "preflight_start.png"
    observation = await backend.observe(screenshot_path)
    passed, failure_reason = audit_start_screen(observation)
    return PreflightResult(
        passed=passed,
        failure_reason=failure_reason,
        observation=observation,
        screenshot_path=screenshot_path,
        takeover_start_screen_audited=True,
    )


def build_live_handoff_packet(
    *,
    config: S2LiveSmokeConfig,
    observation: Observation,
    screenshot_path: Path,
    recent_actions: Sequence[Mapping[str, Any]],
    known_bad_actions: Sequence[Mapping[str, Any]],
    s2_step_index: int,
) -> dict[str, Any]:
    return {
        "packet_version": "s2_live_handoff_v1",
        "live_smoke_id": _live_smoke_id(config),
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
            "page_summary": str(observation.extra.get("state_summary", ""))[:600],
            "visible_text": str(observation.extra.get("visible_text", ""))[:800],
        },
        "s1_history_summary": {
            "recent_actions": list(recent_actions)[-5:],
            "failure_mode": "manual_seeded_state",
            "s1_failure_hypothesis": [
                "The operator started S2 from a bounded U0 recovery state."
            ],
            "do_not_repeat": [
                "Do not enter order, passenger, checkout, payment, login, account, permission, or coupon purchase flows."
            ],
            "known_bad_actions": list(known_bad_actions)[-5:],
        },
        "controller": {
            "route": "S2_TAKEOVER",
            "takeover_reason": "manual_live_smoke",
            "remaining_budget": {
                "steps": max(0, config.max_s2_steps - s2_step_index),
                "tokens": config.max_s2_total_tokens,
                "wall_time_s": config.max_s2_wall_time_s,
            },
            "forbidden_actions": FORBIDDEN_ACTIONS,
        },
        "required_output": {
            "schema": "s2_live_action_json",
            "allowed_routes": ["continue", "done", "halt", "human_confirm"],
        },
    }
```

- [ ] **Step 4: Run tests to verify preflight passes**

Run:

```bash
pytest tests/agent/test_s2_live_takeover_smoke.py -q
```

Expected: PASS for the three tests in this task.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git status --short
git add nanobot/agent/s2_live_takeover_smoke.py tests/agent/test_s2_live_takeover_smoke.py
git diff --cached --name-status
git commit -m "feat: add s2 live smoke preflight"
```

Expected staged files:

```text
A	nanobot/agent/s2_live_takeover_smoke.py
A	tests/agent/test_s2_live_takeover_smoke.py
```

## Task 2: S2 Message Building, Verifier, Progress Evidence, And Safety Gates

**Files:**

- Modify: `nanobot/agent/s2_live_takeover_smoke.py`
- Modify: `tests/agent/test_s2_live_takeover_smoke.py`

- [ ] **Step 1: Add failing tests for verifier, progress, done, unsafe, and repeated actions**

Append to `tests/agent/test_s2_live_takeover_smoke.py`:

```python
from nanobot.agent.s2_live_takeover_smoke import (
    build_s2_live_messages,
    ctrip_date_search_verifier,
    forbidden_action_hit_for_live_smoke,
    known_bad_action_repeated_for_live_smoke,
    progress_evidence,
)
from nanobot.agent.s2_capability_smoke import adapt_s2_action_output


def test_s2_live_messages_include_image_and_packet(tmp_path: Path) -> None:
    screenshot = tmp_path / "screen.png"
    screenshot.write_bytes(PNG_1X1)
    packet = {"packet_version": "s2_live_handoff_v1", "task": {"risk_level": "U0"}}

    messages = build_s2_live_messages(packet=packet, screenshot_path=screenshot)

    rendered = json.dumps(messages, ensure_ascii=False)
    assert "image_url" in rendered
    assert "s2_live_handoff_v1" in rendered
    assert "bounded U0 Ctrip date/search smoke" in rendered


def test_ctrip_verifier_requires_route_date_result_and_non_sensitive_page() -> None:
    success = _obs(
        visible_text=(
            "携程 机票 上海 至 广州 2026-06-05 6月5日 "
            "航班列表 东方航空 南方航空 价格"
        )
    )
    failure = _obs(
        visible_text="携程 机票 上海 至 广州 2026-06-05 乘机人 提交订单 支付"
    )

    ok = ctrip_date_search_verifier(success)
    bad = ctrip_date_search_verifier(failure)

    assert ok.verified_success is True
    assert ok.failure_reason is None
    assert "target_date" in ok.evidence
    assert bad.verified_success is False
    assert bad.failure_reason == "sensitive_flow_page"


def test_progress_evidence_detects_hash_change_and_result_list(tmp_path: Path) -> None:
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    before.write_bytes(PNG_1X1)
    after.write_bytes(PNG_1X1 + b"changed")
    before_obs = _obs(visible_text="携程 选择日期 2027年6月")
    after_obs = _obs(visible_text="携程 上海 广州 2026-06-05 航班列表 价格")

    result = progress_evidence(
        before_observation=before_obs,
        after_observation=after_obs,
        before_screenshot=before,
        after_screenshot=after,
    )

    assert result.progressed is True
    assert "screenshot_hash_changed" in result.evidence
    assert "flight_result_list_visible" in result.evidence


def test_done_route_requires_verifier_success() -> None:
    candidate = adapt_s2_action_output(
        {
            "route": "done",
            "action": {"type": "done", "arguments": {"status": "success"}},
            "reason": "looks complete",
            "semantic_target": "completion",
            "safety_check": {"side_effect": False, "requires_human_confirm": False},
        }
    )
    incomplete_obs = _obs(visible_text="携程 选择日期 2027年6月")

    verdict = ctrip_date_search_verifier(incomplete_obs)

    assert candidate.route == "done"
    assert verdict.verified_success is False
    assert verdict.failure_reason == "verifier_unknown"


def test_live_forbidden_action_blocks_booking_or_payment_language() -> None:
    candidate = adapt_s2_action_output(
        {
            "route": "continue",
            "action": {"type": "click", "arguments": {"x": 500, "y": 900, "relative": True}},
            "reason": "tap submit order to continue booking",
            "semantic_target": "提交订单",
            "safety_check": {"side_effect": False, "requires_human_confirm": False},
        }
    )

    assert forbidden_action_hit_for_live_smoke(candidate) is True


def test_known_bad_action_repetition_detects_same_click() -> None:
    candidate = adapt_s2_action_output(
        {
            "route": "continue",
            "action": {"type": "click", "arguments": {"x": 502, "y": 602, "relative": True}},
            "reason": "tap same calendar cell",
            "semantic_target": "same failed date",
            "safety_check": {"side_effect": False, "requires_human_confirm": False},
        }
    )
    known_bad_actions = [
        {
            "action": {
                "type": "click",
                "arguments": {"x": 500, "y": 600, "relative": True},
            }
        }
    ]

    assert known_bad_action_repeated_for_live_smoke(candidate.action, known_bad_actions) is True
```

- [ ] **Step 2: Run these tests to verify they fail**

Run:

```bash
pytest tests/agent/test_s2_live_takeover_smoke.py -q
```

Expected: FAIL with missing helper imports.

- [ ] **Step 3: Implement message, verifier, progress, and live safety helpers**

Append these helpers to `nanobot/agent/s2_live_takeover_smoke.py`:

```python
def build_s2_live_messages(
    *,
    packet: Mapping[str, Any],
    screenshot_path: Path,
) -> list[dict[str, Any]]:
    raw = screenshot_path.read_bytes()
    mime = detect_image_mime(raw)
    if mime is None:
        raise S2CapabilityError(f"Unsupported image file: {screenshot_path}")
    label = (
        "S2 live handoff packet follows. Return exactly one JSON action for "
        "this current screenshot and packet.\n\n"
        f"Handoff packet:\n{json.dumps(packet, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": S2_LIVE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_image_content_blocks(
                raw, mime, str(screenshot_path), label
            ),
        },
    ]


def ctrip_date_search_verifier(observation: Observation) -> CtripVerifierResult:
    text = _observation_text(observation)
    if any(keyword.lower() in text for keyword in SENSITIVE_FLOW_KEYWORDS):
        return CtripVerifierResult(False, "sensitive_flow_page", [])

    evidence: list[str] = []
    if "上海" in text and "广州" in text:
        evidence.append("route")
    if "2026-06-05" in text or "06-05" in text or "6月5" in text:
        evidence.append("target_date")
    if any(word in text for word in ["航班", "flight", "价格", "起飞", "到达"]):
        evidence.append("flight_result_list")

    if {"route", "target_date", "flight_result_list"}.issubset(set(evidence)):
        return CtripVerifierResult(True, None, evidence)
    return CtripVerifierResult(False, "verifier_unknown", evidence)


def progress_evidence(
    *,
    before_observation: Observation,
    after_observation: Observation,
    before_screenshot: Path,
    after_screenshot: Path,
) -> ProgressEvidence:
    evidence: list[str] = []
    if _file_sha256(before_screenshot) != _file_sha256(after_screenshot):
        evidence.append("screenshot_hash_changed")

    before_text = _observation_text(before_observation)
    after_text = _observation_text(after_observation)
    before_score = _date_progress_score(before_text)
    after_score = _date_progress_score(after_text)
    if after_score > before_score:
        evidence.append("target_date_more_visible")
    if any(word in after_text for word in ["航班", "flight", "价格", "起飞", "到达"]):
        evidence.append("flight_result_list_visible")
    if "2027" in before_text and "2027" not in after_text:
        evidence.append("moved_away_from_known_bad_calendar_state")
    if len(ctrip_date_search_verifier(after_observation).evidence) > len(
        ctrip_date_search_verifier(before_observation).evidence
    ):
        evidence.append("verifier_evidence_improved")
    return ProgressEvidence(progressed=bool(evidence), evidence=evidence)


def forbidden_action_hit_for_live_smoke(candidate: Any) -> bool:
    if candidate.action is not None and candidate.action.action_type in {"done", "wait", "back", "home"}:
        action_type = candidate.action.action_type
    else:
        action_type = candidate.action.action_type if candidate.action is not None else ""
    text = " ".join(
        part
        for part in [
            action_type,
            candidate.reason,
            candidate.semantic_target,
            candidate.action.text if candidate.action is not None else "",
        ]
        if part
    ).lower()
    return any(keyword.lower() in text for keyword in SENSITIVE_FLOW_KEYWORDS)


def known_bad_action_repeated_for_live_smoke(
    candidate_action: Action | None,
    known_bad_actions: Sequence[Mapping[str, Any]],
) -> bool:
    if candidate_action is None:
        return False
    for entry in known_bad_actions:
        action_payload = entry.get("action") if isinstance(entry, Mapping) else None
        if not isinstance(action_payload, Mapping):
            continue
        action_type = action_payload.get("type", action_payload.get("action_type", ""))
        arguments = action_payload.get("arguments", {})
        if not isinstance(arguments, Mapping):
            arguments = {}
        try:
            bad_candidate = adapt_s2_action_output(
                {
                    "route": "continue",
                    "action": {"type": action_type, "arguments": dict(arguments)},
                    "reason": "known bad action",
                    "semantic_target": "known bad action",
                    "safety_check": {"side_effect": False, "requires_human_confirm": False},
                }
            )
        except Exception:
            continue
        if bad_candidate.action is not None and _actions_repeat(candidate_action, bad_candidate.action):
            return True
    return False


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _date_progress_score(text: str) -> int:
    score = 0
    for marker in ["2026", "06-05", "6月5", "上海", "广州"]:
        if marker.lower() in text:
            score += 1
    return score
```

- [ ] **Step 4: Run tests to verify helper behavior**

Run:

```bash
pytest tests/agent/test_s2_live_takeover_smoke.py -q
```

Expected: PASS for all tests written through Task 2.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add nanobot/agent/s2_live_takeover_smoke.py tests/agent/test_s2_live_takeover_smoke.py
git diff --cached --name-status
git commit -m "feat: add s2 live smoke gates"
```

Expected staged files:

```text
M	nanobot/agent/s2_live_takeover_smoke.py
M	tests/agent/test_s2_live_takeover_smoke.py
```

## Task 3: Live Takeover Loop With Fake Backend

**Files:**

- Modify: `nanobot/agent/s2_live_takeover_smoke.py`
- Modify: `tests/agent/test_s2_live_takeover_smoke.py`

- [ ] **Step 1: Add fake provider and failing loop tests**

Append to `tests/agent/test_s2_live_takeover_smoke.py`:

```python
from nanobot.agent.s2_live_takeover_smoke import run_s2_live_takeover_smoke
from nanobot.providers.base import LLMResponse


class FakeProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    async def chat_with_retry(self, **kwargs) -> LLMResponse:
        self.calls.append(kwargs)
        return LLMResponse(
            content=self.responses.pop(0),
            finish_reason="stop",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        )


@pytest.mark.asyncio
async def test_run_loop_executes_safe_continue_then_verifies_done(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            _obs(visible_text="携程 选择日期 2027年6月"),
            _obs(visible_text="携程 上海 广州 2026-06-05 航班列表 价格"),
        ]
    )
    provider = FakeProvider(
        [
            """{
                "route": "continue",
                "action": {
                    "type": "click",
                    "arguments": {"x": 500, "y": 620, "relative": true}
                },
                "reason": "select the visible target date cell",
                "semantic_target": "2026-06-05 date selection",
                "final_answer": {"required": false, "text": "", "evidence": []},
                "safety_check": {"side_effect": false, "requires_human_confirm": false}
            }""",
            """{
                "route": "done",
                "action": {"type": "done", "arguments": {"status": "success"}},
                "reason": "flight results for the target date are visible",
                "semantic_target": "verified flight result list",
                "final_answer": {"required": false, "text": "", "evidence": []},
                "safety_check": {"side_effect": false, "requires_human_confirm": false}
            }""",
        ]
    )

    report = await run_s2_live_takeover_smoke(
        backend=backend,
        provider=provider,
        model="qwen3.5-397b-a17b",
        config=_config(tmp_path),
    )

    assert report["result"] == "verified_success"
    assert report["verified_success"] is True
    assert report["failure_reason"] is None
    assert report["s2_steps"] == 2
    assert len(backend.execute_calls) == 1
    assert backend.execute_calls[0].action_type == "tap"
    assert report["manual_setup_excluded_from_metrics"] is True
    assert report["progress_evidence"]


@pytest.mark.asyncio
async def test_run_loop_blocks_unsafe_action_without_execution(tmp_path: Path) -> None:
    backend = FakeBackend([_obs(visible_text="携程 选择日期 2027年6月")])
    provider = FakeProvider(
        [
            """{
                "route": "continue",
                "action": {
                    "type": "click",
                    "arguments": {"x": 500, "y": 900, "relative": true}
                },
                "reason": "tap submit order",
                "semantic_target": "提交订单",
                "final_answer": {"required": false, "text": "", "evidence": []},
                "safety_check": {"side_effect": false, "requires_human_confirm": false}
            }"""
        ]
    )

    report = await run_s2_live_takeover_smoke(
        backend=backend,
        provider=provider,
        model="qwen3.5-397b-a17b",
        config=_config(tmp_path),
    )

    assert report["result"] == "failed"
    assert report["failure_reason"] == "safety_blocked_action"
    assert backend.execute_calls == []


@pytest.mark.asyncio
async def test_run_loop_rejects_known_bad_repeated_action(tmp_path: Path) -> None:
    backend = FakeBackend([_obs(visible_text="携程 选择日期 2027年6月")])
    provider = FakeProvider(
        [
            """{
                "route": "continue",
                "action": {
                    "type": "click",
                    "arguments": {"x": 500, "y": 600, "relative": true}
                },
                "reason": "tap same failed date",
                "semantic_target": "same failed calendar cell",
                "final_answer": {"required": false, "text": "", "evidence": []},
                "safety_check": {"side_effect": false, "requires_human_confirm": false}
            }"""
        ]
    )

    report = await run_s2_live_takeover_smoke(
        backend=backend,
        provider=provider,
        model="qwen3.5-397b-a17b",
        config=_config(tmp_path),
        known_bad_actions=[
            {
                "action": {
                    "type": "click",
                    "arguments": {"x": 500, "y": 600, "relative": True},
                }
            }
        ],
    )

    assert report["result"] == "failed"
    assert report["failure_reason"] == "repeated_action"
    assert backend.execute_calls == []


@pytest.mark.asyncio
async def test_run_loop_stops_after_no_progress(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            _obs(visible_text="携程 选择日期 2027年6月"),
            _obs(visible_text="携程 选择日期 2027年6月"),
        ]
    )
    provider = FakeProvider(
        [
            """{
                "route": "continue",
                "action": {
                    "type": "wait",
                    "arguments": {"duration_ms": 100}
                },
                "reason": "wait for loading",
                "semantic_target": "same calendar state",
                "final_answer": {"required": false, "text": "", "evidence": []},
                "safety_check": {"side_effect": false, "requires_human_confirm": false}
            }"""
        ]
    )

    report = await run_s2_live_takeover_smoke(
        backend=backend,
        provider=provider,
        model="qwen3.5-397b-a17b",
        config=_config(tmp_path),
    )

    assert report["result"] == "failed"
    assert report["failure_reason"] == "no_progress"
    assert len(backend.execute_calls) == 1


@pytest.mark.asyncio
async def test_run_loop_enforces_three_step_budget(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            _obs(visible_text="携程 选择日期 2027年6月"),
            _obs(visible_text="携程 选择日期 2026年6月"),
            _obs(visible_text="携程 选择日期 2026年6月 6月5"),
            _obs(visible_text="携程 上海 广州"),
        ]
    )
    provider = FakeProvider(
        [
            """{
                "route": "continue",
                "action": {"type": "swipe", "arguments": {"x": 500, "y": 800, "x2": 500, "y2": 200, "relative": true}},
                "reason": "move toward target month",
                "semantic_target": "2026 calendar",
                "final_answer": {"required": false, "text": "", "evidence": []},
                "safety_check": {"side_effect": false, "requires_human_confirm": false}
            }""",
            """{
                "route": "continue",
                "action": {"type": "click", "arguments": {"x": 500, "y": 600, "relative": true}},
                "reason": "select target date",
                "semantic_target": "6月5",
                "final_answer": {"required": false, "text": "", "evidence": []},
                "safety_check": {"side_effect": false, "requires_human_confirm": false}
            }""",
            """{
                "route": "continue",
                "action": {"type": "click", "arguments": {"x": 820, "y": 920, "relative": true}},
                "reason": "open result list",
                "semantic_target": "flight search results",
                "final_answer": {"required": false, "text": "", "evidence": []},
                "safety_check": {"side_effect": false, "requires_human_confirm": false}
            }""",
        ]
    )

    report = await run_s2_live_takeover_smoke(
        backend=backend,
        provider=provider,
        model="qwen3.5-397b-a17b",
        config=_config(tmp_path),
    )

    assert report["s2_steps"] == 3
    assert report["result"] == "failed"
    assert report["failure_reason"] == "s2_budget_exhausted"
    assert len(backend.execute_calls) == 3
```

- [ ] **Step 2: Run loop tests to verify they fail**

Run:

```bash
pytest tests/agent/test_s2_live_takeover_smoke.py -q
```

Expected: FAIL with missing `run_s2_live_takeover_smoke`.

- [ ] **Step 3: Implement the live loop**

Append to `nanobot/agent/s2_live_takeover_smoke.py`:

```python
async def run_s2_live_takeover_smoke(
    *,
    backend: DeviceBackend,
    provider: LLMProvider,
    model: str,
    config: S2LiveSmokeConfig,
    known_bad_actions: Sequence[Mapping[str, Any]] = (),
    max_tokens: int = 512,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = config.run_dir
    screenshots_dir = run_dir / "screenshots"
    trace_path = run_dir / "trace.jsonl"
    report_path = run_dir / "report.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    trace_rows: list[dict[str, Any]] = []
    progress_rows: list[str] = []
    usage = {"s1": 0, "s2": 0, "total": 0}

    preflight = await preflight_s2_live_smoke(backend=backend, config=config)
    if not preflight.passed:
        report = _summary_report(
            config=config,
            result="failed",
            failure_reason=preflight.failure_reason or "preflight_failed",
            s2_steps=0,
            usage=usage,
            started=started,
            model_reported_success=False,
            verified_success=False,
            trace_path=trace_path,
            progress_evidence=progress_rows,
        )
        _write_json(trace_path, [{"event": "s2_takeover_end", **report}])
        _write_json_file(report_path, report)
        return report

    assert preflight.observation is not None
    assert preflight.screenshot_path is not None
    current_observation = preflight.observation
    current_screenshot = preflight.screenshot_path
    trace_rows.append(
        {
            "event": "s2_takeover_start",
            "phase": "s2_takeover",
            "live_smoke_id": _live_smoke_id(config),
            "task_family": config.task_family,
            "takeover_start_mode": TAKEOVER_START_MODE,
            "takeover_reason": "manual_live_smoke",
            "manual_setup_excluded_from_metrics": True,
            "setup_description": config.setup_description,
            "takeover_start_screen_audited": preflight.takeover_start_screen_audited,
            "s1_steps_before_takeover": 0,
            "screenshot_path": str(current_screenshot),
            "foreground_app": current_observation.foreground_app,
            "budget": {
                "max_s2_steps": config.max_s2_steps,
                "max_s2_total_tokens": config.max_s2_total_tokens,
                "max_s2_wall_time_s": config.max_s2_wall_time_s,
            },
        }
    )

    model_reported_success = False
    verified_success = False
    failure_reason: str | None = None
    result = "failed"

    for step_index in range(1, config.max_s2_steps + 1):
        step_started = time.perf_counter()
        packet = build_live_handoff_packet(
            config=config,
            observation=current_observation,
            screenshot_path=current_screenshot,
            recent_actions=[],
            known_bad_actions=known_bad_actions,
            s2_step_index=step_index - 1,
        )
        raw_content = ""
        finish_reason = None
        step_usage: dict[str, int] = {}
        route: str | None = None
        action_summary = None
        executed = False
        screenshot_after: Path | None = None
        progress = ProgressEvidence(False, [])
        try:
            messages = build_s2_live_messages(packet=packet, screenshot_path=current_screenshot)
            response = await provider.chat_with_retry(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=0,
            )
            raw_content = response.content or ""
            finish_reason = response.finish_reason
            step_usage = dict(response.usage or {})
            usage["s2"] += int(step_usage.get("total_tokens", 0))
            usage["total"] = usage["s1"] + usage["s2"]
            if finish_reason != "stop":
                raise S2CapabilityError(f"Provider returned finish_reason={finish_reason!r}.")

            payload = extract_json_object(raw_content)
            candidate = adapt_s2_action_output(payload)
            route = candidate.route
            action_summary = describe_action(candidate.action) if candidate.action else None

            if not safety_filter_passed(candidate) or forbidden_action_hit_for_live_smoke(candidate):
                failure_reason = "safety_blocked_action"
                break
            if known_bad_action_repeated_for_live_smoke(candidate.action, known_bad_actions):
                failure_reason = "repeated_action"
                break
            if route in {"halt", "human_confirm"}:
                result = "halted"
                failure_reason = route
                break
            if route == "done":
                model_reported_success = True
                verdict = ctrip_date_search_verifier(current_observation)
                verified_success = verdict.verified_success
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
            screenshot_after = screenshots_dir / f"step_{step_index:03d}_after.png"
            next_observation = await backend.observe(screenshot_after)
            progress = progress_evidence(
                before_observation=current_observation,
                after_observation=next_observation,
                before_screenshot=current_screenshot,
                after_screenshot=screenshot_after,
            )
            progress_rows.extend(progress.evidence)
            current_observation = next_observation
            current_screenshot = screenshot_after
            if not progress.progressed:
                failure_reason = "no_progress"
                break
        except Exception as exc:
            failure_reason = "execution_error"
            raw_content = raw_content or f"{type(exc).__name__}: {exc}"
            break
        finally:
            trace_rows.append(
                {
                    "event": "s2_takeover_step",
                    "phase": "s2_takeover",
                    "s2_step_index": step_index,
                    "actor_model": model,
                    "screenshot_before": str(current_screenshot),
                    "s2_output_raw": _scrub_raw_output(raw_content),
                    "route": route,
                    "action_summary": action_summary,
                    "executed": executed,
                    "screenshot_after": str(screenshot_after) if screenshot_after else None,
                    "finish_reason": finish_reason,
                    "model_latency_s": float(step_usage.get("latency_s", 0.0)),
                    "step_latency_s": time.perf_counter() - step_started,
                    "usage": step_usage,
                    "progress_evidence": progress.evidence,
                    "budget_remaining": {
                        "steps": max(0, config.max_s2_steps - step_index),
                    },
                }
            )

        if verified_success:
            break

    s2_steps = len([row for row in trace_rows if row.get("event") == "s2_takeover_step"])
    if failure_reason is None and not verified_success:
        failure_reason = "s2_budget_exhausted"
    report = _summary_report(
        config=config,
        result=result if verified_success else "failed",
        failure_reason=failure_reason,
        s2_steps=s2_steps,
        usage=usage,
        started=started,
        model_reported_success=model_reported_success,
        verified_success=verified_success,
        trace_path=trace_path,
        progress_evidence=progress_rows,
    )
    trace_rows.append(
        {
            "event": "s2_takeover_end",
            "phase": "s2_takeover",
            "model_reported_success": model_reported_success,
            "controller_verified_success": verified_success,
            "user_observed_success": None,
            "result": report["result"],
            "failure_reason": report["failure_reason"],
            "final_answer": None,
            "trace_path": str(trace_path),
        }
    )
    _write_json(trace_path, trace_rows)
    _write_json_file(report_path, report)
    return report


def _summary_report(
    *,
    config: S2LiveSmokeConfig,
    result: str,
    failure_reason: str | None,
    s2_steps: int,
    usage: Mapping[str, int],
    started: float,
    model_reported_success: bool,
    verified_success: bool,
    trace_path: Path,
    progress_evidence: Sequence[str],
) -> dict[str, Any]:
    return {
        "task": config.task_instruction,
        "task_family": config.task_family,
        "result": result,
        "manual_setup_excluded_from_metrics": True,
        "setup_description": config.setup_description,
        "takeover_start_screen_audited": True,
        "s2_takeover_started": True,
        "s2_called": s2_steps > 0,
        "s2_steps": s2_steps,
        "total_steps": s2_steps,
        "tokens": dict(usage),
        "latency": {
            "wall_clock_s": time.perf_counter() - started,
            "s2_model_latency_s": 0.0,
        },
        "cross_app": False,
        "failure_reason": failure_reason,
        "takeover_reason": "manual_live_smoke",
        "trigger_evidence": ["manual_live_smoke"],
        "progress_evidence": list(dict.fromkeys(progress_evidence)),
        "verified_success": verified_success,
        "model_reported_success": model_reported_success,
        "final_answer": None,
        "trace_path": str(trace_path),
    }


def _write_json(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _scrub_raw_output(content: str) -> str:
    if len(content) <= 1200:
        return content
    return content[:1200] + "...[truncated]"
```

- [ ] **Step 4: Run loop tests**

Run:

```bash
pytest tests/agent/test_s2_live_takeover_smoke.py -q
```

Expected: PASS for all S2-3 unit tests.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add nanobot/agent/s2_live_takeover_smoke.py tests/agent/test_s2_live_takeover_smoke.py
git diff --cached --name-status
git commit -m "feat: add s2 live smoke loop"
```

Expected staged files:

```text
M	nanobot/agent/s2_live_takeover_smoke.py
M	tests/agent/test_s2_live_takeover_smoke.py
```

## Task 4: CLI Entry Point And Artifact Hygiene Checks

**Files:**

- Modify: `nanobot/agent/s2_live_takeover_smoke.py`
- Modify: `tests/agent/test_s2_live_takeover_smoke.py`

- [ ] **Step 1: Add failing tests for CLI argument parsing and local artifact output**

Append to `tests/agent/test_s2_live_takeover_smoke.py`:

```python
from nanobot.agent.s2_live_takeover_smoke import parse_args


def test_parse_args_requires_manual_setup_description_and_run_dir(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--task",
            "在携程查询2026年6月5日上海到广州的机票，只看到结果列表即可",
            "--success-criteria",
            "Must show route, date, and result list.",
            "--recovery-objective",
            "Recover from calendar to result list.",
            "--setup-description",
            "operator-created Ctrip calendar state",
            "--run-dir",
            str(tmp_path / "run"),
        ]
    )

    assert args.task.startswith("在携程查询")
    assert args.setup_description == "operator-created Ctrip calendar state"
    assert args.run_dir == tmp_path / "run"


@pytest.mark.asyncio
async def test_run_loop_writes_local_trace_and_report_artifacts(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            _obs(visible_text="携程 选择日期 2027年6月"),
            _obs(visible_text="携程 上海 广州 2026-06-05 航班列表 价格"),
        ]
    )
    provider = FakeProvider(
        [
            """{
                "route": "continue",
                "action": {"type": "click", "arguments": {"x": 500, "y": 620, "relative": true}},
                "reason": "select target date",
                "semantic_target": "2026-06-05",
                "final_answer": {"required": false, "text": "", "evidence": []},
                "safety_check": {"side_effect": false, "requires_human_confirm": false}
            }""",
            """{
                "route": "done",
                "action": {"type": "done", "arguments": {"status": "success"}},
                "reason": "target result list visible",
                "semantic_target": "flight result list",
                "final_answer": {"required": false, "text": "", "evidence": []},
                "safety_check": {"side_effect": false, "requires_human_confirm": false}
            }""",
        ]
    )
    config = _config(tmp_path)

    report = await run_s2_live_takeover_smoke(
        backend=backend,
        provider=provider,
        model="qwen3.5-397b-a17b",
        config=config,
    )

    trace_path = Path(report["trace_path"])
    report_path = config.run_dir / "report.json"
    assert trace_path.is_file()
    assert report_path.is_file()
    trace_text = trace_path.read_text(encoding="utf-8")
    assert "s2_takeover_start" in trace_text
    assert "s2_takeover_step" in trace_text
    assert "s2_takeover_end" in trace_text
```

- [ ] **Step 2: Run tests to verify CLI parsing fails**

Run:

```bash
pytest tests/agent/test_s2_live_takeover_smoke.py -q
```

Expected: FAIL with missing `parse_args`.

- [ ] **Step 3: Implement CLI parser and provider/backend loading**

Append to `nanobot/agent/s2_live_takeover_smoke.py`:

```python
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
    return parser.parse_args(argv)


async def _amain(argv: Sequence[str] | None = None) -> int:
    from nanobot.config.loader import load_config, resolve_config_env_vars
    from nanobot.providers.factory import build_gui_s2_provider_snapshot
    from opengui.backends.ios_wda import WdaBackend

    args = parse_args(argv)
    config_obj = resolve_config_env_vars(load_config(args.config))
    snapshot = build_gui_s2_provider_snapshot(config_obj)
    if snapshot is None:
        raise SystemExit("No GUI S2 model configured. Set gui.s2Model and gui.s2Provider.")
    gui_config = getattr(config_obj, "gui")
    backend = WdaBackend(wda_url=gui_config.ios.wda_url)
    smoke_config = S2LiveSmokeConfig(
        task_instruction=args.task,
        success_criteria=args.success_criteria,
        recovery_objective=args.recovery_objective,
        setup_description=args.setup_description,
        run_dir=args.run_dir,
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
```

- [ ] **Step 4: Run focused tests and import check**

Run:

```bash
pytest tests/agent/test_s2_live_takeover_smoke.py -q
python -m nanobot.agent.s2_live_takeover_smoke --help
```

Expected:

- pytest PASS
- help output includes `Run a manual-gated U0 S2 live takeover smoke.`

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add nanobot/agent/s2_live_takeover_smoke.py tests/agent/test_s2_live_takeover_smoke.py
git diff --cached --name-status
git commit -m "feat: add s2 live smoke cli"
```

Expected staged files:

```text
M	nanobot/agent/s2_live_takeover_smoke.py
M	tests/agent/test_s2_live_takeover_smoke.py
```

## Task 5: Final Verification And Non-Live Acceptance

**Files:**

- Modify only if verification reveals a defect: `nanobot/agent/s2_live_takeover_smoke.py`
- Modify only if verification reveals a defect: `tests/agent/test_s2_live_takeover_smoke.py`

- [ ] **Step 1: Run focused S2 tests**

Run:

```bash
pytest tests/agent/test_s2_capability_smoke.py tests/agent/test_s2_handoff_offline_dry_run.py tests/agent/test_s2_live_takeover_smoke.py -q
```

Expected: all tests PASS.

- [ ] **Step 2: Run lint on touched files**

Run:

```bash
ruff check nanobot/agent/s2_live_takeover_smoke.py tests/agent/test_s2_live_takeover_smoke.py
```

Expected: `All checks passed!`

- [ ] **Step 3: Prove no live device action was run during unit verification**

Run:

```bash
git diff --name-only
git diff --cached --name-status
```

Expected:

- unstaged unrelated WIP may remain in the seven existing files
- cached diff is empty after the Task 4 commit
- no trace/report/screenshot artifacts are staged

- [ ] **Step 4: Run CLI help only**

Run:

```bash
python -m nanobot.agent.s2_live_takeover_smoke --help
```

Expected: help text prints and no WDA/iOS/device call occurs.

- [ ] **Step 5: Report implementation boundary**

Final implementation report must state:

- S2-3 harness implemented
- unit tests use fake backend/provider only
- no production controller or monitor changes
- no WDA/ADB/HDC/iOS/Android backend implementation changes
- no live Ctrip task executed
- no raw screenshots, full traces, or private S2 outputs committed
- current branch still has the pre-existing unrelated WIP unstaged

## Manual Live Smoke Command After Human Approval

Do not run this command during implementation verification. It is included so the next live-smoke operator has the exact invocation after S2-3 code review passes and the Ctrip current screen is manually prepared.

```bash
python -m nanobot.agent.s2_live_takeover_smoke \
  --task '在携程查询2026年6月5日上海到广州的机票，只看到结果列表即可，不进入预订/乘机人/支付/订单流程' \
  --success-criteria 'Must show Shanghai to Guangzhou, 2026-06-05 or 6月5日, and visible flight result list/cards; must not be on order/passenger/payment/checkout/coupon/login-modification page.' \
  --recovery-objective 'Recover from the current Ctrip calendar/search state to the target date and visible flight result list.' \
  --setup-description 'operator-created Ctrip calendar/search state; excluded from S2 metrics' \
  --run-dir /tmp/s2_live_ctrip_2026_06_05
```

Expected live result format:

```json
{
  "result": "verified_success | failed | halted",
  "manual_setup_excluded_from_metrics": true,
  "s2_steps": 0,
  "failure_reason": null,
  "verified_success": false,
  "model_reported_success": false,
  "progress_evidence": [],
  "trace_path": "/tmp/s2_live_ctrip_2026_06_05/trace.jsonl"
}
```

## Self-Review Checklist

- S2-3 spec coverage:
  - manual-gated start: Task 1 and Task 3
  - U0 Ctrip date/search only: Task 1 config and Task 2 verifier
  - max three S2 actions: Task 3 budget test
  - no production controller/monitor changes: scope lock and all tasks
  - safety blocks above S2: Task 2 and Task 3
  - `route=done` verifier: Task 2 and Task 3
  - manual setup excluded from metrics: Task 1 and Task 3
  - progress/no-progress evidence: Task 2 and Task 3
  - artifact hygiene: Task 4 and Task 5
- Type consistency:
  - `S2LiveSmokeConfig`, `PreflightResult`, `CtripVerifierResult`, and `ProgressEvidence` are defined before use.
  - `run_s2_live_takeover_smoke()` takes injected `DeviceBackend` and `LLMProvider`.
  - tests import only symbols defined in the plan.
- Execution safety:
  - no test imports a real backend
  - CLI help is the only command allowed during non-live verification
  - manual live command is documented but not part of automated verification
