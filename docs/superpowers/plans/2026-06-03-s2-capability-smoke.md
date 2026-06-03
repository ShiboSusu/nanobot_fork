# S2 Capability Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline-first smoke test that determines whether the configured GUI S2 model can act as a visual GUI executor.

**Architecture:** Add one focused module, `nanobot/agent/s2_capability_smoke.py`, with pure helpers for S2 action JSON parsing, screenshot message construction, fake-provider-testable smoke execution, and a small `python -m` CLI. The smoke uses saved screenshots only; it does not change the controller, monitor, GUI agent loop, WDA, ADB, or any live takeover behavior.

**Tech Stack:** Python 3.11, dataclasses, argparse, existing nanobot provider factory, existing `nanobot.utils.helpers` image helpers, existing `opengui.action.parse_action`, pytest.

---

## Scope Guard

This plan implements only Stage 1 from the S2 takeover spec:

- Endpoint/model call through the configured GUI S2 provider.
- Screenshot-to-action prompting on saved screenshots.
- Image-use contrast check.
- JSON schema parse check.
- OpenGUI action adapter check.
- Safety-filter check.
- JSON report output.

This plan must not:

- modify `opengui/agent.py`;
- modify `opengui/autonomy_monitor.py`;
- modify controller routing;
- execute any GUI action;
- call WDA, ADB, HDC, or iOS device APIs;
- run live takeover;
- run U0/U1 pilot tasks.

## File Structure

- Create `nanobot/agent/s2_capability_smoke.py`
  - Owns the S2 smoke dataclasses, JSON extraction, S2 action schema adapter, safety filter, multimodal message builder, smoke runner, report serialization, and CLI entrypoint.
  - It depends on provider interfaces and `opengui.action.parse_action`, but has no dependency on GUI backends.

- Create `tests/agent/test_s2_capability_smoke.py`
  - Unit tests action schema parsing, safety filtering, screenshot message construction, contrast detection, and runner behavior with a fake provider.
  - Tests must not use network, WDA, ADB, or real model endpoints.

No existing implementation files should be modified for S2-1.

---

### Task 1: S2 Action Schema Adapter

**Files:**
- Create: `nanobot/agent/s2_capability_smoke.py`
- Test: `tests/agent/test_s2_capability_smoke.py`

- [ ] **Step 1: Write failing tests for action schema parsing**

Create `tests/agent/test_s2_capability_smoke.py` with these tests:

```python
from __future__ import annotations

import pytest

from opengui.action import Action

from nanobot.agent.s2_capability_smoke import (
    S2CapabilityError,
    adapt_s2_action_output,
    extract_json_object,
    safety_filter_passed,
)


def test_extract_json_object_strips_markdown_wrapper() -> None:
    content = '```json\n{"route":"halt","reason":"blocked"}\n```'

    assert extract_json_object(content) == {"route": "halt", "reason": "blocked"}


def test_adapts_continue_click_to_opengui_action() -> None:
    candidate = adapt_s2_action_output({
        "route": "continue",
        "action": {
            "type": "click",
            "arguments": {"x": 500, "y": 300, "relative": True},
        },
        "reason": "Tap the visible search box.",
        "semantic_target": "focus_search_box",
        "safety_check": {
            "side_effect": False,
            "requires_human_confirm": False,
        },
    })

    assert candidate.route == "continue"
    assert candidate.action == Action(
        action_type="tap",
        x=500.0,
        y=300.0,
        relative=True,
    )
    assert candidate.reason == "Tap the visible search box."
    assert candidate.semantic_target == "focus_search_box"
    assert safety_filter_passed(candidate) is True


def test_adapts_type_action_to_input_text() -> None:
    candidate = adapt_s2_action_output({
        "route": "continue",
        "action": {
            "type": "type",
            "arguments": {"text": "广州", "auto_enter": False},
        },
        "reason": "Enter the destination city.",
        "semantic_target": "set_destination",
        "safety_check": {
            "side_effect": False,
            "requires_human_confirm": False,
        },
    })

    assert candidate.action == Action(
        action_type="input_text",
        text="广州",
        auto_enter=False,
    )


def test_rejects_natural_language_hint_without_json_action() -> None:
    with pytest.raises(S2CapabilityError, match="No JSON object"):
        extract_json_object("Click the visible 5th day.")


def test_safety_filter_blocks_side_effect_action() -> None:
    candidate = adapt_s2_action_output({
        "route": "continue",
        "action": {
            "type": "click",
            "arguments": {"x": 500, "y": 300, "relative": True},
        },
        "reason": "Tap submit.",
        "semantic_target": "submit_form",
        "safety_check": {
            "side_effect": True,
            "requires_human_confirm": True,
        },
    })

    assert safety_filter_passed(candidate) is False


def test_done_route_requires_done_action() -> None:
    candidate = adapt_s2_action_output({
        "route": "done",
        "action": {"type": "done", "arguments": {"status": "success"}},
        "reason": "The requested item is visible and verified.",
        "semantic_target": "verified_success",
        "safety_check": {
            "side_effect": False,
            "requires_human_confirm": False,
        },
    })

    assert candidate.action == Action(action_type="done", status="success")
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
pytest tests/agent/test_s2_capability_smoke.py -q
```

Expected: fails during import because `nanobot.agent.s2_capability_smoke` does not exist.

- [ ] **Step 3: Implement the minimal schema adapter**

Create `nanobot/agent/s2_capability_smoke.py` with this initial content:

```python
"""Offline S2 capability smoke utilities.

This module verifies whether the configured GUI S2 model can act as a visual
GUI executor. It does not execute GUI actions and does not import GUI backends.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nanobot.providers.base import LLMProvider
from nanobot.utils.helpers import build_image_content_blocks, detect_image_mime
from opengui.action import Action, ActionError, parse_action


ALLOWED_ROUTES = frozenset({"continue", "done", "halt", "human_confirm"})
ACTION_TYPE_ALIASES = {
    "click": "tap",
    "type": "input_text",
    "swipe": "swipe",
    "wait": "wait",
    "back": "back",
    "home": "home",
    "done": "done",
}


class S2CapabilityError(ValueError):
    """Raised when an S2 smoke response is not executable enough to test."""


@dataclass(frozen=True)
class S2ActionCandidate:
    route: str
    action: Action | None
    reason: str
    semantic_target: str
    side_effect: bool
    requires_human_confirm: bool
    raw: dict[str, Any]


def extract_json_object(content: str) -> dict[str, Any]:
    """Extract the first JSON object from an S2 response."""
    text = re.sub(r"(?is)<think>.*?</think>", "", content or "")
    text = re.sub(r"(?is)```(?:json)?", "", text).replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise S2CapabilityError("No JSON object found in S2 response.")
    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise S2CapabilityError(f"Invalid JSON from S2 response: {exc}") from exc
    if not isinstance(payload, dict):
        raise S2CapabilityError("S2 response JSON must be an object.")
    return payload


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _adapt_action(action_payload: Mapping[str, Any] | None, route: str) -> Action | None:
    if route in {"halt", "human_confirm"}:
        return None
    if not isinstance(action_payload, Mapping):
        raise S2CapabilityError("S2 response missing action object.")

    raw_type = str(action_payload.get("type") or "").strip().lower()
    if not raw_type:
        raise S2CapabilityError("S2 action missing type.")
    action_type = ACTION_TYPE_ALIASES.get(raw_type, raw_type)

    args = action_payload.get("arguments") or {}
    if not isinstance(args, Mapping):
        raise S2CapabilityError("S2 action arguments must be an object.")

    parsed_payload = {"action_type": action_type, **dict(args)}
    if route == "done":
        parsed_payload.setdefault("action_type", "done")
        parsed_payload.setdefault("status", "success")

    try:
        return parse_action(parsed_payload)
    except ActionError as exc:
        raise S2CapabilityError(f"S2 action cannot adapt to OpenGUI Action: {exc}") from exc


def adapt_s2_action_output(payload: Mapping[str, Any]) -> S2ActionCandidate:
    """Validate S2 action JSON and adapt it to an OpenGUI Action."""
    route = str(payload.get("route") or "").strip().lower()
    if route not in ALLOWED_ROUTES:
        raise S2CapabilityError(f"Unsupported S2 route: {route!r}.")

    safety = payload.get("safety_check") or {}
    if not isinstance(safety, Mapping):
        raise S2CapabilityError("safety_check must be an object.")

    action = _adapt_action(payload.get("action"), route)
    reason = str(payload.get("reason") or "").strip()
    semantic_target = str(payload.get("semantic_target") or "").strip()
    side_effect = _coerce_bool(safety.get("side_effect", False))
    requires_human_confirm = _coerce_bool(safety.get("requires_human_confirm", False))

    return S2ActionCandidate(
        route=route,
        action=action,
        reason=reason,
        semantic_target=semantic_target,
        side_effect=side_effect,
        requires_human_confirm=requires_human_confirm,
        raw=dict(payload),
    )


def safety_filter_passed(candidate: S2ActionCandidate) -> bool:
    """Return True when a candidate is safe enough for a smoke success."""
    return not candidate.side_effect and not candidate.requires_human_confirm
```

- [ ] **Step 4: Run schema adapter tests and verify they pass**

Run:

```bash
pytest tests/agent/test_s2_capability_smoke.py -q
```

Expected: all tests in the file pass.

- [ ] **Step 5: Commit the schema adapter**

Run:

```bash
git add nanobot/agent/s2_capability_smoke.py tests/agent/test_s2_capability_smoke.py
git commit -m "feat: add s2 action smoke schema"
```

---

### Task 2: Screenshot Message Builder

**Files:**
- Modify: `nanobot/agent/s2_capability_smoke.py`
- Modify: `tests/agent/test_s2_capability_smoke.py`

- [ ] **Step 1: Add failing tests for screenshot prompt construction**

Append these tests to `tests/agent/test_s2_capability_smoke.py`:

```python
from pathlib import Path

from nanobot.agent.s2_capability_smoke import build_s2_action_messages


PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4"
    b"\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05"
    b"\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_build_s2_action_messages_include_image_block(tmp_path: Path) -> None:
    screenshot = tmp_path / "screen.png"
    screenshot.write_bytes(PNG_1X1)

    messages = build_s2_action_messages(
        task="选择 2026-06-05 的出发日期",
        screenshot_path=screenshot,
        case_name="datepicker",
    )

    assert messages[0]["role"] == "system"
    assert "return only JSON" in messages[0]["content"].lower()
    user_content = messages[1]["content"]
    assert user_content[0]["type"] == "image_url"
    assert user_content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert user_content[1]["type"] == "text"
    assert "datepicker" in user_content[1]["text"]


def test_build_s2_action_messages_rejects_non_image(tmp_path: Path) -> None:
    screenshot = tmp_path / "screen.txt"
    screenshot.write_text("not image", encoding="utf-8")

    with pytest.raises(S2CapabilityError, match="Unsupported image"):
        build_s2_action_messages(
            task="选择日期",
            screenshot_path=screenshot,
            case_name="not_image",
        )
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
pytest tests/agent/test_s2_capability_smoke.py -q
```

Expected: fails because `build_s2_action_messages` is not implemented.

- [ ] **Step 3: Implement the screenshot message builder**

Append this code to `nanobot/agent/s2_capability_smoke.py`:

```python
S2_ACTION_SYSTEM_PROMPT = """You are the S2 GUI takeover capability smoke model.
Given a task and a screenshot, return only JSON using this schema:
{
  "route": "continue | done | halt | human_confirm",
  "action": {
    "type": "click | type | swipe | wait | back | home | done",
    "arguments": {}
  },
  "reason": "short trace-grounded reason for this action",
  "semantic_target": "constraint or subgoal this action advances",
  "safety_check": {
    "side_effect": false,
    "requires_human_confirm": false
  }
}
Do not include markdown, prose, or hidden reasoning. Do not propose payment,
purchase, send, submit, delete, account modification, privacy toggles, or
sensitive permission grants.
"""


def build_s2_action_messages(
    *,
    task: str,
    screenshot_path: Path,
    case_name: str,
) -> list[dict[str, Any]]:
    """Build a screenshot plus task prompt for the S2 action smoke."""
    raw = screenshot_path.read_bytes()
    mime = detect_image_mime(raw)
    if not mime:
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
                raw,
                mime,
                str(screenshot_path),
                label,
            ),
        },
    ]
```

- [ ] **Step 4: Run screenshot message tests and verify they pass**

Run:

```bash
pytest tests/agent/test_s2_capability_smoke.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the message builder**

Run:

```bash
git add nanobot/agent/s2_capability_smoke.py tests/agent/test_s2_capability_smoke.py
git commit -m "feat: build s2 smoke image prompts"
```

---

### Task 3: Offline Smoke Runner With Fake Provider

**Files:**
- Modify: `nanobot/agent/s2_capability_smoke.py`
- Modify: `tests/agent/test_s2_capability_smoke.py`

- [ ] **Step 1: Add failing tests for smoke runner and image contrast**

Append these tests to `tests/agent/test_s2_capability_smoke.py`:

```python
from nanobot.providers.base import LLMResponse
from nanobot.agent.s2_capability_smoke import (
    S2SmokeCase,
    actions_are_materially_different,
    run_s2_capability_smoke,
)


class FakeProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    async def chat_with_retry(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("No fake responses left.")
        return LLMResponse(content=self.responses.pop(0), usage={"total_tokens": 10})


def test_actions_are_materially_different() -> None:
    first = adapt_s2_action_output({
        "route": "continue",
        "action": {"type": "click", "arguments": {"x": 100, "y": 200}},
        "reason": "tap date",
        "semantic_target": "select_date",
        "safety_check": {"side_effect": False, "requires_human_confirm": False},
    })
    second = adapt_s2_action_output({
        "route": "continue",
        "action": {"type": "back", "arguments": {}},
        "reason": "leave wrong page",
        "semantic_target": "recover_page",
        "safety_check": {"side_effect": False, "requires_human_confirm": False},
    })

    assert actions_are_materially_different(first, second) is True


async def test_run_s2_capability_smoke_passes_with_contrasting_actions(tmp_path: Path) -> None:
    screen_a = tmp_path / "a.png"
    screen_b = tmp_path / "b.png"
    screen_a.write_bytes(PNG_1X1)
    screen_b.write_bytes(PNG_1X1)
    provider = FakeProvider([
        '{"route":"continue","action":{"type":"click","arguments":{"x":100,"y":200}},'
        '"reason":"tap visible day","semantic_target":"select_date",'
        '"safety_check":{"side_effect":false,"requires_human_confirm":false}}',
        '{"route":"continue","action":{"type":"back","arguments":{}},'
        '"reason":"wrong screen","semantic_target":"recover_screen",'
        '"safety_check":{"side_effect":false,"requires_human_confirm":false}}',
    ])

    report = await run_s2_capability_smoke(
        provider=provider,
        model="qwen3.5-397b-a17b",
        task="选择 2026-06-05 的出发日期",
        case_a=S2SmokeCase(name="date_picker", screenshot_path=screen_a),
        case_b=S2SmokeCase(name="wrong_page", screenshot_path=screen_b),
    )

    assert report.model == "qwen3.5-397b-a17b"
    assert report.schema_parse_success is True
    assert report.action_adapter_success is True
    assert report.unsafe_action_filter_pass is True
    assert report.image_use_contrast_pass is True
    assert len(provider.calls) == 2


async def test_run_s2_capability_smoke_fails_contrast_when_actions_match(tmp_path: Path) -> None:
    screen_a = tmp_path / "a.png"
    screen_b = tmp_path / "b.png"
    screen_a.write_bytes(PNG_1X1)
    screen_b.write_bytes(PNG_1X1)
    same_response = (
        '{"route":"continue","action":{"type":"click","arguments":{"x":100,"y":200}},'
        '"reason":"tap visible day","semantic_target":"select_date",'
        '"safety_check":{"side_effect":false,"requires_human_confirm":false}}'
    )
    provider = FakeProvider([same_response, same_response])

    report = await run_s2_capability_smoke(
        provider=provider,
        model="qwen3.5-397b-a17b",
        task="选择 2026-06-05 的出发日期",
        case_a=S2SmokeCase(name="date_picker", screenshot_path=screen_a),
        case_b=S2SmokeCase(name="wrong_page", screenshot_path=screen_b),
    )

    assert report.image_use_contrast_pass is False
```

- [ ] **Step 2: Run the runner tests and verify they fail**

Run:

```bash
pytest tests/agent/test_s2_capability_smoke.py -q
```

Expected: fails because `S2SmokeCase`, `run_s2_capability_smoke`, and `actions_are_materially_different` are not implemented.

- [ ] **Step 3: Implement runner dataclasses and smoke execution**

Append this code to `nanobot/agent/s2_capability_smoke.py`:

```python
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


def actions_are_materially_different(
    first: S2ActionCandidate,
    second: S2ActionCandidate,
) -> bool:
    """Return True when two S2 actions differ enough for image-use contrast."""
    if first.route != second.route:
        return True
    if (first.action is None) != (second.action is None):
        return True
    if first.action is None or second.action is None:
        return first.semantic_target != second.semantic_target
    return (
        first.action.action_type,
        first.action.x,
        first.action.y,
        first.action.x2,
        first.action.y2,
        first.action.text,
        first.action.status,
        first.semantic_target,
    ) != (
        second.action.action_type,
        second.action.x,
        second.action.y,
        second.action.x2,
        second.action.y2,
        second.action.text,
        second.action.status,
        second.semantic_target,
    )


async def _run_case(
    *,
    provider: LLMProvider,
    model: str,
    task: str,
    case: S2SmokeCase,
    max_tokens: int,
) -> tuple[S2SmokeCaseResult, S2ActionCandidate | None]:
    messages = build_s2_action_messages(
        task=task,
        screenshot_path=case.screenshot_path,
        case_name=case.name,
    )
    started = time.perf_counter()
    try:
        response = await provider.chat_with_retry(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=0,
        )
        latency_s = time.perf_counter() - started
        payload = extract_json_object(response.content or "")
        candidate = adapt_s2_action_output(payload)
        filter_pass = safety_filter_passed(candidate)
        return (
            S2SmokeCaseResult(
                name=case.name,
                screenshot_path=str(case.screenshot_path),
                raw_content=response.content or "",
                route=candidate.route,
                action_type=candidate.action.action_type if candidate.action else None,
                reason=candidate.reason,
                semantic_target=candidate.semantic_target,
                schema_parse_success=True,
                action_adapter_success=True,
                unsafe_action_filter_pass=filter_pass,
                error=None,
                latency_s=latency_s,
                usage=dict(response.usage or {}),
            ),
            candidate,
        )
    except Exception as exc:
        latency_s = time.perf_counter() - started
        return (
            S2SmokeCaseResult(
                name=case.name,
                screenshot_path=str(case.screenshot_path),
                raw_content="",
                route=None,
                action_type=None,
                reason="",
                semantic_target="",
                schema_parse_success=False,
                action_adapter_success=False,
                unsafe_action_filter_pass=False,
                error=f"{type(exc).__name__}: {exc}",
                latency_s=latency_s,
                usage={},
            ),
            None,
        )


def _merge_usage(results: Sequence[S2SmokeCaseResult]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for result in results:
        for key, value in result.usage.items():
            merged[key] = merged.get(key, 0) + int(value)
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
    """Run saved-screenshot S2 capability smoke without touching any device."""
    started = time.perf_counter()
    result_a, candidate_a = await _run_case(
        provider=provider,
        model=model,
        task=task,
        case=case_a,
        max_tokens=max_tokens,
    )
    case_results = [result_a]
    candidate_b: S2ActionCandidate | None = None
    if case_b is not None:
        result_b, candidate_b = await _run_case(
            provider=provider,
            model=model,
            task=task,
            case=case_b,
            max_tokens=max_tokens,
        )
        case_results.append(result_b)

    image_contrast = False
    if candidate_a is not None and candidate_b is not None:
        image_contrast = actions_are_materially_different(candidate_a, candidate_b)

    return S2SmokeReport(
        model=model,
        task=task,
        cases=case_results,
        schema_parse_success=all(item.schema_parse_success for item in case_results),
        action_adapter_success=all(item.action_adapter_success for item in case_results),
        unsafe_action_filter_pass=all(item.unsafe_action_filter_pass for item in case_results),
        image_use_contrast_pass=image_contrast,
        duration_s=time.perf_counter() - started,
        usage=_merge_usage(case_results),
    )
```

- [ ] **Step 4: Run runner tests and verify they pass**

Run:

```bash
pytest tests/agent/test_s2_capability_smoke.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the runner**

Run:

```bash
git add nanobot/agent/s2_capability_smoke.py tests/agent/test_s2_capability_smoke.py
git commit -m "feat: run offline s2 capability smoke"
```

---

### Task 4: CLI Entrypoint For Real Endpoint Smoke

**Files:**
- Modify: `nanobot/agent/s2_capability_smoke.py`
- Modify: `tests/agent/test_s2_capability_smoke.py`

- [ ] **Step 1: Add failing tests for argument parsing**

Append this test to `tests/agent/test_s2_capability_smoke.py`:

```python
from nanobot.agent.s2_capability_smoke import parse_args


def test_parse_args_accepts_required_smoke_inputs(tmp_path: Path) -> None:
    screen_a = tmp_path / "a.png"
    screen_b = tmp_path / "b.png"
    output = tmp_path / "report.json"

    args = parse_args([
        "--task",
        "选择 2026-06-05 的出发日期",
        "--screenshot-a",
        str(screen_a),
        "--screenshot-b",
        str(screen_b),
        "--output",
        str(output),
        "--max-tokens",
        "256",
    ])

    assert args.task == "选择 2026-06-05 的出发日期"
    assert args.screenshot_a == screen_a
    assert args.screenshot_b == screen_b
    assert args.output == output
    assert args.max_tokens == 256
```

- [ ] **Step 2: Run CLI parser test and verify it fails**

Run:

```bash
pytest tests/agent/test_s2_capability_smoke.py -q
```

Expected: fails because `parse_args` is not implemented.

- [ ] **Step 3: Implement parser, async main, and module entrypoint**

Append this code to `nanobot/agent/s2_capability_smoke.py`:

```python
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run saved-screenshot S2 visual GUI executor capability smoke.",
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--screenshot-a", type=Path, required=True)
    parser.add_argument("--screenshot-b", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args(argv)


async def _amain(argv: Sequence[str] | None = None) -> int:
    from nanobot.config.loader import load_config, resolve_config_env_vars
    from nanobot.providers.factory import build_gui_s2_provider_snapshot

    args = parse_args(argv)
    config = resolve_config_env_vars(load_config(args.config))
    snapshot = build_gui_s2_provider_snapshot(config)
    if snapshot is None:
        raise SystemExit("No GUI S2 model configured. Set gui.s2Model and gui.s2Provider.")

    report = await run_s2_capability_smoke(
        provider=snapshot.provider,
        model=snapshot.model,
        task=args.task,
        case_a=S2SmokeCase(name="screenshot_a", screenshot_path=args.screenshot_a),
        case_b=(
            S2SmokeCase(name="screenshot_b", screenshot_path=args.screenshot_b)
            if args.screenshot_b else None
        ),
        max_tokens=args.max_tokens,
    )
    output = json.dumps(report.to_json_dict(), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_amain(argv))


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run parser tests and smoke unit tests**

Run:

```bash
pytest tests/agent/test_s2_capability_smoke.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit CLI entrypoint**

Run:

```bash
git add nanobot/agent/s2_capability_smoke.py tests/agent/test_s2_capability_smoke.py
git commit -m "feat: add s2 capability smoke cli"
```

---

### Task 5: Verification And Real Smoke Command

**Files:**
- No new implementation files unless tests reveal a defect in files from Tasks 1-4.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
pytest tests/agent/test_s2_capability_smoke.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run import smoke**

Run:

```bash
python -m nanobot.agent.s2_capability_smoke --help
```

Expected: exits 0 and prints arguments including `--task`, `--screenshot-a`, `--screenshot-b`, and `--output`.

- [ ] **Step 3: Run real S2 capability smoke on saved screenshots**

Use saved screenshots only. Do not touch the phone.

Run:

```bash
python -m nanobot.agent.s2_capability_smoke \
  --task "选择 2026-06-05 的出发日期；如果截图不是日期选择器，请给出恢复动作" \
  --screenshot-a /Users/su/.nanobot/workspace/gui_runs/2026-06-03_211449_156721/App_6_5_1780492721775_2/screenshots/step_013.png \
  --screenshot-b /Users/su/.nanobot/workspace/gui_runs/2026-06-03_205838_440722/gui_task_1780491518442_0/screenshots/step_001.png \
  --output /tmp/s2_capability_smoke.json \
  --max-tokens 512
```

Expected: command writes `/tmp/s2_capability_smoke.json`. The report should contain:

```json
{
  "model": "qwen3.5-397b-a17b",
  "schema_parse_success": true,
  "action_adapter_success": true,
  "unsafe_action_filter_pass": true,
  "image_use_contrast_pass": true
}
```

If the command fails, classify the failing layer exactly:

- provider config missing;
- endpoint unavailable;
- image payload unsupported;
- S2 output not JSON;
- schema parse failed;
- action adapter failed;
- safety filter failed;
- contrast failed because actions did not change across screenshots.

- [ ] **Step 4: Confirm no live GUI files were changed**

Run:

```bash
git diff -- opengui/agent.py opengui/autonomy_monitor.py nanobot/agent/tools/gui.py
```

Expected: no diff from this task. Pre-existing unrelated diffs may remain in the working tree, but S2-1 should not add or modify these files.

- [ ] **Step 5: Commit verification follow-up only if files changed**

If Task 5 required fixes, commit only the smoke files:

```bash
git add nanobot/agent/s2_capability_smoke.py tests/agent/test_s2_capability_smoke.py
git commit -m "test: verify s2 capability smoke"
```

If no files changed, do not create an empty commit.

---

## Self-Review

Spec coverage:

- S2 capability gate: Tasks 2-5.
- Image-use contrast test: Task 3 and Task 5.
- Schema parse success: Task 1 and Task 3.
- Action adapter success: Task 1 and Task 3.
- Safety filter success: Task 1 and Task 3.
- No controller/live takeover changes: Scope Guard and Task 5 Step 4.

Gaps:

- This plan does not implement handoff packets, offline dry-run recovery on failed traces, controller routes, or live takeover. Those are later stages in the spec and are intentionally out of scope for S2-1.

Execution boundary:

- If real S2 smoke shows that the endpoint does not use screenshots, stop after reporting the failing layer. Do not implement text-only fallback in this task.
- If real S2 smoke shows parseable text-only planning but no visual action capability, record that result and defer architecture changes to the next design discussion.
