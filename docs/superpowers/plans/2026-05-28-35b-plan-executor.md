# 35B PlanExecutor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an opt-in 35B `PlannerPlan` + `PlanExecutor` path that executes validated subtasks sequentially while preserving policy, monitor, and S2 boundaries.

**Architecture:** Extend `MainPlanner` to parse full typed plans while keeping the existing `RouteDecision` fallback API. Add a focused host-level `PlanExecutor` that owns subtask orchestration and delegates execution back to existing tools/routes. Wire it behind `gui.plannerSubtasksEnabled` so the current single-route behavior remains the default fallback until real iOS smoke tests pass.

**Tech Stack:** Python 3.12, dataclasses/enums, pytest/pytest-asyncio, existing `AgentLoop`, `CostAwareProblemRouter`, `PolicyDecision`, `gui_task` tool, OpenAI-compatible providers.

---

## File Structure

- Modify `nanobot/agent/main_planner.py`
  - Add `PlannerSubtask`, `PlannerPlan`, and full-plan parsing.
  - Keep `plan()` returning `RouteDecision` for backward compatibility.
  - Add `plan_full()` returning `PlannerPlan`.

- Create `nanobot/agent/plan_executor.py`
  - Own sequential subtask execution.
  - Own subtask result aggregation and validation.
  - Depend on injected policy checker and injected subtask dispatcher, not concrete GUI internals.

- Modify `nanobot/config/schema.py`
  - Add `gui.plannerSubtasksEnabled: bool = False`.

- Modify `nanobot/agent/loop.py`
  - Instantiate/use `PlanExecutor` only when planner and `plannerSubtasksEnabled` are enabled.
  - Add small dispatch adapter methods that call existing tool/system/gui route code.

- Modify `scripts/nanobot-ios-start`
  - Add `NANOBOT_PLANNER_SUBTASKS_ENABLED` env override.
  - Write `gui.plannerSubtasksEnabled` into runtime config.

- Add/modify tests:
  - `tests/agent/test_main_planner.py`
  - `tests/agent/test_plan_executor.py`
  - `tests/agent/test_cost_aware_routing_integration.py`
  - `tests/providers/test_model_role_routing.py`

---

### Task 1: Add Typed PlannerPlan Parsing

**Files:**
- Modify: `nanobot/agent/main_planner.py`
- Test: `tests/agent/test_main_planner.py`

- [ ] **Step 1: Write failing parser tests**

Add these tests to `tests/agent/test_main_planner.py`:

```python
def test_parse_full_plan_with_multiple_subtasks() -> None:
    planner = MainPlanner(provider=None, model=None, config=PlannerConfig(enabled=True))

    plan = planner.parse_plan(
        """
        {
          "route": "plan",
          "confidence": 0.92,
          "reason": "App playback needs app-local actions.",
          "subtasks": [
            {
              "id": "open_bilibili",
              "route": "system_action",
              "task": "Open Bilibili",
              "system_action": "open_app",
              "success_condition": "Bilibili is in foreground",
              "risk_level": "low"
            },
            {
              "id": "search_video",
              "route": "gui_task",
              "task": "Search Bilibili for 罗翔 刑法课",
              "success_condition": "Search results are visible",
              "risk_level": "low"
            },
            {
              "id": "play_video",
              "route": "gui_task",
              "task": "Open a matching video and start playback",
              "success_condition": "Video is playing",
              "risk_level": "low"
            }
          ],
          "risk_notes": ["public video playback"]
        }
        """,
        original_task="在B站播放罗翔的刑法课视频。",
    )

    assert plan.original_task == "在B站播放罗翔的刑法课视频。"
    assert plan.confidence == 0.92
    assert plan.reason == "App playback needs app-local actions."
    assert plan.risk_notes == ("public video playback",)
    assert [subtask.id for subtask in plan.subtasks] == [
        "open_bilibili",
        "search_video",
        "play_video",
    ]
    assert plan.subtasks[0].route == RouteKind.SYSTEM_ACTION
    assert plan.subtasks[0].system_action == "open_app"
    assert plan.subtasks[1].route == RouteKind.GUI
    assert plan.subtasks[1].success_condition == "Search results are visible"


def test_parse_full_plan_synthesizes_subtask_for_legacy_route() -> None:
    planner = MainPlanner(provider=None, model=None, config=PlannerConfig(enabled=True))

    plan = planner.parse_plan(
        '{"route":"gui_task","confidence":0.9,"reason":"App task","subtasks":[]}',
        original_task="在抖音里搜索旅行Vlog",
    )

    assert plan.route == RouteKind.GUI
    assert len(plan.subtasks) == 1
    assert plan.subtasks[0].id == "subtask_1"
    assert plan.subtasks[0].route == RouteKind.GUI
    assert plan.subtasks[0].task == "在抖音里搜索旅行Vlog"


def test_parse_full_plan_rejects_unsupported_subtask_route() -> None:
    planner = MainPlanner(provider=None, model=None, config=PlannerConfig(enabled=True))

    with pytest.raises(ValueError, match="unsupported subtask route"):
        planner.parse_plan(
            '{"route":"plan","confidence":0.9,"reason":"bad",'
            '"subtasks":[{"route":"shell","task":"run rm"}]}',
            original_task="bad",
        )


@pytest.mark.asyncio
async def test_plan_full_uses_content_json_without_native_tool_calls() -> None:
    provider = SimpleNamespace(
        chat_with_retry=AsyncMock(return_value=LLMResponse(
            content='{"route":"plan","confidence":0.9,"reason":"ok",'
            '"subtasks":[{"route":"web_search","tool":"web_search","task":"查询深圳天气"}]}'
        ))
    )
    planner = MainPlanner(
        provider=provider,
        model="qwen3.6-35b-a3b",
        config=PlannerConfig(enabled=True),
    )

    plan = await planner.plan_full("查询深圳天气", available_tools={"web_search", "gui_task"})

    assert plan.subtasks[0].route == RouteKind.TOOL_CALL
    call = provider.chat_with_retry.await_args
    assert call.kwargs["tools"] is None
    assert call.kwargs["tool_choice"] is None
    assert call.kwargs["model"] == "qwen3.6-35b-a3b"
```

- [ ] **Step 2: Run parser tests and verify failure**

Run:

```bash
uv run pytest tests/agent/test_main_planner.py -q
```

Expected: failures for missing `parse_plan`, `plan_full`, and new plan dataclasses.

- [ ] **Step 3: Implement typed plan models and parser**

In `nanobot/agent/main_planner.py`, add these dataclasses near `PlannerConfig`:

```python
_RISK_LEVELS = frozenset({"low", "medium", "high"})


@dataclass(frozen=True)
class PlannerSubtask:
    id: str
    route: RouteKind
    task: str
    tool: str | None = None
    system_action: str | None = None
    validator: str | None = None
    success_condition: str | None = None
    risk_level: str = "low"


@dataclass(frozen=True)
class PlannerPlan:
    original_task: str
    route: RouteKind
    confidence: float
    reason: str
    subtasks: tuple[PlannerSubtask, ...]
    risk_notes: tuple[str, ...] = ()
```

Extend `_ROUTE_MAP`:

```python
"plan": RouteKind.GUI,
"web_search": RouteKind.TOOL_CALL,
"web_fetch": RouteKind.TOOL_CALL,
```

Add helpers:

```python
    def parse_plan(self, content: str, *, original_task: str) -> PlannerPlan:
        payload = self._parse_json_payload(content)
        route = self._route_from_payload(payload, key="route")
        confidence = self._confidence(payload.get("confidence"))
        if confidence < self.config.confidence_threshold:
            raise ValueError(
                f"low confidence: {confidence:.3f} < {self.config.confidence_threshold:.3f}"
            )
        reason = str(payload.get("reason") or "35B planner selected this plan.").strip()
        subtasks = tuple(self._parse_subtasks(payload, default_route=route, original_task=original_task))
        risk_notes = tuple(str(note).strip() for note in payload.get("risk_notes") or () if str(note).strip())
        return PlannerPlan(
            original_task=original_task,
            route=route,
            confidence=confidence,
            reason=reason,
            subtasks=subtasks,
            risk_notes=risk_notes,
        )

    async def plan_full(
        self,
        task: str,
        *,
        available_tools: set[str] | frozenset[str],
    ) -> PlannerPlan:
        response = await self._call_planner(task, available_tools=available_tools)
        return self.parse_plan(response.content or "", original_task=task)

    async def _call_planner(
        self,
        task: str,
        *,
        available_tools: set[str] | frozenset[str],
    ):
        if self.provider is None:
            raise RuntimeError("planner provider is not configured")
        messages = self._messages(task, available_tools=available_tools)
        return await asyncio.wait_for(
            self.provider.chat_with_retry(
                messages,
                tools=None,
                model=self.model,
                max_tokens=self.config.max_tokens,
                temperature=0.0,
                tool_choice=None,
            ),
            timeout=self.config.timeout_seconds,
        )
```

Move the existing `messages` literal into `_messages()` and have `plan()` call `plan_full()`:

```python
    async def plan(...):
        plan = await self.plan_full(task, available_tools=available_tools)
        return self.plan_to_route_decision(plan)
```

Add conversion and parsing helpers:

```python
    def plan_to_route_decision(self, plan: PlannerPlan) -> RouteDecision:
        first = plan.subtasks[0] if plan.subtasks else None
        task = first.task if first else plan.original_task
        if plan.route == RouteKind.TOOL_CALL:
            return RouteDecision(... same as existing tool branch ...)
        if plan.route == RouteKind.SYSTEM_ACTION:
            return RouteDecision(... task=task, intent=first.system_action if first else "open_app" ...)
        if plan.route == RouteKind.GUI:
            return RouteDecision(... requires_gui=True, routed_task=task ...)
        return RouteDecision(route=plan.route, reason=plan.reason, policy=PolicyDecision(PolicyAction.ALLOW))
```

Use these route helpers:

```python
    def _route_from_payload(self, payload: dict[str, Any], *, key: str) -> RouteKind:
        route_raw = str(payload.get(key) or "").strip().casefold()
        route = self._ROUTE_MAP.get(route_raw)
        if route is None:
            raise ValueError(f"unsupported route: {route_raw or '<missing>'}")
        return route

    def _route_from_subtask(self, payload: dict[str, Any]) -> RouteKind:
        route_raw = str(payload.get("route") or payload.get("tool") or "").strip().casefold()
        route = self._ROUTE_MAP.get(route_raw)
        if route is None:
            raise ValueError(f"unsupported subtask route: {route_raw or '<missing>'}")
        return route

    def _parse_subtasks(
        self,
        payload: dict[str, Any],
        *,
        default_route: RouteKind,
        original_task: str,
    ) -> list[PlannerSubtask]:
        items = payload.get("subtasks")
        parsed: list[PlannerSubtask] = []
        if isinstance(items, list):
            for index, item in enumerate(items, start=1):
                if isinstance(item, dict):
                    parsed.append(self._parse_subtask(item, index=index))
        if parsed:
            return parsed
        return [PlannerSubtask(id="subtask_1", route=default_route, task=original_task)]

    def _parse_subtask(self, payload: dict[str, Any], *, index: int) -> PlannerSubtask:
        route = self._route_from_subtask(payload)
        risk_level = str(payload.get("risk_level") or "low").strip().lower()
        if risk_level not in _RISK_LEVELS:
            risk_level = "medium"
        task = str(payload.get("task") or "").strip()
        if not task:
            raise ValueError(f"subtask {index} missing task")
        return PlannerSubtask(
            id=self._subtask_id(payload.get("id"), index=index),
            route=route,
            task=task,
            tool=str(payload["tool"]).strip() if payload.get("tool") else None,
            system_action=str(payload["system_action"]).strip() if payload.get("system_action") else None,
            validator=str(payload["validator"]).strip() if payload.get("validator") else None,
            success_condition=str(payload["success_condition"]).strip() if payload.get("success_condition") else None,
            risk_level=risk_level,
        )

    @staticmethod
    def _subtask_id(value: Any, *, index: int) -> str:
        text = str(value or "").strip()
        return text or f"subtask_{index}"
```

Update `__all__`:

```python
__all__ = ["MainPlanner", "PlannerConfig", "PlannerPlan", "PlannerSubtask"]
```

- [ ] **Step 4: Run parser tests**

Run:

```bash
uv run pytest tests/agent/test_main_planner.py -q
```

Expected: all tests in the file pass.

- [ ] **Step 5: Commit parser work**

```bash
git add nanobot/agent/main_planner.py tests/agent/test_main_planner.py
git commit -m "feat: parse 35b planner subtasks"
```

---

### Task 2: Add PlanExecutor Core

**Files:**
- Create: `nanobot/agent/plan_executor.py`
- Test: `tests/agent/test_plan_executor.py`

- [ ] **Step 1: Write failing PlanExecutor tests**

Create `tests/agent/test_plan_executor.py`:

```python
from __future__ import annotations

from typing import Any

import pytest

from nanobot.agent.cost_aware_router import RouteKind
from nanobot.agent.main_planner import PlannerPlan, PlannerSubtask
from nanobot.agent.plan_executor import (
    PlanExecutionStatus,
    PlanExecutor,
    SubtaskExecution,
    SubtaskStatus,
)
from opengui.policy import PolicyAction, PolicyDecision


def _plan(*subtasks: PlannerSubtask) -> PlannerPlan:
    return PlannerPlan(
        original_task="test task",
        route=RouteKind.GUI,
        confidence=0.9,
        reason="test",
        subtasks=subtasks,
    )


@pytest.mark.asyncio
async def test_plan_executor_runs_subtasks_in_order() -> None:
    calls: list[str] = []

    async def dispatch(subtask: PlannerSubtask) -> SubtaskExecution:
        calls.append(subtask.id)
        return SubtaskExecution(status=SubtaskStatus.SUCCESS, output=f"ok:{subtask.id}")

    executor = PlanExecutor(
        policy_check=lambda task: PolicyDecision(PolicyAction.ALLOW),
        dispatch=dispatch,
    )

    result = await executor.execute(_plan(
        PlannerSubtask(id="open", route=RouteKind.SYSTEM_ACTION, task="open app"),
        PlannerSubtask(id="search", route=RouteKind.GUI, task="search"),
    ))

    assert result.status == PlanExecutionStatus.SUCCESS
    assert calls == ["open", "search"]
    assert [item.subtask.id for item in result.subtasks] == ["open", "search"]
    assert "2/2 subtasks completed" in result.summary


@pytest.mark.asyncio
async def test_plan_executor_stops_before_policy_blocked_subtask() -> None:
    calls: list[str] = []

    async def dispatch(subtask: PlannerSubtask) -> SubtaskExecution:
        calls.append(subtask.id)
        return SubtaskExecution(status=SubtaskStatus.SUCCESS, output="should not run")

    executor = PlanExecutor(
        policy_check=lambda task: PolicyDecision(
            PolicyAction.ASK_HUMAN_CONFIRM,
            reason="financial data",
            categories=("financial",),
        ),
        dispatch=dispatch,
    )

    result = await executor.execute(_plan(
        PlannerSubtask(id="credit", route=RouteKind.GUI, task="查看白条额度"),
    ))

    assert result.status == PlanExecutionStatus.HUMAN_CONFIRM
    assert calls == []
    assert "financial" in result.summary


@pytest.mark.asyncio
async def test_plan_executor_stops_after_failed_subtask() -> None:
    calls: list[str] = []

    async def dispatch(subtask: PlannerSubtask) -> SubtaskExecution:
        calls.append(subtask.id)
        if subtask.id == "search":
            return SubtaskExecution(
                status=SubtaskStatus.FAILED,
                output="",
                error="stagnation_detected",
            )
        return SubtaskExecution(status=SubtaskStatus.SUCCESS, output="ok")

    executor = PlanExecutor(
        policy_check=lambda task: PolicyDecision(PolicyAction.ALLOW),
        dispatch=dispatch,
    )

    result = await executor.execute(_plan(
        PlannerSubtask(id="open", route=RouteKind.SYSTEM_ACTION, task="open app"),
        PlannerSubtask(id="search", route=RouteKind.GUI, task="search"),
        PlannerSubtask(id="play", route=RouteKind.GUI, task="play"),
    ))

    assert result.status == PlanExecutionStatus.BLOCKED
    assert calls == ["open", "search"]
    assert result.subtasks[-1].error == "stagnation_detected"
    assert "search" in result.summary


@pytest.mark.asyncio
async def test_plan_executor_can_skip_missing_app_when_allowed() -> None:
    async def dispatch(subtask: PlannerSubtask) -> SubtaskExecution:
        return SubtaskExecution(
            status=SubtaskStatus.SKIPPED,
            output="",
            error="missing_app",
        )

    executor = PlanExecutor(
        policy_check=lambda task: PolicyDecision(PolicyAction.ALLOW),
        dispatch=dispatch,
        allow_skip_missing_apps=True,
    )

    result = await executor.execute(_plan(
        PlannerSubtask(id="open_youku", route=RouteKind.GUI, task="打开优酷"),
    ))

    assert result.status == PlanExecutionStatus.SKIPPED
    assert "missing_app" in result.summary
```

- [ ] **Step 2: Run executor tests and verify failure**

```bash
uv run pytest tests/agent/test_plan_executor.py -q
```

Expected: import failure because `nanobot.agent.plan_executor` does not exist.

- [ ] **Step 3: Implement PlanExecutor**

Create `nanobot/agent/plan_executor.py`:

```python
"""Sequential host-level execution for 35B planner subtasks."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum

from nanobot.agent.main_planner import PlannerPlan, PlannerSubtask
from opengui.policy import PolicyAction, PolicyDecision


class SubtaskStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    NEEDS_USER = "needs_user"


class PlanExecutionStatus(str, Enum):
    SUCCESS = "success"
    BLOCKED = "blocked"
    HUMAN_CONFIRM = "human_confirm"
    NEEDS_USER = "needs_user"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class SubtaskExecution:
    status: SubtaskStatus
    output: str
    error: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SubtaskResult:
    subtask: PlannerSubtask
    status: SubtaskStatus
    output: str
    error: str | None = None
    policy: PolicyDecision | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanExecutionResult:
    status: PlanExecutionStatus
    summary: str
    subtasks: tuple[SubtaskResult, ...]


PolicyCheck = Callable[[str], PolicyDecision]
SubtaskDispatch = Callable[[PlannerSubtask], Awaitable[SubtaskExecution]]


class PlanExecutor:
    """Run validated planner subtasks in order through injected host dispatch."""

    def __init__(
        self,
        *,
        policy_check: PolicyCheck,
        dispatch: SubtaskDispatch,
        allow_skip_missing_apps: bool = False,
    ) -> None:
        self._policy_check = policy_check
        self._dispatch = dispatch
        self._allow_skip_missing_apps = allow_skip_missing_apps

    async def execute(self, plan: PlannerPlan) -> PlanExecutionResult:
        results: list[SubtaskResult] = []
        total = len(plan.subtasks)
        for subtask in plan.subtasks:
            policy = self._policy_check(subtask.task)
            if not policy.allowed:
                categories = ", ".join(policy.categories) or "sensitive_action"
                return PlanExecutionResult(
                    status=PlanExecutionStatus.HUMAN_CONFIRM,
                    summary=f"Subtask {subtask.id} requires human confirmation: {categories}",
                    subtasks=tuple(results),
                )

            execution = await self._dispatch(subtask)
            result = SubtaskResult(
                subtask=subtask,
                status=execution.status,
                output=execution.output,
                error=execution.error,
                policy=policy,
                metadata=execution.metadata,
            )
            results.append(result)

            if execution.status == SubtaskStatus.SUCCESS:
                continue
            if execution.status == SubtaskStatus.NEEDS_USER:
                return PlanExecutionResult(
                    status=PlanExecutionStatus.NEEDS_USER,
                    summary=f"Subtask {subtask.id} needs user input: {execution.output}",
                    subtasks=tuple(results),
                )
            if (
                execution.status == SubtaskStatus.SKIPPED
                and self._allow_skip_missing_apps
                and execution.error == "missing_app"
            ):
                return PlanExecutionResult(
                    status=PlanExecutionStatus.SKIPPED,
                    summary=f"Subtask {subtask.id} skipped: {execution.error}",
                    subtasks=tuple(results),
                )
            return PlanExecutionResult(
                status=PlanExecutionStatus.BLOCKED,
                summary=f"Subtask {subtask.id} failed: {execution.error or execution.output}",
                subtasks=tuple(results),
            )

        return PlanExecutionResult(
            status=PlanExecutionStatus.SUCCESS,
            summary=f"{len(results)}/{total} subtasks completed.",
            subtasks=tuple(results),
        )


__all__ = [
    "PlanExecutionResult",
    "PlanExecutionStatus",
    "PlanExecutor",
    "SubtaskExecution",
    "SubtaskResult",
    "SubtaskStatus",
]
```

- [ ] **Step 4: Run executor tests**

```bash
uv run pytest tests/agent/test_plan_executor.py -q
```

Expected: all tests in `test_plan_executor.py` pass.

- [ ] **Step 5: Commit executor core**

```bash
git add nanobot/agent/plan_executor.py tests/agent/test_plan_executor.py
git commit -m "feat: add planner subtask executor"
```

---

### Task 3: Wire PlanExecutor Into AgentLoop Behind A Gate

**Files:**
- Modify: `nanobot/config/schema.py`
- Modify: `nanobot/agent/loop.py`
- Modify: `tests/providers/test_model_role_routing.py`
- Modify: `tests/agent/test_cost_aware_routing_integration.py`

- [ ] **Step 1: Add failing config test**

In `tests/providers/test_model_role_routing.py`, extend `test_gui_config_accepts_explicit_s2_model_and_provider`:

```python
"plannerSubtasksEnabled": True,
```

and assert:

```python
assert cfg.planner_subtasks_enabled is True
```

Add a default test:

```python
def test_gui_config_disables_planner_subtasks_by_default() -> None:
    assert GuiConfig().planner_subtasks_enabled is False
```

- [ ] **Step 2: Add failing AgentLoop integration tests**

In `tests/agent/test_cost_aware_routing_integration.py`, add:

```python
@pytest.mark.asyncio
async def test_35b_planner_subtask_queue_executes_system_then_gui(tmp_path: Path) -> None:
    loop = _make_loop_with_planner(
        tmp_path,
        '{"route":"plan","confidence":0.92,"reason":"App playback",'
        '"subtasks":['
        '{"id":"open_bilibili","route":"system_action","task":"Open Bilibili","system_action":"open_app"},'
        '{"id":"search_video","route":"gui_task","task":"Search Bilibili for 罗翔 刑法课"}'
        ']}',
    )
    loop.gui_config.planner_subtasks_enabled = True
    gui_tool = _FakeGuiTaskTool()
    loop.tools.register(gui_tool)

    response = await loop._process_message(
        InboundMessage(
            channel="cli",
            sender_id="u1",
            chat_id="cli-chat",
            content="在B站播放罗翔的刑法课视频",
        )
    )

    assert response is not None
    assert "2/2 subtasks completed" in response.content
    assert gui_tool.calls == [
        {"task": "Open Bilibili"},
        {"task": "Search Bilibili for 罗翔 刑法课"},
    ]


@pytest.mark.asyncio
async def test_planner_subtask_policy_block_stops_before_gui(tmp_path: Path) -> None:
    loop = _make_loop_with_planner(
        tmp_path,
        '{"route":"plan","confidence":0.92,"reason":"financial",'
        '"subtasks":[{"id":"credit","route":"gui_task","task":"查看京东金融白条额度"}]}',
    )
    loop.gui_config.planner_subtasks_enabled = True
    gui_tool = _FakeGuiTaskTool()
    loop.tools.register(gui_tool)

    response = await loop._process_message(
        InboundMessage(
            channel="cli",
            sender_id="u1",
            chat_id="cli-chat",
            content="在京东金融里查看一下我的白条总额度是多少",
        )
    )

    assert response is not None
    assert "需要你确认或接管" in response.content or "requires human confirmation" in response.content
    assert gui_tool.calls == []
```

- [ ] **Step 3: Run integration tests and verify failure**

```bash
uv run pytest tests/providers/test_model_role_routing.py tests/agent/test_cost_aware_routing_integration.py -q
```

Expected: config field and PlanExecutor wiring tests fail.

- [ ] **Step 4: Add config field**

In `nanobot/config/schema.py`, add to `GuiConfig` near planner fields:

```python
planner_subtasks_enabled: bool = False
```

- [ ] **Step 5: Wire PlanExecutor in AgentLoop**

In `nanobot/agent/loop.py`, import:

```python
from nanobot.agent.main_planner import MainPlanner, PlannerConfig, PlannerPlan, PlannerSubtask
from nanobot.agent.plan_executor import (
    PlanExecutionResult,
    PlanExecutionStatus,
    PlanExecutor,
    SubtaskExecution,
    SubtaskStatus,
)
```

Add a helper:

```python
    async def _execute_planner_plan(self, plan: PlannerPlan) -> str:
        executor = PlanExecutor(
            policy_check=lambda task: self.router.classify(
                task,
                available_tools=set(self.tools.tool_names),
            ).policy,
            dispatch=self._dispatch_planner_subtask,
        )
        result = await executor.execute(plan)
        if result.status == PlanExecutionStatus.HUMAN_CONFIRM:
            return "这个子任务涉及敏感操作，需要你确认或接管后我才能继续。"
        return result.summary

    async def _dispatch_planner_subtask(self, subtask: PlannerSubtask) -> SubtaskExecution:
        if subtask.route == RouteKind.SYSTEM_ACTION:
            decision = RouteDecision(
                route=RouteKind.SYSTEM_ACTION,
                reason=f"Planner subtask {subtask.id}",
                policy=PolicyDecision(PolicyAction.ALLOW),
                system_action={
                    "backend": None,
                    "task": subtask.task,
                    "intent": subtask.system_action or "open_app",
                },
            )
            output = await self._execute_system_action_route(decision)
            return self._subtask_execution_from_output(output)
        if subtask.route == RouteKind.GUI:
            decision = RouteDecision(
                route=RouteKind.GUI,
                reason=f"Planner subtask {subtask.id}",
                policy=PolicyDecision(PolicyAction.ALLOW),
                requires_gui=True,
                routed_task=subtask.task,
            )
            output = await self._execute_gui_route(decision, original_task=subtask.task)
            return self._subtask_execution_from_output(output)
        if subtask.route == RouteKind.TOOL_CALL:
            return SubtaskExecution(
                status=SubtaskStatus.NEEDS_USER,
                output="tool subtasks are executed through the normal agent loop in V0",
                error="tool_subtask_not_directly_dispatched",
            )
        if subtask.route == RouteKind.S2:
            return SubtaskExecution(
                status=SubtaskStatus.NEEDS_USER,
                output="S2 subtask dispatch is reserved for replan/takeover.",
                error="s2_subtask_not_directly_dispatched",
            )
        return SubtaskExecution(
            status=SubtaskStatus.FAILED,
            output="",
            error=f"unsupported_subtask_route:{subtask.route.value}",
        )

    @staticmethod
    def _subtask_execution_from_output(output: str) -> SubtaskExecution:
        lowered = output.lower()
        if "error:" in lowered or "blocked" in lowered or "stagnation_detected" in lowered:
            return SubtaskExecution(status=SubtaskStatus.FAILED, output=output, error="subtask_failed")
        return SubtaskExecution(status=SubtaskStatus.SUCCESS, output=output)
```

In `_process_message`, replace the current planner route call with gated full-plan execution:

```python
            if self._main_planner is not None and self.gui_config.planner_subtasks_enabled:
                try:
                    plan = await self._main_planner.plan_full(
                        msg.content,
                        available_tools=set(self.tools.tool_names),
                    )
                    content = await self._execute_planner_plan(plan)
                    self._save_direct_router_turn(session, msg.content, content)
                    return OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=content,
                        metadata=dict(msg.metadata or {}),
                    )
                except Exception as exc:
                    logger.warning("35B plan executor failed; falling back to route mode: {}", exc)
```

Keep the existing `_plan_problem_route()` path immediately after this block as fallback.

- [ ] **Step 6: Run integration tests**

```bash
uv run pytest tests/providers/test_model_role_routing.py tests/agent/test_cost_aware_routing_integration.py tests/agent/test_plan_executor.py tests/agent/test_main_planner.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit AgentLoop wiring**

```bash
git add nanobot/config/schema.py nanobot/agent/loop.py tests/providers/test_model_role_routing.py tests/agent/test_cost_aware_routing_integration.py
git commit -m "feat: wire planner subtask executor"
```

---

### Task 4: Launcher Flag, Full Verification, And Manual Smoke

**Files:**
- Modify: `scripts/nanobot-ios-start`
- Optional Test: `tests/providers/test_model_role_routing.py`

- [ ] **Step 1: Add launcher runtime field**

In `scripts/nanobot-ios-start`, add near planner env vars:

```bash
PLANNER_SUBTASKS_ENABLED="${NANOBOT_PLANNER_SUBTASKS_ENABLED:-0}"
```

Add to usage env list:

```bash
  NANOBOT_PLANNER_SUBTASKS_ENABLED=$PLANNER_SUBTASKS_ENABLED
```

Add to `ensure_config_runtime()` Python block:

```python
gui["plannerSubtasksEnabled"] = "$PLANNER_SUBTASKS_ENABLED" not in {"0", "false", "False", ""}
```

- [ ] **Step 2: Run shell syntax check**

```bash
bash -n scripts/nanobot-ios-start
```

Expected: exit code 0.

- [ ] **Step 3: Run focused test suite**

```bash
uv run pytest \
  tests/agent/test_main_planner.py \
  tests/agent/test_plan_executor.py \
  tests/agent/test_cost_aware_routing_integration.py \
  tests/providers/test_model_role_routing.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 4: Run lint**

```bash
uv run ruff check \
  nanobot/agent/main_planner.py \
  nanobot/agent/plan_executor.py \
  nanobot/agent/loop.py \
  nanobot/config/schema.py \
  tests/agent/test_main_planner.py \
  tests/agent/test_plan_executor.py \
  tests/agent/test_cost_aware_routing_integration.py \
  tests/providers/test_model_role_routing.py
```

Expected: `All checks passed!`

- [ ] **Step 5: Commit launcher flag**

```bash
git add scripts/nanobot-ios-start
git commit -m "chore: add planner subtask launcher flag"
```

- [ ] **Step 6: Manual iOS smoke with fresh sessions**

Start runtime:

```bash
cd /Users/su/Documents/Codes/nanobot_fork
NANOBOT_PLANNER_SUBTASKS_ENABLED=1 scripts/nanobot-ios-start preflight
```

Run safe smoke tasks with fresh sessions:

```bash
NANOBOT_PLANNER_SUBTASKS_ENABLED=1 scripts/nanobot-ios-start guiclaw "打开设置" -- --session cli:smoke_settings_20260528
NANOBOT_PLANNER_SUBTASKS_ENABLED=1 scripts/nanobot-ios-start guiclaw "在B站播放罗翔的刑法课视频。" -- --session cli:smoke_bili_luoxiang_20260528
NANOBOT_PLANNER_SUBTASKS_ENABLED=1 scripts/nanobot-ios-start guiclaw "查询一下今天深圳的天气" -- --session cli:smoke_weather_20260528
```

Expected:

- settings task uses `system_action` and completes;
- Bilibili task plans multiple subtasks and either plays video or reports missing app/blocker without infinite loop;
- weather task stays tool-first and does not invoke GUI;
- all GUI runs produce trace paths under `~/.nanobot/workspace/gui_runs`.

- [ ] **Step 7: Push branch**

```bash
git status --short --branch
git push origin feat/cost-aware-router-v0
```

Expected: branch is clean and pushed.

---

## Self-Review Checklist

- Spec coverage:
  - Typed `PlannerPlan` and `PlannerSubtask`: Task 1.
  - Sequential `PlanExecutor`: Task 2.
  - Policy re-check per subtask: Task 2 and Task 3.
  - Existing dispatcher reuse: Task 3.
  - Opt-in rollout: Task 3 and Task 4.
  - Startup config: Task 4.
  - Real smoke tests with fresh sessions: Task 4.

- Placeholder scan:
  - No `TODO`, `TBD`, or unspecified "add tests" steps.
  - Each task has concrete paths, commands, and expected results.

- Type consistency:
  - `PlannerSubtask`, `PlannerPlan`, `SubtaskExecution`, and `PlanExecutionResult` names are consistent across tasks.
  - `planner_subtasks_enabled` maps to config key `plannerSubtasksEnabled`.

