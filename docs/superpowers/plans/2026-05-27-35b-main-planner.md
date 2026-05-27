# 35B Main Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 35B content-JSON main planner that semantically routes user tasks before execution while preserving host policy and monitor controls.

**Architecture:** Add a focused `nanobot.agent.main_planner` module that calls a configured LLM provider with no native tool calls, parses a structured JSON plan, and converts it into host `RouteDecision`s. `AgentLoop` runs policy first, then uses the planner when configured; GUI planner routes are executed directly through `gui_task`, tool routes inject a route hint, and malformed or low-confidence plans fall back to the deterministic router.

**Tech Stack:** Python 3.12, Pydantic config schema, existing `LLMProvider`, pytest, ruff.

---

### Task 1: Planner Contract And Parser

**Files:**
- Create: `nanobot/agent/main_planner.py`
- Test: `tests/agent/test_main_planner.py`

- [ ] **Step 1: Write failing parser tests**

```python
from __future__ import annotations

import pytest

from nanobot.agent.cost_aware_router import RouteKind
from nanobot.agent.main_planner import MainPlanner, PlannerConfig


def test_parse_gui_plan_to_route_decision() -> None:
    planner = MainPlanner(provider=None, model=None, config=PlannerConfig(enabled=True))
    decision = planner.parse_decision(
        '{"route":"gui_task","confidence":0.91,"reason":"App internal task",'
        '"subtasks":[{"route":"gui_task","task":"在抖音里搜索旅行Vlog"}]}',
        original_task="在抖音里搜索旅行Vlog",
    )

    assert decision.route == RouteKind.GUI
    assert decision.requires_gui is True
    assert decision.reason == "App internal task"
    assert decision.suggested_tools == ()


def test_parse_rejects_low_confidence() -> None:
    planner = MainPlanner(provider=None, model=None, config=PlannerConfig(enabled=True, confidence_threshold=0.65))

    with pytest.raises(ValueError, match="low confidence"):
        planner.parse_decision(
            '{"route":"gui_task","confidence":0.2,"reason":"not sure","subtasks":[]}',
            original_task="打开设置",
        )
```

- [ ] **Step 2: Run tests and verify red**

Run: `uv run pytest tests/agent/test_main_planner.py -q`

Expected: import failure for `nanobot.agent.main_planner`.

- [ ] **Step 3: Implement minimal parser**

Create `nanobot/agent/main_planner.py` with `PlannerConfig`, `MainPlanner`, JSON extraction, route mapping, confidence validation, and `RouteDecision` creation.

- [ ] **Step 4: Run tests and verify green**

Run: `uv run pytest tests/agent/test_main_planner.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add nanobot/agent/main_planner.py tests/agent/test_main_planner.py
git commit -m "feat: add 35b main planner contract"
```

### Task 2: Planner Runtime Configuration

**Files:**
- Modify: `nanobot/config/schema.py`
- Modify: `nanobot/providers/factory.py`
- Modify: `nanobot/cli/commands.py`
- Modify: `nanobot/nanobot.py`
- Test: `tests/providers/test_model_role_routing.py`
- Test: `tests/test_opengui_p3_nanobot.py`

- [ ] **Step 1: Write failing config tests**

Add assertions that `GuiConfig` accepts `plannerEnabled`, `plannerModel`, `plannerProvider`, `plannerConfidenceThreshold`, `plannerMaxTokens`, and `plannerTimeoutSeconds`, and that `build_gui_planner_provider_snapshot` keeps planner separate from GUI S1 and S2.

- [ ] **Step 2: Run tests and verify red**

Run: `uv run pytest tests/providers/test_model_role_routing.py tests/test_opengui_p3_nanobot.py::test_agent_loop_registers_gui_tool_with_gui_runtime_override -q`

Expected: missing field/helper failures.

- [ ] **Step 3: Implement config wiring**

Add planner fields to `GuiConfig`, add `build_gui_planner_provider_snapshot(config)`, and pass `planner_provider/planner_model` into `AgentLoop` from CLI and SDK setup.

- [ ] **Step 4: Run tests and verify green**

Run: `uv run pytest tests/providers/test_model_role_routing.py tests/test_opengui_p3_nanobot.py::test_agent_loop_registers_gui_tool_with_gui_runtime_override -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add nanobot/config/schema.py nanobot/providers/factory.py nanobot/cli/commands.py nanobot/nanobot.py tests/providers/test_model_role_routing.py tests/test_opengui_p3_nanobot.py
git commit -m "feat: wire 35b planner runtime config"
```

### Task 3: AgentLoop Integration

**Files:**
- Modify: `nanobot/agent/loop.py`
- Test: `tests/agent/test_cost_aware_routing_integration.py`

- [ ] **Step 1: Write failing integration tests**

Add tests where a mock planner returns `gui_task` and `tool_call`. GUI routes should call `gui_task` directly; tool routes should inject a planner hint into the main LLM prompt. Add one policy test showing sensitive tasks are blocked before the planner is called.

- [ ] **Step 2: Run tests and verify red**

Run: `uv run pytest tests/agent/test_cost_aware_routing_integration.py -q`

Expected: planner constructor args and behavior are not implemented.

- [ ] **Step 3: Implement loop integration**

Extend `AgentLoop.__init__` with planner provider/model/config, instantiate `MainPlanner` when enabled, run policy first, call the planner, and fall back to `CostAwareProblemRouter` on planner errors. Add direct GUI execution for planner GUI routes.

- [ ] **Step 4: Run tests and verify green**

Run: `uv run pytest tests/agent/test_cost_aware_routing_integration.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add nanobot/agent/loop.py tests/agent/test_cost_aware_routing_integration.py
git commit -m "feat: route host tasks through 35b planner"
```

### Task 4: Verification

**Files:**
- Modify only if previous tests reveal regressions.

- [ ] **Step 1: Run focused tests**

Run: `uv run pytest tests/agent/test_main_planner.py tests/agent/test_cost_aware_router.py tests/agent/test_cost_aware_routing_integration.py tests/providers/test_model_role_routing.py tests/test_opengui_p3_nanobot.py -q`

Expected: all pass.

- [ ] **Step 2: Run lint**

Run: `uv run ruff check nanobot/agent/main_planner.py nanobot/agent/loop.py nanobot/config/schema.py nanobot/providers/factory.py nanobot/cli/commands.py nanobot/nanobot.py tests/agent/test_main_planner.py tests/agent/test_cost_aware_routing_integration.py tests/providers/test_model_role_routing.py tests/test_opengui_p3_nanobot.py`

Expected: all checks pass.

- [ ] **Step 3: Push**

```bash
git push origin feat/cost-aware-router-v0
```
