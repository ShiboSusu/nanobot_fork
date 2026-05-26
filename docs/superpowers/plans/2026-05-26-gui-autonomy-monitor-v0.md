# GUI Autonomy Monitor V0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a rule-based autonomy monitor that stops or escalates risky GUI trajectories before the small GUI model keeps repeating low-value actions.

**Architecture:** Keep cost-aware task routing in `nanobot/agent/cost_aware_router.py`, keep sensitive-action policy in `opengui/policy.py`, and add step-level trajectory monitoring in a new `opengui/autonomy.py` module. `GuiAgent` will feed low-cost execution signals into the monitor after each step and return a blocked `AgentResult` when cumulative risk or obvious failure signals exceed budget.

**Tech Stack:** Python dataclasses/enums, existing OpenGUI `Action`/`Observation`/`GuiAgent`, pytest async tests, existing dry-run backend.

---

### Task 1: Add Autonomy Monitor Unit

**Files:**
- Create: `opengui/autonomy.py`
- Test: `tests/test_opengui_autonomy.py`

- [ ] **Step 1: Write unit tests for decisions**

Create `tests/test_opengui_autonomy.py` with tests for:
- low-risk step returns `S1_EXECUTE`;
- action/observe failure returns `S2_TAKEOVER`;
- repeated unchanged actions eventually exceed the cumulative budget;
- safety policy signal returns `HUMAN_CONFIRM`.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/opt/anaconda3/envs/nanobot/bin/python -m pytest tests/test_opengui_autonomy.py -q
```

Expected: import failure for `opengui.autonomy`.

- [ ] **Step 3: Implement monitor dataclasses and scoring**

Create `AutonomyDecisionType`, `AutonomySignal`, `AutonomyDecision`, `AutonomyMonitorConfig`, `AutonomyMonitorState`, and `AutonomyMonitor.evaluate(...)`.

V0 signals:
- `action_failed`
- `observe_failed`
- `screen_unchanged`
- `repeated_action`
- `step_budget_pressure`
- `safety_risk`
- `cumulative_risk_over_budget`

- [ ] **Step 4: Run unit tests**

Run:

```bash
/opt/anaconda3/envs/nanobot/bin/python -m pytest tests/test_opengui_autonomy.py -q
```

Expected: pass.

### Task 2: Wire Monitor Into GuiAgent

**Files:**
- Modify: `opengui/agent.py`
- Test: `tests/test_opengui_autonomy.py`

- [ ] **Step 1: Add integration tests**

Add async tests that use `DryRunBackend` and scripted LLM responses:
- repeated unchanged `wait` steps stop with `autonomy_monitor_intervention`;
- successful changed-screen trajectory still completes;
- trace entry includes `autonomy_monitor` metadata.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
/opt/anaconda3/envs/nanobot/bin/python -m pytest tests/test_opengui_autonomy.py -q
```

Expected: failures because `GuiAgent` does not call the monitor yet.

- [ ] **Step 3: Add monitor constructor dependency**

Add optional `autonomy_monitor` to `GuiAgent.__init__`. If omitted, create default `AutonomyMonitor`.

- [ ] **Step 4: Build per-step signal payload**

After `_run_step`, compute:
- `action_failed` from `result.tool_result.startswith("Action failed:")`;
- `observe_failed` from `result.next_observation is None` for non-terminal actions;
- `screen_unchanged` using existing screen fingerprint helpers;
- `repeated_action` using an action signature of action type, coordinates, text, and keys;
- `step_budget_pressure` from `step_index / max_steps`.

- [ ] **Step 5: Apply monitor decisions**

If decision is `S1_EXECUTE`, continue. If decision is `HUMAN_CONFIRM`, reuse existing intervention path. If decision is `S2_HINT` or `S2_TAKEOVER`, return a blocked `AgentResult` with `error="autonomy_monitor_intervention"` and include the decision reason in trace.

- [ ] **Step 6: Run integration tests**

Run:

```bash
/opt/anaconda3/envs/nanobot/bin/python -m pytest tests/test_opengui_autonomy.py tests/test_opengui.py -k 'stagnation or autonomy or ios_settings_task_uses_direct_bundle_launch' -q
```

Expected: pass.

### Task 3: Per-Stage iOS Smoke Checks

**Files:**
- No code changes unless failures reveal bugs.

- [ ] **Step 1: Smoke after monitor unit tests**

Run:

```bash
/Volumes/T9/Mac/bin/nanobot-ios-start status
/Volumes/T9/Mac/bin/nanobot-ios-start agent -m "帮我用手机打开设置"
```

Expected: WDA is ready and iOS Settings opens through the direct system-action path.

- [ ] **Step 2: Smoke after GuiAgent wiring**

Run:

```bash
/Volumes/T9/Mac/bin/nanobot-ios-start agent -m "帮我用手机打开设置"
```

Expected: same result as Step 1. If this breaks, the monitor wiring affected a simple real-device route and must be fixed before continuing.

### Task 4: Preserve Shared CLI/Telegram Route

**Files:**
- Modify only if needed: `nanobot/agent/loop.py`
- Test: `tests/agent/test_cost_aware_routing_integration.py`

- [ ] **Step 1: Re-run existing shared-route tests**

Run:

```bash
/opt/anaconda3/envs/nanobot/bin/python -m pytest tests/agent/test_cost_aware_routing_integration.py -q
```

Expected: pass without changes. Do not move routing out of `AgentLoop`.

### Task 5: Final Runtime Smoke Matrix

**Files:**
- No code changes unless failures reveal bugs.

- [ ] **Step 1: Check WDA and router status**

Run:

```bash
/Volumes/T9/Mac/bin/nanobot-ios-start status
```

Expected: WDA ready at `http://127.0.0.1:8100`; model route config visible.

- [ ] **Step 2: Verify system-action path**

Run:

```bash
/Volumes/T9/Mac/bin/nanobot-ios-start agent -m "帮我用手机打开设置"
```

Expected: opens iOS Settings through system action, not visual icon search.

- [ ] **Step 3: Verify policy path**

Run:

```bash
/Volumes/T9/Mac/bin/nanobot-ios-start agent -m "帮我登录支付宝并完成付款"
```

Expected: asks for human confirmation and does not execute GUI actions.

- [ ] **Step 4: Verify query path**

Run:

```bash
/Volumes/T9/Mac/bin/nanobot-ios-start agent -m "查一下今天深圳天气"
```

Expected: uses tool-call route hint and does not call GUI unless device state is needed.

### Task 6: Commit And Handoff

**Files:**
- Modify: `docs/cost-aware-router-v0-handoff.md` or create a new monitor handoff doc.

- [ ] **Step 1: Run final targeted tests**

Run:

```bash
/opt/anaconda3/envs/nanobot/bin/python -m pytest \
  tests/test_opengui_autonomy.py \
  tests/agent/test_cost_aware_router.py \
  tests/agent/test_cost_aware_routing_integration.py \
  tests/test_opengui_p15_intervention.py \
  -q
```

Expected: pass.

- [ ] **Step 2: Check diff hygiene**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended files changed.

- [ ] **Step 3: Commit**

Run:

```bash
git add opengui/autonomy.py opengui/agent.py tests/test_opengui_autonomy.py docs/superpowers/plans/2026-05-26-gui-autonomy-monitor-v0.md
git commit -m "feat: add gui autonomy monitor v0"
```
