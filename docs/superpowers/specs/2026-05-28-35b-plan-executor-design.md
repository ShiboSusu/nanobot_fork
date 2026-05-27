# 35B PlanExecutor And Subtask Queue Design

## Goal

Upgrade the current 35B main planner from a route selector into a bounded task planner that can decompose complex user requests into executable subtasks, while preserving the existing safety and monitor boundaries.

The target behavior is:

```text
user task
  -> policy hard gate
  -> 35B MainPlanner emits a typed plan with subtasks[]
  -> PlanExecutor runs subtasks one by one
  -> each subtask uses the cheapest safe executor: tool, system_action, gui_task, ask_user, or S2
  -> each subtask is validated before moving on
  -> failure triggers S2 replan, S2 takeover, ask_user, or halt
```

This is not a wholesale replacement with the colleague's `general_e2e` implementation. We keep our host policy, routing, iOS backend, AutonomyMonitor, and S2 boundary. We absorb the useful parts of that branch: content-only action contracts, strict single-action outputs, normalized action aliases, relative-coordinate conventions, and bounded visual history.

## Current Gap

The current code already has:

- 35B `MainPlanner` after policy;
- `RouteDecision` for `tool_call`, `system_action`, `gui_task`, `ask_user`/`s2`;
- direct host dispatch for GUI/system routes;
- GUI S1 execution with `AutonomyMonitor`;
- S2 guidance when the monitor escalates.

The missing layer is a durable subtask queue. `MainPlanner` can ask for `subtasks`, but `parse_decision()` currently uses only the first subtask text to form a single `routed_task`. Complex tasks are therefore still executed as one large GUI instruction, which makes them fragile and hard to recover.

## Non-Goals

- Do not bypass the main agent or policy layer.
- Do not let a model directly execute tool calls outside host validation.
- Do not implement learned/calibrated risk routing in this step.
- Do not make 397B the default executor for all tasks.
- Do not perform irreversible user-account actions without policy or human confirmation.

## Planner Contract

Add typed plan objects in `nanobot/agent/main_planner.py`.

```text
PlannerPlan
  original_task: str
  route: RouteKind
  confidence: float
  reason: str
  subtasks: tuple[PlannerSubtask, ...]
  risk_notes: tuple[str, ...]

PlannerSubtask
  id: str
  route: RouteKind
  task: str
  tool: str | None
  system_action: str | None
  validator: str | None
  success_condition: str | None
  risk_level: "low" | "medium" | "high"
```

The planner still uses content-only JSON, not native tool calls. Expected JSON:

```json
{
  "route": "plan",
  "confidence": 0.82,
  "reason": "The request needs app-local search and playback.",
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
      "success_condition": "Search results for 罗翔 刑法课 are visible",
      "risk_level": "low"
    },
    {
      "id": "play_video",
      "route": "gui_task",
      "task": "Open a relevant 罗翔刑法课 video and start playback",
      "success_condition": "A matching video is playing",
      "risk_level": "low"
    }
  ],
  "risk_notes": []
}
```

For simple tasks, the planner may emit one subtask. For ambiguous tasks, it emits one `ask_user` subtask. For public queries, it emits tool subtasks rather than GUI subtasks.

## PlanExecutor

Add `nanobot/agent/plan_executor.py` with one host-level executor. It owns only orchestration, not low-level GUI operations.

Responsibilities:

- validate planner output into `PlannerPlan`;
- run subtasks sequentially;
- re-run policy on each subtask before execution;
- dispatch each subtask to an existing host capability;
- record `SubtaskResult` with status, output, trace path, error, and validation;
- stop on blocked/high-risk subtasks;
- call S2 for replan/takeover only when a subtask fails or cumulative plan risk becomes too high.

Execution rules:

```text
for subtask in plan.subtasks:
    policy = classify_policy(subtask.task)
    if policy requires confirmation:
        stop with HUMAN_CONFIRM

    result = execute_subtask(subtask)
    validation = validate_subtask(subtask, result)

    if validation passed:
        continue

    if can_replan:
        ask S2 for a replacement suffix plan
    else:
        stop with blocked summary
```

The executor returns a concise final answer plus structured metadata for logs/tests.

## Subtask Dispatch

Initial dispatch should reuse existing code:

- `tool_call`: route hint into the normal main agent tool loop, or directly execute a whitelisted tool when the subtask names one;
- `system_action`: call existing `_execute_system_action_route`;
- `gui_task`: call existing `gui_task` tool with the subtask task string;
- `ask_user`: return a user question and pause;
- `s2`: ask 397B for a replan/takeover decision.

The first implementation should not create a separate GUI backend. It should pass smaller, clearer tasks into the current GUI S1 executor. This keeps the blast radius low and lets the existing monitor/S2 hint code keep working.

## Absorbing General E2E

Use the colleague branch as a contract reference in two places:

1. GUI profile contract
   - Prefer content-only `Thought:` and `Action:` JSON for local/non-tool-call models.
   - Keep one action per response.
   - Normalize aliases like `tap`, `press`, and `touch` to a canonical action.
   - Keep normalized relative coordinates in a predictable range.

2. Visual history discipline
   - Keep the latest N screenshots for context.
   - Replace older screenshots with text placeholders.
   - Include previous action summaries and tool results compactly.

Do not copy the whole `GeneralE2EAgentMCP` class into the host. The host needs a subtask queue; the GUI agent needs a stable action contract. Those are separate concerns.

## Validation

V0 validators are rule-based and intentionally simple:

- `system_action` open-app validator checks foreground bundle/app label where available;
- `gui_task` validator consumes the GUI tool's `success`, `summary`, `error`, `latest_screenshot_path`, and current app;
- `tool_call` validator checks non-empty result and absence of obvious tool error;
- `ask_user` validator treats the task as paused, not failed;
- safety/policy validator always wins over model confidence.

If a validator is uncertain, the executor should stop with `needs_verification` or escalate to S2 instead of pretending success.

## Error Handling

Plan-level failures:

- malformed JSON: fall back to current `RouteDecision` behavior;
- no subtasks: synthesize one subtask from the top-level route;
- low confidence: fall back or ask S2 depending on risk;
- unsupported route/tool: ask user or halt with an explicit unsupported-route message.

Subtask-level failures:

- tool failure: retry once when cheap and idempotent, then S2 replan;
- GUI blocked/stagnation: let `AutonomyMonitor` and GUI result surface the reason, then S2 replan or halt;
- policy violation: human confirmation/halt;
- missing app: mark subtask skipped only if the original instruction allows skipping missing apps; otherwise ask user;
- irreversible action: stop before the final submit/payment/send/delete step.

## Safety Boundaries

The PlanExecutor must never treat a 35B plan as authorization. It must re-check:

- original task policy before planning;
- every subtask policy before execution;
- high-risk GUI actions before final execution;
- account/privacy/financial/external-send actions before proceeding.

Examples that must not auto-execute:

- buying a train ticket;
- calling a taxi;
- booking Disney tickets;
- changing WeChat nickname;
- posting to Moments;
- reading credit/financial额度;
- cancelling favorites or changing profile text.

Those can be navigated only up to a safe confirmation point when explicitly allowed, and should otherwise stop for human confirmation.

## Testing

Unit tests:

- parse multi-subtask planner JSON into `PlannerPlan`;
- synthesize a one-subtask plan for legacy planner output;
- reject unsupported routes;
- preserve policy block before any subtask execution;
- execute tool/system/gui subtasks through mocked dispatchers in order;
- stop after a failed subtask;
- request S2 replan after GUI stagnation;
- skip missing app only when `allow_skip_missing_apps` is true.

Integration tests:

- "在B站播放罗翔的刑法课视频" plans into open/search/play subtasks;
- "查询深圳天气" stays tool-first and does not use GUI;
- "打开设置" uses system action;
- "买一张高铁票" stops before purchase/confirmation;
- legacy `RouteDecision` fallback still works when planner fails.

Manual smoke tests:

- start `scripts/nanobot-ios-start preflight`;
- run each task with a fresh `--session`;
- verify trace paths are created for GUI subtasks;
- verify missing apps are reported as skipped/blocked instead of causing infinite GUI loops.

## Rollout

Implement in three small commits:

1. Add planner plan/subtask models and parser tests.
2. Add `PlanExecutor` with mocked dispatcher tests and wire it behind `gui.plannerSubtasksEnabled`.
3. Enable the subtask executor in `nanobot-ios-start`, then run real iOS smoke tests.

Default rollout should be opt-in until real-device smoke tests pass. The existing single-route behavior remains as fallback.

