# S2 U0 Live Takeover Smoke Design

Date: 2026-06-04

## Purpose

S2-3 is the first stage that allows S2 to execute real GUI actions. It is a
live smoke test, not a pilot and not production controller integration.

The goal is narrow:

Given a real current GUI state for one U0 task, S2 should consume the same
compact handoff packet proven in S2-2, produce parseable GUI action JSON, pass
the safety and repetition gates, execute at most three actions through the
existing GUI backend, and stop with verified success or a clean failure.

S2-3 is allowed only because:

- S2-1 passed the saved-screenshot visual/action capability smoke;
- S2-2 passed the offline handoff recovery dry-run;
- human audit accepted at least one sufficient-history recovery action;
- the first accepted recovery family is a Ctrip date/calendar recovery, not a
  setting modification, payment, purchase, send, or account task.

S2-3 does not prove that hybrid takeover improves success rate. It only proves
that a bounded S2 takeover loop can operate safely on one live U0 task.

## Recommended Approach

Use a manual-gated live takeover smoke.

This means the operator or smoke harness deliberately starts S2 takeover from a
known U0 current screen state. The production controller does not automatically
route to S2 in this stage.

Rejected for S2-3:

- full production controller integration;
- automatic monitor-triggered takeover for arbitrary GUI tasks;
- U0/U1 pilot evaluation;
- information-query final-answer tasks such as 12306;
- setting/status-read tasks such as Bilibili privacy settings;
- any task requiring send, submit, purchase, payment, deletion, account change,
  privacy toggle, or sensitive permission grant.

Rationale:

The first live smoke must isolate whether S2 can act safely from the current
screen. If it also introduces automatic controller routing, broad task
selection, and semantic success evaluation, failures become hard to attribute.

## First Smoke Task Family

The first live task family should be:

`U0 date-constrained travel search recovery`

Preferred concrete family:

- Ctrip flight date/calendar recovery;
- target route: Shanghai to Guangzhou;
- target date: 2026-06-05;
- stop at visible flight search results;
- never enter booking, order, passenger, payment, coupon purchase, login
  modification, or checkout flows.

This family is chosen because S2-2 produced a human-accepted recovery action
from a Ctrip calendar state showing 2027 May/June while the target was 2026-06-05.

Do not use 12306 as the first live smoke. S2-2 showed that S2 may return
`route=done` when ticket rows are visible, but the user-facing answer extraction
path is not yet implemented. That is a separate information-query terminal
protocol problem.

Do not use Bilibili privacy/status-read as the first live smoke. S2-2 showed a
known-bad repeated tap on the video page, and setting/status-read tasks require
a stricter no-toggle verifier.

## Live Definition

Allowed in S2-3:

- observe the current device screen through the existing GUI backend;
- read the current run trace and screenshots;
- build a compact handoff packet;
- call the configured S2 endpoint with the current screenshot and handoff
  packet;
- parse, adapt, safety-check, and repetition-check the S2 action;
- execute at most three safe S2 actions through the existing GUI backend;
- write trace and JSONL report artifacts;
- stop on verified success, clean halt, safety block, human-confirm route,
  repeated action, no-progress, or budget exhaustion.

Forbidden in S2-3:

- U1/U2/U3 tasks;
- production controller routing changes;
- broad monitor logic changes;
- repeated retry loops;
- live takeover for more than one task in the same smoke run;
- S2 handoff back to S1 after takeover starts;
- send, submit, pay, purchase, delete, account modification, privacy toggle,
  sensitive permission grant, or any action that enters those flows;
- model-reported success as the sole success criterion.

## Preflight Baseline Gate

S2-3 must not start unless all baseline checks pass immediately before the live
smoke.

Required checks:

1. Working directory status is reported:
   - `git status -sb`
   - `git diff --name-only`
   - staged files are empty or limited to the S2-3 implementation files.
2. Device bridge is reachable:
   - WDA/iOS status is reachable for iOS tasks;
   - no ADB/HDC/Android path is used for the iOS Ctrip smoke.
3. A fresh screenshot can be captured and saved.
4. A benign input event can be executed only if the smoke operator explicitly
   confirms it is part of the selected U0 setup path.
5. S1 endpoint is reachable if S1 is used to create the current state.
6. S2 endpoint is reachable and reports the expected model:
   `qwen3.5-397b-a17b`.
7. Existing S2 action parser, adapter, and safety filter are importable.
8. The selected task is classified as U0 and does not require sensitive side
   effects.
9. A run directory is allocated before any live action is executed.

If any preflight check fails, S2-3 must stop with `preflight_failed` and no S2
action execution.

## Takeover Start Modes

S2-3 supports one start mode initially:

`MANUAL_TAKEOVER_FROM_CURRENT_STATE`

The operator prepares or reaches a current Ctrip calendar/search state, then
starts the S2 takeover smoke from that screen. The harness observes the current
screen, builds the handoff packet, and starts S2. This mode tests live S2
execution without adding production controller routing.

Future modes are out of scope for the first smoke:

- `S1_TO_MANUAL_TAKEOVER`: S1 runs first, then the operator pauses and starts
  S2 from the observed state.
- `S1_TO_CONTROLLER_TAKEOVER`: monitor/controller routes to S2 automatically.

The first mode is intentionally conservative. It still exercises live
screenshot observation, S2 visual reasoning, parser/adapter/safety gates,
backend execution, trace writing, and post-action observation.

## Handoff Packet

S2-3 reuses the S2-2 packet shape and adds live-run metadata.

Required fields:

```json
{
  "packet_version": "s2_live_handoff_v1",
  "live_smoke_id": "stable run id",
  "takeover_start_mode": "MANUAL_TAKEOVER_FROM_CURRENT_STATE",
  "task": {
    "instruction": "original U0 instruction",
    "task_family": "date_travel_search",
    "success_criteria": "route/date/result-list verifier criteria",
    "risk_level": "U0"
  },
  "current_state": {
    "screenshot_path": "/absolute/path/to/current.png",
    "foreground_app": "ctrip.com or app bundle",
    "page_summary": "short state summary",
    "visible_text": "short OCR or trace-visible text summary when available"
  },
  "s1_history_summary": {
    "recent_actions": [],
    "failure_mode": "wrong_page | date_mismatch | stagnation | manual_seeded_state",
    "s1_failure_hypothesis": [
      "why this current state needs recovery"
    ],
    "do_not_repeat": [
      "known bad scroll direction or repeated tap pattern"
    ],
    "known_bad_actions": []
  },
  "controller": {
    "route": "S2_TAKEOVER",
    "takeover_reason": "manual_live_smoke",
    "remaining_budget": {
      "steps": 3,
      "tokens": 20000,
      "wall_time_s": 180
    },
    "forbidden_actions": [
      "send",
      "submit",
      "pay",
      "purchase",
      "delete",
      "account_change",
      "privacy_toggle",
      "sensitive_permission_grant"
    ]
  },
  "required_output": {
    "schema": "s2_live_action_json",
    "allowed_routes": [
      "continue",
      "done",
      "halt",
      "human_confirm"
    ]
  }
}
```

Do not pass full S1 traces to S2. Use at most the most recent five actions and a
compact failure summary. This preserves the S2-2 anti-anchoring rule.

## S2 Output Schema

S2 must return one JSON object per live step:

```json
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
```

For the first Ctrip date/search smoke, `final_answer.required` should be false.

For later information-query tasks, `final_answer.required` must be true and the
controller must reject `route=done` if the answer text does not contain the
requested user-facing information. This is included now because S2-2 showed
that 12306 can reach ticket rows while still failing to return the information
to the user. However, information-query tasks remain out of scope for the first
S2-3 live smoke.

`route=done` is a candidate terminal state, not verified success. The controller
must still run the verifier before reporting success.

## Step Loop

At each S2 takeover step:

1. Observe the current screen and save a screenshot.
2. Build or update the live handoff packet.
3. Call S2 with the current screenshot and compact packet.
4. Parse the S2 JSON.
5. Adapt the action to OpenGUI `Action`.
6. Run the safety filter.
7. Run the forbidden action check.
8. Run the known-bad/repeated action check.
9. If route is `halt` or `human_confirm`, stop cleanly without execution.
10. If route is `done`, run the task-family verifier.
11. If route is `continue`, execute the action.
12. Observe the next screen.
13. Record trace fields and update progress evidence.
14. Stop if verified success, budget exhaustion, no-progress, repeated action,
    unsafe action, app mismatch that cannot be safely recovered, or execution
    error occurs.

S2 must not hand the task back to S1 during the smoke.

## Budgets

Hard limits for the first smoke:

- `max_s2_steps = 3`;
- `max_s2_calls = 3`;
- `max_s2_total_tokens = 20000`;
- `max_s2_wall_time_s = 180`;
- `max_no_progress_steps = 1`;
- `max_repeated_action = 0` for known S1 bad actions;
- `max_same_s2_action_repeat = 0`;
- one task per run.

Budget exhaustion returns failure:

```json
{
  "verified_success": false,
  "failure_reason": "s2_budget_exhausted"
}
```

It must not be reported as partial success.

## Safety Rules

Safety policy sits above both models.

The following categories are hard blocks in S2-3:

- send or publish;
- submit forms with external effect;
- pay, purchase, book, order, recharge, checkout;
- delete, clear, uninstall, cancel irreversible state;
- account, password, phone, real-name, login-device, payment-method changes;
- privacy toggles or setting modifications;
- sensitive permission grants;
- ambiguous confirm/continue/agree dialogs with unknown consequences.

Allowed low-risk actions in the Ctrip smoke:

- back or close date picker;
- tap date/calendar/search UI;
- swipe calendar;
- wait for loading;
- navigate within flight search results;
- stop on result list.

If S2 proposes a hard-blocked action, the smoke stops with
`safety_blocked_action` and no execution of that action.

## Verifier For First Smoke

The first verifier is task-family-specific and rule-first.

For Ctrip date/search smoke, verified success requires evidence that:

- foreground app is Ctrip or the expected Ctrip web/app context;
- route is Shanghai to Guangzhou, or the task-specific equivalent;
- target date is visible as `06-05`, `6月5日`, or `2026-06-05`;
- a flight result list or flight card is visible;
- the current page is not an order, passenger, payment, checkout, coupon
  purchase, or login-modification page.

If `route=done` occurs before these conditions are met, record:

```json
{
  "model_reported_success": true,
  "controller_verified_success": false,
  "failure_reason": "false_done"
}
```

If a page clearly shows visible flight result cards but the route/date cannot be
verified from the screen, the smoke should stop with `verifier_unknown`, not
success.

## Information-Query Terminal Protocol

S2-3 includes the output field for final answers but does not test it live in
the first smoke.

For later ticket/list/information-query tasks, success requires:

- `final_answer.required = true`;
- `final_answer.text` contains the requested user-facing information;
- `final_answer.evidence` cites visible screen fields or trace-visible text;
- controller/verifier confirms required slots are present.

Examples:

- 12306: train number, departure/arrival time, origin/destination, visible
  seat/price/availability when present.
- Ranking/list: rank number and complete item title.
- Setting/status-read: setting label and observed state, with no toggle.

Until this protocol is implemented and tested, information-query tasks must not
be used as the first live takeover smoke.

## Trace Schema

Every S2-3 run must write a trace row for takeover start, each S2 step, and
takeover end.

Required takeover-start fields:

```json
{
  "event": "s2_takeover_start",
  "phase": "s2_takeover",
  "live_smoke_id": "...",
  "task_family": "date_travel_search",
  "takeover_start_mode": "MANUAL_TAKEOVER_FROM_CURRENT_STATE",
  "takeover_reason": "manual_live_smoke",
  "s1_steps_before_takeover": 0,
  "screenshot_path": "...",
  "foreground_app": "...",
  "budget": {
    "max_s2_steps": 3,
    "max_s2_total_tokens": 20000,
    "max_s2_wall_time_s": 180
  }
}
```

Required per-step fields:

```json
{
  "event": "s2_takeover_step",
  "phase": "s2_takeover",
  "s2_step_index": 1,
  "actor_model": "qwen3.5-397b-a17b",
  "screenshot_before": "...",
  "s2_output_raw": "{}",
  "route": "continue | done | halt | human_confirm",
  "action": {},
  "action_adapter_success": true,
  "safety_filter_pass": true,
  "forbidden_action_hit": false,
  "known_bad_action_repeated": false,
  "executed": true,
  "screenshot_after": "...",
  "model_latency_s": 0.0,
  "step_latency_s": 0.0,
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  },
  "budget_remaining": {}
}
```

Required takeover-end fields:

```json
{
  "event": "s2_takeover_end",
  "phase": "s2_takeover",
  "model_reported_success": false,
  "controller_verified_success": false,
  "user_observed_success": null,
  "failure_reason": "verified_success | false_done | safety_blocked_action | s2_budget_exhausted | no_progress | repeated_action | verifier_unknown | execution_error",
  "final_answer": null,
  "trace_path": "..."
}
```

## Report Schema

The smoke report should include one summary JSON object:

```json
{
  "task": "selected U0 instruction",
  "task_family": "date_travel_search",
  "result": "verified_success | failed | halted",
  "s2_takeover_started": true,
  "s2_called": true,
  "s2_steps": 0,
  "total_steps": 0,
  "tokens": {
    "s1": 0,
    "s2": 0,
    "total": 0
  },
  "latency": {
    "wall_clock_s": 0.0,
    "s2_model_latency_s": 0.0
  },
  "cross_app": false,
  "failure_reason": null,
  "takeover_reason": "manual_live_smoke",
  "trigger_evidence": [],
  "verified_success": false,
  "model_reported_success": false,
  "final_answer": null,
  "trace_path": "..."
}
```

This report intentionally supports the same statistics table used in earlier
manual task tracking: result, S2 usage, tokens, latency, model latency,
cross-app status, total steps, failure reason, takeover trigger, and trace path.

## Acceptance Criteria

S2-3 spec is ready for implementation when:

1. S2-1 capability smoke is available and passing.
2. S2-2 offline handoff report has at least one human-accepted
   sufficient-history recovery action.
3. The first live smoke is limited to one U0 Ctrip date/search recovery task.
4. `max_s2_steps = 3` is a hard limit.
5. Production controller routing changes are explicitly out of scope.
6. Safety blocks are defined above S2.
7. `route=done` requires verifier success.
8. Information-query final-answer protocol is defined but excluded from the
   first live smoke.
9. Trace and report schemas are sufficient to audit each S2 action.
10. No S2-3 implementation starts until this spec is reviewed and approved.

The S2-3 live smoke itself is accepted only when:

1. All preflight checks pass.
2. Exactly one U0 task is attempted.
3. S2 takeover starts from a real current screen.
4. At most three S2 actions are executed.
5. Every S2 output parses and adapts before execution.
6. No unsafe or forbidden action is executed.
7. Trace rows include takeover start, per-step, and takeover end events.
8. The final result is one of:
   - controller-verified success;
   - clean halt with a concrete failure reason.
9. The report does not rely on model-reported success alone.
10. No U1/U2/U3 task is attempted.

## Implementation Scope For Next Plan

Allowed future implementation files:

- `nanobot/agent/s2_live_takeover_smoke.py`;
- `tests/agent/test_s2_live_takeover_smoke.py`;
- small additions to `nanobot/agent/s2_handoff_offline_dry_run.py` only if a
  helper must be shared instead of duplicated;
- this S2-3 spec for review fixes.

Forbidden implementation touch points for the first S2-3 smoke:

- production controller routing files;
- broad monitor logic;
- WDA, ADB, HDC, iOS, Android, or desktop backend implementations;
- `opengui/agent.py`, unless a later reviewed plan proves that a minimal public
  hook is unavoidable;
- U0/U1 pilot harness;
- task-family semantic verifier framework beyond the Ctrip smoke verifier;
- Bilibili setting/status-read live takeover;
- 12306 or other information-query live takeover.

## Completion Boundary

Passing S2-3 means:

- S2 can execute a bounded live recovery loop on one U0 task family;
- trace and report artifacts can audit what S2 did;
- the system is ready to design a controller-triggered U0 smoke or a small U0
  pilot.

Passing S2-3 does not mean:

- production hybrid takeover is enabled;
- monitor routing is tuned;
- S2 is safe for U1/U2/U3;
- information-query final-answer tasks are solved;
- Bilibili setting/status-read tasks are safe;
- the FastSlow success-cost claim is proven.

## Risks

The live device may fail before the model is tested. This is why the preflight
gate must separate device failure from S2 failure.

The selected Ctrip state may already be too close to success or too polluted to
recover within three steps. The report should preserve that outcome instead of
changing the task mid-run.

S2 may return `done` early. That is a false-done candidate unless the verifier
passes.

S2 may propose a safe-looking navigation action that repeats S1's failed
pattern. The known-bad action gate should prefer false negatives over unsafe or
unproductive repetition.

The first smoke may pass due to a simple action and still fail to generalize.
That is acceptable. Generalization is measured only in later U0/U1 pilot stages.
