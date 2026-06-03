# S2 Takeover Design

Date: 2026-06-03

## Purpose

This document defines a practical S2 takeover design for the GUI agent stack.
The goal is not to replace the current S1 GUI agent with a larger model by
default. The goal is to build a controlled hybrid policy that can rescue tasks
when S1 has become unreliable, while keeping cost, latency, safety, and
evaluation measurable.

The current system is:

1. The user sends a task.
2. The router chooses `gui_task` when the task depends on device/app state.
3. OpenGUI runs S1 as the GUI actor.
4. The monitor observes cheap execution signals such as repeated actions,
   step-budget pressure, app mismatch, and stagnation.
5. When risk rises, S2 returns a recovery hint.
6. The hint is injected into the next S1 prompt.
7. S1 continues, or the attempt fails and retry starts another S1 run.

This is useful as a baseline, but it is not takeover. S2 currently has no
execution authority. If S1 has already shown that it cannot handle the current
GUI state, another hint often just produces a slower version of the same
failure pattern.

## Research Claim

The system should test a specific claim:

`HYBRID_TAKEOVER` should achieve a higher verified success rate than
`S1_ONLY`, with lower cost and latency than `S2_ONLY_FROM_START`, on tasks
where S1 is usually sufficient but sometimes fails in recoverable GUI states.

This claim is not automatically true. If S1 leaves the device in a polluted
state, then S2 takeover from that state may be harder than S2 starting from a
clean initial state. Let:

- `p9` be the success rate of S1 from a clean initial state.
- `p397` be the success rate of S2 from a clean initial state.
- `q` be the success rate of S2 from the state left after S1 fails.

Hybrid success is approximately:

`p_hybrid = p9 + (1 - p9) * q`

The central engineering problem is therefore to make `q` high enough and cheap
enough. That requires more than model switching. It requires a different
control policy, a clean handoff, bounded execution, and verified success.

## Non-Goals

- Do not use S2 for every GUI task by default.
- Do not make entropy or disagreement the only routing signal.
- Do not allow S2 to override safety policy.
- Do not treat model-reported `done` as verified success.
- Do not implement live S2 takeover before proving that the S2 endpoint can
  act as a GUI executor.

## Execution Modes

The evaluation harness should support these modes:

`S1_ONLY`: S1 runs the task from the initial state. S2 is disabled.

`S2_ONLY_FROM_START`: S2 runs the task from the initial state. This measures
the larger model's clean-state upper reference.

`HYBRID_GUIDE_ONLY`: S1 runs first. S2 may provide hints, but cannot execute
actions. This is the current hint baseline.

`S2_FROM_S1_FAILED_STATE`: S1 first creates a failure state. S2 then starts
from that saved state without the full hybrid controller. This isolates S2's
ability to recover from polluted GUI states.

`HYBRID_TAKEOVER`: S1 runs first. The controller routes to S2 takeover when
monitor evidence and semantic verification indicate that S1 is unlikely to
recover cheaply. Once S2 takes over, S2 owns the task until verified success,
safe halt, human confirmation, or budget exhaustion.

## Architecture

The design separates four responsibilities.

### Monitor

The monitor produces typed evidence. It should not directly decide that a task
is complete, and it should not rely on a single signal.

Initial evidence classes:

- `repeated_action`: the same action type or target repeats without progress.
- `stagnation`: screenshots or page summaries do not meaningfully change.
- `app_mismatch`: foreground app differs from the expected business app.
- `step_budget_pressure`: remaining S1 budget is low.
- `unsafe_action_candidate`: action may send, submit, pay, delete, authorize,
  modify account state, or modify privacy settings.
- `completion_quality_risk`: S1 is about to call `done` without task evidence.
- `constraint_violation`: a semantic verifier detected a broken task
  constraint.
- `uncertainty`: action disagreement, coordinate variance, or low confidence.

Entropy and action disagreement belong under `uncertainty`. They are evidence
of instability, not a sufficient intervention rule. Low entropy can be stable
wrong behavior. High entropy can be multiple valid actions.

### Semantic Verifier

The semantic verifier checks task-specific constraints using the task, current
screen, recent history, and, when available, OCR/accessibility summaries.

Initial verifier templates:

- Date tasks: verify that the target date is computed correctly and visible on
  the result page or selected in the date picker.
- Ranking/list tasks: verify that the agent entered a full list/ranking page or
  saw explicit rank numbers and full item text.
- Ticket/route tasks: verify origin, destination, date, and visible ticket or
  flight information; never proceed to purchase.
- Setting/status-read tasks: verify the correct setting page and observed
  target state; never toggle the setting.
- Information extraction tasks: verify that the final answer contains the
  requested information, not merely that a page was opened.

The verifier should produce:

```json
{
  "verdict": "pass | fail | unknown",
  "reason_class": "date_mismatch | list_preview_only | missing_info | setting_not_read | route_mismatch | unsafe | unknown",
  "evidence": "short trace-grounded explanation"
}
```

### Controller

The controller converts monitor evidence and verifier output into one route:

- `S1_CONTINUE`
- `S2_VERIFY`
- `S2_GUIDE`
- `S2_TAKEOVER`
- `HUMAN_CONFIRM`
- `HALT`
- `BLOCK`

Initial route rules:

- If an action candidate is payment, purchase, destructive, external send,
  account modification, privacy modification, or sensitive permission grant:
  route to `HUMAN_CONFIRM`, `HALT`, or `BLOCK`.
- If S1 calls `done` and the semantic verifier does not pass: route to
  `S2_TAKEOVER` or `HALT`, depending on safety.
- If `S2_GUIDE` has already been used in the current attempt and there is no
  semantic progress afterward: route to `S2_TAKEOVER` or `HALT`.
- If repeated actions or stagnation exceed the configured threshold: route to
  `S2_TAKEOVER`.
- If the foreground app drifts and S1 does not recover within a small budget:
  route to `S2_TAKEOVER`.
- If uncertainty is high but the task is low-risk and no semantic constraint is
  broken: route to `S2_VERIFY` or one cautious S1 continuation.
- If the page is progressing and no safety or semantic issue is detected: route
  to `S1_CONTINUE`.

`S2_GUIDE` is allowed at most once per attempt in the first version. Repeated
hint loops are not allowed as the recovery strategy.

### S2 Takeover Executor

S2 takeover is an execution phase, not a prompt note. When the route is
`S2_TAKEOVER`, S2 receives a structured handoff packet and directly produces
GUI actions until the task ends or a budget/safety boundary is reached.

S2 takeover must use a different control policy from S1:

- reconstruct the task goal and success criteria before acting;
- identify what S1 has already completed and what it likely broke;
- prefer verification before action on date, ranking, ticket, and setting
  tasks;
- output one parseable GUI action per step;
- explain which constraint the action is trying to restore or satisfy;
- never execute sensitive side-effect actions without controller approval;
- complete the user-facing answer when the task asks for information.

Once takeover starts, the task is not handed back to S1. S2 is responsible for
verified completion, safe stop, human confirmation, or explicit failure.

## S2 Capability Gate

Before live takeover, the system must verify S2 capability.

Required checks:

1. Endpoint health and model availability.
2. Text-only JSON verdict stability.
3. Screenshot plus task to JSON verdict.
4. Screenshot plus task to next GUI action.
5. Schema parse success over repeated calls.
6. Latency measurement.
7. Confirmation that screenshot input is actually supported.

If S2 cannot consume screenshots, it cannot be treated as a GUI takeover actor.
It may still serve as a text-only verifier/planner using OCR or screen
summaries, but the live actor role should remain disabled.

## Handoff Packet

S2 should not receive the entire S1 trace by default. Full traces can contain
wrong paths that anchor the larger model. The handoff should summarize only the
state needed for recovery.

Packet schema:

```json
{
  "task": {
    "instruction": "...",
    "risk_level": "U0 | U1 | U2 | U3",
    "success_criteria": "..."
  },
  "current_state": {
    "foreground_app": "...",
    "screenshot_path": "...",
    "screen_summary": "...",
    "visible_text": "..."
  },
  "s1_history_summary": {
    "recent_actions": [
      {
        "step": 12,
        "action": "scroll down",
        "result": "no semantic progress"
      }
    ],
    "failure_pattern": "..."
  },
  "monitor_evidence": {
    "signals": ["repeated_action", "step_budget_pressure"],
    "semantic_verifier": {
      "verdict": "fail",
      "reason_class": "date_mismatch",
      "evidence": "..."
    }
  },
  "controller": {
    "route": "S2_TAKEOVER",
    "takeover_reason": "...",
    "remaining_budget": {
      "steps": 8,
      "tokens": 20000,
      "wall_time_s": 180
    },
    "forbidden_actions": [
      "payment",
      "purchase",
      "send",
      "submit",
      "delete",
      "account_change",
      "privacy_toggle",
      "sensitive_permission_grant"
    ]
  },
  "required_output": {
    "schema": "gui_action_json",
    "allowed_routes": ["continue", "done", "halt", "human_confirm"]
  }
}
```

## Takeover Budgets

S2 takeover must be bounded.

Initial defaults:

- `max_s2_takeover_steps`: 8 for normal U0/U1 tasks.
- `max_s2_takeover_tokens`: configured per deployment.
- `max_s2_wall_time_s`: configured per deployment.
- `max_repeated_action`: 1 repeated action unless the screen changed.
- `max_no_progress_steps`: 2.

Budget exhaustion returns a verified failure with a reason such as
`s2_budget_exhausted`, not model-reported success.

## Trace Requirements

Every step should identify the actor and phase.

Minimum trace fields:

```json
{
  "phase": "s1 | s2_takeover",
  "actor_model": "qwen3-vl-9b | qwen3.5-397b-a17b",
  "controller_route": "S1_CONTINUE | S2_TAKEOVER | ...",
  "takeover_reason": "...",
  "monitor_evidence": {},
  "semantic_verifier": {},
  "model_reported_success": false,
  "controller_verified_success": false,
  "failure_reason": null,
  "s2_step_index": 0,
  "s2_budget_remaining": {}
}
```

Task reports should distinguish:

- model-reported success;
- controller-verified success;
- user-observed success, when manually audited;
- false done;
- unsafe block;
- takeover salvage.

## Evaluation Metrics

Primary metrics:

- `verified_success_rate`
- `cost_normalized_success`
- `takeover_salvage_rate`
- `false_done_rate`

Secondary metrics:

- total tokens;
- S1 tokens;
- S2 tokens;
- wall-clock latency;
- model latency;
- total steps;
- S2 takeover steps;
- S2 call count;
- takeover rate;
- unsafe block count;
- human confirmation count.

`takeover_salvage_rate` is defined as the fraction of tasks where S1 triggered
a takeover condition and S2 takeover achieved verified success.

## Implementation Stages

Stage 0: Baseline hygiene

- Verify device connection, screenshot, input event, model endpoint, parser, and
  trace writing before running takeover experiments.

Stage 1: S2 capability smoke

- Use saved screenshots and tasks.
- Do not operate the device.
- Confirm S2 can emit parseable action JSON.

Stage 2: Handoff packet and offline dry run

- Build handoff packets from failed traces.
- Ask S2 for next actions offline.
- Human-audit whether actions are plausible recovery steps.

Stage 3: U0 live takeover smoke

- Use one low-risk task.
- Limit S2 to 3 steps.
- Require phase trace and controller route trace.
- Stop cleanly on unsafe or budget failure.

Stage 4: U0/U1 pilot

- Compare `S1_ONLY`, `S2_ONLY_FROM_START`, `HYBRID_GUIDE_ONLY`,
  `S2_FROM_S1_FAILED_STATE`, and `HYBRID_TAKEOVER`.
- Use verified success, not model-reported success.

Stage 5: Broader pilot

- Run 20-30 tasks covering ranking/list, date, tickets/routes, setting/status
  read, information extraction, and simple cross-app tasks.

Stage 6: Research-grade evaluation

- Expand beyond the pilot set.
- Report ablations and confidence intervals.

## Initial Task Families

Use a small debug set before the 20-30 task pilot:

- 2 setting/status-read tasks.
- 2 ranking/list or information-extraction tasks.
- 1 date-constrained search task.
- 1 simple cross-app task.

Do not include payment, purchase, external send, account modification, or
privacy toggling in the initial live takeover smoke.

## Risks

S2 may not support visual input. In that case, takeover must remain disabled
until a screen-summary path is proven reliable.

S1 may pollute the state so badly that takeover is more expensive than starting
over. This is why `S2_FROM_S1_FAILED_STATE` must be measured separately.

Controller rules may over-trigger takeover. The pilot must measure takeover
rate and cost-normalized success, not only raw success.

Verifier rules may be too narrow. The first version should cover only task
families observed in the pilot, then expand based on failures.

Long takeover can create expensive loops. Hard budgets and progress checks are
mandatory.

Safety policy must sit above both S1 and S2. A stronger model must not get
permission to pay, submit, delete, send, authorize, or modify privacy/account
settings without explicit user confirmation.

## Acceptance Criteria For The First Milestone

The first milestone is complete when:

1. S2 capability smoke reports whether the 397B endpoint can act as a visual
   GUI executor.
2. Failed traces can be converted into handoff packets.
3. S2 can produce parseable offline recovery actions for at least three saved
   traces.
4. The trace schema can represent `phase=s2_takeover`, controller route,
   takeover reason, budgets, and verified success fields.
5. No live S2 action is executed before passing the capability and offline dry
   run gates.
