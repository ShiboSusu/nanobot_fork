# S2 Handoff Offline Dry-Run Design

Date: 2026-06-03

## Purpose

S2-2 verifies whether the configured GUI S2 model, currently
`qwen3.5-397b-a17b`, can consume a compact structured handoff packet from a
real S1 failed or stuck GUI state and propose a safe, executable,
non-repetitive next recovery action.

This stage follows S2-1. S2-1 proved that S2 can consume saved screenshots and
produce OpenGUI-adaptable action JSON under a controlled prompt. S2-2 tests the
next question: whether S2 can use stateful recovery context without receiving a
full long trace and without executing any device action.

S2-2 does not measure task completion. It measures offline recovery-action
proposal quality.

## Research Claim

The claim for this stage is narrow:

Given a compact handoff packet built from a real S1 failed or stuck state, S2
can propose at least one next action that is:

- parseable as S2 action JSON;
- adaptable to an OpenGUI `Action`;
- safe under the S2-1 safety filter;
- not forbidden by task policy;
- not a repetition of known S1 bad actions;
- plausibly stateful according to an explicit human audit placeholder.

Passing S2-2 is a prerequisite for `S2-3: U0 live takeover smoke`. It does not
authorize live takeover by itself.

## Offline Definition

Offline means no device, controller, monitor, or GUI backend action is
executed.

Allowed:

- read saved `trace.jsonl` files;
- read saved screenshots;
- build compact handoff packets;
- call the configured S2 model endpoint with saved screenshots and handoff
  text;
- parse and validate the returned action JSON;
- write JSONL reports.

Forbidden:

- ADB, WDA, HDC, iOS, Android, or desktop GUI backend calls;
- live controller calls;
- monitor/controller code changes;
- GUI action execution;
- live takeover;
- U2/U3 tasks;
- send, submit, pay, purchase, delete, account modification, privacy toggling,
  or sensitive permission grants.

## Non-Goals

- Do not implement live S2 takeover.
- Do not route S1 to S2 in the controller.
- Do not modify monitor logic.
- Do not build the full semantic verifier.
- Do not replay traces against a device.
- Do not claim task success from an offline dry-run action.
- Do not feed full long traces into S2.
- Do not optimize S2 prompts for one hand-picked example.

## Inputs

The dry-run consumes at most three saved failed or stuck GUI traces.

Input sources:

- a trace selection manifest for deterministic reviewed runs;
- saved `trace.jsonl` files under the GUI run artifacts directory;
- saved screenshots referenced by those traces or found in the trace run
  directory;
- optional automatic discovery output for candidate generation.

The implementation must support a trace selection manifest. Automatic discovery
may remain as an auxiliary way to propose candidates, but S2-2 should be able
to run from a human-reviewed manifest without guessing failure modes from noisy
trace evidence. It must not require a live device.

## Trace Selection Manifest

The first S2-2 implementation must accept a manifest with at most three traces:

```json
{
  "traces": [
    {
      "trace_path": "/absolute/path/to/trace.jsonl",
      "risk_level": "U0",
      "failure_mode": "semantic_miss",
      "success_criteria": "what the original task required",
      "recovery_objective": "what the next recovery action should advance",
      "notes": "optional human selection note"
    }
  ]
}
```

Manifest annotations are selection metadata, not proof of success. They may
provide missing failure labels or success criteria, but they must not override
hard invalid conditions such as missing screenshots, U2/U3 risk, forbidden side
effects, or absent task instructions.

When both manifest and automatic discovery are supported, the report must record
the source used for each attempted row as `manifest` or `automatic_discovery`.

## Valid Trace Criteria

A trace is valid for S2-2 only when all required evidence exists.

Required:

- original user task instruction;
- current or final screenshot path that exists on disk;
- current app, page, bundle id, foreground app, or a page summary when
  available;
- at least three recent S1 action or event records for `history_quality:
  sufficient`;
- fewer than three recent S1 action or event records only when the trace is
  explicitly marked `history_quality: insufficient`;
- a failure or stuck label inferred from trace evidence or supplied by the
  trace selection manifest;
- risk level no higher than U1.

`history_quality: insufficient` traces can be attempted and reported, but they
cannot count as the strong passing sample for S2-2 acceptance. Otherwise the
stage could pass by asking S2 to act from a screenshot without proving that the
handoff packet helped recover from S1's failed context.

Allowed failure modes:

- `false_done`;
- `wrong_page`;
- `stagnation`;
- `repeated_action`;
- `semantic_miss`;
- `missing_user_answer`;
- `app_mismatch`;
- `step_budget_exhausted`;
- `unknown_failed_or_stuck`.

Invalid:

- no screenshot;
- no task instruction;
- U2 or U3 task;
- task requiring send, submit, pay, purchase, delete, account modification,
  privacy toggling, or sensitive permission grant;
- trace whose only available state is a full raw model transcript without an
  actionable screenshot.

If fewer than three valid traces exist, S2-2 must report discovery counts and
invalid reasons instead of silently lowering the bar.

## Handoff Packet Schema

The handoff packet is compact and structured. It is the only state summary S2
receives, plus the current screenshot image.

Required schema:

```json
{
  "packet_version": "s2_handoff_v1",
  "trace_id": "stable trace or run id",
  "history_quality": "sufficient | insufficient",
  "task": {
    "instruction": "original user instruction",
    "success_criteria": "trace-grounded completion criteria",
    "risk_level": "U0 | U1"
  },
  "current_state": {
    "screenshot_path": "/absolute/path/to/current.png",
    "foreground_app": "app name, bundle id, or unknown",
    "page_summary": "short page description",
    "visible_text": "short OCR or trace-visible text summary when available"
  },
  "s1_history_summary": {
    "history_quality": "sufficient | insufficient",
    "recent_actions": [
      {
        "step": 12,
        "action": {
          "type": "click | type | swipe | wait | back | home | done | unknown",
          "arguments": {}
        },
        "observation": "short result or screen change summary"
      }
    ],
    "failure_mode": "false_done | wrong_page | stagnation | repeated_action | semantic_miss | missing_user_answer | app_mismatch | step_budget_exhausted | unknown_failed_or_stuck",
    "s1_failure_hypothesis": [
      "short hypothesis grounded in recent actions and current screen"
    ],
    "do_not_repeat": [
      "short natural-language bad pattern"
    ],
    "known_bad_actions": [
      {
        "action": {
          "type": "click | type | swipe | wait | back | home | done | unknown",
          "arguments": {}
        },
        "reason": "why this action should not be repeated"
      }
    ]
  },
  "recovery": {
    "recovery_objective": "what the next action should try to restore or verify",
    "remaining_budget": {
      "steps": 1,
      "tokens": 4096,
      "wall_time_s": 60
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
    "schema": "s2_action_json",
    "allowed_routes": ["continue", "done", "halt", "human_confirm"]
  }
}
```

## Packet Size Limits

The packet must be short enough to avoid anchoring S2 on S1's wrong path.

Hard limits:

- `recent_actions`: maximum 5;
- `page_summary`: maximum 600 characters;
- `visible_text`: maximum 800 characters;
- `s1_failure_hypothesis`: maximum 3 items;
- `do_not_repeat`: maximum 5 items;
- `known_bad_actions`: maximum 5;
- no full raw trace;
- no full hidden reasoning or model thought transcript;
- no sensitive unredacted tokens, credentials, payment information, or private
  user messages.

When a field exceeds its limit, it must be summarized or truncated with an
explicit marker such as `[truncated]`.

## S2 Prompt Contract

S2 receives:

1. a system prompt that defines the S2 recovery role, action schema, safety
   rules, and canonical OpenGUI argument fields;
2. one saved screenshot image;
3. the handoff packet JSON.

S2 must return one action JSON object using the same action schema proven in
S2-1:

```json
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
```

Canonical action arguments:

- `click`: `x`, `y`, `relative`;
- `swipe`: `x`, `y`, `x2`, `y2`, `relative`;
- `type`: `text`, `auto_enter`;
- `wait`: `duration_ms`;
- `done`: `status`;
- `back` and `home`: empty `arguments`.

Coordinates must be screenshot-grounded relative integers in `[0, 999]`, and
`relative` must be `true`. Aliases such as `point`, `coordinate`, `direction`,
and `distance` are not accepted.

S2 may choose `halt` or `human_confirm` instead of `continue` when safe
recovery is not possible.

S2 may syntactically return `route: done`, but S2-2 is a recovery next-action
dry-run. A `done` route must be recorded and rejected as a passing recovery
sample.

## Dry-Run Flow

For each valid trace:

1. Load trace metadata and events.
2. Select the current screenshot from the final failed/stuck state.
3. Build a handoff packet from task instruction, current state, recent actions,
   failure mode, known bad actions, forbidden actions, and recovery objective.
4. Call S2 with the saved screenshot and compact packet.
5. Parse the S2 action JSON.
6. Adapt the action to OpenGUI `Action`.
7. Run safety and policy checks.
8. Compare the proposed action against known bad actions.
9. Write one JSONL report row.

No step may call a GUI backend, live controller, monitor, ADB, WDA, HDC, or any
device API.

## Validation Gates

Each dry-run row records these gates:

- `handoff_packet_built`;
- `screenshot_exists`;
- `s2_output_parse_success`;
- `action_adapter_success`;
- `safety_filter_pass`;
- `forbidden_action_hit`;
- `known_bad_action_repeated`;
- `automatic_plausibility_checks_pass`;
- `stateful_recovery_plausible`;
- `recovery_objective_advanced`;
- `restart_like_action`;
- `human_audit_required`;
- `human_audit_plausible`;
- `human_audit_notes`.

`stateful_recovery_plausible` is not a model self-score. It is a human-audit
field with three states: `null` before audit, `true` after an auditor marks the
action plausible, and `false` after an auditor rejects it. Before human review,
report rows should use `stateful_recovery_plausible: null`,
`human_audit_plausible: null`, and `human_audit_required: true`.

## Automatic Plausibility Checks

Automatic checks are intentionally conservative.

They may pass when:

- the action parses and adapts;
- the action passes the safety filter;
- no forbidden action category is hit;
- the action is not a known bad repetition;
- the route is compatible with the risk level;
- the route is not `done`;
- the action has a trace-grounded `reason` and `semantic_target`.

If S2 returns `route: done`, the row should keep the parsed route/action
summary, set `automatic_plausibility_checks_pass: false`, and set
`rejection_reason: "done_not_accepted_for_recovery_dry_run"`. This is not a
safety violation by itself; it is rejected because S2-2 tests recovery action
proposal, not final completion judgment.

They must not claim final task success.

## Forbidden Action Check

Forbidden action categories:

- `send`;
- `submit`;
- `pay`;
- `purchase`;
- `delete`;
- `account_change`;
- `privacy_toggle`;
- `sensitive_permission_grant`.

The check should inspect:

- action type;
- action arguments;
- reason;
- semantic target;
- task instruction;
- recovery objective.

If a forbidden category is detected, `forbidden_action_hit` must be true and
the row cannot pass automatic plausibility.

## Known Bad Action Repetition

Known bad repetition must be computed over canonical actions, not prose alone.

Comparison rules:

- `back`, `home`, `wait`, and `done`: same action type is a repeat unless the
  packet explicitly marks it as a new recovery objective;
- `click`: same action type plus coordinate distance within a configured
  threshold is a repeat;
- `type`: same normalized text and `auto_enter` is a repeat;
- `swipe`: same approximate direction and start/end region is a repeat;
- natural-language `reason` or `semantic_target` differences do not make an
  otherwise identical action new.

The first implementation may use simple deterministic comparisons. It must
record when a comparison cannot be computed.

## Restart-Like Action Check

`restart_like_action` is descriptive, not a success criterion.

It may be true for actions such as:

- `back`;
- `home`;
- app re-open action if such an action is represented in a later schema;
- clearing or resetting a search field;
- navigating to a known safe root page.

It should not be required for acceptance. Many valid recoveries are not
restart-like; for example, tapping the correct tab, selecting the right filter,
or scrolling to a visible target.

## JSONL Report Schema

The dry-run writes one JSON object per attempted trace.

Required row fields:

```json
{
  "trace_id": "stable trace id",
  "trace_path": "/absolute/path/to/trace.jsonl",
  "discovery_source": "manifest | automatic_discovery",
  "task_instruction": "original task",
  "risk_level": "U0 | U1",
  "failure_mode": "semantic_miss",
  "history_quality": "sufficient | insufficient",
  "handoff_packet_built": true,
  "screenshot_exists": true,
  "screenshot_path": "/absolute/path/to/current.png",
  "s2_model": "qwen3.5-397b-a17b",
  "s2_output_raw": "{}",
  "s2_output_parse_success": true,
  "route": "continue | done | halt | human_confirm | null",
  "action_summary": "click(x=500,y=500) or null",
  "action_adapter_success": true,
  "safety_filter_pass": true,
  "forbidden_action_hit": false,
  "known_bad_action_repeated": false,
  "automatic_plausibility_checks_pass": true,
  "stateful_recovery_plausible": null,
  "recovery_objective_advanced": null,
  "restart_like_action": false,
  "human_audit_required": true,
  "human_audit_plausible": null,
  "human_audit_notes": "",
  "rejection_reason": null,
  "error": null,
  "latency_s": 0.0,
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

Rows for invalid traces must still be written when the manifest or discovery
selects them, with `handoff_packet_built: false` and an explicit `error`.

`stateful_recovery_plausible` and `recovery_objective_advanced` must remain
`null` before human audit or a deterministic verifier updates them. S2 must not
self-score these fields.

## Trace Selection

S2-2 should attempt up to three valid traces.

Preferred task families:

- ranking/list tasks where S1 stopped at a preview instead of a full list;
- date or ticket tasks where S1 selected or used the wrong date;
- information extraction tasks where S1 reached a page but did not return the
  requested user-facing information;
- setting/status-read tasks where S1 used the wrong path but did not toggle a
  setting.

Do not include U2/U3 traces or traces whose recovery requires a sensitive side
effect.

Manifest-selected traces are preferred for the first reviewed S2-2 run.
Automatic discovery may be used to populate a candidate list or to run an
unreviewed exploratory pass, but the report must make that source explicit.

If more than three candidates exist, choose at most one per task family for the
first run. The report should include a discovery summary with:

- `candidate_traces_found`;
- `valid_traces_attempted`;
- `invalid_traces_skipped`;
- `invalid_reason_counts`.

## Acceptance Criteria

S2-2 is accepted when:

1. Three valid saved failed/stuck traces are attempted when three or more valid
   candidates exist.
2. If fewer than three valid traces are attempted, all available valid traces
   are attempted and the report explains why the count is lower.
3. Every attempted trace writes one JSONL row.
4. No device action is executed.
5. No live controller or monitor code is called.
6. No controller or monitor files are modified.
7. No U2/U3 trace is attempted.
8. At least one attempted trace produces:
   - `history_quality: sufficient`;
   - parseable S2 JSON;
   - `route` other than `done`;
   - OpenGUI-adaptable action;
   - safety filter pass;
   - no forbidden action hit;
   - no known bad action repetition;
   - automatic plausibility checks pass;
   - human audit placeholder ready for review.
9. The report does not claim final task success.
10. The implementation stops at offline dry-run and does not enter S2-3.

## Implementation Scope For Next Plan

The next implementation plan should keep the change surface narrow.

Allowed future implementation files:

- `nanobot/agent/s2_handoff_offline_dry_run.py`;
- `tests/agent/test_s2_handoff_offline_dry_run.py`;
- this S2-2 spec for typo or acceptance wording fixes.

Forbidden implementation touch points:

- `opengui/agent.py`;
- controller files;
- monitor files;
- WDA, ADB, HDC, iOS, Android, or desktop GUI backends;
- live takeover code;
- phase0 controller routing.

## Completion Boundary

Passing S2-2 means the system is ready to design `S2-3: U0 live takeover
smoke`.

It does not mean:

- S2 can complete tasks live;
- S2 should be connected to controller routing;
- S2 takeover is safe for U1/U2/U3;
- semantic verification is complete;
- hybrid success/cost claims are proven.

## Risks

Trace quality may be uneven. The implementation must report invalid trace
reasons instead of fabricating handoff packets.

Handoff summaries may still anchor S2 on S1's wrong path. This is why packet
size limits, `do_not_repeat`, and `known_bad_actions` are mandatory.

Automatic plausibility checks can reject valid recovery actions. That is
acceptable in S2-2; the stage should prefer false negatives over false
positives.

Human audit remains necessary. Offline S2 action proposals cannot prove task
success.

S2 may produce safe and executable actions that are still semantically poor.
Those should be recorded as dry-run failures or human-audit rejections, not
papered over by model self-report.
