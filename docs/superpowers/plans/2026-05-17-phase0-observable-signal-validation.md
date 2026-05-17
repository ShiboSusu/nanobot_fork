# Phase 0 Observable Signal Validation Plan

## Background

Phase 0 is no longer centered on provider-native `reasoning_tokens`.

Under the current Qwen3-VL-8B-Instruct + vLLM serving setup, provider-native `reasoning_tokens` is not available as a usable runtime signal. Signal 1 backend feasibility result is therefore **NO-GO** for the current S1 backend.

Current normal GUI execution is restored with `qwen3-vl-8b`, provider `custom`, API base `http://127.0.0.1:8001/v1`, and `enable_thinking=false`. The normal vLLM service is running without `--reasoning-parser qwen3`, and the chat smoke test returned non-null `content`.

## Signal 1 backend feasibility result: NO-GO

Signal 1 cannot depend on provider-native `reasoning_tokens` for this backend. This does not imply that Qwen3-VL cannot reason, and it does not imply that reasoning-token-style signals are useless in every serving setup. It only means this backend cannot expose that signal in a usable way for the Phase 0 pilot.

## New Phase 0 objective

Phase 0 now validates observable runtime signals for possible step-level Fast/Slow routing.

This is pilot evidence only. Do not make paper-level claims from Phase 0. Qwen3-VL-8B remains S1. S2 remains a future slow planner / verifier / reflector. There is no Controller implementation in Phase 0.

## Observable runtime signal taxonomy

Signals must be analyzed by layer first, then in combined trigger policies.

A. Model self-report signals:

- `confidence`
- `need_slow_planner`
- `uncertainty_reason`

B. Static/task-level risk signals:

- `task_risk_level`

C. Rule-based step risk signals:

- `step_predicted_risk_level`
- `rule_based_step_risk_level`
- `action_type_risk`
- `app_sensitive_action`

D. Environment/execution anomaly signals:

- `repeated_action`
- `stagnation_count`
- `foreground_app_mismatch`
- `post_action_no_observable_change`
- `action_parse_failure`
- `execution_error`

`confidence` is self-reported belief, not a trusted calibrated probability. `task_risk_level`, `step_predicted_risk_level`, and `rule_based_step_risk_level` must not be mixed. They represent different layers and should be evaluated separately before any combined policy analysis.

## Prompt/parser constraints from Task 3A

The current `agentProfile` is `default`, which uses native tool calling. In this profile, `parse_action()` receives the tool-call arguments dict; it does not directly parse assistant text for action.

Non-default profiles such as `qwen3vl` may parse `<tool_call>` from assistant text. Runtime signal design must therefore be profile-aware. Do not assume that placing `<runtime_signal>` before `<tool_call>` is always safe.

Future Task 3B must compare:

- A: `runtime_signal` before `<tool_call>`
- B: `runtime_signal` after `<tool_call>`
- C: side-channel / `model_snapshot` trace recording that does not enter action parser input

## Trace architecture and outer-inner trace alignment

Old wrong assumption:

`outer trace_*.jsonl model_output contains complete raw model output`

Correct design:

Outer `trace_*.jsonl` preserves:

- `token_usage`
- `duration`
- `chat_latency`
- `ttft`
- `action`
- `observation`
- action summary

Inner `trace.jsonl` preserves:

- `raw_content`
- `assistant_message`
- `tool_calls`
- `parsed_action`
- richer model snapshot fields

Task 4 must align outer and inner traces by `step_index`, execution order, and session path. Task 4 must not rely only on outer trace `model_output` to parse runtime signals.

## Env/config loading requirements

Future pilot scripts must resolve environment-backed config values with:

```python
cfg = resolve_config_env_vars(load_config(path))
```

They must not use `load_config()` alone. `direct_eval.py` previously did not resolve env vars, so future scripts should not blindly copy that pattern.

## U2 safety policy

U2 tasks are skipped by default. Runner-layer enforcement is required, and prompt-only safety is insufficient. Real U2 execution requires explicit flags and constraints.

Minimum future Task 4 controls:

- `--allow-u2`
- `--dry-run-u2`
- `--allowed-contact`
- `--message-prefix`
- `--allow-task-id`

Default behavior is to skip U2 unless explicitly enabled.

## Step-level trigger features vs outcome proxies

Avoid circular reasoning. Trigger features and outcome proxies must be kept separate:

```json
{
  "trigger_features": {
    "self_report": {},
    "risk": {},
    "execution_state": {}
  },
  "outcome_proxies": {
    "execution_error": false,
    "action_parse_failure": false,
    "post_action_no_observable_change": false,
    "judge_derived_not_advanced": null
  }
}
```

`repeated_action`, `stagnation_count`, and `foreground_app_mismatch` can be trigger features. Do not use the same field as both trigger and outcome label when claiming predictive value. `post_action_no_observable_change` is a rule-based proxy. `judge_derived_not_advanced` is judge-derived and must be labeled separately.

## Pilot data collection design

Future Task 4 should:

- Read `eval/datasets/phase0_validation.csv`.
- Use `eval/phase0_config.json`.
- Use `AgentLoop` initialization, not loose pseudocode.
- Use `resolve_config_env_vars(load_config(path))`.
- Write incremental JSONL.
- Skip U2 by default.
- Collect both outer trace and inner trace.
- Align traces by step.
- Preserve raw model output only in local uncommitted artifacts.
- Avoid committing raw traces or per-task JSONL.

## Analysis design

Analyze:

- `confidence` vs outcome proxies
- calibration: ECE / Brier / reliability curve
- risk bucket failure rate
- `need_slow_planner` precision / recall
- anomaly signal precision / recall
- combined trigger slow-call rate / failure capture rate

Prompt tax A/B:

- `runtime_signal` off
- `runtime_signal` on

Ablations:

- self-report only
- risk only
- anomaly only
- self-report + risk
- risk + anomaly
- combined

This is pilot evidence, not final paper-level evidence.

## Decision criteria

If signals show no trend, revise signal design before Controller. If `confidence` is poorly calibrated but anomaly/risk works, Controller v0 should use anomaly + risk first. If combined trigger captures failures with acceptable slow-call rate, proceed to Controller v0.

Do not proceed to Controller based only on anecdotal examples.

## Connection to Phase 1 Controller / Phase 2 Slow Correction / Phase 3 Skill Evolution

Phase 1 introduces the step-level Fast/Slow Controller. Phase 2 adds slow correction and targeted exploration. Phase 3 adds skill distillation and skill self-evolution.

Controller safety order:

```python
if high_risk:
    verify_or_slow()
elif skill_match and skill_verified and risk_low:
    execute_skill()
    post_skill_verify()
elif confidence_low:
    slow()
elif anomaly:
    recover()
else:
    fast()
```

Skill hit must not bypass high-risk verification.

## Non-goals for this phase

This task does not:

- implement `runtime_signal`
- modify prompt/parser
- run pilot
- run U2 tasks
- implement Controller
- make paper-level claims
