# GUI Autonomy Monitor V0 Handoff

Date: 2026-05-26

Branch: `feat/gui-autonomy-monitor-v0`

Repository: `ShiboSusu/nanobot_fork`

## Goal

Add a low-cost, rule-based autonomy monitor to improve GUIClaw/OpenGUI reliability without making every step call the large model.

This V0 monitor does not train a verifier and does not enable skill execution. It only watches observable execution signals and stops the small GUI actor when continuing would likely waste steps or compound errors.

## Architecture

- `nanobot/agent/cost_aware_router.py`
  - Still owns task-level routing before model/tool cost is spent.
  - Query tasks prefer tool calls.
  - Simple system tasks such as opening iOS Settings use direct system actions.
  - Sensitive tasks go to human confirmation.

- `opengui/policy.py`
  - Still owns deterministic safety policy.
  - Login, payment, delete, permission, privacy, external send, and account-security categories remain blocked behind confirmation.

- `opengui/autonomy.py`
  - New step-level autonomy monitor.
  - Inputs: task, step index, max steps, action, action failure, observe failure, screen unchanged, repeated action, progress observed, optional policy decision.
  - Output: `AutonomyDecision`.

- `opengui/agent.py`
  - Wires the monitor into `GuiAgent._run_once` after each executed step and before appending history.
  - Records monitor metadata into trace/model snapshots under `autonomy_monitor`.
  - Blocks with `error="autonomy_monitor_intervention"` when the monitor returns `S2_HINT`, `S2_TAKEOVER`, `HUMAN_CONFIRM`, or `HALT`.

## V0 Decisions

- `S1_EXECUTE`: continue small-model GUI execution.
- `CHEAP_VERIFY`: recorded as a warning-level decision; V0 continues because no cheap verifier is wired yet.
- `S2_HINT`: stop the current GUI attempt and ask the slow path to verify/recover.
- `S2_TAKEOVER`: stop because the current step has high immediate risk.
- `HUMAN_CONFIRM`: stop for human confirmation.
- `HALT`: stop.

## Signals

- `action_failed`
- `observe_failed`
- `screen_unchanged`
- `repeated_action`
- `step_budget_pressure`
- `safety_risk`
- `cumulative_risk_over_budget`

Important detail: if the screen changed after the action, `progress_observed=True`, and repeated action is not counted as a risk signal for that step. This avoids killing legitimate wait/load flows.

## Verified Commands

Unit and integration:

```bash
/opt/anaconda3/envs/nanobot/bin/python -m pytest tests/test_opengui_autonomy.py -q
```

Result: `7 passed`.

OpenGUI regression around monitor, stagnation, and direct iOS route:

```bash
/opt/anaconda3/envs/nanobot/bin/python -m pytest \
  tests/test_opengui_autonomy.py \
  tests/test_opengui.py \
  -k 'stagnation or autonomy or ios_settings_task_uses_direct_bundle_launch' \
  -q
```

Result: `14 passed, 89 deselected`.

Router and intervention regression:

```bash
/opt/anaconda3/envs/nanobot/bin/python -m pytest \
  tests/test_opengui_autonomy.py \
  tests/agent/test_cost_aware_router.py \
  tests/agent/test_cost_aware_routing_integration.py \
  tests/test_opengui_p15_intervention.py \
  -q
```

Result: `24 passed, 4 warnings`.

Diff hygiene:

```bash
git diff --check
```

Result: clean.

## Real iOS Smoke Results

Status:

```bash
/Volumes/T9/Mac/bin/nanobot-ios-start status
```

Verified:

- Remote vLLM ready on `8000`.
- Local small model endpoint ready at `http://127.0.0.1:18000/v1`.
- WDA ready at `http://127.0.0.1:8100`.
- Main model: `qwen3.5-397b-a17b`.
- GUI model: `qwen3.5-9b`.

System action:

```bash
/Volumes/T9/Mac/bin/nanobot-ios-start agent -m "帮我用手机打开设置"
```

Verified output:

```text
Status: completed
Done: Directly opened iOS app com.apple.Preferences
Remaining: none
Current: com.apple.Preferences 402x874
```

Sensitive policy:

```bash
/Volumes/T9/Mac/bin/nanobot-ios-start agent -m "帮我登录支付宝并完成付款"
```

Verified: blocked with `login_or_auth, payment_or_purchase`; no GUI action executed.

Query/tool-first:

```bash
/Volumes/T9/Mac/bin/nanobot-ios-start agent -m "查一下今天深圳天气"
```

Verified: answered weather via tool-first route; no GUI fallback was needed.

## Next Steps

1. Wire an actual cheap verifier behind `CHEAP_VERIFY`.
2. Add optional S2 hint/recovery call when `S2_HINT` is returned instead of stopping at V0.
3. Add trace-derived labels for V1 learned step router.
4. Keep skill execution disabled until validators and policy gates are strong enough.
