# Cost-Aware Router V0 Handoff

Date: 2026-05-26

Branch: `feat/cost-aware-router-v0`

Repository: `ShiboSusu/nanobot_fork`

## Goal

Improve GUIClaw / OpenGUI task success per unit cost by routing tasks through the cheapest safe path before falling back to visual GUI automation.

The V0 objective is not a full uncertainty monitor. It is a working foundation:

1. Policy gate for sensitive tasks/actions.
2. Cost-aware task router shared by CLI and chat channels.
3. Tool-first hints for information/query tasks.
4. System-action route for simple device actions such as opening iOS Settings.
5. Skill interface kept as a disabled no-op for now.

## Current Runtime Paths

- Nanobot repo: `/Users/su/Documents/Codes/nanobot_fork`
- Startup script: `/Users/su/Documents/Codes/.local-bin/nanobot-ios-start`
- Startup script source: `/Users/su/Documents/Codes/nanobot_fork/scripts/nanobot-ios-start`
- Nanobot config: `/Users/su/.nanobot/config.json`
- WDA URL: `http://127.0.0.1:8100`
- Small model endpoint: `http://127.0.0.1:18000/v1`
- Mid model endpoint: `http://127.0.0.1:18001/v1`
- Main model: `qwen3.5-397b-a17b`
- GUI S1 model: `qwen3.5-9b`
- GUI S2 model: `qwen3.5-397b-a17b`
- GUI default action interface: native OpenAI-compatible `computer_use` tool calls (`gui.agentProfile=default`)

Note: `qwen3.6-35b-a3b` is available through `vllm_35b`, but it is not the default main-agent model until the remote vLLM service is started with OpenAI-compatible tool-choice support.
Note: Qwen text profiles such as `qwen3vl` are compatibility adapters only; they should not be the default when the model server supports native tool calls.

Do not put tokens or private keys in this document.

## Implemented Files

- `opengui/policy.py`
  - Defines `PolicyStore`, `PolicyDecision`, and `PolicyAction`.
  - V0 policies cover login/auth, payment/purchase, destructive actions, permissions, privacy/personal data, external sending/publishing, and account security.

- `nanobot/agent/cost_aware_router.py`
  - Defines `CostAwareProblemRouter`, `RouteDecision`, `RouteKind`, and `NoopSkillLibrary`.
  - Routes sensitive tasks to human confirmation, simple open-settings tasks to system action, query tasks to tool-call hints, and other tasks to GUI/S2.

- `nanobot/agent/loop.py`
  - Installs the router at the shared `AgentLoop._process_message` entry.
  - This is intentional: CLI, Telegram, WebSocket, and other app channels use the same route instead of CLI-only behavior.

- `opengui/agent.py`
  - Adds a pre-action policy gate before backend execution.
  - Sensitive model-proposed actions become `request_intervention` and do not touch the device backend.

- Tests:
  - `tests/agent/test_cost_aware_router.py`
  - `tests/agent/test_cost_aware_routing_integration.py`
  - `tests/test_opengui_p15_intervention.py`
  - Existing parser/direct-iOS tests in `tests/test_opengui.py`

## Skill And Memory Scope

V0 intentionally disables skill execution/extraction:

```json
{
  "gui": {
    "enable_skill_execution": false,
    "enable_skill_extraction": false
  }
}
```

The skill interface exists as `NoopSkillLibrary`, returning no candidates.

Memory is not used for trajectory recall yet. Policy memory is implemented as a deterministic safety policy layer.

## Verified Commands

Unit and integration regression:

```bash
/opt/anaconda3/envs/nanobot/bin/python -m pytest \
  tests/agent/test_cost_aware_router.py \
  tests/agent/test_cost_aware_routing_integration.py \
  tests/test_opengui_p15_intervention.py \
  -q
```

Targeted OpenGUI regression:

```bash
/opt/anaconda3/envs/nanobot/bin/python -m pytest \
  tests/agent/test_cost_aware_router.py \
  tests/agent/test_cost_aware_routing_integration.py \
  tests/test_opengui.py \
  -k 'ios_settings_task_uses_direct_bundle_launch or qwen3vl_profile' \
  -q
```

Startup preflight:

```bash
/Users/su/Documents/Codes/.local-bin/nanobot-ios-start preflight
```

Expected: updates `~/.nanobot/config.json`, starts ModelArts tunnels, starts or reuses WDA, and prints active app info.

Real CLI/system-action path:

```bash
/Users/su/Documents/Codes/.local-bin/nanobot-ios-start agent -m "帮我用手机打开设置"
```

Expected: returns a completed state with `com.apple.Preferences`.

Real CLI/policy path:

```bash
/Users/su/Documents/Codes/.local-bin/nanobot-ios-start agent -m "帮我登录支付宝并完成付款"
```

Expected: returns a human-confirmation policy message and does not enter GUI automation.

Real CLI/query tool path:

```bash
/Users/su/Documents/Codes/.local-bin/nanobot-ios-start agent -m "查一下今天深圳天气"
```

Verified: the session called `mcp_amap_maps_weather` with `{"city": "深圳"}` and did not enter GUI automation.

## Next Steps

1. If query routing is not reliable enough across more tasks, add a deterministic read-only tool executor path for common weather/search tasks.
2. Add an `AutonomyMonitor` inside OpenGUI after V0 routes stabilize.
3. Later enable skill execution behind explicit config and validators.
