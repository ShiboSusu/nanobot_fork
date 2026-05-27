# 35B Main Planner Design

## Goal

Use the 35B A3B model as the main task planner/router at the host entry point, so task routing is based on semantic understanding rather than an ever-growing keyword list. The planner should recover the better behavior from the previous `general_e2e` style while keeping the current system's safety boundaries.

The design goal is efficient task completion at low cost:

- public information queries should use tools first;
- simple device actions should use deterministic system actions when safe;
- GUI-only tasks should be delegated to the S1 GUI executor;
- private, financial, destructive, auth, or external-send tasks must still pass policy gates;
- 397B remains System 2 for high-risk recovery, re-planning, or takeover.

## Architecture

The host task flow becomes:

```text
user task
  -> policy hard gate
  -> 35B Main Planner
  -> plan schema validation
  -> host execution router
  -> tool / system_action / gui_task(S1) / ask_user / S2
  -> autonomy monitor
  -> 397B S2 hint or takeover when needed
```

Policy remains outside model control. The 35B planner may recommend a route, but the host validates the plan and can override or stop it.

## Planner Contract

The planner uses content-only JSON, not provider-native tool calls. This keeps it compatible with vLLM/OpenAI-compatible endpoints that do not fully support `tool_choice`.

Expected JSON:

```json
{
  "route": "tool_call | system_action | gui_task | ask_user | s2",
  "confidence": 0.0,
  "reason": "short reason",
  "subtasks": [
    {
      "route": "web_search | web_fetch | weather | system_action | gui_task | ask_user | s2",
      "task": "self-contained subtask",
      "tool": "optional tool name",
      "system_action": "optional system intent"
    }
  ],
  "risk_notes": ["optional notes"]
}
```

The parser accepts fenced JSON or raw JSON, rejects unknown routes, clamps confidence to `[0, 1]`, and falls back to the current deterministic router when output is missing, malformed, or low-confidence.

## Routing Rules

The planner should be the default semantic entry point after policy:

- Weather, trains, facts, current events, and other public lookup tasks route to tools.
- App-internal tasks like "search in Douyin" or "play a Bilibili video" route to GUI.
- Safe one-shot actions like opening Settings or Safari can route to system actions.
- Ambiguous or under-specified tasks route to `ask_user`.
- Complex tasks can be decomposed into subtasks, but the host executes only supported subtasks.

The host still re-runs policy on the original task and each subtask before execution.

## Error Handling

Fallbacks are explicit:

- planner timeout or provider error: use deterministic router;
- invalid JSON: use deterministic router;
- low confidence below threshold: use deterministic router or S2 depending on risk;
- policy violation in plan/subtask: require human confirmation;
- unsupported route/tool: return `ask_user` or deterministic fallback;
- repeated planner failures: disable semantic planner for that task run.

## Configuration

Add GUI/agent config fields:

- `plannerModel`: default empty;
- `plannerProvider`: default empty;
- `plannerEnabled`: default false until manually enabled;
- `plannerConfidenceThreshold`: default `0.65`;
- `plannerMaxTokens`: default `512`;
- `plannerTimeoutSeconds`: default small bounded timeout.

For the local setup, point `plannerModel/plannerProvider` to the 35B A3B endpoint. Keep `s2Model/s2Provider` on 397B.

## Testing

Add unit tests for:

- public query -> tool route;
- in-app search -> GUI route;
- sensitive account query blocked before planner route can execute;
- malformed planner output falls back to deterministic router;
- low-confidence planner output falls back;
- 35B planner uses content-only JSON, not native tool calls;
- host rejects unsupported routes and tools;
- subtask policy re-check blocks private or financial subtasks.

Integration tests should use a mock planner provider first. Real 35B endpoint testing can be a separate manual smoke test after the contract is stable.
