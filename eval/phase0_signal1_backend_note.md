# Phase 0 Signal 1 Backend Feasibility Note

Date: 2026-05-17

## Context

Phase 0 originally considered provider-native `reasoning_tokens` as a possible runtime signal for deciding when the fast GUI agent should escalate to a slow planner. This note records the backend feasibility result for the current S1 backend only.

This is a backend feasibility note, not a paper-level claim about all models, all serving stacks, or the general usefulness of reasoning signals.

## Tested Backend

- Model path: `/home/ma-user/work/ShiboSu/models/Qwen3-VL-8B-Instruct`
- Served model name: `qwen3-vl-8b`
- Remote vLLM port: `8001`
- Local Phase 0 endpoint: `http://127.0.0.1:8001/v1`
- Local tunnel is required. Use a template such as:

```bash
ssh -i <MODELARTS_KEY_PATH> -L 8001:127.0.0.1:8001 <USER>@<MODELARTS_HOST>
```

## Diagnostic Parser Mode Result

The backend was tested in a diagnostic vLLM mode with `--reasoning-parser qwen3`. vLLM logs confirmed `reasoning_parser='qwen3'`.

Observed behavior in this diagnostic mode:

- A smoke test can produce structured `message.reasoning`.
- Through the nanobot provider path, the same routed text can appear as `reasoning_content`.
- `completion_tokens_details.reasoning_tokens` stayed `0`.
- Top-level `usage.reasoning_tokens` was absent.
- With `enable_thinking=true`, `message.content` can become `null`.

The `message.reasoning` field is therefore evidence of vLLM parser routing behavior. It is not a usable runtime token-count signal for this backend.

## Normal GUI Execution Implication

Diagnostic parser/thinking mode should not be used for the GUI pilot unless a chat smoke test proves that `message.content` remains non-null and contains the action output expected by the GUI parser.

For normal GUI execution, the serving mode and config must avoid routing normal action text into `reasoning` with `content=null`, because that can break GUI action parsing.

## Conclusion

Under the current Qwen3-VL-8B-Instruct plus vLLM serving setup, provider-native `reasoning_tokens` is not available as a usable runtime signal for this S1 backend.

Signal 1 (`reasoning_tokens`) is therefore NO-GO for this S1 backend under the current serving setup. The next Phase 0 direction should validate observable runtime signals rather than running a 22-task `reasoning_tokens` pilot.
