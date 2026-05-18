# Entropy-Aware Uncertainty Signal Design

## Motivation

Phase 0 has pivoted from provider-native `reasoning_tokens` to observable runtime uncertainty signals. The current S1 path is `qwen3-vl-8b` served through the `qwen3vl` text profile, with runtime self-report signals available as an experiment but not yet validated as reliable routing evidence.

Self-reported confidence is useful but not reliable enough alone. A model can be overconfident, especially in short GUI tasks where it may repeatedly report values such as `0.9` or `1.0` even when the screen is ambiguous, the target is partially hidden, or the chosen coordinate is brittle. Confidence should therefore be treated as one claim from S1, not as calibrated probability.

Entropy and disagreement signals should supplement confidence, risk, and execution anomaly features. They should not replace all other signals. A robust Monitor should combine several views of uncertainty:

- what the model says about its uncertainty,
- how stable the proposed action is across samples,
- whether the action is risky,
- whether execution state shows stagnation or mismatch,
- whether trace quality is sufficient for analysis.

## Core Principle

GUI uncertainty should be action-level, not raw-text-level.

Raw token entropy can be informative, but a GUI agent ultimately executes parsed actions. Two text completions can look different while producing the same safe action, and two similar strings can produce materially different taps or typed arguments. The Monitor should therefore prioritize uncertainty over the normalized action representation.

Preferred signals:

- `action_entropy`
- `sample_disagreement`
- `action_type_disagreement`
- `argument_disagreement`
- `target_disagreement`
- `coordinate_variance`

Raw token entropy is optional and lower priority. It may become useful if the serving endpoint exposes stable log probabilities, but it should not be the primary uncertainty layer for Phase 0.

## Signal Schema

The entropy-aware Monitor should extend `trigger_features` while preserving the trigger/outcome separation already established by Task 4C.

Recommended shape:

```json
{
  "trigger_features": {
    "self_report": {
      "confidence": null,
      "need_slow_planner": null,
      "uncertainty_reason": null
    },
    "entropy": {
      "logprob_entropy": null,
      "action_entropy": null,
      "sample_disagreement": null,
      "action_type_disagreement": null,
      "argument_disagreement": null,
      "target_disagreement": null,
      "coordinate_variance": null,
      "num_samples": 1,
      "entropy_method": "none"
    },
    "risk": {
      "task_risk_level": "U0",
      "step_predicted_risk_level": null,
      "rule_based_step_risk_level": "U0"
    },
    "execution_state": {
      "repeated_action": false,
      "stagnation_count": 0,
      "foreground_app_mismatch": false,
      "no_screen_change_after_action": null
    }
  },
  "outcome_proxies": {
    "execution_error": false,
    "action_parse_failure": false,
    "post_action_no_observable_change": null,
    "judge_derived_not_advanced": null
  }
}
```

Field meanings:

- `logprob_entropy`: token-level entropy if endpoint log probability data is available.
- `action_entropy`: entropy over normalized parsed actions or action clusters.
- `sample_disagreement`: fraction or rate of samples that disagree with the modal parsed action.
- `action_type_disagreement`: disagreement over action type, such as `tap`, `type`, `swipe`, `wait`, or `done`.
- `argument_disagreement`: disagreement over non-coordinate action arguments, such as typed text, app name, or status.
- `target_disagreement`: disagreement over semantic target when target labels or detected UI elements are available.
- `coordinate_variance`: variance or normalized spread over coordinates for coordinate-bearing actions.
- `num_samples`: number of model samples used to estimate disagreement.
- `entropy_method`: source of the estimate, such as `none`, `logprobs`, `multi_sample`, or `parsed_action_disagreement`.

## Estimating Entropy

### Option A: Logprob-Based Entropy

Logprob-based entropy uses token probability data from the model endpoint to estimate uncertainty in generated text.

Endpoint dependency:

- Requires the serving endpoint to expose log probability fields.
- Depends on stable behavior from the Qwen3-VL + vLLM-compatible serving stack.
- May be unavailable or expensive for vision-language chat completions.

Cost and latency:

- Potentially lower than multi-sampling if log probabilities are returned in a single request.
- Can increase response size and post-processing overhead.

Robustness for `qwen3vl` text profile:

- Useful only if log probabilities correspond cleanly to the assistant text that contains the action.
- Fragile when action markup, natural language, and formatting tokens dominate token-level uncertainty.

Mapping to GUI action uncertainty:

- Weak to moderate. Token uncertainty does not necessarily map to action uncertainty.
- A low-entropy text sequence can still produce a wrong coordinate.
- A high-entropy text sequence can still parse to the same action.

Failure modes:

- Endpoint does not expose usable log probabilities.
- Entropy is dominated by harmless formatting variation.
- Entropy is low for confidently wrong actions.
- Logprob fields differ across backend versions.

### Option B: Multi-Sample Action Disagreement

Multi-sample action disagreement queries S1 multiple times for the same observation and instruction, parses each candidate action, and measures disagreement over parsed actions.

Endpoint dependency:

- Requires only normal generation and existing action parsing.
- Does not require log probability support.
- Works with the current `qwen3vl` text-profile path if each sample still emits parseable `<tool_call>` content.

Cost and latency:

- More expensive than single-step fast execution.
- Latency scales with the number of samples unless requests can be parallelized.
- Should not run on every normal fast step.

Robustness for `qwen3vl` text profile:

- Stronger than raw token entropy because it operates after parsing.
- Directly tests whether the model converges on the same GUI action.
- Requires strict handling of parse failures as uncertainty, not as silent missing data.

Mapping to GUI action uncertainty:

- Strong. Disagreement over action type, arguments, target, or coordinates maps naturally to action uncertainty.
- Coordinate variance is especially relevant for tap and drag actions.
- Argument disagreement is especially relevant for typed text and app-opening tasks.

Failure modes:

- Sampling temperature too low may hide uncertainty.
- Sampling temperature too high may create artificial disagreement.
- All samples may agree on a wrong action.
- Multiple plausible actions may all be safe, so disagreement does not always imply failure.
- Repeated sampling can change device state if samples are executed; samples must be non-executing candidate generations.

### Option C: Parsed-Action Disagreement Without Logprobs

Parsed-action disagreement without endpoint log probabilities is a constrained version of Option B. It may use either multiple generated candidates or repeated trace snapshots, but only the parsed action outputs are compared.

Endpoint dependency:

- No log probability dependency.
- Depends on existing parser output and trace capture.

Cost and latency:

- Lower than full semantic verifier calls.
- Similar to or slightly lower than multi-sample action disagreement if only parsed summaries are retained.

Robustness for `qwen3vl` text profile:

- Good if parser outputs are stable and preserved in inner trace.
- Sensitive to trace quality and inner/outer alignment.

Mapping to GUI action uncertainty:

- Strong for action type and argument disagreement.
- Moderate for target disagreement unless target labels are available.
- Strong for coordinate variance if coordinate-bearing actions are normalized.

Failure modes:

- Parser failures may collapse into missing data unless explicitly counted.
- Different text outputs can parse to superficially similar actions with different real effects.
- Coordinate comparisons need screen-size normalization.

## Recommended Default

Use multi-sample action disagreement as the primary entropy proxy in Phase 0 experiments.

Rationale:

- It aligns with GUI action uncertainty rather than raw text uncertainty.
- It does not depend on log probability support from the current endpoint.
- It works with `qwen3vl` text-profile traces if candidate actions remain parseable.
- It can be analyzed alongside self-report confidence to detect overconfidence.

Treat log probabilities as optional. If the Huawei/vLLM-compatible endpoint exposes stable log probability data for the relevant chat completions, record `logprob_entropy` as a secondary signal. Do not block Phase 0 on logprob availability.

Do not sample every step during normal Fast execution. Entropy sampling should be used only:

1. in offline pilot runs, where latency is acceptable and actions can be controlled, or
2. when the Monitor already suspects high risk or anomaly, such as high rule-based risk, repeated actions, stagnation, foreground-app mismatch, or low confidence.

Normal fast execution should remain single-pass until pilot evidence shows when the extra calls are worth their cost.

## Calibration And Thresholds

Do not set final routing thresholds yet. Thresholds must be selected from pilot evidence, not hand-picked from anecdotes.

Candidate confidence thresholds:

- `T_conf` in `{0.5, 0.6, 0.7, 0.8}`

Candidate action entropy buckets:

- low: near-zero action entropy or full agreement
- medium: moderate disagreement over coordinates or arguments
- high: action type disagreement, target disagreement, or repeated parse failures

Candidate sample disagreement thresholds:

- `0.33`: at least roughly one third of samples disagree with the modal action
- `0.66`: most samples disagree or no stable modal action emerges

Pilot analysis should sweep combinations such as:

- confidence only,
- entropy only,
- risk only,
- anomaly only,
- confidence + entropy,
- entropy + risk,
- entropy + anomaly,
- confidence + entropy + risk + anomaly.

Final thresholds should optimize failure capture rate subject to acceptable slow-call rate. They must be reported with sample size, task mix, and trace-quality filters.

## Avoiding Circular Reasoning

Entropy and disagreement are trigger features. Outcome proxies must remain separate.

Do not use `sample_disagreement` itself as the failure label proving that `sample_disagreement` works. For example, it is circular to say an action failed because samples disagreed and then claim disagreement predicts failure.

Valid outcome proxies include:

- execution errors,
- action parse failures,
- post-action no observable change,
- judge-derived not advanced,
- task level clean success or clean failure,
- verifier-derived issue labels in later phases.

When analyzing entropy, compare trigger features against these separate outcomes. If an entropy signal only correlates with its own construction artifact, it should not be used for routing.

## Integration With Future S2

S2 is future work and is not connected in this task. The planned S2 model is Qwen3.5-397B-A17B on a Huawei cluster, but this design does not add credentials, endpoints, connection code, or routing code.

Future integration order:

1. High-risk or high-disagreement actions trigger an S2 verifier.
2. Repeated or stagnated actions trigger S2 recovery.
3. S2 planner is introduced only after verifier and recovery paths are stable.

Recommended Monitor behavior:

```python
if high_risk and high_disagreement:
    verify_with_s2()
elif anomaly_repeated_or_stagnated:
    recover_with_s2()
elif low_confidence and high_action_entropy:
    verify_with_s2()
else:
    continue_fast()
```

The S2 verifier should inspect the proposed action and observation before execution when possible. S2 recovery should be reserved for post-action anomalies, repeated actions, or stagnation. S2 planning should not be the first slow path because planning introduces broader behavioral changes and needs stronger safety controls.

## Required Future Smoke Tests

Future tests should be run before any broader pilot. Do not run them in this task.

- no-runtime-signal baseline,
- runtime signal only,
- entropy sampling only,
- runtime signal + entropy,
- verify that action parsing still works,
- verify trace records all samples or compact sample summaries,
- verify no U2 execution,
- verify no signal fields are inserted into action arguments,
- verify parse failures are counted as uncertainty,
- verify repeated samples are generated without executing candidate actions.

Minimum trace requirements for entropy sampling:

- base step observation identifier,
- number of samples,
- candidate raw outputs or sanitized summaries,
- candidate parsed actions,
- parse status per sample,
- modal action,
- disagreement metrics,
- chosen action,
- whether chosen action equals modal action.

Raw screenshots and raw model outputs should remain local uncommitted artifacts unless a later privacy review approves a redacted export format.

## Non-Goals

This task does not:

- implement a Controller,
- connect S2,
- add S2 credentials or endpoint configuration,
- run U2 tasks,
- run the full pilot,
- modify GUI prompt, parser, model adapter, or trace recorder internals,
- claim that entropy proves uncertainty,
- set final routing thresholds,
- replace risk or execution anomaly signals with entropy alone.

## Open Questions

- What sample count gives a useful signal without unacceptable latency: 3, 5, or another value?
- Should candidate samples use deterministic low-temperature perturbation or normal sampling temperature?
- How should coordinate variance be normalized across device resolutions and screenshot scaling?
- Should parse failures count as maximum disagreement or as a separate uncertainty feature?
- Can inner trace coverage remain sufficient when multiple candidate samples are captured but only one action is executed?
