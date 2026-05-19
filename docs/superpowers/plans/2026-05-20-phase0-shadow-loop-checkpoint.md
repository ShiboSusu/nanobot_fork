# Phase 0 Shadow Loop Checkpoint

## Summary

Phase 0 is still on the intended FastSlow-GUI main line:

```text
real U0/U1 shadow-mode
-> runtime/controller signals
-> S2 offline verifier
-> controller dry-run closed loop
```

The recent work did not implement live intervention, did not connect S2 to live GUI control, did not run U2, and did not run a full pilot. It strengthened the shadow-mode evidence path and made the offline loop diagnosable.

This is pilot evidence only. Do not report task success rates, controller performance, or paper-level claims from these records.

## Current Evidence Slice

Use the scoped summaries, not unscoped artifact tails, because the ignored JSONL files contain older schema records, missing-env checks, and S2 retry artifacts.

Current real shadow slice:

```bash
/opt/anaconda3/envs/nanobot/bin/python eval/phase0_observable_signal_pilot.py \
  --summarize-output eval/phase0_observable_results.jsonl \
  --summarize-since-line 29
```

Observed summary for lines 29-31:

```text
Total records: 3
Risk distribution: {'U0': 3}
Runner clean success count: 2
Semantic success count: 0
Semantic failure count: 3
Termination reason distribution: {'stagnation_detected': 1, 'completed': 2}
Average inner coverage: 1.0
Clean for signal analysis count: 3
Controller route distribution: {'FAST': 2, 'RECOVER': 2, 'VERIFY': 2}
Monitor trigger distribution:
  none: 2
  repeated_region_action: 1
  action_type_run: 1
  semantic_missing_answer: 2
High confidence failed record count: 1
Screenshot failure record count: 0
```

Current S2 offline slice:

```bash
/opt/anaconda3/envs/nanobot/bin/python eval/s2_verifier_client.py \
  --summarize-output eval/phase0_s2_verifier_offline.jsonl \
  --summarize-last 3
```

Observed summary for the latest line 29-31 source records:

```text
Source record line range: 29-31
Decision distribution: {"block": 1, "replan": 2}
Failure risk distribution: {"high": 3}
Reason-for-verification distribution:
  semantic_missing_answer: 2
  stagnation: 1
Allowed-to-execute count: 0
```

Current controller dry-run closed loop:

```bash
/opt/anaconda3/envs/nanobot/bin/python eval/phase0_controller_dry_run.py \
  --input eval/phase0_s2_verifier_offline.jsonl \
  --input-last 3 \
  --output eval/phase0_controller_dry_run.jsonl \
  --summary
```

Observed route summary:

```text
route_distribution: {"BLOCK": 1, "SLOW": 2}
verifier_decision_distribution: {"block": 1, "replan": 2}
skipped_unsafe_count: 0
unusable_trace_count: 0
```

## Task-Level Observations

| source line | task_id | shadow outcome | key signal | S2 offline decision | dry-run route |
| ---: | --- | --- | --- | --- | --- |
| 29 | `ChromeSearchBeijingWeatherTask` | `stagnation_detected`, semantic failure | repeated tap / `stagnation`, high confidence `1.0` | `replan`, failure `high` | `SLOW` |
| 30 | `CheckPuchasedItem` | runner clean, semantic failure | empty `done(success)` / `semantic_missing_answer` | `replan`, failure `high` | `SLOW` |
| 31 | `RecentTotalExpenseTask` | runner clean, semantic failure | empty `done(success)` / `semantic_missing_answer` | `block`, failure `high` | `BLOCK` |

Interpretation:

- Trace extraction is usable for this slice: all three records have `inner_coverage=1.0` and `clean_for_signal_analysis=true`.
- Runner-clean completion is not semantic completion. `CheckPuchasedItem` and `RecentTotalExpenseTask` finished with `done(success)` but lacked the required answer.
- Self-reported confidence remains unreliable. Chrome had confidence `1.0` while stuck in repeated taps and then failed.
- `semantic_missing_answer`, `repeated_region_action`, and `action_type_run` are useful shadow signals for diagnosability.
- S2 offline verifier can classify these failures as high failure risk when semantic/controller diagnostics are included in the request.
- Controller dry-run maps S2 `replan` to `SLOW` and `block` to `BLOCK`, giving a complete offline shadow loop.

## What Changed Recently

Recent commits kept to analysis/shadow infrastructure:

- `c3c954a7` flags missing-answer semantic failures in shadow routing.
- `dc9dfa91` summarizes controller shadow triggers.
- `00750b3f` summarizes monitor anomaly signals.
- `48b2e77e` scopes observable output summaries.
- `b0bb261e` selects S2 offline smoke records by physical JSONL line.
- `9ddff9fd` includes semantic/controller diagnostics in S2 requests.
- `90030b58` summarizes S2 offline verifier output.
- `1cbad8e2` scopes controller dry-run inputs.
- `d5a2cf9f` records S2 offline source provenance.

These are still on the main line because they reduce ambiguity in the shadow pipeline and prevent false claims from mixed old/new artifact records.

## Important Caveats

- The sample is tiny and biased toward debugging. It is not an evaluation set.
- S2 offline output is not ground truth. It is a verifier candidate whose behavior still needs human audit and larger scoped samples.
- The ignored S2 output contains missing-env and non-JSON retry artifacts. Always use `source_record` and scoped summaries.
- S2 has not been connected to live GUI control.
- Controller has not intervened in live execution.
- U2 remains blocked.
- No formal threshold, ROC, calibration, or success-rate claim is supported yet.

## Mainline Status

The main line has advanced from:

```text
real-device U0 diagnosis in progress
```

to:

```text
real U0/U1 shadow records usable
-> semantic failure guard active
-> monitor anomalies summarized
-> S2 offline verifier enriched with semantic/controller diagnostics
-> controller dry-run closed loop demonstrated on latest 3-record U0 slice
```

## Recommended Next Gate

Do not implement live intervention yet.

The first 3-record U0-only version of this gate is now met for diagnosability. It is not enough for formal thresholds or live intervention. Before any live S2/controller action, either stop for human audit and design review, or collect at most two additional safe U0/U1 records if there are genuinely safe candidates:

- 5 U0/U1 records maximum for this scoped batch.
- No U2.
- No messages, email, WeChat, purchases, payments, account/security changes, destructive settings, or full pilot.
- Runtime signal on.
- Controller shadow on.
- S2 only offline after the GUI run.
- Report:
  - trace quality,
  - semantic outcome,
  - monitor trigger distribution,
  - high-confidence failure count,
  - S2 decision distribution,
  - controller dry-run route distribution,
  - unsafe-action concerns.

Minimum gate to consider the next design step:

- All selected records have diagnosable trace output or clear environment hard-gate reason.
- Empty final `done` is never counted as semantic success.
- S2 offline request includes `semantic_outcome` and selected-step `controller` diagnostics.
- Controller dry-run output is scoped to matching S2 records via `source_record` provenance.
- At least one repeated-action/stagnation failure and one empty-answer failure remain explainable end-to-end.
