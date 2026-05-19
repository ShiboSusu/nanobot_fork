# Phase 0 Shadow Batch Audit

## Decision

Stop real-device data collection for this micro-batch and do a design audit before any additional live runs.

The line 29-31 U0 slice is enough to validate the current shadow-loop plumbing:

```text
real GUI run
-> runtime/controller shadow signals
-> semantic outcome guard
-> S2 offline verifier
-> controller dry-run route
```

It is not enough to support intervention, thresholds, success-rate claims, or S2/controller performance claims.

## Evidence

The scoped batch contains three real U0 records:

| source line | task_id | runner outcome | semantic outcome | monitor trigger | S2 offline | dry-run route |
| ---: | --- | --- | --- | --- | --- | --- |
| 29 | `ChromeSearchBeijingWeatherTask` | failed with `stagnation_detected` | false | `repeated_region_action`, `action_type_run` | `replan`, failure `high` | `SLOW` |
| 30 | `CheckPuchasedItem` | clean `done(success)` | false | `semantic_missing_answer` | `replan`, failure `high` | `SLOW` |
| 31 | `RecentTotalExpenseTask` | clean `done(success)` | false | `semantic_missing_answer` | `block`, failure `high` | `BLOCK` |

All three records are `clean_for_signal_analysis=true` with `inner_coverage=1.0`, so the current trace path is usable for diagnosis.

The key result is not task success. The key result is that semantic failure and monitor anomalies now survive through the offline loop.

## Why Not Add Two More Live Records Now

The remaining U0 tasks have real side effects:

- `SetAlarmTask` changes the user's alarms.
- `TakeSelfieTask` creates a photo on the device.

The U1 candidates are also real device-setting changes:

- brightness maximum/minimum,
- font/icon scale changes,
- wallpaper changes,
- flight mode changes.

These are not U2, but they are still live device mutations. Since the 3-record diagnosability gate is already met, additional live runs would add operational risk before the next design question is settled.

The current runner safety filter blocks U2 and obvious external side effects such as messages, purchases, payments, account changes, and destructive settings. It does not currently classify local-but-real mutations such as alarms, camera captures, brightness changes, wallpaper changes, or flight mode as a separate hard-gated category. That gap should be handled before broadening real-device collection.

## Design Issues Exposed

1. Self-reported confidence remains unreliable.

   The Chrome record had confidence `1.0` while repeated taps and stagnation were happening.

2. `done(success)` is only runner finalization.

   Both answer-required Taobao tasks ended with empty `done(success)` and semantic failure.

3. S2 offline decisions are not yet a policy.

   The same `semantic_missing_answer` trigger produced `replan` for one record and `block` for another. That may be reasonable, but it should be audited before any live controller uses it.

4. Controller dry-run is correctly scoped, but still descriptive.

   The dry-run proves routing records can be produced from S2 output and provenance. It does not prove that those routes are the right interventions.

## Recommended Next Task

Task 8AF: U0/U1 Safety Gate and Offline Route Policy Audit

Goal:

Review the latest scoped batch, reclassify remaining U0/U1 live-task side effects, and define how Phase 0 should interpret verifier decisions before any live intervention.

Inputs:

- observable records with source lines 29-31,
- S2 offline verifier records whose `source_record.line_number` is 29-31,
- controller dry-run output scoped to those S2 records.

Questions:

- Should `semantic_missing_answer` after empty `done(success)` default to `SLOW`, `RECOVER`, `ASK_USER`, or `BLOCK` in shadow policy?
- Should S2 `block` be accepted directly for U0 answer-missing failures, or normalized to a non-destructive recovery route?
- Should local device mutations such as alarm/camera/brightness/wallpaper/flight-mode tasks be allowed in Phase 0 live shadow, reclassified, or hard-gated until a reset/cleanup protocol exists?
- Which fields are required before moving from dry-run routing to any live intervention proposal?
- Do we need to store verifier rationale in the offline JSONL for human audit?

Acceptance:

- No GUI run.
- No S2 live GUI connection.
- No U2.
- No token or endpoint written to repo.
- A short written route-policy note or code-level TODO exists before the next live-data run.
