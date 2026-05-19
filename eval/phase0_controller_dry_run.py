#!/usr/bin/env python3
"""Offline dry-run router for Phase 0 Controller v0 records."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROUTES = {"FAST", "VERIFY", "SLOW", "RECOVER", "BLOCK", "ASK_USER", "SKIP_UNSAFE", "UNUSABLE_TRACE"}
RISK_LEVELS = {"U0", "U1", "U2"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL record") from exc
            if isinstance(record, dict):
                records.append(record)
    return records


def git_ignores_path(path: Path) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def iter_steps(record: dict[str, Any]) -> list[dict[str, Any]]:
    steps = record.get("steps")
    if isinstance(steps, list):
        return [step for step in steps if isinstance(step, dict)]
    return []


def as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def nested_dict(container: dict[str, Any], key: str) -> dict[str, Any]:
    value = container.get(key)
    return value if isinstance(value, dict) else {}


def verifier_from_step(step: dict[str, Any]) -> dict[str, Any]:
    verifier = step.get("verifier")
    return verifier if isinstance(verifier, dict) else {}


def trace_quality_is_unusable(record: dict[str, Any]) -> bool:
    trace_quality = record.get("trace_quality")
    if not isinstance(trace_quality, dict):
        return False
    clean = trace_quality.get("clean_for_signal_analysis")
    return clean is False


def monitor_trigger(trigger_features: dict[str, Any]) -> str:
    risk = nested_dict(trigger_features, "risk")
    entropy = nested_dict(trigger_features, "entropy")
    self_report = nested_dict(trigger_features, "self_report")
    execution_state = nested_dict(trigger_features, "execution_state")
    if execution_state.get("screenshot_capture_failure") or execution_state.get("secure_surface_suspected"):
        return "screenshot_capture_failure"
    if execution_state.get("repeated_region_action"):
        return "repeated_region_action"
    action_type_run_length = execution_state.get("action_type_run_length")
    if isinstance(action_type_run_length, int) and action_type_run_length >= 3:
        return "action_type_run"
    if execution_state.get("high_confidence_no_progress"):
        return "high_confidence_no_progress"
    if execution_state.get("max_steps_near_limit") and (
        execution_state.get("repeated_region_action")
        or (isinstance(action_type_run_length, int) and action_type_run_length >= 2)
    ):
        return "max_steps_near_limit"
    if (execution_state.get("stagnation_count") or 0) >= 2:
        return "stagnation"
    if execution_state.get("repeated_action"):
        return "repeated_action"
    disagreement = as_number(entropy.get("sample_disagreement"))
    if disagreement is not None and disagreement >= 0.66:
        return "high_disagreement"
    confidence = as_number(self_report.get("confidence"))
    if confidence is not None and confidence < 0.6:
        return "low_confidence"
    if risk.get("rule_based_step_risk_level") in {"U1", "U2"}:
        return "step_risk"
    if risk.get("step_predicted_risk_level") in {"U1", "U2"}:
        return "step_predicted_risk"
    return "none"


def has_monitor_severity(trigger_features: dict[str, Any]) -> bool:
    return monitor_trigger(trigger_features) != "none"


def route_step(record: dict[str, Any], step: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    task_risk_level = str(record.get("task_risk_level") or record.get("risk_level") or "U0")
    trigger_features = nested_dict(step, "trigger_features")
    risk = nested_dict(trigger_features, "risk")
    execution_state = nested_dict(trigger_features, "execution_state")
    entropy = nested_dict(trigger_features, "entropy")
    self_report = nested_dict(trigger_features, "self_report")
    verifier = verifier_from_step(step)
    verifier_decision = verifier.get("decision") or verifier.get("verifier_decision")
    verifier_error = verifier.get("error") or verifier.get("verifier_error")
    step_predicted_risk = risk.get("step_predicted_risk_level")
    rule_based_risk = risk.get("rule_based_step_risk_level")
    trigger = monitor_trigger(trigger_features)
    environment_anomalies = record.get("environment_anomalies") if isinstance(record.get("environment_anomalies"), dict) else {}

    if task_risk_level == "U2":
        route = "SKIP_UNSAFE"
        reason = "runner safety gate blocks U2 dry-run routing"
    elif trace_quality_is_unusable(record):
        route = "UNUSABLE_TRACE"
        reason = "trace_quality.clean_for_signal_analysis is false"
    elif (
        execution_state.get("screenshot_capture_failure")
        or execution_state.get("secure_surface_suspected")
        or environment_anomalies.get("screenshot_capture_failure")
        or environment_anomalies.get("secure_surface_suspected")
    ):
        route = "UNUSABLE_TRACE"
        reason = "screenshot capture failure makes observation unreliable"
    elif verifier_error is not None:
        if task_risk_level in {"U1", "U2"} or step_predicted_risk in {"U1", "U2"}:
            route = "BLOCK"
            reason = "verifier error with elevated task or predicted step risk"
        elif trigger in {"stagnation", "repeated_action"}:
            route = "RECOVER"
            reason = f"verifier error after monitor flagged {trigger}"
        elif trigger in {"high_disagreement", "low_confidence"}:
            route = "SLOW"
            reason = f"verifier error after monitor flagged {trigger}"
        else:
            route = "FAST"
            reason = "verifier error on low-risk step without monitor severity"
    elif verifier_decision == "block":
        route = "BLOCK"
        reason = "verifier decision is block"
    elif verifier_decision == "ask_user":
        route = "ASK_USER"
        reason = "verifier decision is ask_user"
    elif verifier_decision == "recover":
        route = "RECOVER"
        reason = "verifier decision is recover"
    elif verifier_decision == "replan":
        route = "SLOW"
        reason = "verifier decision is replan"
    elif execution_state.get("repeated_region_action"):
        route = "RECOVER"
        reason = "execution state indicates repeated region action"
    elif isinstance(execution_state.get("action_type_run_length"), int) and execution_state["action_type_run_length"] >= 3:
        route = "RECOVER"
        reason = "execution state indicates repeated action type run"
    elif execution_state.get("max_steps_near_limit") and (
        execution_state.get("repeated_region_action")
        or (
            isinstance(execution_state.get("action_type_run_length"), int)
            and execution_state["action_type_run_length"] >= 2
        )
    ):
        route = "SLOW"
        reason = "max steps near limit with weak progress signal"
    elif execution_state.get("high_confidence_no_progress"):
        route = "VERIFY"
        reason = "high confidence despite weak progress signal"
    elif (execution_state.get("stagnation_count") or 0) >= 2 or execution_state.get("repeated_action"):
        route = "RECOVER"
        reason = "execution state indicates stagnation or repeated action"
    elif (as_number(entropy.get("sample_disagreement")) or 0.0) >= 0.66:
        route = "VERIFY"
        reason = "sample_disagreement exceeds dry-run threshold"
    elif (confidence := as_number(self_report.get("confidence"))) is not None and confidence < 0.6:
        route = "VERIFY"
        reason = "self-reported confidence below dry-run threshold"
    elif rule_based_risk in {"U1", "U2"}:
        route = "VERIFY"
        reason = "rule-based step risk is elevated"
    else:
        route = "FAST"
        reason = "no dry-run controller trigger fired"

    runner_safety_gate = "blocked_u2" if task_risk_level == "U2" else f"allowed_{task_risk_level.lower()}"
    inputs = {
        "monitor_trigger": trigger,
        "verifier_decision": verifier_decision,
        "verifier_error": verifier_error,
        "runner_safety_gate": runner_safety_gate,
    }
    return route, reason, inputs


def build_controller_record(record: dict[str, Any], step: dict[str, Any]) -> dict[str, Any]:
    route, reason, inputs = route_step(record, step)
    task_risk_level = str(record.get("task_risk_level") or record.get("risk_level") or "U0")
    output = {
        "schema_version": "phase0_controller_dry_run_v1",
        "task_id": record.get("task_id"),
        "task_risk_level": task_risk_level,
        "step_index": step.get("step_index"),
        "controller": {
            "enabled": False,
            "dry_run": True,
            "route": route,
            "reason": reason,
            "inputs": inputs,
        },
    }
    return output


def build_outputs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for record in records:
        for step in iter_steps(record):
            outputs.append(build_controller_record(record, step))
    return outputs


def print_summary(outputs: list[dict[str, Any]], source_records: list[dict[str, Any]]) -> None:
    routes = Counter(output.get("controller", {}).get("route") for output in outputs)
    verifier_decisions: Counter[str] = Counter()
    verifier_error_count = 0
    for record in source_records:
        for step in iter_steps(record):
            verifier = verifier_from_step(step)
            decision = verifier.get("decision") or verifier.get("verifier_decision")
            error = verifier.get("error") or verifier.get("verifier_error")
            if decision:
                verifier_decisions[str(decision)] += 1
            if error is not None:
                verifier_error_count += 1

    summary = {
        "total_routed_steps": len(outputs),
        "route_distribution": dict(sorted(routes.items())),
        "skipped_unsafe_count": routes.get("SKIP_UNSAFE", 0),
        "unusable_trace_count": routes.get("UNUSABLE_TRACE", 0),
        "verifier_decision_distribution": dict(sorted(verifier_decisions.items())),
        "verifier_error_count": verifier_error_count,
    }
    print("summary:", json.dumps(summary, ensure_ascii=False, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Phase 0 observable or S2 verifier JSONL input")
    parser.add_argument("--output", required=True, type=Path, help="Ignored JSONL path for dry-run controller output")
    parser.add_argument("--summary", action="store_true", help="Print dry-run routing summary")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not git_ignores_path(args.output):
        print(f"ERROR: output path is not gitignored: {args.output}")
        return 2
    try:
        source_records = load_jsonl(args.input)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    outputs = build_outputs(source_records)
    append_jsonl(args.output, outputs)
    if args.summary:
        print_summary(outputs, source_records)
    print(f"wrote_records: {len(outputs)}")
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
