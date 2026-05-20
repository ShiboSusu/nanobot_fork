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


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("positive integer required") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("positive integer required")
    return parsed


def load_jsonl_with_lines(path: Path) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL record") from exc
            if isinstance(record, dict):
                records.append((line_no, record))
    return records


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [record for _line_no, record in load_jsonl_with_lines(path)]


def apply_input_filters(
    records_with_lines: list[tuple[int, dict[str, Any]]],
    *,
    since_line: int | None,
    last: int | None,
) -> list[tuple[int, dict[str, Any]]]:
    if since_line is not None:
        records_with_lines = [(line_no, record) for line_no, record in records_with_lines if line_no >= since_line]
    if last is not None:
        records_with_lines = records_with_lines[-last:]
    return records_with_lines


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


def hard_gate_reason(record: dict[str, Any], step: dict[str, Any]) -> str | None:
    task_risk_level = str(record.get("task_risk_level") or record.get("risk_level") or "U0")
    if task_risk_level == "U2":
        return "blocked_u2"

    trace_quality = record.get("trace_quality")
    if isinstance(trace_quality, dict) and trace_quality.get("clean_for_signal_analysis") is False:
        quality_warning = trace_quality.get("quality_warning")
        if quality_warning in {"low_inner_coverage", "trace_missing"}:
            return str(quality_warning)
        return "trace_quality_unusable"

    trigger_features = nested_dict(step, "trigger_features")
    execution_state = nested_dict(trigger_features, "execution_state")
    environment_anomalies = record.get("environment_anomalies") if isinstance(record.get("environment_anomalies"), dict) else {}
    if execution_state.get("screenshot_capture_failure") or environment_anomalies.get("screenshot_capture_failure"):
        return "screenshot_capture_failure"
    if execution_state.get("secure_surface_suspected") or environment_anomalies.get("secure_surface_suspected"):
        return "secure_surface_suspected"
    return None


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


def without_observation_hard_gate_signals(trigger_features: dict[str, Any]) -> dict[str, Any]:
    execution_state = nested_dict(trigger_features, "execution_state")
    if not execution_state:
        return trigger_features
    sanitized_execution_state = dict(execution_state)
    sanitized_execution_state.pop("screenshot_capture_failure", None)
    sanitized_execution_state.pop("secure_surface_suspected", None)
    sanitized_trigger_features = dict(trigger_features)
    sanitized_trigger_features["execution_state"] = sanitized_execution_state
    return sanitized_trigger_features


def has_monitor_severity(trigger_features: dict[str, Any]) -> bool:
    return monitor_trigger(trigger_features) != "none"


def semantic_missing_answer_trigger(record: dict[str, Any], step: dict[str, Any]) -> bool:
    verifier = verifier_from_step(step)
    if verifier.get("reason_for_verification") == "semantic_missing_answer":
        return True

    action = step.get("action")
    if not isinstance(action, dict):
        return False
    return (
        record.get("semantic_task_success") is False
        and record.get("semantic_success_source") == "answer_presence_guard"
        and record.get("semantic_success_reason") == "missing_required_final_answer"
        and action.get("action_type") == "done"
        and action.get("status") == "success"
    )


def _route_step(record: dict[str, Any], step: dict[str, Any], *, apply_observation_hard_gates: bool) -> tuple[str, str, dict[str, Any]]:
    task_risk_level = str(record.get("task_risk_level") or record.get("risk_level") or "U0")
    trigger_features = nested_dict(step, "trigger_features")
    if not apply_observation_hard_gates:
        trigger_features = without_observation_hard_gate_signals(trigger_features)
    risk = nested_dict(trigger_features, "risk")
    execution_state = nested_dict(trigger_features, "execution_state")
    entropy = nested_dict(trigger_features, "entropy")
    self_report = nested_dict(trigger_features, "self_report")
    verifier = verifier_from_step(step)
    verifier_decision = verifier.get("decision") or verifier.get("verifier_decision")
    verifier_error = verifier.get("error") or verifier.get("verifier_error")
    verifier_safety_risk = verifier.get("safety_risk")
    verifier_requires_image_context = verifier.get("requires_image_context")
    step_predicted_risk = risk.get("step_predicted_risk_level")
    rule_based_risk = risk.get("rule_based_step_risk_level")
    trigger = monitor_trigger(trigger_features)
    if semantic_missing_answer_trigger(record, step):
        trigger = "semantic_missing_answer"
    normalized_verifier_decision = verifier_decision
    if (
        verifier_decision == "block"
        and trigger == "semantic_missing_answer"
        and task_risk_level == "U0"
        and verifier_safety_risk in {None, "U0"}
    ):
        normalized_verifier_decision = "replan"
    environment_anomalies = record.get("environment_anomalies") if isinstance(record.get("environment_anomalies"), dict) else {}

    if task_risk_level == "U2":
        route = "SKIP_UNSAFE"
        reason = "runner safety gate blocks U2 dry-run routing"
    elif apply_observation_hard_gates and trace_quality_is_unusable(record):
        route = "UNUSABLE_TRACE"
        reason = "trace_quality.clean_for_signal_analysis is false"
    elif apply_observation_hard_gates and (
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
    elif normalized_verifier_decision == "block":
        route = "BLOCK"
        reason = "verifier decision is block"
    elif normalized_verifier_decision == "ask_user":
        route = "ASK_USER"
        reason = "verifier decision is ask_user"
    elif normalized_verifier_decision == "recover" and verifier_requires_image_context is True:
        route = "SLOW"
        reason = "verifier recover requires image context before executable recovery"
    elif normalized_verifier_decision == "recover":
        route = "RECOVER"
        reason = "verifier decision is recover"
    elif normalized_verifier_decision == "replan":
        route = "SLOW"
        if verifier_decision == "block" and trigger == "semantic_missing_answer":
            reason = "verifier block normalized to slow route for U0 semantic missing answer"
        else:
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
    elif trigger == "semantic_missing_answer":
        route = "VERIFY"
        reason = "semantic guard found missing required final answer"
    else:
        route = "FAST"
        reason = "no dry-run controller trigger fired"

    runner_safety_gate = "blocked_u2" if task_risk_level == "U2" else f"allowed_{task_risk_level.lower()}"
    inputs = {
        "monitor_trigger": trigger,
        "verifier_decision": verifier_decision,
        "verifier_safety_risk": verifier_safety_risk,
        "verifier_requires_image_context": verifier_requires_image_context,
        "normalized_verifier_decision": normalized_verifier_decision,
        "verifier_error": verifier_error,
        "runner_safety_gate": runner_safety_gate,
    }
    return route, reason, inputs


def route_step(record: dict[str, Any], step: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    return _route_step(record, step, apply_observation_hard_gates=True)


def route_step_diagnostics(record: dict[str, Any], step: dict[str, Any]) -> dict[str, Any]:
    route, reason, inputs = route_step(record, step)
    raw_route, raw_reason, raw_inputs = _route_step(record, step, apply_observation_hard_gates=False)
    return {
        "route": route,
        "reason": reason,
        "inputs": inputs,
        "raw_monitor_route": raw_route,
        "raw_monitor_reason": raw_reason,
        "raw_monitor_inputs": raw_inputs,
        "hard_gate_reason": hard_gate_reason(record, step),
    }


def build_controller_record(record: dict[str, Any], step: dict[str, Any]) -> dict[str, Any]:
    diagnostics = route_step_diagnostics(record, step)
    task_risk_level = str(record.get("task_risk_level") or record.get("risk_level") or "U0")
    output = {
        "schema_version": "phase0_controller_dry_run_v1",
        "task_id": record.get("task_id"),
        "task_risk_level": task_risk_level,
        "step_index": step.get("step_index"),
        "controller": {
            "enabled": False,
            "dry_run": True,
            **diagnostics,
        },
    }
    return output


def build_outputs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for record in records:
        for step in iter_steps(record):
            outputs.append(build_controller_record(record, step))
    return outputs


def print_summary(
    outputs: list[dict[str, Any]],
    source_records: list[dict[str, Any]],
    *,
    input_line_range: list[int] | None,
    input_filters: dict[str, int | None],
) -> None:
    routes = Counter(output.get("controller", {}).get("route") for output in outputs)
    image_context_recovery_guard_count = 0
    for output in outputs:
        controller = output.get("controller") if isinstance(output.get("controller"), dict) else {}
        inputs = controller.get("inputs") if isinstance(controller.get("inputs"), dict) else {}
        if (
            controller.get("route") == "SLOW"
            and inputs.get("verifier_decision") == "recover"
            and inputs.get("verifier_requires_image_context") is True
        ):
            image_context_recovery_guard_count += 1
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
        "image_context_recovery_guard_count": image_context_recovery_guard_count,
        "verifier_decision_distribution": dict(sorted(verifier_decisions.items())),
        "verifier_error_count": verifier_error_count,
        "input_line_range": input_line_range,
        "input_filters": input_filters,
    }
    print("summary:", json.dumps(summary, ensure_ascii=False, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Phase 0 observable or S2 verifier JSONL input")
    parser.add_argument("--output", required=True, type=Path, help="Ignored JSONL path for dry-run controller output")
    parser.add_argument("--summary", action="store_true", help="Print dry-run routing summary")
    parser.add_argument("--input-last", type=positive_int, help="Route only the last N valid JSONL records after other filters")
    parser.add_argument("--input-since-line", type=positive_int, help="Route valid records from this physical JSONL line onward")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not git_ignores_path(args.output):
        print(f"ERROR: output path is not gitignored: {args.output}")
        return 2
    try:
        records_with_lines = load_jsonl_with_lines(args.input)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    records_with_lines = apply_input_filters(
        records_with_lines,
        since_line=args.input_since_line,
        last=args.input_last,
    )
    source_records = [record for _line_no, record in records_with_lines]
    outputs = build_outputs(source_records)
    append_jsonl(args.output, outputs)
    if args.summary:
        input_line_range = [records_with_lines[0][0], records_with_lines[-1][0]] if records_with_lines else None
        print_summary(
            outputs,
            source_records,
            input_line_range=input_line_range,
            input_filters={"since_line": args.input_since_line, "last": args.input_last},
        )
    print(f"wrote_records: {len(outputs)}")
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
