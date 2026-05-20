from __future__ import annotations

import json
import sys

import pytest

from eval import phase0_controller_dry_run as dry_run


def verifier_record(task_id: str, decision: str | None = None) -> dict[str, object]:
    verifier = {"called": True, "decision": decision, "error": None}
    return {
        "task_id": task_id,
        "task_risk_level": "U0",
        "steps": [
            {
                "step_index": 1,
                "trigger_features": {
                    "risk": {"rule_based_step_risk_level": "U0", "step_predicted_risk_level": "U0"},
                    "execution_state": {},
                },
                "verifier": verifier,
            }
        ],
    }


def s2_offline_semantic_missing_answer_record(
    *,
    task_id: str = "RecentTotalExpenseTask",
    task_risk_level: str = "U0",
    decision: str = "block",
    safety_risk: str | None = "U0",
) -> dict[str, object]:
    return {
        "features": {"controller": False, "runtime_signal": True, "s2_verifier": True},
        "schema_version": "phase0_observable_v2",
        "selected_step_index": 1,
        "source_record": {
            "input_path": "eval/phase0_observable_results.jsonl",
            "line_number": 31,
            "selection": "record_line",
        },
        "task_id": task_id,
        "task_risk_level": task_risk_level,
        "task_source": "dataset",
        "steps": [
            {
                "outcome_proxies": {
                    "action_parse_failure": False,
                    "execution_error": False,
                    "judge_derived_not_advanced": None,
                    "post_action_no_observable_change": None,
                },
                "step_index": 1,
                "trigger_features": {
                    "execution_state": {
                        "action_type_run_length": 1,
                        "high_confidence_no_progress": False,
                        "repeated_action": False,
                        "repeated_region_action": False,
                        "screenshot_capture_failure": False,
                        "secure_surface_suspected": False,
                        "stagnation_count": 0,
                    },
                    "risk": {
                        "rule_based_step_risk_level": "U0",
                        "step_predicted_risk_level": None,
                        "task_risk_level": task_risk_level,
                    },
                    "self_report": {
                        "confidence": None,
                        "need_slow_planner": None,
                        "runtime_signal_parse_error": None,
                        "uncertainty_reason": None,
                    },
                },
                "verifier": {
                    "allowed_to_execute_s1_action": False,
                    "called": True,
                    "decision": decision,
                    "error": None,
                    "failure_risk": "high",
                    "reason_for_verification": "semantic_missing_answer",
                    "safety_risk": safety_risk,
                },
            }
        ],
    }


def semantic_missing_answer_record(decision: str = "block", safety_risk: str | None = "U0") -> dict[str, object]:
    return {
        "task_id": "RecentTotalExpenseTask",
        "task_risk_level": "U0",
        "semantic_task_success": False,
        "semantic_success_source": "answer_presence_guard",
        "semantic_success_reason": "missing_required_final_answer",
        "trace_quality": {"clean_for_signal_analysis": True},
        "steps": [
            {
                "step_index": 1,
                "action": {"action_type": "done", "status": "success"},
                "trigger_features": {
                    "risk": {"rule_based_step_risk_level": "U0", "step_predicted_risk_level": "U0"},
                    "execution_state": {},
                },
                "verifier": {
                    "called": True,
                    "decision": decision,
                    "error": None,
                    "safety_risk": safety_risk,
                    "failure_risk": "high",
                    "allowed_to_execute_s1_action": False,
                },
            }
        ],
    }


def run_main(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    input_path,
    *extra_args: str,
) -> tuple[list[dict[str, object]], dict[str, object], str]:
    output_path = tmp_path / "phase0_controller_dry_run.jsonl"
    monkeypatch.setattr(dry_run, "git_ignores_path", lambda path: True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase0_controller_dry_run.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--summary",
            *extra_args,
        ],
    )

    exit_code = dry_run.main()

    stdout = capsys.readouterr().out
    summary_line = next(line for line in stdout.splitlines() if line.startswith("summary: "))
    outputs = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert exit_code == 0
    return outputs, json.loads(summary_line.removeprefix("summary: ")), stdout


def test_input_last_routes_only_last_valid_record(tmp_path, monkeypatch, capsys) -> None:
    input_path = tmp_path / "phase0_s2_verifier_offline.jsonl"
    records = [
        verifier_record("line1", "pass"),
        verifier_record("line2", "replan"),
        verifier_record("line3", "block"),
    ]
    input_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    outputs, summary, stdout = run_main(tmp_path, monkeypatch, capsys, input_path, "--input-last", "1")

    assert [output["task_id"] for output in outputs] == ["line3"]
    assert outputs[0]["controller"]["route"] == "BLOCK"
    assert summary["total_routed_steps"] == 1
    assert summary["route_distribution"] == {"BLOCK": 1}
    assert summary["input_line_range"] == [3, 3]
    assert summary["input_filters"] == {"last": 1, "since_line": None}
    assert "wrote_records: 1" in stdout


def test_input_since_line_filters_by_physical_jsonl_line(tmp_path, monkeypatch, capsys) -> None:
    input_path = tmp_path / "phase0_s2_verifier_offline.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(verifier_record("line1", "pass")),
                "",
                json.dumps(verifier_record("line3", "replan")),
                json.dumps(verifier_record("line4", "block")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    outputs, summary, _stdout = run_main(tmp_path, monkeypatch, capsys, input_path, "--input-since-line", "3")

    assert [output["task_id"] for output in outputs] == ["line3", "line4"]
    assert [output["controller"]["route"] for output in outputs] == ["SLOW", "BLOCK"]
    assert summary["route_distribution"] == {"BLOCK": 1, "SLOW": 1}
    assert summary["input_line_range"] == [3, 4]
    assert summary["input_filters"] == {"last": None, "since_line": 3}


def test_input_filters_apply_since_line_before_last(tmp_path, monkeypatch, capsys) -> None:
    input_path = tmp_path / "phase0_s2_verifier_offline.jsonl"
    records = [
        verifier_record("line1", "pass"),
        verifier_record("line2", "replan"),
        verifier_record("line3", "block"),
    ]
    input_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    outputs, summary, _stdout = run_main(
        tmp_path,
        monkeypatch,
        capsys,
        input_path,
        "--input-since-line",
        "1",
        "--input-last",
        "1",
    )

    assert [output["task_id"] for output in outputs] == ["line3"]
    assert outputs[0]["controller"]["route"] == "BLOCK"
    assert summary["input_line_range"] == [3, 3]
    assert summary["input_filters"] == {"last": 1, "since_line": 1}


def test_input_filter_cli_values_must_be_positive(capsys) -> None:
    parser = dry_run.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--input", "input.jsonl", "--output", "output.jsonl", "--input-last", "0"])

    stderr = capsys.readouterr().err
    assert "positive integer" in stderr


def test_u0_observable_semantic_missing_answer_block_normalizes_to_slow() -> None:
    record = semantic_missing_answer_record(decision="block", safety_risk="U0")
    step = record["steps"][0]

    route, reason, inputs = dry_run.route_step(record, step)

    assert route == "SLOW"
    assert "semantic missing answer" in reason
    assert inputs["monitor_trigger"] == "semantic_missing_answer"
    assert inputs["verifier_decision"] == "block"
    assert inputs["verifier_safety_risk"] == "U0"
    assert inputs["normalized_verifier_decision"] == "replan"


def test_u0_s2_offline_semantic_missing_answer_block_normalizes_to_slow() -> None:
    record = s2_offline_semantic_missing_answer_record(decision="block", safety_risk="U0")
    step = record["steps"][0]

    route, reason, inputs = dry_run.route_step(record, step)

    assert route == "SLOW"
    assert "semantic missing answer" in reason
    assert inputs["monitor_trigger"] == "semantic_missing_answer"
    assert inputs["verifier_decision"] == "block"
    assert inputs["verifier_safety_risk"] == "U0"
    assert inputs["normalized_verifier_decision"] == "replan"


def test_s2_offline_semantic_missing_answer_block_without_safety_risk_normalizes_to_slow() -> None:
    record = s2_offline_semantic_missing_answer_record(decision="block", safety_risk=None)
    step = record["steps"][0]

    route, _reason, inputs = dry_run.route_step(record, step)

    assert route == "SLOW"
    assert inputs["monitor_trigger"] == "semantic_missing_answer"
    assert inputs["normalized_verifier_decision"] == "replan"


def test_u2_verifier_block_remains_safety_blocked() -> None:
    record = s2_offline_semantic_missing_answer_record(
        task_id="UnsafeTask",
        task_risk_level="U2",
        decision="block",
        safety_risk="U2",
    )
    step = record["steps"][0]

    route, reason, inputs = dry_run.route_step(record, step)

    assert route == "SKIP_UNSAFE"
    assert "blocks U2" in reason
    assert inputs["verifier_decision"] == "block"
    assert inputs["verifier_safety_risk"] == "U2"
    assert inputs["normalized_verifier_decision"] == "block"


def test_s2_offline_scoped_summary_normalizes_u0_missing_answer_block(tmp_path, monkeypatch, capsys) -> None:
    input_path = tmp_path / "phase0_s2_verifier_offline.jsonl"
    records = [
        verifier_record("line29", "replan"),
        s2_offline_semantic_missing_answer_record(task_id="line30", decision="replan"),
        s2_offline_semantic_missing_answer_record(task_id="line31", decision="block"),
    ]
    input_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    outputs, summary, _stdout = run_main(tmp_path, monkeypatch, capsys, input_path, "--input-last", "3")

    assert [output["controller"]["route"] for output in outputs] == ["SLOW", "SLOW", "SLOW"]
    assert summary["route_distribution"] == {"SLOW": 3}
    assert summary["verifier_decision_distribution"] == {"block": 1, "replan": 2}
    assert outputs[2]["controller"]["inputs"]["normalized_verifier_decision"] == "replan"


def test_recover_requiring_image_context_stays_non_executable() -> None:
    record = verifier_record("visual_recover", "recover")
    step = record["steps"][0]
    step["verifier"]["safety_risk"] = "U0"
    step["verifier"]["requires_image_context"] = True

    route, reason, inputs = dry_run.route_step(record, step)

    assert route == "SLOW"
    assert "image context" in reason
    assert inputs["verifier_decision"] == "recover"
    assert inputs["verifier_requires_image_context"] is True


def test_summary_reports_image_context_recovery_guard(tmp_path, monkeypatch, capsys) -> None:
    input_path = tmp_path / "phase0_s2_verifier_offline.jsonl"
    record = verifier_record("visual_recover", "recover")
    record["steps"][0]["verifier"]["safety_risk"] = "U0"
    record["steps"][0]["verifier"]["requires_image_context"] = True
    input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    _outputs, summary, _stdout = run_main(tmp_path, monkeypatch, capsys, input_path)

    assert summary["route_distribution"] == {"SLOW": 1}
    assert summary["image_context_recovery_guard_count"] == 1
