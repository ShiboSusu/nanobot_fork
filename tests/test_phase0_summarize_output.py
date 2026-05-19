from __future__ import annotations

import json

import pytest

from eval.phase0_observable_signal_pilot import build_parser, summarize_output


def test_summarize_output_prints_controller_shadow_step_distributions(tmp_path, capsys) -> None:
    output_path = tmp_path / "phase0.jsonl"
    record = {
        "task_risk_level": "U0",
        "termination_reason": "completed",
        "clean_success": True,
        "semantic_task_success": False,
        "trace_quality": {"inner_coverage": 1.0, "quality_warning": None, "clean_for_signal_analysis": True},
        "steps": [
            {
                "controller": {
                    "route": "FAST",
                    "raw_monitor_route": "FAST",
                    "hard_gate_reason": None,
                    "inputs": {"monitor_trigger": None},
                    "raw_monitor_inputs": {"monitor_trigger": None},
                }
            },
            {
                "controller": {
                    "route": "VERIFY",
                    "raw_monitor_route": "VERIFY",
                    "hard_gate_reason": None,
                    "inputs": {"monitor_trigger": "semantic_missing_answer"},
                    "raw_monitor_inputs": {"monitor_trigger": "semantic_missing_answer"},
                }
            },
            {
                "controller": {
                    "route": "UNUSABLE_TRACE",
                    "raw_monitor_route": "VERIFY",
                    "hard_gate_reason": "low_inner_coverage",
                    "inputs": {"monitor_trigger": "semantic_missing_answer"},
                    "raw_monitor_inputs": {"monitor_trigger": "semantic_missing_answer"},
                }
            },
        ],
    }
    output_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    summarize_output(output_path)

    stdout = capsys.readouterr().out
    assert "Controller route distribution: {'FAST': 1, 'VERIFY': 1, 'UNUSABLE_TRACE': 1}" in stdout
    assert "Raw monitor route distribution: {'FAST': 1, 'VERIFY': 2}" in stdout
    assert "Hard gate reason distribution: {'low_inner_coverage': 1}" in stdout
    assert "Monitor trigger distribution: {'semantic_missing_answer': 2}" in stdout
    assert "Raw monitor trigger distribution: {'semantic_missing_answer': 2}" in stdout


def test_summarize_output_prints_monitor_feature_distributions(tmp_path, capsys) -> None:
    output_path = tmp_path / "phase0.jsonl"
    records = [
        {
            "task_risk_level": "U0",
            "termination_reason": "max_steps_exceeded",
            "clean_success": False,
            "semantic_task_success": False,
            "environment_anomalies": {"screenshot_capture_failure": False},
            "steps": [
                {
                    "trigger_features": {
                        "self_report": {"confidence": 0.95},
                        "execution_state": {
                            "repeated_action": True,
                            "repeated_region_action": True,
                            "high_confidence_no_progress": True,
                            "coordinate_bucket_repeat": True,
                            "screen_region_repeat": True,
                        },
                    },
                }
            ],
        },
        {
            "task_risk_level": "U1",
            "termination_reason": "runner_error",
            "clean_success": True,
            "semantic_task_success": True,
            "environment_anomalies": {"screenshot_capture_failure": True},
            "steps": [
                {
                    "trigger_features": {
                        "self_report": {"confidence": 0.4},
                        "execution_state": {
                            "screenshot_capture_failure": True,
                            "secure_surface_suspected": True,
                            "max_steps_near_limit": True,
                        },
                    },
                }
            ],
        },
    ]
    output_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    summarize_output(output_path)

    stdout = capsys.readouterr().out
    assert (
        "Execution monitor feature distribution: "
        "{'repeated_action': 1, 'repeated_region_action': 1, "
        "'high_confidence_no_progress': 1, 'screenshot_capture_failure': 1, "
        "'secure_surface_suspected': 1, 'max_steps_near_limit': 1, "
        "'coordinate_bucket_repeat': 1, 'screen_region_repeat': 1}"
    ) in stdout
    assert "High confidence failed record count: 1" in stdout
    assert "Screenshot failure record count: 1" in stdout


def test_summarize_output_filters_by_physical_since_line(tmp_path, capsys) -> None:
    output_path = tmp_path / "phase0.jsonl"
    records = [
        {"task_risk_level": "U0", "termination_reason": "completed", "clean_success": True},
        {"task_risk_level": "U1", "termination_reason": "runner_error", "clean_success": False},
        {"task_risk_level": "U1", "termination_reason": "completed", "clean_success": True},
    ]
    output_path.write_text(
        "\n".join(
            [
                json.dumps(records[0]),
                "{not-json",
                json.dumps(records[1]),
                json.dumps(records[2]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summarize_output(output_path, summarize_since_line=3)

    stdout = capsys.readouterr().out
    assert "Total records: 2" in stdout
    assert "Summary source line range: 3-4" in stdout
    assert "Summary filters: since_line=3 last=none" in stdout
    assert "Risk distribution: {'U1': 2}" in stdout


def test_summarize_output_filters_to_last_valid_records(tmp_path, capsys) -> None:
    output_path = tmp_path / "phase0.jsonl"
    records = [
        {"task_risk_level": "U0", "termination_reason": "completed", "clean_success": True},
        {"task_risk_level": "U1", "termination_reason": "runner_error", "clean_success": False},
        {"task_risk_level": "U2", "termination_reason": "completed", "clean_success": True},
    ]
    output_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    summarize_output(output_path, summarize_last=1)

    stdout = capsys.readouterr().out
    assert "Total records: 1" in stdout
    assert "Summary source line range: 3-3" in stdout
    assert "Summary filters: since_line=none last=1" in stdout
    assert "Risk distribution: {'U2': 1}" in stdout


def test_summarize_output_applies_since_line_before_last(tmp_path, capsys) -> None:
    output_path = tmp_path / "phase0.jsonl"
    records = [
        {"task_risk_level": "U0", "termination_reason": "completed", "clean_success": True},
        {"task_risk_level": "U1", "termination_reason": "runner_error", "clean_success": False},
        {"task_risk_level": "U2", "termination_reason": "completed", "clean_success": True},
    ]
    output_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    summarize_output(output_path, summarize_since_line=1, summarize_last=1)

    stdout = capsys.readouterr().out
    assert "Total records: 1" in stdout
    assert "Summary source line range: 3-3" in stdout
    assert "Summary filters: since_line=1 last=1" in stdout
    assert "Risk distribution: {'U2': 1}" in stdout


def test_summarize_filter_cli_values_must_be_positive(capsys) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--summarize-output", "phase0.jsonl", "--summarize-last", "0"])

    stderr = capsys.readouterr().err
    assert "positive integer" in stderr
