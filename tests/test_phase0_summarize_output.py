from __future__ import annotations

import json

from eval.phase0_observable_signal_pilot import summarize_output


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
