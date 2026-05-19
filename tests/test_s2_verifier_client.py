from __future__ import annotations

import json

import pytest

from eval import s2_verifier_client as s2


def test_load_record_at_line_reads_valid_dict_on_physical_line(tmp_path) -> None:
    input_path = tmp_path / "phase0.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "line-1", "task_risk_level": "U0"}),
                "{not-json",
                json.dumps({"task_id": "line-3", "task_risk_level": "U1"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    record = s2.load_record_at_line(input_path, 3)

    assert record == {"task_id": "line-3", "task_risk_level": "U1"}


def test_run_offline_smoke_rejects_invalid_record_line(tmp_path, monkeypatch, capsys) -> None:
    input_path = tmp_path / "phase0.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "line-1", "task_risk_level": "U0"}),
                "{not-json",
                json.dumps({"task_id": "line-3", "task_risk_level": "U1"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("MA_INTRANET_URL", raising=False)
    monkeypatch.delenv("MA_TOKEN", raising=False)

    exit_code = s2.run_offline_smoke(input_path, latest=False, record_line=2, step_index=None, timeout_s=1.0, write_output=None)

    assert exit_code == 2
    assert "ERROR: invalid JSON on physical line 2" in capsys.readouterr().out


def test_run_offline_smoke_uses_record_line_without_env(tmp_path, monkeypatch, capsys) -> None:
    input_path = tmp_path / "phase0.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "line-1", "task_risk_level": "U0"}),
                json.dumps(
                    {
                        "task_id": "line-2",
                        "task_risk_level": "U1",
                        "steps": [{"step_index": 0, "action": {"action_type": "done"}}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("MA_INTRANET_URL", raising=False)
    monkeypatch.delenv("MA_TOKEN", raising=False)

    exit_code = s2.run_offline_smoke(input_path, latest=False, record_line=2, step_index=None, timeout_s=1.0, write_output=None)

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "selected_task_id: line-2" in stdout
    assert f"record_source: {input_path}:2" in stdout
    assert '"verifier_error": "missing_env:MA_INTRANET_URL,MA_TOKEN"' in stdout


def test_build_request_from_phase0_record_includes_semantic_missing_answer_diagnostics() -> None:
    record = {
        "task_id": "ChromeSearchBeijingWeatherTask",
        "instruction": "Use Chrome to search for Beijing highest temperature today. ONLY give a integer number.",
        "task_risk_level": "U0",
        "clean_success": False,
        "answer_required": True,
        "final_answer_present": False,
        "semantic_task_success": False,
        "semantic_success_source": "answer_presence_guard",
        "semantic_success_reason": "missing_required_final_answer",
        "trace_quality": {"clean_for_signal_analysis": True},
        "steps": [
            {
                "step_index": 5,
                "model_output": "Done.",
                "action": {"action_type": "done", "status": "success", "answer": ""},
                "trigger_features": {
                    "self_report": {"confidence": 0.95},
                    "risk": {"task_risk_level": "U0", "rule_based_step_risk_level": "U0"},
                    "execution_state": {"stagnation_count": 0},
                },
                "controller": {
                    "route": "VERIFY",
                    "reason": "semantic guard found missing required final answer",
                    "inputs": {"monitor_trigger": "semantic_missing_answer"},
                    "raw_monitor_route": "VERIFY",
                    "raw_monitor_inputs": {"monitor_trigger": "semantic_missing_answer"},
                    "hard_gate_reason": None,
                },
            }
        ],
    }

    request = s2.build_request_from_phase0_record(record)

    assert request.reason_for_verification == "semantic_missing_answer"
    assert request.monitor_signals["semantic_outcome"] == {
        "answer_required": True,
        "final_answer_present": False,
        "semantic_task_success": False,
        "semantic_success_source": "answer_presence_guard",
        "semantic_success_reason": "missing_required_final_answer",
    }
    assert request.monitor_signals["controller"] == {
        "route": "VERIFY",
        "raw_monitor_route": "VERIFY",
        "monitor_trigger": "semantic_missing_answer",
        "raw_monitor_trigger": "semantic_missing_answer",
        "hard_gate_reason": None,
        "reason": "semantic guard found missing required final answer",
    }
    assert "missing_required_final_answer" in request.current_observation["text_summary"]
    assert "final_answer_present: False" in request.current_observation["text_summary"]


def test_build_request_from_phase0_record_keeps_stagnation_reason_and_controller_diagnostics() -> None:
    record = {
        "task_id": "AdjustBrightnessMaximumTask",
        "instruction": "Set brightness to maximum.",
        "task_risk_level": "U1",
        "clean_success": False,
        "answer_required": False,
        "final_answer_present": False,
        "semantic_task_success": None,
        "semantic_success_source": "none",
        "semantic_success_reason": "not_evaluated",
        "termination_reason": "stagnation_detected",
        "steps": [
            {
                "step_index": 14,
                "action": {"action_type": "tap", "x": 164.0, "y": 476.0},
                "trigger_features": {
                    "self_report": {"confidence": 0.95},
                    "risk": {"task_risk_level": "U1", "rule_based_step_risk_level": "U1"},
                    "execution_state": {"stagnation_count": 4},
                },
                "controller": {
                    "route": "RECOVER",
                    "reason": "execution state indicates repeated region action",
                    "inputs": {"monitor_trigger": "repeated_region_action"},
                    "raw_monitor_route": "RECOVER",
                    "raw_monitor_inputs": {"monitor_trigger": "repeated_region_action"},
                    "hard_gate_reason": None,
                },
            }
        ],
    }

    request = s2.build_request_from_phase0_record(record)

    assert request.reason_for_verification == "stagnation"
    assert request.monitor_signals["controller"] == {
        "route": "RECOVER",
        "raw_monitor_route": "RECOVER",
        "monitor_trigger": "repeated_region_action",
        "raw_monitor_trigger": "repeated_region_action",
        "hard_gate_reason": None,
        "reason": "execution state indicates repeated region action",
    }


def test_parser_rejects_latest_with_record_line(capsys) -> None:
    parser = s2.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--offline-smoke", "--input", "phase0.jsonl", "--latest", "--record-line", "3"])

    stderr = capsys.readouterr().err
    assert "not allowed with argument" in stderr
