from __future__ import annotations

import json
import sys

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


def verifier_output_record(
    *,
    task_risk_level: str,
    called: bool,
    decision: str,
    safety_risk: str,
    failure_risk: str,
    reason_for_verification: str,
    allowed: bool,
    latency_s: float,
    prompt_tokens: int,
    completion_tokens: int,
    error: str | None = None,
) -> dict:
    return {
        "task_risk_level": task_risk_level,
        "steps": [
            {
                "verifier": {
                    "called": called,
                    "decision": decision,
                    "safety_risk": safety_risk,
                    "failure_risk": failure_risk,
                    "reason_for_verification": reason_for_verification,
                    "allowed_to_execute_s1_action": allowed,
                    "latency_s": latency_s,
                    "token_usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    },
                    "error": error,
                }
            }
        ],
    }


def write_verifier_output_jsonl(path) -> None:
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    verifier_output_record(
                        task_risk_level="U0",
                        called=True,
                        decision="pass",
                        safety_risk="U0",
                        failure_risk="low",
                        reason_for_verification="low_confidence",
                        allowed=True,
                        latency_s=1.25,
                        prompt_tokens=10,
                        completion_tokens=2,
                    )
                ),
                "{not-json",
                json.dumps(
                    verifier_output_record(
                        task_risk_level="U1",
                        called=True,
                        decision="block",
                        safety_risk="U2",
                        failure_risk="high",
                        reason_for_verification="stagnation",
                        allowed=False,
                        latency_s=2.5,
                        prompt_tokens=20,
                        completion_tokens=4,
                        error="auth_error",
                    )
                ),
                json.dumps(
                    verifier_output_record(
                        task_risk_level="U0",
                        called=True,
                        decision="pass",
                        safety_risk="U0",
                        failure_risk="medium",
                        reason_for_verification="semantic_missing_answer",
                        allowed=True,
                        latency_s=0.25,
                        prompt_tokens=5,
                        completion_tokens=1,
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_main_summarize_output_reports_verifier_distributions(tmp_path, monkeypatch, capsys) -> None:
    output_path = tmp_path / "phase0_s2_verifier_offline.jsonl"
    write_verifier_output_jsonl(output_path)
    monkeypatch.setattr(sys, "argv", ["s2_verifier_client.py", "--summarize-output", str(output_path)])

    exit_code = s2.main()

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Summary source line range: 1-4" in stdout
    assert "Summary filters: since_line=none last=none" in stdout
    assert "Total records: 3" in stdout
    assert 'Task risk distribution: {"U0": 2, "U1": 1}' in stdout
    assert "Called count: 3" in stdout
    assert "Error count: 1" in stdout
    assert 'Decision distribution: {"block": 1, "pass": 2}' in stdout
    assert 'Safety risk distribution: {"U0": 2, "U2": 1}' in stdout
    assert 'Failure risk distribution: {"high": 1, "low": 1, "medium": 1}' in stdout
    assert (
        'Reason-for-verification distribution: {"low_confidence": 1, '
        '"semantic_missing_answer": 1, "stagnation": 1}'
    ) in stdout
    assert "Allowed-to-execute count: 2" in stdout
    assert "Total latency seconds: 4.0" in stdout
    assert "Total prompt/completion tokens: 35/7" in stdout


def test_main_summarize_output_applies_since_line_before_last(tmp_path, monkeypatch, capsys) -> None:
    output_path = tmp_path / "phase0_s2_verifier_offline.jsonl"
    write_verifier_output_jsonl(output_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "s2_verifier_client.py",
            "--summarize-output",
            str(output_path),
            "--summarize-since-line",
            "3",
            "--summarize-last",
            "1",
        ],
    )

    exit_code = s2.main()

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Summary source line range: 4-4" in stdout
    assert "Summary filters: since_line=3 last=1" in stdout
    assert "Total records: 1" in stdout
    assert 'Task risk distribution: {"U0": 1}' in stdout
    assert 'Decision distribution: {"pass": 1}' in stdout
    assert "Total prompt/completion tokens: 5/1" in stdout
