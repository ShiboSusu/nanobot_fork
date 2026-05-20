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


def test_run_offline_smoke_writes_record_line_source_record(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "phase0.jsonl"
    output_path = tmp_path / "phase0_s2_verifier_offline.jsonl"
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
    monkeypatch.setattr(s2, "git_ignores_path", lambda path: True)

    exit_code = s2.run_offline_smoke(
        input_path,
        latest=False,
        record_line=2,
        step_index=None,
        timeout_s=1.0,
        write_output=output_path,
    )

    assert exit_code == 0
    output_record = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_record["source_record"] == {
        "input_path": str(input_path),
        "line_number": 2,
        "selection": "record_line",
    }


def test_run_offline_smoke_writes_latest_source_record_line(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "phase0.jsonl"
    output_path = tmp_path / "phase0_s2_verifier_offline.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "line-1", "task_risk_level": "U0"}),
                "{not-json",
                json.dumps(
                    {
                        "task_id": "line-3",
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
    monkeypatch.setattr(s2, "git_ignores_path", lambda path: True)

    exit_code = s2.run_offline_smoke(
        input_path,
        latest=True,
        record_line=None,
        step_index=None,
        timeout_s=1.0,
        write_output=output_path,
    )

    assert exit_code == 0
    output_record = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_record["source_record"] == {
        "input_path": str(input_path),
        "line_number": 3,
        "selection": "latest",
    }


def test_run_offline_smoke_writes_synthetic_fallback_source_record(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "phase0_s2_verifier_offline.jsonl"
    monkeypatch.delenv("MA_INTRANET_URL", raising=False)
    monkeypatch.delenv("MA_TOKEN", raising=False)
    monkeypatch.setattr(s2, "git_ignores_path", lambda path: True)

    exit_code = s2.run_offline_smoke(
        None,
        latest=False,
        record_line=None,
        step_index=None,
        timeout_s=1.0,
        write_output=output_path,
    )

    assert exit_code == 0
    output_record = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_record["source_record"] == {
        "input_path": None,
        "line_number": None,
        "selection": "synthetic_fallback",
    }


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


def test_verifier_metadata_from_result_preserves_rationale_fields() -> None:
    response = s2.S2VerifierResponse(
        decision="recover",
        safety_risk="U1",
        failure_risk="medium",
        reason="The proposed tap repeats a prior ineffective action.",
        evidence=["same coordinates were tapped twice", "stagnation_count is 2"],
        suggested_next_step="Open app switcher and verify current screen before acting.",
        allowed_to_execute_s1_action=False,
        requires_image_context=True,
        confidence=0.73,
    )
    result = s2.S2VerifierCallResult(
        ok=True,
        response=response,
        latency_s=1.234,
        token_usage={"prompt_tokens": 11, "completion_tokens": 7},
        error=None,
        model="s2-test",
    )

    metadata = s2.verifier_metadata_from_result(result, reason_for_verification="stagnation")

    assert metadata["verifier_reason"] == "The proposed tap repeats a prior ineffective action."
    assert metadata["verifier_evidence"] == ["same coordinates were tapped twice", "stagnation_count is 2"]
    assert metadata["verifier_suggested_next_step"] == "Open app switcher and verify current screen before acting."
    assert metadata["verifier_requires_image_context"] is True
    assert metadata["verifier_confidence"] == 0.73


def test_verifier_metadata_from_error_result_uses_conservative_rationale_defaults() -> None:
    result = s2.S2VerifierCallResult(
        ok=False,
        response=None,
        latency_s=0.0,
        token_usage=None,
        error="missing_env:MA_INTRANET_URL,MA_TOKEN",
        model="s2-test",
    )

    metadata = s2.verifier_metadata_from_result(result, reason_for_verification="stagnation")
    block = s2.verifier_block_from_metadata(metadata)

    assert metadata["verifier_reason"] is None
    assert metadata["verifier_evidence"] == []
    assert metadata["verifier_suggested_next_step"] is None
    assert metadata["verifier_requires_image_context"] is None
    assert metadata["verifier_confidence"] is None
    assert block["reason"] is None
    assert block["evidence"] == []
    assert block["suggested_next_step"] is None
    assert block["requires_image_context"] is None
    assert block["confidence"] is None


def test_verifier_block_from_metadata_exposes_rationale_fields() -> None:
    metadata = {
        "verifier_called": True,
        "verifier_mode": "text_only",
        "reason_for_verification": "stagnation",
        "verifier_decision": "recover",
        "safety_risk": "U1",
        "failure_risk": "medium",
        "allowed_to_execute_s1_action": False,
        "verifier_latency_s": 1.234,
        "verifier_token_usage": {"prompt_tokens": 11, "completion_tokens": 7},
        "verifier_error": None,
        "s2_model": "s2-test",
        "s2_endpoint_route": "/invocations",
        "verifier_reason": "The proposed tap repeats a prior ineffective action.",
        "verifier_evidence": ["same coordinates were tapped twice", "stagnation_count is 2"],
        "verifier_suggested_next_step": "Open app switcher and verify current screen before acting.",
        "verifier_requires_image_context": True,
        "verifier_confidence": 0.73,
    }

    block = s2.verifier_block_from_metadata(metadata)

    assert block["reason"] == "The proposed tap repeats a prior ineffective action."
    assert block["evidence"] == ["same coordinates were tapped twice", "stagnation_count is 2"]
    assert block["suggested_next_step"] == "Open app switcher and verify current screen before acting."
    assert block["requires_image_context"] is True
    assert block["confidence"] == 0.73


def test_build_sanitized_offline_record_includes_verifier_rationale_fields() -> None:
    record = {"task_id": "task-1", "task_risk_level": "U1", "steps": []}
    request = s2.S2VerifierRequest(
        task_id="task-1",
        instruction="Evaluate proposed GUI action.",
        risk_level="U1",
        current_observation={"text_summary": "Screen appears unchanged.", "screenshot_path": None, "foreground_app": None},
        s1_proposed_action={"action_type": "tap", "x": 100, "y": 200},
        recent_steps=[],
        monitor_signals={},
        verification_mode="text_only",
        reason_for_verification="stagnation",
    )
    selected_step = {
        "step_index": 4,
        "trigger_features": {"execution_state": {"stagnation_count": 2}},
        "outcome_proxies": {},
    }
    metadata = {
        "verifier_called": True,
        "verifier_mode": "text_only",
        "reason_for_verification": "stagnation",
        "verifier_decision": "recover",
        "safety_risk": "U1",
        "failure_risk": "medium",
        "allowed_to_execute_s1_action": False,
        "verifier_latency_s": 1.234,
        "verifier_token_usage": {"prompt_tokens": 11, "completion_tokens": 7},
        "verifier_error": None,
        "s2_model": "s2-test",
        "s2_endpoint_route": "/invocations",
        "verifier_reason": "The proposed tap repeats a prior ineffective action.",
        "verifier_evidence": ["same coordinates were tapped twice", "stagnation_count is 2"],
        "verifier_suggested_next_step": "Open app switcher and verify current screen before acting.",
        "verifier_requires_image_context": True,
        "verifier_confidence": 0.73,
    }

    output_record = s2.build_sanitized_offline_record(record, request, selected_step, metadata)

    assert output_record["steps"][0]["verifier"]["reason"] == "The proposed tap repeats a prior ineffective action."
    assert output_record["steps"][0]["verifier"]["evidence"] == [
        "same coordinates were tapped twice",
        "stagnation_count is 2",
    ]
    assert (
        output_record["steps"][0]["verifier"]["suggested_next_step"]
        == "Open app switcher and verify current screen before acting."
    )
    assert output_record["steps"][0]["verifier"]["requires_image_context"] is True
    assert output_record["steps"][0]["verifier"]["confidence"] == 0.73


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
    reason: str | None = None,
    evidence: list[str] | None = None,
    suggested_next_step: str | None = None,
    requires_image_context: bool | None = None,
    confidence: float | None = None,
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
                    "reason": reason,
                    "evidence": evidence or [],
                    "suggested_next_step": suggested_next_step,
                    "requires_image_context": requires_image_context,
                    "confidence": confidence,
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
                        reason="Low confidence but action is safe.",
                        evidence=["confidence below threshold"],
                        suggested_next_step="Proceed with caution.",
                        requires_image_context=False,
                        confidence=0.7,
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
                        reason="Final answer is missing.",
                        evidence=["final_answer_present is False", "semantic guard fired"],
                        suggested_next_step="Find the answer before done.",
                        requires_image_context=True,
                        confidence=0.9,
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
    assert "Verifier rationale count: 2" in stdout
    assert "Verifier evidence item count: 3" in stdout
    assert "Verifier suggested-next-step count: 2" in stdout
    assert "Requires-image-context count: 1" in stdout
    assert "Recovery-design candidate count: 0" in stdout
    assert "Total latency seconds: 4.0" in stdout
    assert "Total prompt/completion tokens: 35/7" in stdout


def test_main_summarize_output_reports_recovery_design_candidates(tmp_path, monkeypatch, capsys) -> None:
    output_path = tmp_path / "phase0_s2_verifier_offline.jsonl"
    records = [
        verifier_output_record(
            task_risk_level="U0",
            called=True,
            decision="recover",
            safety_risk="U0",
            failure_risk="high",
            reason_for_verification="stagnation",
            allowed=False,
            latency_s=1.0,
            prompt_tokens=10,
            completion_tokens=4,
            reason="The current action repeats a failed tap but a local back action can recover.",
            evidence=["same action repeated", "no external side effect"],
            suggested_next_step="Press back once and inspect the current screen.",
            requires_image_context=False,
            confidence=0.8,
        ),
        verifier_output_record(
            task_risk_level="U0",
            called=True,
            decision="recover",
            safety_risk="U0",
            failure_risk="high",
            reason_for_verification="stagnation",
            allowed=False,
            latency_s=1.0,
            prompt_tokens=10,
            completion_tokens=4,
            reason="Need visual grounding before recovery.",
            evidence=["target location unknown"],
            suggested_next_step="Tap the visible search box.",
            requires_image_context=True,
            confidence=0.8,
        ),
        verifier_output_record(
            task_risk_level="U0",
            called=True,
            decision="replan",
            safety_risk="U0",
            failure_risk="high",
            reason_for_verification="semantic_missing_answer",
            allowed=False,
            latency_s=1.0,
            prompt_tokens=10,
            completion_tokens=4,
            reason="The answer is missing.",
            evidence=["final_answer_present is False"],
            suggested_next_step="Find the answer before finishing.",
            requires_image_context=False,
            confidence=0.8,
        ),
    ]
    output_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["s2_verifier_client.py", "--summarize-output", str(output_path)])

    exit_code = s2.main()

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Recovery-design candidate count: 1" in stdout
    assert 'Recovery-design blocker distribution: {"non_recover_decision": 1, "requires_image_context": 1}' in stdout
    assert (
        'Recovery-design blocker details: [{"blockers": ["requires_image_context"], '
        '"line": 2, "source_line": null, "task_id": null}, '
        '{"blockers": ["non_recover_decision"], "line": 3, "source_line": null, "task_id": null}]'
    ) in stdout
    assert "Requires-image-context count: 1" in stdout


def test_main_summarize_output_reports_source_record_line_range(tmp_path, monkeypatch, capsys) -> None:
    output_path = tmp_path / "phase0_s2_verifier_offline.jsonl"
    records = [
        verifier_output_record(
            task_risk_level="U0",
            called=True,
            decision="pass",
            safety_risk="U0",
            failure_risk="low",
            reason_for_verification="low_confidence",
            allowed=True,
            latency_s=1.0,
            prompt_tokens=10,
            completion_tokens=2,
        ),
        verifier_output_record(
            task_risk_level="U1",
            called=True,
            decision="block",
            safety_risk="U2",
            failure_risk="high",
            reason_for_verification="stagnation",
            allowed=False,
            latency_s=2.0,
            prompt_tokens=20,
            completion_tokens=4,
        ),
    ]
    records[0]["source_record"] = {"input_path": "phase0.jsonl", "line_number": 2, "selection": "record_line"}
    records[1]["source_record"] = {"input_path": "phase0.jsonl", "line_number": 4, "selection": "latest"}
    output_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["s2_verifier_client.py", "--summarize-output", str(output_path)])

    exit_code = s2.main()

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Source record line range: 2-4" in stdout


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
