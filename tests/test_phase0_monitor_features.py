from __future__ import annotations

from eval.phase0_controller_dry_run import build_controller_record, route_step
from eval.phase0_observable_signal_pilot import (
    attach_shadow_controller,
    detect_screenshot_capture_failure,
    enrich_monitor_features,
)


def test_enrich_monitor_features_detects_repeated_regions_and_high_confidence() -> None:
    steps = [
        {
            "step_index": 0,
            "action": {"action_type": "tap", "x": 280.0, "y": 25.0, "relative": True},
            "trigger_features": {
                "self_report": {"confidence": 0.95},
                "execution_state": {},
            },
        },
        {
            "step_index": 1,
            "action": {"action_type": "tap", "x": 281.0, "y": 25.0, "relative": True},
            "trigger_features": {
                "self_report": {"confidence": 0.95},
                "execution_state": {},
            },
        },
        {
            "step_index": 2,
            "action": {"action_type": "tap", "x": 281.0, "y": 24.0, "relative": True},
            "trigger_features": {
                "self_report": {"confidence": 0.95},
                "execution_state": {},
            },
        },
    ]

    enrich_monitor_features(steps, max_steps=3)

    execution_state = steps[2]["trigger_features"]["execution_state"]
    assert execution_state["action_type_run_length"] == 3
    assert execution_state["coordinate_bucket"] == "top_left"
    assert execution_state["coordinate_bucket_repeat"] is True
    assert execution_state["screen_region"] == "top"
    assert execution_state["screen_region_repeat"] is True
    assert execution_state["repeated_region_action"] is True
    assert execution_state["max_steps_remaining"] == 0
    assert execution_state["max_steps_near_limit"] is True
    assert execution_state["high_confidence_no_progress"] is True


def test_max_steps_near_limit_alone_does_not_trigger_high_confidence_no_progress() -> None:
    steps = [
        {
            "step_index": 0,
            "action": {"action_type": "tap", "x": 100.0, "y": 100.0, "relative": True},
            "trigger_features": {"self_report": {"confidence": 0.95}, "execution_state": {}},
        },
        {
            "step_index": 1,
            "action": {"action_type": "swipe", "x": 800.0, "y": 800.0, "relative": True},
            "trigger_features": {"self_report": {"confidence": 0.95}, "execution_state": {}},
        },
    ]

    enrich_monitor_features(steps, max_steps=2)

    execution_state = steps[1]["trigger_features"]["execution_state"]
    assert execution_state["max_steps_remaining"] == 0
    assert execution_state["max_steps_near_limit"] is True
    assert execution_state["repeated_region_action"] is False
    assert execution_state["action_type_run_length"] == 1
    assert execution_state["high_confidence_no_progress"] is False


def test_detect_screenshot_capture_failure_from_adb_error() -> None:
    assert detect_screenshot_capture_failure(
        "AdbError: adb failed (exit 1): adb shell screencap -p /data/local/tmp/__opengui_cap.png"
    )
    assert not detect_screenshot_capture_failure("max_steps_exceeded")


def test_route_step_recovers_on_repeated_region_action() -> None:
    record = {"task_id": "example", "task_risk_level": "U0", "trace_quality": {"clean_for_signal_analysis": True}}
    step = {
        "trigger_features": {
            "risk": {"rule_based_step_risk_level": "U0", "step_predicted_risk_level": "U0"},
            "execution_state": {"repeated_region_action": True, "action_type_run_length": 2},
        }
    }

    route, reason, inputs = route_step(record, step)

    assert route == "RECOVER"
    assert "repeated region" in reason
    assert inputs["monitor_trigger"] == "repeated_region_action"


def test_route_step_marks_screenshot_failure_unusable() -> None:
    record = {
        "task_id": "example",
        "task_risk_level": "U0",
        "trace_quality": {"clean_for_signal_analysis": True},
        "environment_anomalies": {"screenshot_capture_failure": True, "secure_surface_suspected": True},
    }
    step = {
        "trigger_features": {
            "risk": {"rule_based_step_risk_level": "U0"},
            "execution_state": {"screenshot_capture_failure": True, "secure_surface_suspected": True},
        }
    }

    route, reason, inputs = route_step(record, step)

    assert route == "UNUSABLE_TRACE"
    assert "screenshot" in reason
    assert inputs["monitor_trigger"] == "screenshot_capture_failure"


def test_shadow_controller_exposes_raw_route_under_low_inner_coverage_gate() -> None:
    record = {
        "task_id": "example",
        "task_risk_level": "U0",
        "trace_quality": {"clean_for_signal_analysis": False, "quality_warning": "low_inner_coverage"},
        "steps": [
            {
                "step_index": 0,
                "trigger_features": {
                    "risk": {"rule_based_step_risk_level": "U0", "step_predicted_risk_level": "U0"},
                    "execution_state": {"repeated_region_action": True, "action_type_run_length": 2},
                },
            }
        ],
    }

    attach_shadow_controller(record)

    controller = record["steps"][0]["controller"]
    assert controller["route"] == "UNUSABLE_TRACE"
    assert controller["raw_monitor_route"] == "RECOVER"
    assert controller["raw_monitor_reason"] == "execution state indicates repeated region action"
    assert controller["raw_monitor_inputs"]["monitor_trigger"] == "repeated_region_action"
    assert controller["hard_gate_reason"] == "low_inner_coverage"
    assert controller["inputs"]["monitor_trigger"] == "repeated_region_action"


def test_shadow_controller_verifies_clean_empty_done_semantic_failure() -> None:
    record = {
        "task_id": "example",
        "task_risk_level": "U0",
        "trace_quality": {"clean_for_signal_analysis": True},
        "semantic_task_success": False,
        "semantic_success_source": "answer_presence_guard",
        "semantic_success_reason": "missing_required_final_answer",
        "steps": [
            {
                "step_index": 0,
                "action": {"action_type": "done", "status": "success", "answer": ""},
                "trigger_features": {
                    "risk": {"rule_based_step_risk_level": "U0", "step_predicted_risk_level": "U0"},
                    "execution_state": {},
                },
            }
        ],
    }

    attach_shadow_controller(record)

    controller = record["steps"][0]["controller"]
    assert controller["route"] == "VERIFY"
    assert controller["raw_monitor_route"] == "VERIFY"
    assert controller["hard_gate_reason"] is None
    assert controller["inputs"]["monitor_trigger"] == "semantic_missing_answer"
    assert controller["raw_monitor_inputs"]["monitor_trigger"] == "semantic_missing_answer"
    assert "missing required final answer" in controller["reason"]
    assert "missing required final answer" in controller["raw_monitor_reason"]


def test_shadow_controller_keeps_trace_gate_but_exposes_empty_done_semantic_raw_route() -> None:
    record = {
        "task_id": "example",
        "task_risk_level": "U0",
        "trace_quality": {"clean_for_signal_analysis": False, "quality_warning": "low_inner_coverage"},
        "semantic_task_success": False,
        "semantic_success_source": "answer_presence_guard",
        "semantic_success_reason": "missing_required_final_answer",
        "steps": [
            {
                "step_index": 0,
                "action": {"action_type": "done", "status": "success", "answer": ""},
                "trigger_features": {
                    "risk": {"rule_based_step_risk_level": "U0", "step_predicted_risk_level": "U0"},
                    "execution_state": {},
                },
            }
        ],
    }

    attach_shadow_controller(record)

    controller = record["steps"][0]["controller"]
    assert controller["route"] == "UNUSABLE_TRACE"
    assert controller["raw_monitor_route"] == "VERIFY"
    assert controller["hard_gate_reason"] == "low_inner_coverage"
    assert controller["inputs"]["monitor_trigger"] == "semantic_missing_answer"
    assert controller["raw_monitor_inputs"]["monitor_trigger"] == "semantic_missing_answer"
    assert controller["raw_monitor_reason"] == "semantic guard found missing required final answer"


def test_shadow_controller_exposes_raw_route_under_screenshot_gate() -> None:
    record = {
        "task_id": "example",
        "task_risk_level": "U0",
        "trace_quality": {"clean_for_signal_analysis": True, "quality_warning": None},
        "environment_anomalies": {"screenshot_capture_failure": True, "secure_surface_suspected": True},
        "steps": [
            {
                "step_index": 0,
                "trigger_features": {
                    "risk": {"rule_based_step_risk_level": "U0", "step_predicted_risk_level": "U0"},
                    "execution_state": {
                        "repeated_region_action": True,
                        "action_type_run_length": 2,
                        "screenshot_capture_failure": True,
                        "secure_surface_suspected": True,
                    },
                },
            }
        ],
    }

    attach_shadow_controller(record)

    controller = record["steps"][0]["controller"]
    assert controller["route"] == "UNUSABLE_TRACE"
    assert controller["raw_monitor_route"] == "RECOVER"
    assert controller["raw_monitor_reason"] == "execution state indicates repeated region action"
    assert controller["raw_monitor_inputs"]["monitor_trigger"] == "repeated_region_action"
    assert controller["hard_gate_reason"] == "screenshot_capture_failure"
    assert controller["inputs"]["monitor_trigger"] == "screenshot_capture_failure"


def test_shadow_controller_clean_u1_step_has_no_hard_gate() -> None:
    record = {
        "task_id": "example",
        "task_risk_level": "U1",
        "trace_quality": {"clean_for_signal_analysis": True, "quality_warning": None},
        "steps": [
            {
                "step_index": 0,
                "trigger_features": {
                    "risk": {"rule_based_step_risk_level": "U1", "step_predicted_risk_level": "U1"},
                    "execution_state": {},
                },
            }
        ],
    }

    attach_shadow_controller(record)

    controller = record["steps"][0]["controller"]
    assert controller["route"] == "VERIFY"
    assert controller["raw_monitor_route"] == "VERIFY"
    assert controller["raw_monitor_reason"] == "rule-based step risk is elevated"
    assert controller["raw_monitor_inputs"]["monitor_trigger"] == "step_risk"
    assert controller["hard_gate_reason"] is None
    assert controller["inputs"]["monitor_trigger"] == "step_risk"


def test_shadow_controller_u2_keeps_safety_gate_for_raw_route() -> None:
    record = {
        "task_id": "example",
        "task_risk_level": "U2",
        "trace_quality": {"clean_for_signal_analysis": True, "quality_warning": None},
        "steps": [
            {
                "step_index": 0,
                "trigger_features": {
                    "risk": {"rule_based_step_risk_level": "U2", "step_predicted_risk_level": "U2"},
                    "execution_state": {"repeated_region_action": True},
                },
            }
        ],
    }

    attach_shadow_controller(record)

    controller = record["steps"][0]["controller"]
    assert controller["route"] == "SKIP_UNSAFE"
    assert controller["raw_monitor_route"] == "SKIP_UNSAFE"
    assert controller["hard_gate_reason"] == "blocked_u2"
    assert controller["raw_monitor_inputs"]["runner_safety_gate"] == "blocked_u2"


def test_shadow_controller_trace_missing_hard_gate_reason() -> None:
    record = {
        "task_id": "example",
        "task_risk_level": "U0",
        "trace_quality": {"clean_for_signal_analysis": False, "quality_warning": "trace_missing"},
        "steps": [{"step_index": 0, "trigger_features": {"risk": {}, "execution_state": {}}}],
    }

    attach_shadow_controller(record)

    controller = record["steps"][0]["controller"]
    assert controller["route"] == "UNUSABLE_TRACE"
    assert controller["hard_gate_reason"] == "trace_missing"


def test_shadow_controller_generic_trace_quality_hard_gate_reason() -> None:
    record = {
        "task_id": "example",
        "task_risk_level": "U0",
        "trace_quality": {"clean_for_signal_analysis": False, "quality_warning": "unexpected_quality_warning"},
        "steps": [{"step_index": 0, "trigger_features": {"risk": {}, "execution_state": {}}}],
    }

    attach_shadow_controller(record)

    controller = record["steps"][0]["controller"]
    assert controller["route"] == "UNUSABLE_TRACE"
    assert controller["hard_gate_reason"] == "trace_quality_unusable"


def test_shadow_controller_secure_surface_only_hard_gate_reason() -> None:
    record = {
        "task_id": "example",
        "task_risk_level": "U0",
        "trace_quality": {"clean_for_signal_analysis": True, "quality_warning": None},
        "steps": [
            {
                "step_index": 0,
                "trigger_features": {
                    "risk": {"rule_based_step_risk_level": "U0"},
                    "execution_state": {"secure_surface_suspected": True, "repeated_region_action": True},
                },
            }
        ],
    }

    attach_shadow_controller(record)

    controller = record["steps"][0]["controller"]
    assert controller["route"] == "UNUSABLE_TRACE"
    assert controller["hard_gate_reason"] == "secure_surface_suspected"
    assert controller["raw_monitor_inputs"]["monitor_trigger"] == "repeated_region_action"


def test_build_controller_record_includes_shadow_diagnostics() -> None:
    record = {
        "task_id": "example",
        "task_risk_level": "U0",
        "trace_quality": {"clean_for_signal_analysis": False, "quality_warning": "low_inner_coverage"},
    }
    step = {
        "step_index": 0,
        "trigger_features": {
            "risk": {"rule_based_step_risk_level": "U0"},
            "execution_state": {"repeated_region_action": True},
        },
    }

    output = build_controller_record(record, step)

    controller = output["controller"]
    assert controller["route"] == "UNUSABLE_TRACE"
    assert controller["raw_monitor_route"] == "RECOVER"
    assert controller["raw_monitor_inputs"]["monitor_trigger"] == "repeated_region_action"
    assert controller["hard_gate_reason"] == "low_inner_coverage"
