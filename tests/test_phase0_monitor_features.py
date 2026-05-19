from __future__ import annotations

from eval.phase0_controller_dry_run import route_step
from eval.phase0_observable_signal_pilot import (
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
