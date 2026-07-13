"""
Unit tests for opengui.trajectory module.

Covers:
- TrajectoryRecorder: event sequencing (metadata first, result last),
  phase tracking, start/finish lifecycle, and error paths.
- TrajectorySummarizer: returns non-empty string from mocked LLM.

All tests are network-free and device-free; external I/O is replaced by
injected fakes following the established P0/_FakeEmbedder patterns.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from opengui.interfaces import LLMResponse
from opengui.trajectory.recorder import ExecutionPhase, TrajectoryRecorder
from opengui.trajectory.summarizer import TrajectorySummarizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ScriptedLLM:
    """Minimal LLMProvider fake: pops canned string responses in order."""

    def __init__(self, *responses: str) -> None:
        self._queue = list(responses)

    async def chat(self, messages, tools=None, tool_choice=None) -> LLMResponse:
        if not self._queue:
            raise AssertionError("No scripted responses left.")
        return LLMResponse(content=self._queue.pop(0))


# ---------------------------------------------------------------------------
# TrajectoryRecorder tests
# ---------------------------------------------------------------------------

def test_trajectory_recorder_start_creates_file(tmp_path: Path) -> None:
    """start() returns a Path that exists on disk."""
    rec = TrajectoryRecorder(output_dir=tmp_path, task="open settings", platform="android")
    trace_path = rec.start()

    assert isinstance(trace_path, Path)
    assert trace_path.exists()
    assert trace_path.suffix == ".jsonl"


def test_trajectory_recorder_event_order(tmp_path: Path) -> None:
    """After start + 2 steps + finish, JSONL has metadata first and result last."""
    rec = TrajectoryRecorder(output_dir=tmp_path, task="open wifi", platform="android")
    path = rec.start()

    rec.record_step(action={"action_type": "tap", "x": 100, "y": 200}, model_output="tap icon")
    rec.record_step(action={"action_type": "done"}, model_output="done")
    rec.finish(success=True)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line) for line in lines]
    types = [e["type"] for e in events]

    assert types[0] == "metadata", f"Expected metadata first, got: {types}"
    assert types[-1] == "result", f"Expected result last, got: {types}"

    metadata = events[0]
    assert metadata["task"] == "open wifi"
    assert metadata["platform"] == "android"

    result = events[-1]
    assert result["success"] is True
    assert result["total_steps"] == 2

    step_events = [e for e in events if e["type"] == "step"]
    assert len(step_events) == 2, f"Expected 2 step events, got: {len(step_events)}"


def test_trajectory_recorder_set_phase(tmp_path: Path) -> None:
    """set_phase() changes subsequent step events to the new phase."""
    rec = TrajectoryRecorder(output_dir=tmp_path, task="test task", platform="macos")
    path = rec.start()

    # Step 1: default phase (AGENT)
    rec.record_step(action={"action_type": "tap", "x": 50, "y": 50}, model_output="first tap")
    # Switch phase to SKILL
    rec.set_phase(ExecutionPhase.SKILL, reason="matched skill")
    # Step 2: should be recorded with SKILL phase
    rec.record_step(action={"action_type": "tap", "x": 80, "y": 80}, model_output="second tap")
    rec.finish(success=True)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line) for line in lines]
    step_events = [e for e in events if e["type"] == "step"]

    assert len(step_events) == 2
    assert step_events[0]["phase"] == "agent", (
        f"First step should be 'agent', got: {step_events[0]['phase']}"
    )
    assert step_events[1]["phase"] == "skill", (
        f"Second step should be 'skill', got: {step_events[1]['phase']}"
    )


def test_trajectory_recorder_step_details_are_persisted(tmp_path: Path) -> None:
    """record_step() should persist lightweight observation fields."""
    rec = TrajectoryRecorder(output_dir=tmp_path, task="search video", platform="android")
    path = rec.start()

    rec.record_step(
        action={"action_type": "input_text", "text": "we are the world"},
        model_output="输入搜索词",
        screenshot_path="/screenshots/step_001.png",
        foreground_app="com.example.app",
        screen_width=1080,
        screen_height=1920,
        platform="android",
    )
    rec.finish(success=True)

    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    step_event = next(event for event in events if event["type"] == "step")

    assert step_event["observation"]["foreground_app"] == "com.example.app"
    assert step_event["observation"]["app"] == "com.example.app"
    assert step_event["observation"]["screen_width"] == 1080
    assert step_event["observation"]["screen_height"] == 1920
    assert step_event["observation"]["platform"] == "android"
    assert step_event["observation"]["screenshot_path"] == "/screenshots/step_001.png"
    assert "prompt" not in step_event
    assert "model_response" not in step_event
    assert "execution" not in step_event


def test_trajectory_recorder_records_rpr_step_fields(tmp_path: Path) -> None:
    rec = TrajectoryRecorder(output_dir=tmp_path, task="rpr", platform="android")
    path = rec.start()
    screen = tmp_path / "screen.png"
    screen.write_bytes(b"screen-one")

    rec.record_step(
        action={"action_type": "tap", "x": 100, "y": 200},
        model_output="tap",
        screenshot_path=str(screen),
        foreground_app="com.example.app",
        token_usage={
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "cached_tokens": 2,
        },
        arm="s1_fast",
        model_name="qwen3.5-9b",
        reasoning_effort="none",
        action_repr="tap at (100, 200)",
    )
    rec.finish(success=True)

    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    step_event = next(event for event in events if event["type"] == "step")

    assert step_event["arm"] == "s1_fast"
    assert step_event["model_name"] == "qwen3.5-9b"
    assert step_event["reasoning_effort"] == "none"
    assert step_event["token_in"] == 10
    assert step_event["token_out"] == 3
    assert step_event["token_think"] is None
    assert step_event["token_cached"] == 2
    assert step_event["screen_hash"] == hashlib.sha256(b"screen-one").hexdigest()
    assert step_event["prev_screen_changed"] is False
    assert step_event["foreground_app"] == "com.example.app"
    assert step_event["action_type"] == "tap"
    assert step_event["action_repr"] == "tap at (100, 200)"


def test_trajectory_recorder_record_step_lazy_starts(tmp_path: Path) -> None:
    """record_step() can lazy-start for MobileGym per-step sessions."""
    rec = TrajectoryRecorder(output_dir=tmp_path, task="test", platform="android")

    rec.record_step(action={"action_type": "tap", "x": 0, "y": 0})

    assert rec.path is not None
    events = [json.loads(line) for line in rec.path.read_text(encoding="utf-8").splitlines()]
    assert [event["type"] for event in events] == ["metadata", "step"]


def test_trajectory_recorder_finish_failure(tmp_path: Path) -> None:
    """finish(success=False, error=...) writes result with success=False and error field."""
    rec = TrajectoryRecorder(output_dir=tmp_path, task="failing task", platform="android")
    path = rec.start()

    rec.record_step(action={"action_type": "wait"}, model_output="waiting")
    rec.finish(success=False, error="timeout")

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line) for line in lines]
    result = events[-1]

    assert result["type"] == "result"
    assert result["success"] is False
    assert result["error"] == "timeout"
    assert result["total_steps"] == 1


def test_trajectory_recorder_metadata_fields(tmp_path: Path) -> None:
    """metadata event includes task, platform, and initial_phase fields."""
    rec = TrajectoryRecorder(output_dir=tmp_path, task="check battery", platform="ios")
    path = rec.start()
    rec.finish(success=True)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line) for line in lines]
    metadata = events[0]

    assert metadata["type"] == "metadata"
    assert metadata["task"] == "check battery"
    assert metadata["platform"] == "ios"
    assert "initial_phase" in metadata
    assert "timestamp" in metadata


# ---------------------------------------------------------------------------
# TrajectorySummarizer tests
# ---------------------------------------------------------------------------

async def test_trajectory_summarizer_returns_string() -> None:
    """summarize_events() returns the strict GUI state note from the LLM."""
    canned = (
        "Status: completed\n"
        "Done: Opened settings and finished the flow.\n"
        "Remaining: none\n"
        "Current: Settings screen\n"
        "Resume: No further action needed."
    )
    llm = _ScriptedLLM(canned)
    summarizer = TrajectorySummarizer(llm)

    events = [
        {"type": "metadata", "task": "open settings", "platform": "android"},
        {
            "type": "step",
            "step_index": 0,
            "action": {"action_type": "tap"},
            "model_output": "tap settings icon",
        },
        {"type": "result", "success": True, "duration_s": 1.2, "error": None},
    ]

    summary = await summarizer.summarize_events(events)

    assert isinstance(summary, str)
    assert summary == canned
    assert summary.startswith("Status: completed")


async def test_trajectory_summarizer_empty_events_returns_empty_string() -> None:
    """summarize_events([]) returns '' without calling the LLM."""
    llm = _ScriptedLLM()  # no responses — would raise if called
    summarizer = TrajectorySummarizer(llm)

    result = await summarizer.summarize_events([])

    assert result == ""
