from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

from eval import phase0_observable_signal_pilot as pilot
from eval.phase0_observable_signal_pilot import (
    Phase0Task,
    TracePaths,
    extract_trace,
    find_newest_trace_paths,
    load_jsonl,
)


def write_jsonl(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")


def outer_attempt_events(first_count: int, second_count: int) -> list[dict]:
    events: list[dict] = [{"type": "attempt_start", "attempt": 0}]
    for index in range(first_count):
        events.append(
            {
                "type": "step",
                "step_index": index,
                "action": {"action_type": "tap", "attempt": 0, "n": index},
                "model_output": f"outer attempt 0 step {index}",
            }
        )
    events.append(
        {
            "type": "attempt_result",
            "attempt": 0,
            "success": False,
            "error": "max_steps_exceeded",
            "steps_taken": first_count,
        }
    )
    events.append({"type": "retry", "attempt": 0})
    events.append({"type": "attempt_start", "attempt": 1})
    for index in range(second_count):
        events.append(
            {
                "type": "step",
                "step_index": index,
                "action": {"action_type": "tap", "attempt": 1, "n": index},
                "model_output": f"outer attempt 1 step {index}",
            }
        )
    events.append(
        {
            "type": "attempt_result",
            "attempt": 1,
            "success": False,
            "error": "stagnation_detected",
            "steps_taken": second_count,
        }
    )
    return events


def inner_events(count: int, attempt: int, *, rich: bool = True) -> list[dict]:
    events: list[dict] = []
    for index in range(count):
        model_output = {
            "raw_content": f"attempt {attempt} step {index}",
            "parsed_action": {"action_type": "tap", "attempt": attempt, "n": index},
        }
        if rich:
            model_output["tool_calls"] = [{"name": "tap"}]
            model_output["assistant_message"] = "tap"
        events.append({"type": "step", "step_index": index, "model_output": model_output})
    return events


def task() -> Phase0Task:
    return Phase0Task(task_id="attempt-test", instruction="tap", instruction_ch="", risk_level="U0")


def trace_paths(run_dir: Path, outer: Path, inner: Path | None = None) -> TracePaths:
    return TracePaths(
        run_dir=run_dir,
        outer_trace_path=outer,
        inner_trace_path=inner,
        inner_trace_candidates_count=2 if inner is not None else 0,
        selected_inner_trace_score=None,
    )


def test_extract_trace_uses_final_retry_attempt_for_primary_steps(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    outer = run_dir / "trace_outer.jsonl"
    attempt0_inner = run_dir / "candidate_0" / "trace.jsonl"
    attempt1_inner = run_dir / "candidate_1" / "trace.jsonl"
    write_jsonl(outer, outer_attempt_events(first_count=5, second_count=3))
    write_jsonl(attempt0_inner, inner_events(5, attempt=0))
    write_jsonl(attempt1_inner, inner_events(3, attempt=1))

    steps, alignment, quality = extract_trace(task(), trace_paths(run_dir, outer, attempt0_inner))

    assert len(steps) == 3
    assert [step["action"]["attempt"] for step in steps] == [1, 1, 1]
    assert alignment["analysis_scope"] == "final_attempt"
    assert alignment["selected_attempt_index"] == 1
    assert alignment["outer_attempt_count"] == 2
    assert alignment["outer_step_count"] == 3
    assert alignment["inner_step_count"] == 3
    assert quality["inner_coverage"] == 1.0
    assert quality["selected_attempt_outer_step_count"] == 3
    assert quality["selected_attempt_inner_step_count"] == 3


def test_extract_trace_prefers_final_attempt_over_better_inner_candidate(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    outer = run_dir / "trace_outer.jsonl"
    attempt0_inner = run_dir / "candidate_0" / "trace.jsonl"
    attempt1_inner = run_dir / "candidate_1" / "trace.jsonl"
    write_jsonl(outer, outer_attempt_events(first_count=5, second_count=3))
    write_jsonl(attempt0_inner, inner_events(5, attempt=0, rich=True))
    write_jsonl(attempt1_inner, inner_events(3, attempt=1, rich=False))

    steps, alignment, quality = extract_trace(task(), trace_paths(run_dir, outer, attempt0_inner))

    assert len(steps) == 3
    assert alignment["selected_attempt_index"] == 1
    assert alignment["selected_attempt_inner_step_count"] == 3
    assert quality["inner_coverage"] == 1.0
    assert quality["attempts"][1]["selected_inner_path_display"].endswith("candidate_1/trace.jsonl")


def test_extract_trace_uses_chronological_inner_when_suffix_missing_for_final_attempt(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    outer = run_dir / "trace_outer.jsonl"
    attempt0_inner = run_dir / "candidate_0" / "trace.jsonl"
    attempt1_inner = run_dir / "candidate_final" / "trace.jsonl"
    write_jsonl(outer, outer_attempt_events(first_count=5, second_count=3))
    write_jsonl(attempt0_inner, inner_events(5, attempt=0, rich=True))
    write_jsonl(attempt1_inner, inner_events(3, attempt=1, rich=False))

    steps, alignment, quality = extract_trace(task(), trace_paths(run_dir, outer, attempt0_inner))

    assert len(steps) == 3
    assert alignment["selected_attempt_index"] == 1
    assert alignment["selected_inner_trace_path_display"].endswith("candidate_final/trace.jsonl")
    assert quality["selected_inner_trace_path_display"].endswith("candidate_final/trace.jsonl")
    assert quality["inner_coverage"] == 1.0


def test_extract_trace_does_not_reuse_explicit_candidate_for_chronological_fallback(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    outer = run_dir / "trace_outer.jsonl"
    attempt0_inner = run_dir / "candidate_0" / "trace.jsonl"
    attempt1_inner = run_dir / "candidate_final" / "trace.jsonl"
    write_jsonl(outer, outer_attempt_events(first_count=5, second_count=3))
    write_jsonl(attempt1_inner, inner_events(3, attempt=1, rich=False))
    write_jsonl(attempt0_inner, inner_events(5, attempt=0, rich=True))
    newer = attempt0_inner.stat().st_mtime + 10
    os.utime(attempt0_inner, (newer, newer))

    steps, alignment, quality = extract_trace(task(), trace_paths(run_dir, outer, attempt0_inner))

    assert len(steps) == 3
    assert [step["action"]["attempt"] for step in steps] == [1, 1, 1]
    assert alignment["selected_inner_trace_path_display"].endswith("candidate_final/trace.jsonl")
    assert alignment["inner_step_count"] == 3
    assert quality["inner_coverage"] == 1.0


def test_extract_trace_final_attempt_exception_with_no_steps_is_selected(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    outer = run_dir / "trace_outer.jsonl"
    inner = run_dir / "candidate_0" / "trace.jsonl"
    write_jsonl(
        outer,
        [
            {"type": "attempt_start", "attempt": 0},
            {"type": "step", "step_index": 0, "action": {"action_type": "tap", "attempt": 0}},
            {"type": "attempt_result", "attempt": 0, "success": False, "error": "max_steps_exceeded"},
            {"type": "retry", "attempt": 0},
            {"type": "attempt_start", "attempt": 1},
            {"type": "attempt_exception", "attempt": 1, "error_type": "RuntimeError", "error_message": "boom"},
        ],
    )
    write_jsonl(inner, inner_events(1, attempt=0))

    steps, alignment, quality = extract_trace(task(), trace_paths(run_dir, outer, inner))

    assert steps == []
    assert alignment["selected_attempt_index"] == 1
    assert alignment["outer_step_count"] == 0
    assert alignment["outer_attempt_count"] == 2
    assert quality["attempts"][1]["result_error"] == "RuntimeError: boom"


def test_load_jsonl_returns_empty_for_unreadable_file() -> None:
    class UnreadablePath:
        def exists(self) -> bool:
            return True

        def open(self, **kwargs):
            raise OSError("permission denied")

    assert load_jsonl(UnreadablePath()) == []


def test_find_newest_trace_paths_skips_run_dir_that_fails_stat_after_discovery(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "artifacts"
    stale = root / "stale"
    newest = root / "newest"
    stale.mkdir(parents=True)
    newest.mkdir()
    outer = newest / "trace_outer.jsonl"
    write_jsonl(outer, [])

    cfg = SimpleNamespace(gui=SimpleNamespace(artifacts_dir=str(root)), workspace_path=tmp_path)
    original_stat = Path.stat
    calls: dict[Path, int] = {}

    def flaky_stat(path: Path, *args, **kwargs):
        calls[path] = calls.get(path, 0) + 1
        if path == newest and calls[path] > 1:
            raise OSError("disappeared")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", flaky_stat)

    trace_paths = find_newest_trace_paths(cfg, after_ts=0)

    assert trace_paths.run_dir == stale


def test_extract_trace_skips_unreadable_inner_candidates(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    outer = run_dir / "trace_outer.jsonl"
    bad_inner = run_dir / "candidate_0" / "trace.jsonl"
    good_inner = run_dir / "candidate_1" / "trace.jsonl"
    write_jsonl(outer, outer_attempt_events(first_count=1, second_count=3))
    write_jsonl(bad_inner, inner_events(1, attempt=0))
    write_jsonl(good_inner, inner_events(3, attempt=1))

    original_score = pilot.inner_trace_score

    def score_or_fail(path: Path) -> int:
        if path == bad_inner:
            raise OSError("disappeared")
        return original_score(path)

    monkeypatch.setattr(pilot, "inner_trace_score", score_or_fail)

    steps, alignment, quality = extract_trace(task(), trace_paths(run_dir, outer, bad_inner))

    assert len(steps) == 3
    assert alignment["selected_inner_trace_path_display"].endswith("candidate_1/trace.jsonl")
    assert quality["inner_coverage"] == 1.0


def test_extract_trace_no_retry_treats_all_steps_as_attempt_zero(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    outer = run_dir / "trace_outer.jsonl"
    inner = run_dir / "candidate_0" / "trace.jsonl"
    write_jsonl(
        outer,
        [
            {
                "type": "step",
                "step_index": index,
                "action": {"action_type": "tap", "attempt": 0, "n": index},
            }
            for index in range(4)
        ],
    )
    write_jsonl(inner, inner_events(4, attempt=0))

    steps, alignment, quality = extract_trace(task(), trace_paths(run_dir, outer, inner))

    assert len(steps) == 4
    assert alignment["selected_attempt_index"] == 0
    assert alignment["outer_attempt_count"] == 1
    assert alignment["outer_step_count"] == 4
    assert quality["analysis_scope"] == "final_attempt"
