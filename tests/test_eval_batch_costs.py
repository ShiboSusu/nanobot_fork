from __future__ import annotations

import json

from eval.batch.aggregate import summarize_phase
from eval.batch.metrics import parse_trace
from eval.batch.schemas import RunRecord


def test_batch_summary_costs_and_ci(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join([
            json.dumps({
                "type": "step",
                "arm": "s1_fast",
                "model_name": "qwen3.5-9b",
                "token_in": 1_000_000,
                "token_out": 100_000,
                "token_think": None,
                "token_usage": {
                    "prompt_tokens": 1_000_000,
                    "completion_tokens": 100_000,
                    "total_tokens": 1_100_000,
                },
            }),
            json.dumps({
                "type": "step",
                "arm": "s2_slow",
                "model_name": "qwen3.5-397b-a17b",
                "token_in": 200_000,
                "token_out": 10_000,
                "token_think": 5_000,
                "token_usage": {
                    "prompt_tokens": 200_000,
                    "completion_tokens": 10_000,
                    "total_tokens": 215_000,
                },
            }),
        ]) + "\n",
        encoding="utf-8",
    )

    metrics = parse_trace(trace)
    summary = summarize_phase(
        "phase_a",
        [
            RunRecord("task1", 1, "a", str(trace), True, metrics=metrics),
            RunRecord("task2", 1, "b", str(trace), False, metrics=metrics),
        ],
        k=1,
    )

    assert metrics.step_costs_yuan == [0.35, 0.348]
    assert metrics.cost_yuan == 0.698
    assert metrics.cost_note == "think 未计"
    assert summary.pass_at_1 == 0.5
    assert summary.pass_at_1_ci95 == (0.0, 1.0)
    assert summary.total_cost_yuan == 1.396
    assert summary.cost_per_success == 1.396
    assert summary.token_per_success == 2_630_000
