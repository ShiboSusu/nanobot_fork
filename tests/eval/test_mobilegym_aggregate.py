from __future__ import annotations

import json
from pathlib import Path

from eval.mobilegym.aggregate import aggregate_arms, load_run, render_markdown
from eval.mobilegym.cost import apply_costs


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_response(run_dir: Path, task_id: str, trial_id: int, payload: dict) -> None:
    traj_dir = run_dir / "trajectory" / f"{task_id.replace('.', '_')}_t{trial_id}"
    _write_response_in_dir(traj_dir, payload)


def _write_response_in_dir(traj_dir: Path, payload: dict) -> None:
    traj_dir.mkdir(parents=True, exist_ok=True)
    (traj_dir / "trajectory.json").write_text(
        json.dumps(
            [
                {
                    "step": 1,
                    "model_response_path": "step_001_response.txt",
                    "action_type": "COMPLETE",
                    "action_data": {"return": "done"},
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (traj_dir / "step_001_response.txt").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def _base_row(task_id: str, trial_id: int, *, success: bool, judge_error: str | None = None) -> dict:
    return {
        "task_id": task_id,
        "id": task_id,
        "trial_id": trial_id,
        "is_success": success,
        "is_error": judge_error is not None,
        "progress": 1.0 if success else 0.0,
        "execution": {
            "steps": 1,
            "stop_reason": "COMPLETE",
            "runtime_s": 10.0,
            "error": None,
        },
        "judge": {
            "success": success,
            "clean": True,
            "progress": 1.0 if success else 0.0,
            "passed": success,
            "issues": [],
            "warnings": [],
            **({"judge_error": judge_error} if judge_error else {}),
        },
    }


def test_load_run_parses_s2_usage_and_excludes_judge_error_from_sr(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_jsonl(
        run_dir / "results.jsonl",
        [
            _base_row("wechat.TaskA", 0, success=True),
            _base_row("wechat.TaskA", 1, success=False, judge_error="VLM parse failed"),
            _base_row("wechat.TaskB", 0, success=False),
        ],
    )
    _write_response(
        run_dir,
        "wechat.TaskA",
        0,
        {
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            "s2_usage": {
                "enabled": True,
                "hints_used": 1,
                "takeover_used": True,
                "takeover_steps": 2,
                "s1_steps": 3,
                "s2_steps": 2,
                "triggers": [
                    {"step_index": 3, "trigger": "max_steps_near", "mode": "hint"},
                    {"step_index": 4, "trigger": "missing_evidence", "mode": "takeover"},
                ],
                "token_usage": {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55},
            },
        },
    )
    _write_response(
        run_dir,
        "wechat.TaskA",
        1,
        {"token_usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}},
    )
    _write_response(
        run_dir,
        "wechat.TaskB",
        0,
        {"token_usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22}},
    )

    summary = load_run(run_dir, arm="B_fastslow", k=2)

    assert summary["episodes"] == 3
    assert summary["judge_error_episodes"] == 1
    assert summary["valid_episodes"] == 2
    assert summary["episode_sr"] == 0.5
    assert summary["episode_sr_ci95"] != (0.0, 0.0)
    assert summary["pass_at_1"] == 0.5
    assert summary["pass_at_1_ci95"] != (0.0, 0.0)
    assert summary["pass_at_k"] == 0.5
    assert summary["pass_at_k_ci95"] != (0.0, 0.0)
    assert summary["s2_trigger_rate"] == 1 / 3
    assert summary["hint_rate"] == 1 / 3
    assert summary["takeover_rate"] == 1 / 3
    assert summary["trigger_histogram"] == {"max_steps_near": 1, "missing_evidence": 1}
    assert summary["mode_histogram"] == {"hint": 1, "takeover": 1}
    assert summary["s1_total_tokens"] == 88
    assert summary["s2_total_tokens"] == 55
    assert summary["total_tokens"] == 143
    assert summary["s1_total_tokens"] + summary["s2_total_tokens"] == summary["total_tokens"]
    assert summary["s2_token_share"] == 55 / 143


def test_aggregate_arms_reports_takeover_rescues_against_baseline(tmp_path: Path) -> None:
    a_dir = tmp_path / "A"
    b_dir = tmp_path / "B"
    _write_jsonl(a_dir / "results.jsonl", [_base_row("wechat.TaskA", 0, success=False)])
    _write_jsonl(b_dir / "results.jsonl", [_base_row("wechat.TaskA", 0, success=True)])
    _write_response(a_dir, "wechat.TaskA", 0, {"token_usage": {"total_tokens": 100}})
    _write_response(
        b_dir,
        "wechat.TaskA",
        0,
        {
            "token_usage": {"total_tokens": 100},
            "s2_usage": {
                "takeover_used": True,
                "takeover_steps": 1,
                "s2_steps": 1,
                "triggers": [{"trigger": "step_error", "mode": "takeover"}],
                "token_usage": {"total_tokens": 25},
            },
        },
    )

    aggregate = aggregate_arms({"A_s1_only": a_dir, "B_fastslow": b_dir}, k=1)

    assert aggregate["comparisons"]["B_fastslow_vs_A_s1_only"]["takeover_rescue_successes"] == 1
    assert aggregate["comparisons"]["B_fastslow_vs_A_s1_only"]["takeover_episodes"] == 1


def test_pass_at_1_is_mean_single_trial_success_not_first_trial_only(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_jsonl(
        run_dir / "results.jsonl",
        [
            _base_row("wechat.TaskA", 0, success=False),
            _base_row("wechat.TaskA", 1, success=True),
        ],
    )
    _write_response(run_dir, "wechat.TaskA", 0, {"token_usage": {"total_tokens": 10}})
    _write_response(run_dir, "wechat.TaskA", 1, {"token_usage": {"total_tokens": 10}})

    summary = load_run(run_dir, arm="A_s1_only", k=2)

    assert summary["episode_sr"] == 0.5
    assert summary["pass_at_1"] == 0.5
    assert summary["pass_at_k"] == 1.0


def test_apply_costs_uses_actor_split_tokens() -> None:
    aggregate = {
        "arms": {
            "B_fastslow": {
                "successes": 2,
                "s1_model": "9B",
                "s2_model": "397B",
                "s1_prompt_tokens": 1_000_000,
                "s1_completion_tokens": 500_000,
                "s2_prompt_tokens": 200_000,
                "s2_completion_tokens": 100_000,
            }
        }
    }
    priced = apply_costs(
        aggregate,
        prices={
            "9B": {"input": 1.0, "output": 2.0},
            "397B": {"input": 10.0, "output": 20.0},
        },
    )

    arm = priced["arms"]["B_fastslow"]
    assert arm["cost_cny"] == 6.0
    assert arm["cost_per_success"] == 3.0


def test_apply_costs_default_prices_are_nonzero() -> None:
    aggregate = {
        "arms": {
            "s1_fast": {
                "successes": 1,
                "s1_model": "9B",
                "s2_model": None,
                "s1_prompt_tokens": 1_000_000,
                "s1_completion_tokens": 1_000_000,
                "s2_prompt_tokens": 0,
                "s2_completion_tokens": 0,
            }
        }
    }

    arm = apply_costs(aggregate)["arms"]["s1_fast"]

    assert arm["cost_cny"] == 1.7


def test_agent_model_error_is_reported_outside_sr_denominator(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_jsonl(
        run_dir / "results.jsonl",
        [
            _base_row("wechat.TaskA", 0, success=False),
            _base_row("wechat.TaskB", 0, success=True),
        ],
    )
    _write_response(
        run_dir,
        "wechat.TaskA",
        0,
        {"error": "AuthenticationError: api key invalid", "token_usage": {}},
    )
    _write_response(
        run_dir,
        "wechat.TaskB",
        0,
        {"token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
    )

    summary = load_run(run_dir, arm="A_s1_only", k=1)

    assert summary["agent_error_episodes"] == 1
    assert summary["agent_error_rate"] == 0.5
    assert summary["valid_episodes"] == 1
    assert summary["episode_sr"] == 1.0


def test_expected_and_code_agent_failures_stay_in_sr_denominator(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_jsonl(
        run_dir / "results.jsonl",
        [
            _base_row("wechat.TaskA", 0, success=False),
            _base_row("wechat.TaskB", 0, success=False),
            _base_row("wechat.TaskC", 0, success=False),
            _base_row("wechat.TaskD", 0, success=False),
        ],
    )
    _write_response(run_dir, "wechat.TaskA", 0, {"error": "stagnation_detected"})
    _write_response(run_dir, "wechat.TaskB", 0, {"error": "intervention_cancelled: continuing"})
    _write_response(run_dir, "wechat.TaskC", 0, {"error": "Service unavailable: 503"})
    _write_response(run_dir, "wechat.TaskD", 0, {"error": "ValueError: unmapped action"})

    summary = load_run(run_dir, arm="A_s1_only", k=1)

    assert summary["agent_error_episodes"] == 1
    assert summary["code_error_episodes"] == 1
    assert summary["valid_episodes"] == 3
    assert summary["episode_sr"] == 0.0
    assert summary["agent_error_category_histogram"] == {"infra": 1, "our_code": 1}


def test_render_markdown_lists_agent_error_texts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_jsonl(run_dir / "results.jsonl", [_base_row("wechat.TaskA", 0, success=False)])
    _write_response(run_dir, "wechat.TaskA", 0, {"error": "ValueError: unmapped action"})

    markdown = render_markdown({"arms": {"A_s1_only": load_run(run_dir, arm="A_s1_only")}})

    assert "## Agent Error Texts" in markdown
    assert "ValueError: unmapped action" in markdown


def test_load_run_reads_single_trial_trajectory_without_trial_suffix(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_jsonl(run_dir / "results.jsonl", [_base_row("sms.OpenConversationBySender", 0, success=True)])
    _write_response_in_dir(
        run_dir / "trajectory" / "sms_OpenConversationBySender",
        {"token_usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}},
    )

    summary = load_run(run_dir, arm="A_s1_only", k=1)

    assert summary["s1_total_tokens"] == 15


def test_load_run_reads_nested_mobilegym_runs_dir(tmp_path: Path) -> None:
    arm_dir = tmp_path / "s1_fast"
    run_dir = arm_dir / "mobilegym_runs" / "20260706_000000"
    _write_jsonl(run_dir / "results.jsonl", [_base_row("wechat.TaskA", 0, success=True)])
    _write_response(run_dir, "wechat.TaskA", 0, {"token_usage": {"total_tokens": 33}})

    summary = load_run(arm_dir, arm="s1_fast", k=1)

    assert summary["tasks"] == 1
    assert summary["successes"] == 1
    assert summary["total_tokens"] == 33


def test_load_run_uses_step_trace_for_arm_calls_tokens_and_time(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    row = _base_row("wechat.TaskA", 0, success=True)
    row["execution"]["runtime_s"] = None
    row["extra_meta"] = {
        "trace_path": "traces/wechat_TaskA_t0.jsonl",
        "token_usage": {"prompt_tokens": 999, "completion_tokens": 999, "total_tokens": 1998},
    }
    _write_jsonl(run_dir / "results.jsonl", [row])
    _write_jsonl(
        run_dir / "traces" / "wechat_TaskA_t0.jsonl",
        [
            {"type": "step", "arm": "s1_fast", "token_in": 100, "token_out": 10, "duration_s": 2.0},
            {"type": "step", "arm": "s2_slow", "token_in": 200, "token_out": 20, "duration_s": 3.0},
            {"type": "step", "arm": "s1_slow", "token_in": 70, "token_out": 7, "duration_s": 5.0},
        ],
    )

    summary = load_run(run_dir, arm="B_fastslow", k=1)
    markdown = render_markdown({"arms": {"B_fastslow": summary}})

    assert summary["s1_total_tokens"] == 187
    assert summary["s2_total_tokens"] == 220
    assert summary["total_tokens"] == 407
    assert summary["avg_steps"] == 3.0
    assert summary["s2_call_rate"] == 1 / 3
    assert summary["slow_call_rate"] == 2 / 3
    assert summary["avg_time_per_task_s"] == 10.0
    assert summary["episodes_detail"][0]["trace_step_count"] == 3
    assert "avg_steps" in markdown
    assert "time/task" in markdown
    assert "S2_call" in markdown
    assert "slow_call" in markdown
