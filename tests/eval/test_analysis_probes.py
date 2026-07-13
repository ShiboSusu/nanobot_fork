from __future__ import annotations

from eval.analysis.confidence_probe import analyze as analyze_confidence
from eval.analysis.oracle import analyze as analyze_oracle
from eval.mobilegym.aggregate import load_run


def _summary(model: str, episodes: list[dict]) -> dict:
    actor = "s2" if model == "397B" else "s1"
    other = "s1" if actor == "s2" else "s2"
    return {
        "s1_model": "9B" if actor == "s1" else None,
        "s2_model": "397B" if actor == "s2" else None,
        "episode_sr": sum(1 for ep in episodes if ep["success"]) / len(episodes),
        "episodes_detail": [
            {
                **ep,
                f"{actor}_token_usage": ep["usage"],
                f"{other}_token_usage": {},
                "judge_error": False,
                "execution_error": False,
                "agent_error": False,
            }
            for ep in episodes
        ],
    }


def test_oracle_selects_cheapest_successful_arm_per_task() -> None:
    aggregate = {
        "arms": {
            "s1_fast": _summary("9B", [
                {"task_id": "A", "success": True, "usage": {"prompt_tokens": 100, "completion_tokens": 10}},
                {"task_id": "B", "success": False, "usage": {"prompt_tokens": 100, "completion_tokens": 10}},
            ]),
            "s2_slow": _summary("397B", [
                {"task_id": "A", "success": True, "usage": {"prompt_tokens": 100, "completion_tokens": 10}},
                {"task_id": "B", "success": True, "usage": {"prompt_tokens": 100, "completion_tokens": 10}},
            ]),
        }
    }

    result = analyze_oracle(aggregate)

    assert result["tasks"] == 2
    assert result["oracle_successes"] == 2
    assert result["oracle_tasks"][0]["arm"] == "s1_fast"
    assert result["oracle_tasks"][1]["arm"] == "s2_slow"
    assert "optimistic upper bound" in result["note"]


def test_confidence_probe_reports_confident_wrong_and_aurocs() -> None:
    result = analyze_confidence(
        [
            {"correct": True, "temp_consistency": 1.0, "perturb_consistency": 1.0, "action_type": "tap"},
            {"correct": False, "temp_consistency": 1.0, "perturb_consistency": 1.0, "action_type": "tap"},
            {"correct": False, "temp_consistency": 0.34, "perturb_consistency": 0.34, "action_type": "wait"},
            {"correct": True, "temp_actions": ["a", "a", "b"], "perturb_actions": ["a", "a", "a"]},
        ],
        threshold=2 / 3,
    )

    assert result["confusion"] == {
        "confident_correct": 2,
        "confident_wrong": 1,
        "uncertain_correct": 0,
        "uncertain_wrong": 1,
    }
    assert result["high_stakes_confident_wrong_rate_among_wrong"] == 1.0
    assert result["auroc_error_detection"]["combined_uncertainty"] is not None


def test_s2_arm_infers_397b_for_cost_split(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.jsonl").write_text(
        '{"task_id":"A","trial_id":0,"is_success":true,"execution":{},"judge":{}}\n',
        encoding="utf-8",
    )

    summary = load_run(run_dir, arm="s2_slow")

    assert summary["s1_model"] == "397B"
    assert summary["s2_model"] == "397B"
