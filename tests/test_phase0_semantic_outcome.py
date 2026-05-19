from __future__ import annotations

from eval.phase0_observable_signal_pilot import (
    Phase0Task,
    infer_semantic_outcome,
)


def test_answer_required_task_done_without_answer_is_semantic_failure() -> None:
    task = Phase0Task(
        task_id="RecentTotalExpenseTask",
        instruction="",
        instruction_ch="请帮我算一下在淘宝上最近1个月我总共花了多少钱。请只回答一个整数, 不要返回任何其他文本.",
        risk_level="U0",
    )
    steps = [{"action": {"action_type": "done", "status": "success"}}]

    outcome = infer_semantic_outcome(task, steps)

    assert outcome["answer_required"] is True
    assert outcome["final_answer_present"] is False
    assert outcome["semantic_task_success"] is False
    assert outcome["semantic_success_source"] == "answer_presence_guard"
    assert outcome["semantic_success_reason"] == "missing_required_final_answer"


def test_answer_required_task_done_with_answer_remains_unjudged() -> None:
    task = Phase0Task(
        task_id="ChromeSearchBeijingWeatherTask",
        instruction="Use Chrome to search for Beijing highest temperature today. ONLY give a integer number denoted Celsius degree.",
        instruction_ch="",
        risk_level="U0",
    )
    steps = [{"action": {"action_type": "done", "status": "success", "answer": "28"}}]

    outcome = infer_semantic_outcome(task, steps)

    assert outcome["answer_required"] is True
    assert outcome["final_answer_present"] is True
    assert outcome["semantic_task_success"] is None
    assert outcome["semantic_success_source"] == "none"
    assert outcome["semantic_success_reason"] == "final_answer_present_but_unjudged"


def test_non_answer_task_has_no_semantic_label_without_judge() -> None:
    task = Phase0Task(
        task_id="AdjustBrightnessMaximumTask",
        instruction="Set the brightness to maximum.",
        instruction_ch="",
        risk_level="U1",
    )
    steps = [{"action": {"action_type": "done", "status": "success"}}]

    outcome = infer_semantic_outcome(task, steps)

    assert outcome["answer_required"] is False
    assert outcome["semantic_task_success"] is None
    assert outcome["semantic_success_source"] == "none"
    assert outcome["semantic_success_reason"] == "not_evaluated"
