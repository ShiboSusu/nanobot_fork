from __future__ import annotations

from nanobot.agent.gui_task_schema import normalize_gui_task_request
from opengui.evidence import apply_evidence_contract


def test_existing_answer_candidates_preserved() -> None:
    request = normalize_gui_task_request({"task": "打开微博,看看今天热搜榜的第三名是什么"})
    payload = {
        "success": True,
        "answer_candidates": [{"key": "hot_rank_3", "text": "测试事件A"}],
    }

    out = apply_evidence_contract(request=request, payload=payload)

    assert out["success"] is True
    assert out["answer_candidates"][0]["text"] == "测试事件A"


def test_blackboard_meta_becomes_answer_candidate() -> None:
    request = normalize_gui_task_request({"task": "打开微博查热搜第三名，然后不要发送"})
    payload = {"success": True}

    out = apply_evidence_contract(
        request=request,
        payload=payload,
        blackboard_meta={
            "hot_rank_3": {
                "value": "测试事件A",
                "source_subtask": "在微博查看热搜第三名",
                "source_app_hint": "微博",
                "confidence": 0.8,
                "evidence_refs": ["trace.jsonl"],
            }
        },
    )

    assert out["answer_candidates"][0]["key"] == "hot_rank_3"
    assert out["answer_candidates"][0]["text"] == "测试事件A"


def test_model_summary_alone_does_not_extract_answer_candidate() -> None:
    request = normalize_gui_task_request({"task": "打开微博,看看今天热搜榜的第三名是什么"})
    payload = {
        "success": True,
        "model_summary": "微博热搜榜第三名是测试事件A",
    }

    out = apply_evidence_contract(request=request, payload=payload)

    assert out["success"] is False
    assert out["error"] == "missing_required_answer_evidence"
    assert out["answer_candidates"] == []


def test_information_query_without_candidate_is_downgraded() -> None:
    request = normalize_gui_task_request({"task": "打开微博,看看今天热搜榜的第三名是什么"})
    payload = {
        "success": True,
        "summary": "Status: completed Done: 已打开微博",
        "model_summary": "已打开微博页面",
    }

    out = apply_evidence_contract(request=request, payload=payload)

    assert out["success"] is False
    assert out["status"] == "partial"
    assert out["error"] == "missing_required_answer_evidence"
    assert out["uncertainty"]["level"] == "high"


def test_operation_task_does_not_require_answer_candidates() -> None:
    request = normalize_gui_task_request({"task": "打开淘宝搜索蓝牙耳机"})
    payload = {
        "success": True,
        "summary": "Status: completed Done: 已搜索蓝牙耳机",
    }

    out = apply_evidence_contract(request=request, payload=payload)

    assert out["success"] is True
