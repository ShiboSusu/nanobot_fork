from __future__ import annotations

from nanobot.agent.gui_task_schema import normalize_gui_task_request
from opengui.evidence import apply_evidence_contract


def test_page_not_loaded_text_is_not_generic_answer_false_positive() -> None:
    request = normalize_gui_task_request({"task": "打开微博，查询今天热搜榜第三名是什么"})
    payload = {
        "success": True,
        "model_summary": "任务是查询热搜，但页面未加载",
    }

    out = apply_evidence_contract(request=request, payload=payload)

    assert out["success"] is False
    assert out["error"] == "missing_required_answer_evidence"
    assert out["answer_candidates"] == []
