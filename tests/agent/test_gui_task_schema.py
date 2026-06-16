from __future__ import annotations

from nanobot.agent.gui_task_schema import (
    GuiOutputMode,
    GuiTaskType,
    normalize_gui_task_request,
)


def test_normalize_operation_task() -> None:
    request = normalize_gui_task_request({"task": "打开淘宝搜索蓝牙耳机"})

    assert request.task_type == GuiTaskType.OPERATION
    assert request.output_mode == GuiOutputMode.OPERATION_STATUS
    assert request.evidence_requirements.answer_candidates_required is False


def test_normalize_weibo_hot_rank_information_query() -> None:
    request = normalize_gui_task_request({"task": "打开微博,看看今天热搜榜的第三名是什么"})

    assert request.task_type == GuiTaskType.INFORMATION_QUERY
    assert request.output_mode == GuiOutputMode.ANSWER_REQUIRED
    assert request.success_condition.required_key == "answer"
    assert request.success_condition.required_fields == ()
    assert request.evidence_requirements.answer_candidates_required is True


def test_normalize_sensitive_action() -> None:
    request = normalize_gui_task_request({"task": "打开支付宝付款给张三"})

    assert request.task_type == GuiTaskType.SENSITIVE_ACTION
    assert request.output_mode == GuiOutputMode.NEEDS_HUMAN_CONFIRM


def test_normalize_mixed_query_and_action() -> None:
    request = normalize_gui_task_request({"task": "打开微博查热搜第三名，然后发给微信里的张三"})

    assert request.task_type == GuiTaskType.MIXED_QUERY_AND_ACTION
    assert request.output_mode == GuiOutputMode.ANSWER_REQUIRED
    assert request.evidence_requirements.answer_candidates_required is True


def test_sensitive_external_share_synonyms_are_detected() -> None:
    for text in (
        "找通话截图发我",
        "把截图分享给我",
        "share the screenshot with me",
        "转发给张三",
    ):
        request = normalize_gui_task_request({"task": text})

        assert request.task_type == GuiTaskType.SENSITIVE_ACTION
        assert request.output_mode == GuiOutputMode.NEEDS_HUMAN_CONFIRM


def test_explicit_request_payload_is_preserved() -> None:
    request = normalize_gui_task_request(
        {
            "schema_version": "gui_task_request.v1",
            "task": "查一下快递到哪了",
            "task_type": "information_query",
            "output_mode": "answer_required",
            "success_condition": {
                "type": "extract_field",
                "required_key": "logistics_status",
                "description": "获得物流状态",
            },
        }
    )

    assert request.success_condition.required_key == "logistics_status"
