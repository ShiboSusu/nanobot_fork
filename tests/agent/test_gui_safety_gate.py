from __future__ import annotations

from nanobot.agent.gui_safety import GuiSafetyRisk, check_gui_safety


def test_query_task_allowed() -> None:
    decision = check_gui_safety(task="在微博查看今天热搜榜第三名是什么", app_hint="微博")

    assert decision.allowed is True
    assert decision.requires_human_confirm is False


def test_external_send_requires_confirm() -> None:
    decision = check_gui_safety(
        task="在微信把 hot_rank_3 发给张三",
        app_hint="微信",
        known_values={"hot_rank_3": "测试事件A"},
    )

    assert decision.allowed is False
    assert decision.requires_human_confirm is True
    assert decision.risk == GuiSafetyRisk.EXTERNAL_SEND


def test_prepare_before_send_is_allowed_at_subtask_level() -> None:
    decision = check_gui_safety(
        task="在微信准备把 hot_rank_3 发给张三，但停在发送前，不要真的发送",
        app_hint="微信",
        known_values={"hot_rank_3": "测试事件A"},
        stage="before_subtask",
    )

    assert decision.allowed is True


def test_prepare_before_send_is_allowed_at_gui_task_level() -> None:
    decision = check_gui_safety(
        task="打开微信准备发给张三，但停在发送前，不要真的发送",
        app_hint="微信",
        stage="before_gui_task",
    )

    assert decision.allowed is True
    assert decision.requires_human_confirm is False


def test_tap_send_is_blocked_at_action_level_even_if_prepare_task() -> None:
    decision = check_gui_safety(
        task="在微信准备消息但不要真的发送",
        app_hint="微信",
        stage="before_action",
        action_summary="点击发送按钮",
        state_summary="消息编辑完成",
    )

    assert decision.allowed is False
    assert decision.requires_human_confirm is True


def test_tap_send_is_still_blocked_at_action_level_with_prepare_task() -> None:
    decision = check_gui_safety(
        task="打开微信准备发给张三，但停在发送前，不要真的发送",
        app_hint="微信",
        stage="before_action",
        action_summary="点击发送按钮",
        state_summary="消息已编辑完成",
    )

    assert decision.allowed is False
    assert decision.requires_human_confirm is True
    assert decision.risk == GuiSafetyRisk.EXTERNAL_SEND


def test_payment_requires_confirm() -> None:
    decision = check_gui_safety(task="打开支付宝付款给张三")

    assert decision.allowed is False
    assert decision.risk == GuiSafetyRisk.PAYMENT


def test_order_submit_requires_confirm() -> None:
    decision = check_gui_safety(task="提交淘宝订单")

    assert decision.allowed is False
    assert decision.risk == GuiSafetyRisk.ORDER_SUBMIT
