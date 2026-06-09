from __future__ import annotations

from nanobot.agent.cost_aware_router import (
    CostAwareProblemRouter,
    NoopSkillLibrary,
    PolicyAction,
    PolicyStore,
    RouteKind,
)


def test_sensitive_task_requires_human_confirmation() -> None:
    router = CostAwareProblemRouter(policy_store=PolicyStore(), skill_library=NoopSkillLibrary())

    decision = router.classify("帮我登录支付宝并完成付款", available_tools={"gui_task", "web_search"})

    assert decision.route == RouteKind.HUMAN_CONFIRM
    assert decision.policy.action == PolicyAction.ASK_HUMAN_CONFIRM
    assert "login_or_auth" in decision.policy.categories
    assert "payment_or_purchase" in decision.policy.categories


def test_system_action_prefers_device_system_route() -> None:
    router = CostAwareProblemRouter(policy_store=PolicyStore(), skill_library=NoopSkillLibrary())

    decision = router.classify("帮我用手机打开设置", available_tools={"gui_task", "web_search"})

    assert decision.route == RouteKind.SYSTEM_ACTION
    assert decision.system_action == {
        "backend": None,
        "task": "帮我用手机打开设置",
        "intent": "open_settings",
    }
    assert decision.requires_gui is False


def test_system_action_prefers_known_app_open_route() -> None:
    router = CostAwareProblemRouter(policy_store=PolicyStore(), skill_library=NoopSkillLibrary())

    decision = router.classify("帮我用手机打开 Safari app", available_tools={"gui_task", "web_search"})

    assert decision.route == RouteKind.SYSTEM_ACTION
    assert decision.system_action == {
        "backend": None,
        "task": "帮我用手机打开 Safari app",
        "intent": "open_app",
    }
    assert decision.requires_gui is False


def test_system_action_prefers_common_chinese_app_open_route() -> None:
    router = CostAwareProblemRouter(policy_store=PolicyStore(), skill_library=NoopSkillLibrary())

    decision = router.classify("打开微信", available_tools={"gui_task", "web_search"})

    assert decision.route == RouteKind.SYSTEM_ACTION
    assert decision.system_action == {
        "backend": None,
        "task": "打开微信",
        "intent": "open_app",
    }
    assert decision.requires_gui is False


def test_compound_app_task_uses_gui_not_system_or_web_search() -> None:
    router = CostAwareProblemRouter(policy_store=PolicyStore(), skill_library=NoopSkillLibrary())

    decision = router.classify("打开设置，然后点击搜索框", available_tools={"gui_task", "web_search"})

    assert decision.route == RouteKind.GUI
    assert decision.requires_gui is True


def test_non_device_open_task_does_not_use_system_action() -> None:
    router = CostAwareProblemRouter(policy_store=PolicyStore(), skill_library=NoopSkillLibrary())

    decision = router.classify("打开 OpenAI 官网", available_tools={"gui_task", "web_search"})

    assert decision.route != RouteKind.SYSTEM_ACTION


def test_query_task_prefers_tool_call_without_gui() -> None:
    router = CostAwareProblemRouter(policy_store=PolicyStore(), skill_library=NoopSkillLibrary())

    decision = router.classify("查一下今天深圳天气", available_tools={"gui_task", "web_search"})

    assert decision.route == RouteKind.TOOL_CALL
    assert decision.suggested_tools == ("web_search", "web_fetch")
    assert decision.requires_gui is False


def test_in_app_search_uses_gui_not_web_search() -> None:
    router = CostAwareProblemRouter(policy_store=PolicyStore(), skill_library=NoopSkillLibrary())

    decision = router.classify("在抖音里搜索“旅行Vlog”", available_tools={"gui_task", "web_search"})

    assert decision.route == RouteKind.GUI
    assert decision.requires_gui is True


def test_train_schedule_query_prefers_tool_call_without_gui() -> None:
    router = CostAwareProblemRouter(policy_store=PolicyStore(), skill_library=NoopSkillLibrary())

    decision = router.classify("看看今天下午去天津的火车，要快一点的", available_tools={"gui_task", "web_search"})

    assert decision.route == RouteKind.TOOL_CALL
    assert decision.suggested_tools == ("web_search", "web_fetch")
    assert decision.requires_gui is False


def test_financial_account_read_requires_human_confirmation() -> None:
    router = CostAwareProblemRouter(policy_store=PolicyStore(), skill_library=NoopSkillLibrary())

    decision = router.classify("打开支付宝，看看我的蚂蚁森林能量有多少克", available_tools={"gui_task", "web_search"})

    assert decision.route == RouteKind.HUMAN_CONFIRM
    assert decision.policy.action == PolicyAction.ASK_HUMAN_CONFIRM
    assert "financial_or_account_read" in decision.policy.categories


def test_account_presence_read_requires_human_confirmation() -> None:
    router = CostAwareProblemRouter(policy_store=PolicyStore(), skill_library=NoopSkillLibrary())

    decision = router.classify("检查我的QQ在线状态是不是“Q我吧”", available_tools={"gui_task", "web_search"})

    assert decision.route == RouteKind.HUMAN_CONFIRM
    assert decision.policy.action == PolicyAction.ASK_HUMAN_CONFIRM
    assert "financial_or_account_read" in decision.policy.categories


def test_user_authorized_private_account_query_uses_gui_not_web_search() -> None:
    router = CostAwareProblemRouter(policy_store=PolicyStore(), skill_library=NoopSkillLibrary())

    decision = router.classify("查一下我的淘宝订单到哪了", available_tools={"gui_task", "web_search"})

    assert decision.route == RouteKind.GUI
    assert decision.policy.action == PolicyAction.ALLOW
    assert decision.requires_gui is True


def test_user_authorized_read_only_order_logistics_query_uses_gui() -> None:
    router = CostAwareProblemRouter(policy_store=PolicyStore(), skill_library=NoopSkillLibrary())

    decision = router.classify(
        "打开淘宝,进入我的订单页面,查看最近一笔交易的物流信息。",
        available_tools={"gui_task", "web_search"},
    )

    assert decision.route == RouteKind.GUI
    assert decision.policy.action == PolicyAction.ALLOW
    assert decision.requires_gui is True


def test_read_only_published_content_query_uses_gui_not_publish_policy() -> None:
    router = CostAwareProblemRouter(policy_store=PolicyStore(), skill_library=NoopSkillLibrary())

    decision = router.classify(
        "在小红书里搜索“健身餐”,点开第一篇笔记,然后看看作者还发布了哪些其他笔记。",
        available_tools={"gui_task", "web_search"},
    )

    assert decision.route == RouteKind.GUI
    assert decision.policy.action == PolicyAction.ALLOW
    assert decision.requires_gui is True


def test_external_share_task_still_requires_human_confirmation() -> None:
    router = CostAwareProblemRouter(policy_store=PolicyStore(), skill_library=NoopSkillLibrary())

    decision = router.classify(
        "打开抖音,把第一个视频分享给微信好友“Su4o_”",
        available_tools={"gui_task", "web_search"},
    )

    assert decision.route == RouteKind.HUMAN_CONFIRM
    assert decision.policy.action == PolicyAction.ASK_HUMAN_CONFIRM
    assert "external_send_or_publish" in decision.policy.categories


def test_read_only_privacy_setting_check_uses_gui_without_confirmation() -> None:
    router = CostAwareProblemRouter(policy_store=PolicyStore(), skill_library=NoopSkillLibrary())

    decision = router.classify(
        "检查B站安全隐私的空间设置里我的关注列表设置是不是不公开，只查看当前设置，不要修改任何开关",
        available_tools={"gui_task", "web_search"},
    )

    assert decision.route == RouteKind.GUI
    assert decision.policy.action == PolicyAction.ALLOW
    assert decision.requires_gui is True


def test_privacy_setting_change_still_requires_human_confirmation() -> None:
    router = CostAwareProblemRouter(policy_store=PolicyStore(), skill_library=NoopSkillLibrary())

    decision = router.classify(
        "在B站安全隐私的空间设置里开启公开我的关注列表开关",
        available_tools={"gui_task", "web_search"},
    )

    assert decision.route == RouteKind.HUMAN_CONFIRM
    assert decision.policy.action == PolicyAction.ASK_HUMAN_CONFIRM
    assert "privacy_or_personal_data" in decision.policy.categories


def test_jd_baitiao_credit_limit_requires_human_confirmation() -> None:
    router = CostAwareProblemRouter(policy_store=PolicyStore(), skill_library=NoopSkillLibrary())

    decision = router.classify(
        "在京东金融里查看一下我的白条总额度是多少",
        available_tools={"gui_task", "web_search"},
    )

    assert decision.route == RouteKind.HUMAN_CONFIRM
    assert decision.policy.action == PolicyAction.ASK_HUMAN_CONFIRM
    assert "private_account_query" in decision.policy.categories


def test_sensitive_financial_app_open_requires_human_confirmation() -> None:
    router = CostAwareProblemRouter(policy_store=PolicyStore(), skill_library=NoopSkillLibrary())

    decision = router.classify("打开支付宝", available_tools={"gui_task", "web_search"})

    assert decision.route == RouteKind.HUMAN_CONFIRM
    assert decision.policy.action == PolicyAction.ASK_HUMAN_CONFIRM
    assert "sensitive_app_open" in decision.policy.categories


def test_policy_matches_obfuscated_sensitive_chinese_term() -> None:
    router = CostAwareProblemRouter(policy_store=PolicyStore(), skill_library=NoopSkillLibrary())

    decision = router.classify("帮我支 付这个订单", available_tools={"gui_task", "web_search"})

    assert decision.route == RouteKind.HUMAN_CONFIRM
    assert decision.policy.action == PolicyAction.ASK_HUMAN_CONFIRM
    assert "payment_or_purchase" in decision.policy.categories


def test_noop_skill_library_keeps_skill_interface_disabled() -> None:
    library = NoopSkillLibrary()

    assert library.search(task="打开设置", observation=None) == []
