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


def test_noop_skill_library_keeps_skill_interface_disabled() -> None:
    library = NoopSkillLibrary()

    assert library.search(task="打开设置", observation=None) == []
