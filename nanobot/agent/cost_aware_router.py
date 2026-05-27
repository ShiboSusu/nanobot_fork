"""Cost-aware problem routing before the main agent loop spends model tokens."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from opengui.policy import PolicyAction, PolicyDecision, PolicyStore


class RouteKind(str, Enum):
    DIRECT_ANSWER = "direct_answer"
    TOOL_CALL = "tool_call"
    SYSTEM_ACTION = "system_action"
    SKILL = "skill"
    GUI = "gui"
    S2 = "s2"
    HUMAN_CONFIRM = "human_confirm"


@dataclass(frozen=True)
class SkillCandidate:
    skill_id: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


class SkillLibrary(Protocol):
    def search(self, *, task: str, observation: Any | None = None) -> list[SkillCandidate]:
        ...


class NoopSkillLibrary:
    """V0 skill interface: deliberately disabled, but the dependency boundary exists."""

    def search(self, *, task: str, observation: Any | None = None) -> list[SkillCandidate]:
        del task, observation
        return []


@dataclass(frozen=True)
class RouteDecision:
    route: RouteKind
    reason: str
    policy: PolicyDecision = field(default_factory=lambda: PolicyDecision(PolicyAction.ALLOW))
    suggested_tools: tuple[str, ...] = ()
    requires_gui: bool = False
    system_action: dict[str, Any] | None = None
    skill_candidates: tuple[SkillCandidate, ...] = ()


class CostAwareProblemRouter:
    """Classify a user task into the cheapest safe route that can solve it."""

    _QUERY_TERMS = (
        "查", "查询", "搜索", "天气", "新闻", "汇率", "价格", "百科",
        "资料", "信息", "路线", "地址", "几点", "多少", "最新",
        "search", "look up", "weather", "news", "exchange rate", "price",
        "what is", "who is", "when is", "where is", "how many",
    )
    _OPEN_TERMS = ("打开", "开启", "启动", "open", "launch")
    _APP_LOOKUP_TERMS = ("查找", "寻找", "找到", "搜索", "find", "search")
    _APP_TERMS = ("app", "应用", "软件")
    _SETTINGS_TERMS = ("设置", "settings")
    _FOLLOW_UP_TERMS = (
        "然后", "之后", "接着", "再", "并", "并且", "同时",
        "点击", "点一下", "输入", "搜索框", "填写", "发送", "发消息",
        "购买", "付款", "登录", "选择", "切换", "查看", "进入", "改成",
        "调到", "滑动",
        "then", "and then", "tap", "click", "type", "send", "pay",
        "login", "log in", "select", "switch",
    )

    def __init__(
        self,
        *,
        policy_store: PolicyStore | None = None,
        skill_library: SkillLibrary | None = None,
    ) -> None:
        self._policy_store = policy_store or PolicyStore()
        self._skill_library = skill_library or NoopSkillLibrary()

    def classify(
        self,
        task: str,
        *,
        available_tools: set[str] | frozenset[str] | None = None,
    ) -> RouteDecision:
        available = set(available_tools or set())
        normalized = self._normalize(task)

        policy = self._policy_store.match_task(task)
        if not policy.allowed:
            return RouteDecision(
                route=RouteKind.HUMAN_CONFIRM,
                reason="Task-level policy requires user confirmation before autonomous execution.",
                policy=policy,
                requires_gui=False,
            )

        skill_candidates = tuple(self._skill_library.search(task=task, observation=None))
        if skill_candidates:
            return RouteDecision(
                route=RouteKind.SKILL,
                reason="A reusable skill matched the task.",
                policy=policy,
                skill_candidates=skill_candidates,
                requires_gui=False,
            )

        system_intent = self._system_action_intent(normalized)
        if system_intent and "gui_task" in available:
            return RouteDecision(
                route=RouteKind.SYSTEM_ACTION,
                reason="A device system action can solve this without visual exploration.",
                policy=policy,
                requires_gui=False,
                system_action={
                    "backend": None,
                    "task": task,
                    "intent": system_intent,
                },
            )

        if self._is_query_task(normalized) and "web_search" in available:
            return RouteDecision(
                route=RouteKind.TOOL_CALL,
                reason="Information lookup should use tools before GUI automation.",
                policy=policy,
                suggested_tools=("web_search", "web_fetch"),
                requires_gui=False,
            )

        if "gui_task" in available:
            return RouteDecision(
                route=RouteKind.GUI,
                reason="Task may require device/app state; GUI is available as fallback.",
                policy=policy,
                requires_gui=True,
            )

        return RouteDecision(
            route=RouteKind.S2,
            reason="No cheap deterministic route matched; continue through the main agent.",
            policy=policy,
            requires_gui=False,
        )

    def _system_action_intent(self, normalized: str) -> str | None:
        if not normalized or self._has_follow_up_gui_work(normalized):
            return None
        wants_open = any(term in normalized for term in self._OPEN_TERMS)
        wants_app_lookup = (
            any(term in normalized for term in self._APP_LOOKUP_TERMS)
            and any(term in normalized for term in self._APP_TERMS)
        )
        mentions_device_app = any(term in normalized for term in (
            *self._APP_TERMS,
            "手机", "iphone", "ios",
            *self._SETTINGS_TERMS,
        ))
        if not ((wants_open and mentions_device_app) or wants_app_lookup):
            return None
        if any(term in normalized for term in self._SETTINGS_TERMS):
            return "open_settings"
        return "open_app"

    def _has_follow_up_gui_work(self, normalized: str) -> bool:
        if any(term in normalized for term in self._FOLLOW_UP_TERMS):
            return True
        if "搜索" in normalized and not any(term in normalized for term in self._APP_TERMS):
            return True
        return False

    def _is_query_task(self, normalized: str) -> bool:
        if self._has_device_gui_context(normalized):
            return False
        return any(term in normalized for term in self._QUERY_TERMS)

    def _has_device_gui_context(self, normalized: str) -> bool:
        return any(term in normalized for term in (
            "手机", "iphone", "ios", "app", "应用", "软件", "设置", "搜索框",
            "点击", "输入", "打开", "启动", "open", "launch", "tap", "click",
        ))

    @staticmethod
    def _normalize(task: str) -> str:
        return " ".join((task or "").casefold().split())


__all__ = [
    "CostAwareProblemRouter",
    "NoopSkillLibrary",
    "PolicyAction",
    "PolicyDecision",
    "PolicyStore",
    "RouteDecision",
    "RouteKind",
    "SkillCandidate",
    "SkillLibrary",
]
