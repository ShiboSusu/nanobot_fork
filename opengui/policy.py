"""Policy decisions shared by GUI execution and host-level routing.

The policy layer is intentionally conservative and cheap.  It does not try to
remember prior task trajectories; it only detects task/action text that should
not be executed autonomously without the user's confirmation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PolicyAction(str, Enum):
    ALLOW = "allow"
    ASK_HUMAN_CONFIRM = "ask_human_confirm"
    REQUIRE_HUMAN_TAKEOVER = "require_human_takeover"
    HALT = "halt"


@dataclass(frozen=True)
class PolicyDecision:
    action: PolicyAction
    categories: tuple[str, ...] = ()
    reason: str = ""
    matched_terms: tuple[str, ...] = ()
    source: str = "policy"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.action == PolicyAction.ALLOW


@dataclass(frozen=True)
class PolicyRule:
    category: str
    action: PolicyAction
    terms: tuple[str, ...]
    reason: str


class PolicyStore:
    """Small, deterministic policy store for sensitive task/action gating."""

    _DEFAULT_RULES: tuple[PolicyRule, ...] = (
        PolicyRule(
            category="login_or_auth",
            action=PolicyAction.ASK_HUMAN_CONFIRM,
            terms=(
                "登录", "登陆", "验证码", "短信验证码", "动态码", "密码",
                "二次验证", "人脸识别", "指纹", "mfa", "otp", "2fa",
                "password", "passcode", "login", "log in", "sign in",
                "authenticate", "authentication",
            ),
            reason="Login, passwords, and authentication steps require user confirmation.",
        ),
        PolicyRule(
            category="payment_or_purchase",
            action=PolicyAction.ASK_HUMAN_CONFIRM,
            terms=(
                "支付", "付款", "确认支付", "立即支付", "完成支付", "支付密码",
                "转账", "充值", "下单", "购买", "结算", "收银台",
                "pay", "payment", "purchase", "checkout", "transfer money",
                "place order",
            ),
            reason="Payment, purchases, checkout, and money movement require user confirmation.",
        ),
        PolicyRule(
            category="delete_or_irreversible",
            action=PolicyAction.ASK_HUMAN_CONFIRM,
            terms=(
                "删除", "清空", "移除", "注销", "销户", "解绑", "恢复出厂",
                "delete", "remove", "erase", "clear all", "factory reset",
                "deactivate", "close account",
            ),
            reason="Irreversible destructive actions require user confirmation.",
        ),
        PolicyRule(
            category="permission_or_authorization",
            action=PolicyAction.ASK_HUMAN_CONFIRM,
            terms=(
                "允许", "授权", "权限", "同意", "批准", "信任此电脑",
                "allow", "approve", "authorize", "grant permission", "consent",
                "trust this computer",
            ),
            reason="Permission and authorization prompts require user confirmation.",
        ),
        PolicyRule(
            category="privacy_or_personal_data",
            action=PolicyAction.ASK_HUMAN_CONFIRM,
            terms=(
                "身份证", "银行卡", "手机号", "手机号码", "地址", "通讯录",
                "相册", "定位", "隐私", "个人信息", "id card", "bank card",
                "phone number", "address", "contacts", "photos", "location",
                "personal data", "privacy",
            ),
            reason="Private or personal data operations require user confirmation.",
        ),
        PolicyRule(
            category="external_send_or_publish",
            action=PolicyAction.ASK_HUMAN_CONFIRM,
            terms=(
                "发送", "分享", "发布", "提交", "评论", "转发", "群发", "send",
                "share", "publish", "post", "submit", "comment", "forward",
            ),
            reason="External sending, publishing, or submitting requires user confirmation.",
        ),
        PolicyRule(
            category="account_security",
            action=PolicyAction.ASK_HUMAN_CONFIRM,
            terms=(
                "修改密码", "重置密码", "绑定账号", "换绑", "安全中心",
                "change password", "reset password", "bind account",
                "account security",
            ),
            reason="Account security changes require user confirmation.",
        ),
        PolicyRule(
            category="account_profile_update",
            action=PolicyAction.ASK_HUMAN_CONFIRM,
            terms=(
                "修改昵称", "改昵称", "更改昵称", "修改简介", "改简介",
                "更改简介", "编辑个人资料", "更新个人资料",
                "change nickname", "edit bio", "update bio",
                "edit profile", "profile update", "update profile",
            ),
            reason="Account profile changes require user confirmation.",
        ),
        PolicyRule(
            category="financial_or_account_read",
            action=PolicyAction.ASK_HUMAN_CONFIRM,
            terms=(
                "蚂蚁森林", "余额", "账单", "资产", "在线状态", "q我吧",
                "状态是不是", "account balance", "account status",
                "presence status",
            ),
            reason="Financial, account, and presence status reads require user confirmation.",
        ),
        PolicyRule(
            category="private_account_query",
            action=PolicyAction.ASK_HUMAN_CONFIRM,
            terms=(
                "我的订单", "我的淘宝订单", "我的京东订单", "我的美团订单",
                "我的拼多多订单", "我的携程订单", "我的滴滴订单",
                "物流到哪", "快递到哪", "退款进度", "我的优惠券",
                "我的券包", "我的积分", "会员积分", "花呗", "借呗",
                "白条", "京东金融", "总额度", "信用额度", "芝麻信用",
                "医保", "社保", "公积金", "my order", "my refund",
                "my coupon", "my points", "credit limit",
            ),
            reason="Private account, order, benefits, and membership queries require user confirmation.",
        ),
    )

    def __init__(self, rules: tuple[PolicyRule, ...] | None = None) -> None:
        self._rules = self._DEFAULT_RULES if rules is None else rules

    def match_task(self, task: str) -> PolicyDecision:
        return self.match_text(task, source="task")

    def match_action(
        self,
        *,
        task: str,
        action: Any,
        observation: Any | None = None,
        action_summary: str | None = None,
        state_summary: str | None = None,
    ) -> PolicyDecision:
        parts = [
            task,
            getattr(action, "action_type", ""),
            getattr(action, "text", ""),
            action_summary or "",
            state_summary or "",
            getattr(observation, "foreground_app", "") if observation is not None else "",
        ]
        return self.match_text(" ".join(str(part) for part in parts if part), source="action")

    def match_text(self, text: str, *, source: str = "text") -> PolicyDecision:
        normalized = self._normalize(text)
        if not normalized:
            return PolicyDecision(action=PolicyAction.ALLOW, source=source)

        if source == "task" and (
            self._is_explicit_read_only_privacy_setting_check(normalized)
            or self._is_user_authorized_read_only_order_query(normalized)
            or self._is_read_only_published_content_query(normalized)
        ):
            return PolicyDecision(
                action=PolicyAction.ALLOW,
                source=source,
                metadata={"task_level_read_only_exception": True},
            )

        matched_categories: list[str] = []
        matched_terms: list[str] = []
        reasons: list[str] = []
        selected_action = PolicyAction.ALLOW

        for rule in self._rules:
            terms = [term for term in rule.terms if self._contains_term(normalized, term)]
            if not terms:
                continue
            matched_categories.append(rule.category)
            matched_terms.extend(terms)
            reasons.append(rule.reason)
            selected_action = self._max_action(selected_action, rule.action)

        if selected_action == PolicyAction.ALLOW:
            return PolicyDecision(action=PolicyAction.ALLOW, source=source)

        return PolicyDecision(
            action=selected_action,
            categories=tuple(dict.fromkeys(matched_categories)),
            reason=" ".join(dict.fromkeys(reasons)),
            matched_terms=tuple(dict.fromkeys(matched_terms)),
            source=source,
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").casefold()).strip()

    @staticmethod
    def _contains_term(text: str, term: str) -> bool:
        needle = PolicyStore._normalize(term)
        if not needle:
            return False
        if re.fullmatch(r"[a-z0-9_ -]+", needle):
            return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", text) is not None
        compact_text = PolicyStore._compact_obfuscated_cjk(text)
        compact_needle = PolicyStore._compact_obfuscated_cjk(needle)
        if compact_needle == "支付":
            return re.search(r"支付(?!宝)", compact_text) is not None
        return needle in text or compact_needle in compact_text

    @classmethod
    def _is_explicit_read_only_privacy_setting_check(cls, text: str) -> bool:
        compact = cls._compact_obfuscated_cjk(text)
        privacy_terms = (
            "隐私", "公开", "不公开", "关注列表", "粉丝列表", "空间设置",
            "安全隐私", "privacy", "public", "private", "followers", "following",
        )
        check_terms = (
            "检查", "查看", "看看", "确认", "是不是", "是否", "状态",
            "只查看", "只读取", "check", "view", "inspect", "read only",
        )
        read_only_terms = (
            "只查看", "仅查看", "只读取", "仅读取", "不要修改", "不修改",
            "不要更改", "不更改", "不要切换", "不切换", "不要点", "不点",
            "read only", "do not modify", "don't modify", "do not change",
            "don't change", "do not toggle", "don't toggle",
        )
        mutation_terms = (
            "开启", "关闭", "切换", "改成", "设置为", "设为", "更改为",
            "修改为", "打开开关", "关闭开关", "开关打开", "开关关闭",
            "toggle", "enable", "disable", "turn on", "turn off", "change to",
            "set to",
        )
        has_privacy_context = any(cls._contains_term(text, term) for term in privacy_terms)
        has_check_intent = any(cls._contains_term(text, term) for term in check_terms)
        has_read_only_constraint = any(cls._contains_term(text, term) for term in read_only_terms)
        mutation_text = text
        mutation_compact = compact
        for term in read_only_terms:
            normalized_term = cls._normalize(term)
            if normalized_term:
                mutation_text = mutation_text.replace(normalized_term, " ")
            compact_term = cls._compact_obfuscated_cjk(normalized_term)
            if compact_term:
                mutation_compact = mutation_compact.replace(compact_term, "")
        has_mutation_intent = any(
            cls._contains_term(mutation_text, term)
            or cls._compact_obfuscated_cjk(term) in mutation_compact
            for term in mutation_terms
        )
        return (
            has_privacy_context
            and has_check_intent
            and has_read_only_constraint
            and not has_mutation_intent
        )

    @classmethod
    def _is_user_authorized_read_only_order_query(cls, text: str) -> bool:
        read_terms = (
            "查看", "看看", "显示", "查询", "查一下", "检查", "进入",
            "view", "show", "check", "look up", "inspect",
        )
        order_terms = (
            "订单", "交易", "物流", "快递", "运单", "配送", "包裹",
            "order", "transaction", "logistics", "shipment", "delivery", "package",
        )
        mutation_terms = (
            "支付", "付款", "确认支付", "下单", "购买", "结算", "删除",
            "清空", "取消订单", "确认收货", "退款", "退货", "投诉", "评价",
            "pay", "payment", "purchase", "checkout", "delete", "cancel order",
            "confirm receipt", "refund", "return goods", "review",
        )
        return (
            any(cls._contains_term(text, term) for term in read_terms)
            and any(cls._contains_term(text, term) for term in order_terms)
            and not any(cls._contains_term(text, term) for term in mutation_terms)
        )

    @classmethod
    def _is_read_only_published_content_query(cls, text: str) -> bool:
        read_terms = (
            "看看", "查看", "查询", "搜索", "点开", "哪些", "什么",
            "看一下", "view", "check", "search", "what",
        )
        published_content_patterns = (
            r"发布(?:了|过|的)?(?:哪些|什么|其他|更多|笔记|内容|作品|视频)",
            r"(?:哪些|什么|其他|更多).{0,12}发布(?:了|过|的)?",
            r"(?:posted|published).{0,24}(?:notes|content|posts|videos|works)",
        )
        publish_action_terms = (
            "发布到", "发布一篇", "发布笔记", "发布视频", "发表", "提交",
            "发出去", "公开发布", "publish a", "post a", "submit",
        )
        has_read_intent = any(cls._contains_term(text, term) for term in read_terms)
        has_published_content_context = any(
            re.search(pattern, text, flags=re.IGNORECASE)
            for pattern in published_content_patterns
        )
        has_publish_action = any(cls._contains_term(text, term) for term in publish_action_terms)
        return has_read_intent and has_published_content_context and not has_publish_action

    @staticmethod
    def _compact_obfuscated_cjk(text: str) -> str:
        return re.sub(r"[\s\u200b\u200c\u200d\ufeff]+", "", text or "")

    @staticmethod
    def _max_action(left: PolicyAction, right: PolicyAction) -> PolicyAction:
        order = {
            PolicyAction.ALLOW: 0,
            PolicyAction.ASK_HUMAN_CONFIRM: 1,
            PolicyAction.REQUIRE_HUMAN_TAKEOVER: 2,
            PolicyAction.HALT: 3,
        }
        return left if order[left] >= order[right] else right
