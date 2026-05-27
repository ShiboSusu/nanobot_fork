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
                "发送", "发布", "提交", "评论", "转发", "群发", "send",
                "publish", "post", "submit", "comment", "forward",
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
