from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GuiSafetyRisk(str, Enum):
    EXTERNAL_SEND = "external_send"
    PAYMENT = "payment"
    ORDER_SUBMIT = "order_submit"
    DELETE = "delete"
    LOGIN_SECRET = "login_secret"
    PRIVACY_CHANGE = "privacy_change"
    UNKNOWN_SENSITIVE = "unknown_sensitive"


@dataclass(frozen=True)
class GuiSafetyDecision:
    allowed: bool
    requires_human_confirm: bool
    risk: GuiSafetyRisk | None = None
    reason: str = ""
    pending_action: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "requires_human_confirm": self.requires_human_confirm,
            "risk": self.risk.value if self.risk is not None else None,
            "reason": self.reason,
            "pending_action": dict(self.pending_action),
        }


_RISK_KEYWORDS: tuple[tuple[GuiSafetyRisk, tuple[str, ...]], ...] = (
    (
        GuiSafetyRisk.PAYMENT,
        ("付款", "支付", "转账", "收款", "pay", "transfer"),
    ),
    (
        GuiSafetyRisk.ORDER_SUBMIT,
        ("提交订单", "下单", "购买", "确认订单", "提交", "submit order", "submit", "checkout"),
    ),
    (
        GuiSafetyRisk.DELETE,
        ("删除", "清空", "移除", "delete", "remove"),
    ),
    (
        GuiSafetyRisk.LOGIN_SECRET,
        ("验证码", "密码", "支付密码", "人脸", "登录", "授权", "code", "password", "otp"),
    ),
    (
        GuiSafetyRisk.PRIVACY_CHANGE,
        ("隐私", "权限", "公开", "关闭可见", "privacy", "permission"),
    ),
    (
        GuiSafetyRisk.EXTERNAL_SEND,
        ("发送", "发给", "转发", "发消息", "评论", "发帖", "send", "forward", "post", "comment"),
    ),
)

_PREPARE_ONLY_PHRASES = (
    "不要真的发送",
    "不要发送",
    "发送前停住",
    "停在发送前",
    "仅准备",
    "草稿",
    "不提交",
    "不付款",
)


def check_gui_safety(
    *,
    task: str,
    app_hint: str | None = None,
    known_values: dict[str, str] | None = None,
    stage: str = "before_subtask",
    action_summary: str | None = None,
    state_summary: str | None = None,
) -> GuiSafetyDecision:
    known_values = dict(known_values or {})
    task_text = str(task or "")
    stage_text = str(stage or "before_subtask")
    combined = " ".join(
        item
        for item in (task_text, str(action_summary or ""), str(state_summary or ""))
        if item
    )
    risk = _detect_risk(combined)
    if risk is None:
        return GuiSafetyDecision(allowed=True, requires_human_confirm=False)

    if stage_text == "before_subtask" and _is_prepare_only(task_text):
        return GuiSafetyDecision(allowed=True, requires_human_confirm=False)

    pending_action = {
        "type": risk.value,
        "app_hint": app_hint,
        "task": task_text,
        "known_values": known_values,
        "stage": stage_text,
    }
    if action_summary:
        pending_action["action_summary"] = str(action_summary)
    if state_summary:
        pending_action["state_summary"] = str(state_summary)

    return GuiSafetyDecision(
        allowed=False,
        requires_human_confirm=True,
        risk=risk,
        reason=_reason_for_risk(risk),
        pending_action=pending_action,
    )


def _detect_risk(text: str) -> GuiSafetyRisk | None:
    lowered = text.casefold()
    for risk, keywords in _RISK_KEYWORDS:
        if any(keyword.casefold() in lowered for keyword in keywords):
            return risk
    return None


def _is_prepare_only(task: str) -> bool:
    lowered = task.casefold()
    return any(phrase.casefold() in lowered for phrase in _PREPARE_ONLY_PHRASES)


def _reason_for_risk(risk: GuiSafetyRisk) -> str:
    return {
        GuiSafetyRisk.EXTERNAL_SEND: "该 GUI 子任务涉及对外发送，需要用户确认。",
        GuiSafetyRisk.PAYMENT: "该 GUI 子任务涉及付款或转账，需要用户确认。",
        GuiSafetyRisk.ORDER_SUBMIT: "该 GUI 子任务涉及提交订单，需要用户确认。",
        GuiSafetyRisk.DELETE: "该 GUI 子任务涉及删除或清空，需要用户确认。",
        GuiSafetyRisk.LOGIN_SECRET: "该 GUI 子任务涉及登录、授权或敏感凭据，需要用户确认。",
        GuiSafetyRisk.PRIVACY_CHANGE: "该 GUI 子任务涉及隐私或权限变更，需要用户确认。",
        GuiSafetyRisk.UNKNOWN_SENSITIVE: "该 GUI 子任务可能涉及敏感动作，需要用户确认。",
    }.get(risk, "该 GUI 子任务需要用户确认。")
