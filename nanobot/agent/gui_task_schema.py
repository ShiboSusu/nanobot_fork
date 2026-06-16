from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from nanobot.agent.gui_safety import SENSITIVE_ACTION_KEYWORDS
from opengui.skills.normalization import annotate_ios_apps, find_android_app_in_text, resolve_ios_bundle


class GuiTaskType(str, Enum):
    OPERATION = "operation"
    INFORMATION_QUERY = "information_query"
    MIXED_QUERY_AND_ACTION = "mixed_query_and_action"
    SENSITIVE_ACTION = "sensitive_action"


class GuiOutputMode(str, Enum):
    OPERATION_STATUS = "operation_status"
    ANSWER_REQUIRED = "answer_required"
    NEEDS_HUMAN_CONFIRM = "needs_human_confirm"


class GuiSuccessConditionType(str, Enum):
    STATE_REACHED = "state_reached"
    EXTRACT_FIELD = "extract_field"
    NONE = "none"


@dataclass(frozen=True)
class GuiSuccessCondition:
    type: GuiSuccessConditionType = GuiSuccessConditionType.NONE
    required_key: str | None = None
    description: str | None = None
    required_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class GuiEvidenceRequirements:
    answer_candidates_required: bool = False
    visible_text_required: bool = False
    minimum_confidence: float = 0.0


@dataclass(frozen=True)
class GuiSafetyPolicy:
    allow_external_send: bool = False
    allow_payment: bool = False
    allow_submit: bool = False
    allow_delete: bool = False
    allow_login_secret: bool = False


@dataclass(frozen=True)
class GuiTaskRequestV1:
    schema_version: str = "gui_task_request.v1"
    task: str = ""
    task_type: GuiTaskType = GuiTaskType.OPERATION
    output_mode: GuiOutputMode = GuiOutputMode.OPERATION_STATUS
    app_hint: str | None = None
    app_bundle_id: str | None = None
    success_condition: GuiSuccessCondition = field(default_factory=GuiSuccessCondition)
    evidence_requirements: GuiEvidenceRequirements = field(default_factory=GuiEvidenceRequirements)
    safety: GuiSafetyPolicy = field(default_factory=GuiSafetyPolicy)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task": self.task,
            "task_type": self.task_type.value,
            "output_mode": self.output_mode.value,
            "app_hint": self.app_hint,
            "app_bundle_id": self.app_bundle_id,
            "success_condition": {
                "type": self.success_condition.type.value,
                "required_key": self.success_condition.required_key,
                "description": self.success_condition.description,
                "required_fields": list(self.success_condition.required_fields),
            },
            "evidence_requirements": {
                "answer_candidates_required": self.evidence_requirements.answer_candidates_required,
                "visible_text_required": self.evidence_requirements.visible_text_required,
                "minimum_confidence": self.evidence_requirements.minimum_confidence,
            },
            "safety": {
                "allow_external_send": self.safety.allow_external_send,
                "allow_payment": self.safety.allow_payment,
                "allow_submit": self.safety.allow_submit,
                "allow_delete": self.safety.allow_delete,
                "allow_login_secret": self.safety.allow_login_secret,
            },
            "raw": dict(self.raw),
        }


_INFORMATION_QUERY_KEYWORDS = (
    "告诉我",
    "看看",
    "查",
    "查询",
    "是什么",
    "是谁",
    "哪个",
    "哪家",
    "多少",
    "多少钱",
    "第几",
    "第一",
    "第二",
    "第三",
    "热搜",
    "榜",
    "排名",
    "价格",
    "票价",
    "车次",
    "航班",
    "酒店",
    "物流",
    "快递",
    "订单",
    "状态",
    "结果",
    "what",
    "which",
    "who",
    "price",
    "status",
)

_EXPLICIT_GUI_CONTEXT_KEYWORDS = (
    "打开",
    "进入",
    "使用",
    "手机",
    "应用",
    "app",
    "ios",
    "android",
    "安卓",
)


def normalize_gui_task_request(raw: Any) -> GuiTaskRequestV1:
    if isinstance(raw, GuiTaskRequestV1):
        return raw
    if isinstance(raw, dict):
        payload = dict(raw)
    elif isinstance(raw, str):
        payload = {"task": raw}
    else:
        payload = {"task": "" if raw is None else str(raw)}

    task = _clean_optional_text(payload.get("task")) or ""
    app_hint = _clean_optional_text(payload.get("app_hint"))
    app_bundle_id = _clean_optional_text(payload.get("app_bundle_id") or payload.get("appBundleId"))
    app_text = " ".join(part for part in (app_hint or "", app_bundle_id or "", task) if part)
    resolved_bundle = resolve_ios_bundle(app_text)
    if not app_bundle_id and resolved_bundle and resolved_bundle != app_text:
        app_bundle_id = resolved_bundle
    if not app_bundle_id:
        android_package = find_android_app_in_text(app_text)
        if android_package:
            app_bundle_id = android_package
    if not app_hint and app_bundle_id:
        annotated = annotate_ios_apps([app_bundle_id])
        if annotated:
            app_hint = annotated[0].split(": ", 1)[0]
    inferred = _infer_request(task)

    task_type = _coerce_enum(GuiTaskType, payload.get("task_type"), inferred.task_type)
    output_mode = _coerce_enum(GuiOutputMode, payload.get("output_mode"), inferred.output_mode)
    success_condition = _parse_success_condition(
        payload.get("success_condition"),
        default=inferred.success_condition,
    )
    evidence_requirements = _parse_evidence_requirements(
        payload.get("evidence_requirements"),
        default=inferred.evidence_requirements,
    )
    safety = _parse_safety_policy(payload.get("safety"))

    return GuiTaskRequestV1(
        task=task,
        task_type=task_type,
        output_mode=output_mode,
        app_hint=app_hint,
        app_bundle_id=app_bundle_id,
        success_condition=success_condition,
        evidence_requirements=evidence_requirements,
        safety=safety,
        raw=payload,
    )


def _infer_request(task: str) -> GuiTaskRequestV1:
    has_information = _contains_any(task, _INFORMATION_QUERY_KEYWORDS)
    has_sensitive = _contains_any(task, SENSITIVE_ACTION_KEYWORDS)

    success_condition = GuiSuccessCondition()
    evidence_requirements = GuiEvidenceRequirements()
    if has_information:
        success_condition = _inferred_extract_condition(task)
        evidence_requirements = GuiEvidenceRequirements(answer_candidates_required=True)

    if has_information and has_sensitive:
        return GuiTaskRequestV1(
            task=task,
            task_type=GuiTaskType.MIXED_QUERY_AND_ACTION,
            output_mode=GuiOutputMode.ANSWER_REQUIRED,
            success_condition=success_condition,
            evidence_requirements=evidence_requirements,
        )
    if has_information:
        return GuiTaskRequestV1(
            task=task,
            task_type=GuiTaskType.INFORMATION_QUERY,
            output_mode=GuiOutputMode.ANSWER_REQUIRED,
            success_condition=success_condition,
            evidence_requirements=evidence_requirements,
        )
    if has_sensitive:
        return GuiTaskRequestV1(
            task=task,
            task_type=GuiTaskType.SENSITIVE_ACTION,
            output_mode=GuiOutputMode.NEEDS_HUMAN_CONFIRM,
        )
    return GuiTaskRequestV1(task=task)


def requires_gui_task_routing(task: str) -> bool:
    request = normalize_gui_task_request({"task": task})
    if request.task_type in {GuiTaskType.SENSITIVE_ACTION, GuiTaskType.MIXED_QUERY_AND_ACTION}:
        return True
    if request.output_mode == GuiOutputMode.NEEDS_HUMAN_CONFIRM:
        return True
    if (request.app_hint or request.app_bundle_id) and _contains_any(task, _EXPLICIT_GUI_CONTEXT_KEYWORDS):
        return True
    return False


def _inferred_extract_condition(task: str) -> GuiSuccessCondition:
    return GuiSuccessCondition(
        type=GuiSuccessConditionType.EXTRACT_FIELD,
        required_key="answer",
    )


def _parse_success_condition(raw: Any, *, default: GuiSuccessCondition) -> GuiSuccessCondition:
    if isinstance(raw, GuiSuccessCondition):
        return raw
    if not isinstance(raw, dict):
        return default
    required_fields = raw.get("required_fields", ())
    if isinstance(required_fields, str):
        required_fields = (required_fields,)
    elif isinstance(required_fields, list | tuple):
        required_fields = tuple(str(item) for item in required_fields if str(item).strip())
    else:
        required_fields = ()
    return GuiSuccessCondition(
        type=_coerce_enum(
            GuiSuccessConditionType,
            raw.get("type"),
            default.type,
        ),
        required_key=_clean_optional_text(raw.get("required_key")),
        description=_clean_optional_text(raw.get("description")),
        required_fields=required_fields,
    )


def _parse_evidence_requirements(
    raw: Any,
    *,
    default: GuiEvidenceRequirements,
) -> GuiEvidenceRequirements:
    if isinstance(raw, GuiEvidenceRequirements):
        return raw
    if not isinstance(raw, dict):
        return default
    return GuiEvidenceRequirements(
        answer_candidates_required=bool(
            raw.get("answer_candidates_required", default.answer_candidates_required)
        ),
        visible_text_required=bool(raw.get("visible_text_required", default.visible_text_required)),
        minimum_confidence=_float_or_default(raw.get("minimum_confidence"), default.minimum_confidence),
    )


def _parse_safety_policy(raw: Any) -> GuiSafetyPolicy:
    if isinstance(raw, GuiSafetyPolicy):
        return raw
    if not isinstance(raw, dict):
        return GuiSafetyPolicy()
    return GuiSafetyPolicy(
        allow_external_send=bool(raw.get("allow_external_send", False)),
        allow_payment=bool(raw.get("allow_payment", False)),
        allow_submit=bool(raw.get("allow_submit", False)),
        allow_delete=bool(raw.get("allow_delete", False)),
        allow_login_secret=bool(raw.get("allow_login_secret", False)),
    )


def _coerce_enum(enum_cls: type[Enum], value: Any, default: Any) -> Any:
    if isinstance(value, enum_cls):
        return value
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return enum_cls(text)
    except ValueError:
        normalized = text.upper()
        for item in enum_cls:
            if item.name == normalized:
                return item
        return default


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.casefold()
    return any(keyword.casefold() in lowered for keyword in keywords)


def _clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
