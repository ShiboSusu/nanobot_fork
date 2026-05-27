"""Content-JSON main task planner for semantic host-level routing."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

from nanobot.agent.cost_aware_router import PolicyDecision, RouteDecision, RouteKind
from opengui.policy import PolicyAction


@dataclass(frozen=True)
class PlannerConfig:
    enabled: bool = False
    confidence_threshold: float = 0.65
    max_tokens: int = 512
    timeout_seconds: float = 30.0


_RISK_LEVELS = frozenset({"low", "medium", "high"})


@dataclass(frozen=True)
class PlannerSubtask:
    id: str
    route: RouteKind
    task: str
    tool: str | None = None
    system_action: str | None = None
    validator: str | None = None
    success_condition: str | None = None
    risk_level: str = "low"


@dataclass(frozen=True)
class PlannerPlan:
    original_task: str
    route: RouteKind
    confidence: float
    reason: str
    subtasks: tuple[PlannerSubtask, ...]
    risk_notes: tuple[str, ...] = ()


class MainPlanner:
    """35B semantic router that emits content-only JSON plans."""

    _ROUTE_MAP = {
        "plan": RouteKind.GUI,
        "tool_call": RouteKind.TOOL_CALL,
        "tool": RouteKind.TOOL_CALL,
        "web_search": RouteKind.TOOL_CALL,
        "web_fetch": RouteKind.TOOL_CALL,
        "weather": RouteKind.TOOL_CALL,
        "system_action": RouteKind.SYSTEM_ACTION,
        "gui_task": RouteKind.GUI,
        "gui": RouteKind.GUI,
        "s2": RouteKind.S2,
        "ask_user": RouteKind.S2,
    }

    def __init__(
        self,
        *,
        provider: Any | None,
        model: str | None,
        config: PlannerConfig | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.config = config or PlannerConfig()

    def parse_decision(self, content: str, *, original_task: str) -> RouteDecision:
        return self.plan_to_route_decision(
            self.parse_plan(content, original_task=original_task)
        )

    def parse_plan(self, content: str, *, original_task: str) -> PlannerPlan:
        payload = self._parse_json_payload(content)
        route_raw = str(payload.get("route") or "").strip().casefold()
        route = self._route_from_payload(payload)
        confidence = self._confidence(payload.get("confidence"))
        if confidence < self.config.confidence_threshold:
            raise ValueError(
                f"low confidence: {confidence:.3f} < {self.config.confidence_threshold:.3f}"
            )

        reason = str(payload.get("reason") or "35B planner selected this route.").strip()
        risk_notes = self._risk_notes(payload.get("risk_notes"))
        subtasks = self._parse_subtasks(payload, route=route, original_task=original_task)
        if route_raw == "plan" and len(subtasks) == 1:
            route = subtasks[0].route
        return PlannerPlan(
            original_task=original_task,
            route=route,
            confidence=confidence,
            reason=reason,
            subtasks=subtasks,
            risk_notes=risk_notes,
        )

    def plan_to_route_decision(self, plan: PlannerPlan) -> RouteDecision:
        route = plan.route
        subtask = plan.subtasks[0] if plan.subtasks else None
        task = (
            plan.original_task
            if route == RouteKind.GUI and len(plan.subtasks) > 1
            else subtask.task if subtask and subtask.task else plan.original_task
        )

        if route == RouteKind.TOOL_CALL:
            return RouteDecision(
                route=RouteKind.TOOL_CALL,
                reason=plan.reason,
                policy=PolicyDecision(PolicyAction.ALLOW),
                suggested_tools=("web_search", "web_fetch"),
                requires_gui=False,
            )

        if route == RouteKind.SYSTEM_ACTION:
            return RouteDecision(
                route=RouteKind.SYSTEM_ACTION,
                reason=plan.reason,
                policy=PolicyDecision(PolicyAction.ALLOW),
                requires_gui=False,
                system_action={
                    "backend": None,
                    "task": task,
                    "intent": self._system_action_intent(subtask),
                },
            )

        if route == RouteKind.GUI:
            return RouteDecision(
                route=RouteKind.GUI,
                reason=plan.reason,
                policy=PolicyDecision(PolicyAction.ALLOW),
                requires_gui=True,
                routed_task=task,
            )

        return RouteDecision(
            route=route,
            reason=plan.reason,
            policy=PolicyDecision(PolicyAction.ALLOW),
            requires_gui=False,
        )

    async def plan(
        self,
        task: str,
        *,
        available_tools: set[str] | frozenset[str],
    ) -> RouteDecision:
        return self.plan_to_route_decision(
            await self.plan_full(task, available_tools=available_tools)
        )

    async def plan_full(
        self,
        task: str,
        *,
        available_tools: set[str] | frozenset[str],
    ) -> PlannerPlan:
        response = await self._call_planner(task, available_tools=available_tools)
        return self.parse_plan(response.content or "", original_task=task)

    async def _call_planner(
        self,
        task: str,
        *,
        available_tools: set[str] | frozenset[str],
    ) -> Any:
        if self.provider is None:
            raise RuntimeError("planner provider is not configured")
        return await asyncio.wait_for(
            self.provider.chat_with_retry(
                self._messages(task, available_tools=available_tools),
                tools=None,
                model=self.model,
                max_tokens=self.config.max_tokens,
                temperature=0.0,
                tool_choice=None,
            ),
            timeout=self.config.timeout_seconds,
        )

    @staticmethod
    def _messages(
        task: str,
        *,
        available_tools: set[str] | frozenset[str],
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are the main task planner/router for nanobot. "
                    "Choose the cheapest safe route for the user's request. "
                    "Use content-only JSON, no native tool calls. "
                    "Allowed top-level routes: plan, tool_call, system_action, gui_task, ask_user, s2. "
                    "Public information lookup should use tool_call. "
                    "Tasks inside a mobile app should use gui_task. "
                    "Safe one-shot device actions can use system_action. "
                    "Ambiguous tasks should use ask_user. "
                    "Return only JSON with route, confidence, reason, subtasks, and risk_notes."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": task,
                        "available_tools": sorted(available_tools),
                    },
                    ensure_ascii=False,
                ),
            },
        ]

    @classmethod
    def _parse_json_payload(cls, content: str) -> dict[str, Any]:
        text = (content or "").strip()
        if not text:
            raise ValueError("empty planner output")
        text = re.sub(r"(?is)^```(?:json)?\s*", "", text)
        text = re.sub(r"(?is)\s*```$", "", text).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("planner output did not contain a JSON object")
        try:
            payload = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid planner JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("planner JSON must be an object")
        return payload

    @staticmethod
    def _confidence(value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return min(1.0, max(0.0, confidence))

    def _route_from_payload(self, payload: dict[str, Any]) -> RouteKind:
        route_raw = str(payload.get("route") or "").strip().casefold()
        route = self._ROUTE_MAP.get(route_raw)
        if route is None:
            raise ValueError(f"unsupported route: {route_raw or '<missing>'}")
        return route

    def _route_from_subtask(self, payload: dict[str, Any]) -> RouteKind:
        route_raw = str(payload.get("route") or "").strip().casefold()
        if not route_raw:
            route_raw = str(payload.get("tool") or "").strip().casefold()
        if route_raw == "plan":
            raise ValueError("unsupported subtask route: plan")
        route = self._ROUTE_MAP.get(route_raw)
        if route is None:
            raise ValueError(f"unsupported subtask route: {route_raw or '<missing>'}")
        return route

    def _parse_subtasks(
        self,
        payload: dict[str, Any],
        *,
        route: RouteKind,
        original_task: str,
    ) -> tuple[PlannerSubtask, ...]:
        subtasks = payload.get("subtasks")
        parsed: list[PlannerSubtask] = []
        if isinstance(subtasks, list):
            for index, item in enumerate(subtasks, start=1):
                if isinstance(item, dict):
                    parsed.append(self._parse_subtask(item, index=index))
        if parsed:
            return tuple(parsed)
        return (
            PlannerSubtask(
                id="subtask_1",
                route=route,
                task=original_task,
            ),
        )

    def _parse_subtask(
        self,
        payload: dict[str, Any],
        *,
        index: int,
    ) -> PlannerSubtask:
        risk_level = str(payload.get("risk_level") or "low").strip().casefold()
        if risk_level not in _RISK_LEVELS:
            risk_level = "medium"
        task = self._optional_text(payload.get("task"))
        if not task:
            raise ValueError(f"subtask {index} missing task")
        return PlannerSubtask(
            id=self._subtask_id(payload, index=index),
            route=self._route_from_subtask(payload),
            task=task,
            tool=self._optional_text(payload.get("tool")),
            system_action=self._optional_text(payload.get("system_action") or payload.get("intent")),
            validator=self._optional_text(payload.get("validator")),
            success_condition=self._optional_text(payload.get("success_condition")),
            risk_level=risk_level,
        )

    @staticmethod
    def _subtask_id(payload: dict[str, Any], *, index: int) -> str:
        text = MainPlanner._optional_text(payload.get("id"))
        return text or f"subtask_{index}"

    @staticmethod
    def _risk_notes(value: Any) -> tuple[str, ...]:
        if not isinstance(value, list):
            return ()
        return tuple(text for item in value if (text := MainPlanner._optional_text(item)))

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _system_action_intent(subtask: PlannerSubtask | None) -> str:
        intent = subtask.system_action if subtask else None
        text = str(intent or "").strip()
        return text or "open_app"


__all__ = ["MainPlanner", "PlannerConfig", "PlannerPlan", "PlannerSubtask"]
