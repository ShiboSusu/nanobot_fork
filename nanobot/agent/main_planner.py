"""Content-JSON main task planner for semantic host-level routing."""

from __future__ import annotations

import json
import re
import asyncio
from dataclasses import dataclass
from typing import Any

from nanobot.agent.cost_aware_router import PolicyDecision, RouteDecision, RouteKind
from opengui.policy import PolicyAction


@dataclass(frozen=True)
class PlannerConfig:
    enabled: bool = False
    confidence_threshold: float = 0.65
    max_tokens: int = 512
    timeout_seconds: float = 8.0


class MainPlanner:
    """35B semantic router that emits content-only JSON plans."""

    _ROUTE_MAP = {
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
        payload = self._parse_json_payload(content)
        route_raw = str(payload.get("route") or "").strip().casefold()
        route = self._ROUTE_MAP.get(route_raw)
        if route is None:
            raise ValueError(f"unsupported route: {route_raw or '<missing>'}")

        confidence = self._confidence(payload.get("confidence"))
        if confidence < self.config.confidence_threshold:
            raise ValueError(
                f"low confidence: {confidence:.3f} < {self.config.confidence_threshold:.3f}"
            )

        reason = str(payload.get("reason") or "35B planner selected this route.").strip()
        subtask = self._first_subtask(payload)
        task = self._subtask_text(subtask) or original_task

        if route == RouteKind.TOOL_CALL:
            return RouteDecision(
                route=RouteKind.TOOL_CALL,
                reason=reason,
                policy=PolicyDecision(PolicyAction.ALLOW),
                suggested_tools=("web_search", "web_fetch"),
                requires_gui=False,
            )

        if route == RouteKind.SYSTEM_ACTION:
            return RouteDecision(
                route=RouteKind.SYSTEM_ACTION,
                reason=reason,
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
                reason=reason,
                policy=PolicyDecision(PolicyAction.ALLOW),
                requires_gui=True,
                routed_task=task,
            )

        return RouteDecision(
            route=route,
            reason=reason,
            policy=PolicyDecision(PolicyAction.ALLOW),
            requires_gui=False,
        )

    async def plan(
        self,
        task: str,
        *,
        available_tools: set[str] | frozenset[str],
    ) -> RouteDecision:
        if self.provider is None:
            raise RuntimeError("planner provider is not configured")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the main task planner/router for nanobot. "
                    "Choose the cheapest safe route for the user's request. "
                    "Use content-only JSON, no native tool calls. "
                    "Allowed top-level routes: tool_call, system_action, gui_task, ask_user, s2. "
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
        response = await asyncio.wait_for(
            self.provider.chat_with_retry(
                messages,
                tools=None,
                model=self.model,
                max_tokens=self.config.max_tokens,
                temperature=0.0,
                tool_choice=None,
            ),
            timeout=self.config.timeout_seconds,
        )
        return self.parse_decision(response.content or "", original_task=task)

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

    @staticmethod
    def _first_subtask(payload: dict[str, Any]) -> dict[str, Any]:
        subtasks = payload.get("subtasks")
        if isinstance(subtasks, list):
            for item in subtasks:
                if isinstance(item, dict):
                    return item
        return {}

    @staticmethod
    def _subtask_text(subtask: dict[str, Any]) -> str:
        task = subtask.get("task")
        return str(task).strip() if task is not None else ""

    @staticmethod
    def _system_action_intent(subtask: dict[str, Any]) -> str:
        intent = subtask.get("system_action") or subtask.get("intent")
        text = str(intent or "").strip()
        return text or "open_app"


__all__ = ["MainPlanner", "PlannerConfig"]
