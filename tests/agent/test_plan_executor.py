from __future__ import annotations

import pytest

from nanobot.agent.cost_aware_router import RouteKind
from nanobot.agent.main_planner import PlannerPlan, PlannerSubtask
from nanobot.agent.plan_executor import (
    PlanExecutionStatus,
    PlanExecutor,
    SubtaskExecution,
    SubtaskStatus,
)
from opengui.policy import PolicyAction, PolicyDecision


def _plan(*subtasks: PlannerSubtask) -> PlannerPlan:
    return PlannerPlan(
        original_task="test task",
        route=RouteKind.GUI,
        confidence=0.9,
        reason="test",
        subtasks=subtasks,
    )


@pytest.mark.asyncio
async def test_plan_executor_runs_subtasks_in_order() -> None:
    calls: list[str] = []

    async def dispatch(subtask: PlannerSubtask) -> SubtaskExecution:
        calls.append(subtask.id)
        return SubtaskExecution(status=SubtaskStatus.SUCCESS, output=f"ok:{subtask.id}")

    executor = PlanExecutor(
        policy_check=lambda task: PolicyDecision(PolicyAction.ALLOW),
        dispatch=dispatch,
    )

    result = await executor.execute(
        _plan(
            PlannerSubtask(id="open", route=RouteKind.SYSTEM_ACTION, task="open app"),
            PlannerSubtask(id="search", route=RouteKind.GUI, task="search"),
        )
    )

    assert result.status == PlanExecutionStatus.SUCCESS
    assert calls == ["open", "search"]
    assert [item.subtask.id for item in result.subtasks] == ["open", "search"]
    assert "2/2 subtasks completed" in result.summary


@pytest.mark.asyncio
async def test_plan_executor_stops_before_policy_blocked_subtask() -> None:
    calls: list[str] = []

    async def dispatch(subtask: PlannerSubtask) -> SubtaskExecution:
        calls.append(subtask.id)
        return SubtaskExecution(status=SubtaskStatus.SUCCESS, output="should not run")

    executor = PlanExecutor(
        policy_check=lambda task: PolicyDecision(
            PolicyAction.ASK_HUMAN_CONFIRM,
            reason="financial data",
            categories=("financial",),
        ),
        dispatch=dispatch,
    )

    result = await executor.execute(
        _plan(
            PlannerSubtask(id="credit", route=RouteKind.GUI, task="查看白条额度"),
        )
    )

    assert result.status == PlanExecutionStatus.HUMAN_CONFIRM
    assert calls == []
    assert "financial" in result.summary


@pytest.mark.asyncio
async def test_plan_executor_stops_after_failed_subtask() -> None:
    calls: list[str] = []

    async def dispatch(subtask: PlannerSubtask) -> SubtaskExecution:
        calls.append(subtask.id)
        if subtask.id == "search":
            return SubtaskExecution(
                status=SubtaskStatus.FAILED,
                output="",
                error="stagnation_detected",
            )
        return SubtaskExecution(status=SubtaskStatus.SUCCESS, output="ok")

    executor = PlanExecutor(
        policy_check=lambda task: PolicyDecision(PolicyAction.ALLOW),
        dispatch=dispatch,
    )

    result = await executor.execute(
        _plan(
            PlannerSubtask(id="open", route=RouteKind.SYSTEM_ACTION, task="open app"),
            PlannerSubtask(id="search", route=RouteKind.GUI, task="search"),
            PlannerSubtask(id="play", route=RouteKind.GUI, task="play"),
        )
    )

    assert result.status == PlanExecutionStatus.BLOCKED
    assert calls == ["open", "search"]
    assert result.subtasks[-1].error == "stagnation_detected"
    assert "search" in result.summary


@pytest.mark.asyncio
async def test_plan_executor_can_skip_missing_app_when_allowed() -> None:
    async def dispatch(subtask: PlannerSubtask) -> SubtaskExecution:
        return SubtaskExecution(
            status=SubtaskStatus.SKIPPED,
            output="",
            error="missing_app",
        )

    executor = PlanExecutor(
        policy_check=lambda task: PolicyDecision(PolicyAction.ALLOW),
        dispatch=dispatch,
        allow_skip_missing_apps=True,
    )

    result = await executor.execute(
        _plan(
            PlannerSubtask(id="open_youku", route=RouteKind.GUI, task="打开优酷"),
        )
    )

    assert result.status == PlanExecutionStatus.SKIPPED
    assert "missing_app" in result.summary
