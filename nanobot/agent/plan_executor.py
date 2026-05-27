"""Sequential host-level execution for 35B planner subtasks."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum

from nanobot.agent.main_planner import PlannerPlan, PlannerSubtask
from opengui.policy import PolicyAction, PolicyDecision


class SubtaskStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    NEEDS_USER = "needs_user"


class PlanExecutionStatus(str, Enum):
    SUCCESS = "success"
    BLOCKED = "blocked"
    HUMAN_CONFIRM = "human_confirm"
    NEEDS_USER = "needs_user"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class SubtaskExecution:
    status: SubtaskStatus
    output: str
    error: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SubtaskResult:
    subtask: PlannerSubtask
    status: SubtaskStatus
    output: str
    error: str | None = None
    policy: PolicyDecision | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanExecutionResult:
    status: PlanExecutionStatus
    summary: str
    subtasks: tuple[SubtaskResult, ...]


PolicyCheck = Callable[[str], PolicyDecision]
SubtaskDispatch = Callable[[PlannerSubtask], Awaitable[SubtaskExecution]]


class PlanExecutor:
    """Run validated planner subtasks in order through injected host dispatch."""

    def __init__(
        self,
        *,
        policy_check: PolicyCheck,
        dispatch: SubtaskDispatch,
        allow_skip_missing_apps: bool = False,
    ) -> None:
        self._policy_check = policy_check
        self._dispatch = dispatch
        self._allow_skip_missing_apps = allow_skip_missing_apps

    async def execute(self, plan: PlannerPlan) -> PlanExecutionResult:
        results: list[SubtaskResult] = []
        total = len(plan.subtasks)
        for subtask in plan.subtasks:
            policy = self._policy_check(subtask.task)
            if not policy.allowed:
                result = SubtaskResult(
                    subtask=subtask,
                    status=self._subtask_status_from_policy(policy),
                    output=policy.reason,
                    error=policy.action.value,
                    policy=policy,
                    metadata={
                        "categories": policy.categories,
                        "matched_terms": policy.matched_terms,
                    },
                )
                results.append(result)
                categories = ", ".join(policy.categories) or "sensitive_action"
                return PlanExecutionResult(
                    status=self._status_from_policy(policy),
                    summary=(
                        f"Subtask {subtask.id} blocked by policy "
                        f"{policy.action.value}: {categories}"
                    ),
                    subtasks=tuple(results),
                )

            execution = await self._dispatch(subtask)
            result = SubtaskResult(
                subtask=subtask,
                status=execution.status,
                output=execution.output,
                error=execution.error,
                policy=policy,
                metadata=execution.metadata,
            )
            results.append(result)

            if execution.status == SubtaskStatus.SUCCESS:
                continue
            if execution.status == SubtaskStatus.NEEDS_USER:
                return PlanExecutionResult(
                    status=PlanExecutionStatus.NEEDS_USER,
                    summary=f"Subtask {subtask.id} needs user input: {execution.output}",
                    subtasks=tuple(results),
                )
            if (
                execution.status == SubtaskStatus.SKIPPED
                and self._allow_skip_missing_apps
                and execution.error == "missing_app"
            ):
                return PlanExecutionResult(
                    status=PlanExecutionStatus.SKIPPED,
                    summary=f"Subtask {subtask.id} skipped: {execution.error}",
                    subtasks=tuple(results),
                )
            return PlanExecutionResult(
                status=PlanExecutionStatus.BLOCKED,
                summary=f"Subtask {subtask.id} failed: {execution.error or execution.output}",
                subtasks=tuple(results),
            )

        return PlanExecutionResult(
            status=PlanExecutionStatus.SUCCESS,
            summary=f"{len(results)}/{total} subtasks completed.",
            subtasks=tuple(results),
        )

    @staticmethod
    def _status_from_policy(policy: PolicyDecision) -> PlanExecutionStatus:
        if policy.action == PolicyAction.ASK_HUMAN_CONFIRM:
            return PlanExecutionStatus.HUMAN_CONFIRM
        if policy.action == PolicyAction.REQUIRE_HUMAN_TAKEOVER:
            return PlanExecutionStatus.NEEDS_USER
        return PlanExecutionStatus.BLOCKED

    @staticmethod
    def _subtask_status_from_policy(policy: PolicyDecision) -> SubtaskStatus:
        if policy.action == PolicyAction.HALT:
            return SubtaskStatus.FAILED
        return SubtaskStatus.NEEDS_USER


__all__ = [
    "PlanExecutionResult",
    "PlanExecutionStatus",
    "PlanExecutor",
    "SubtaskExecution",
    "SubtaskResult",
    "SubtaskStatus",
]
