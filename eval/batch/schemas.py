"""Dataclasses for the batch eval harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskSpec:
    task_id: str
    instruction: str
    instruction_ch: str = ""

    @property
    def prompt(self) -> str:
        return self.instruction_ch or self.instruction


@dataclass
class RunMetrics:
    """Per-run metrics parsed from a trace.jsonl."""

    steps: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    cost_yuan: float = 0.0
    step_costs_yuan: list[float] = field(default_factory=list)
    cost_note: str = ""
    avg_step_duration_s: float | None = None
    avg_chat_latency_s: float | None = None
    avg_ttft_s: float | None = None
    skill_hit: bool = False
    skill_executed_success: bool = False


@dataclass
class RunRecord:
    """One (task, trial) outcome."""

    task_id: str
    trial_index: int
    instruction: str
    trace_path: str | None
    success: bool
    judge_reason: str = ""
    error: str | None = None
    duration_s: float | None = None
    metrics: RunMetrics = field(default_factory=RunMetrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "trial_index": self.trial_index,
            "instruction": self.instruction,
            "trace_path": self.trace_path,
            "success": self.success,
            "judge_reason": self.judge_reason,
            "error": self.error,
            "duration_s": self.duration_s,
            "metrics": self.metrics.__dict__,
        }


@dataclass
class PhaseSummary:
    """Aggregate metrics for one phase across all tasks × trials."""

    phase: str
    n_tasks: int
    n_trials: int
    pass_at_1: float
    pass_at_1_ci95: tuple[float, float]
    pass_at_2: float
    pass_at_2_ci95: tuple[float, float]
    pass_at_k: float
    pass_at_k_ci95: tuple[float, float]
    skill_hit_rate: float
    avg_steps: float
    avg_prompt_tokens: float
    avg_completion_tokens: float
    avg_total_tokens: float
    total_cost_yuan: float
    avg_cost_yuan: float
    token_per_success: float | None
    cost_per_success: float | None
    cost_note: str
    avg_step_duration_s: float | None
    avg_chat_latency_s: float | None
    avg_ttft_s: float | None
    per_task_pass_at_1: dict[str, float]
    per_task_pass_at_k: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__
