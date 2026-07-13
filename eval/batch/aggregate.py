"""Aggregate per-run records into PhaseSummary + A-vs-B report."""

from __future__ import annotations

from collections import defaultdict
import random
from statistics import mean
from typing import Iterable

from eval.batch.schemas import PhaseSummary, RunRecord


def _avg(xs: list[float | int | None]) -> float | None:
    vals = [float(x) for x in xs if x is not None]
    return mean(vals) if vals else None


def _pass_at_k(successes: list[bool], k: int) -> float:
    """Vanilla pass@k: 1.0 if at least one of the first k trials succeeded."""
    if not successes or k <= 0:
        return 0.0
    return 1.0 if any(successes[:k]) else 0.0


def _bootstrap_ci95(values: list[float], *, n: int = 1000) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(0)
    size = len(values)
    samples = []
    for _ in range(n):
        samples.append(mean(values[rng.randrange(size)] for _ in range(size)))
    samples.sort()
    return (samples[int(0.025 * n)], samples[int(0.975 * n)])


def summarize_phase(
    phase: str,
    records: Iterable[RunRecord],
    k: int,
) -> PhaseSummary:
    by_task: dict[str, list[RunRecord]] = defaultdict(list)
    for r in records:
        by_task[r.task_id].append(r)
    for task_recs in by_task.values():
        task_recs.sort(key=lambda r: r.trial_index)

    per_task_p1: dict[str, float] = {}
    per_task_p2: dict[str, float] = {}
    per_task_pk: dict[str, float] = {}
    for task_id, recs in by_task.items():
        succ = [r.success for r in recs]
        per_task_p1[task_id] = _pass_at_k(succ, 1)
        per_task_p2[task_id] = _pass_at_k(succ, 2)
        per_task_pk[task_id] = _pass_at_k(succ, k)

    flat = [r for recs in by_task.values() for r in recs]
    skill_hits = sum(1 for r in flat if r.metrics.skill_hit)
    total_cost = sum(r.metrics.cost_yuan for r in flat)
    total_tokens = sum(r.metrics.total_tokens for r in flat)
    passed_tasks = sum(per_task_pk.values())
    cost_notes = sorted({r.metrics.cost_note for r in flat if r.metrics.cost_note})

    return PhaseSummary(
        phase=phase,
        n_tasks=len(by_task),
        n_trials=k,
        pass_at_1=mean(per_task_p1.values()) if per_task_p1 else 0.0,
        pass_at_1_ci95=_bootstrap_ci95(list(per_task_p1.values())),
        pass_at_2=mean(per_task_p2.values()) if per_task_p2 else 0.0,
        pass_at_2_ci95=_bootstrap_ci95(list(per_task_p2.values())),
        pass_at_k=mean(per_task_pk.values()) if per_task_pk else 0.0,
        pass_at_k_ci95=_bootstrap_ci95(list(per_task_pk.values())),
        skill_hit_rate=(skill_hits / len(flat)) if flat else 0.0,
        avg_steps=_avg([r.metrics.steps for r in flat]) or 0.0,
        avg_prompt_tokens=_avg([r.metrics.prompt_tokens for r in flat]) or 0.0,
        avg_completion_tokens=_avg([r.metrics.completion_tokens for r in flat]) or 0.0,
        avg_total_tokens=_avg([r.metrics.total_tokens for r in flat]) or 0.0,
        total_cost_yuan=total_cost,
        avg_cost_yuan=_avg([r.metrics.cost_yuan for r in flat]) or 0.0,
        token_per_success=(total_tokens / passed_tasks) if passed_tasks else None,
        cost_per_success=(total_cost / passed_tasks) if passed_tasks else None,
        cost_note="; ".join(cost_notes),
        avg_step_duration_s=_avg([r.metrics.avg_step_duration_s for r in flat]),
        avg_chat_latency_s=_avg([r.metrics.avg_chat_latency_s for r in flat]),
        avg_ttft_s=_avg([r.metrics.avg_ttft_s for r in flat]),
        per_task_pass_at_1=per_task_p1,
        per_task_pass_at_k=per_task_pk,
    )


def render_report(a: PhaseSummary, b: PhaseSummary | None) -> str:
    """Markdown table comparing phase A vs phase B."""

    def fmt(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else "-"

    rows = [
        ("pass@1", a.pass_at_1, b.pass_at_1 if b else None),
        ("pass@2", a.pass_at_2, b.pass_at_2 if b else None),
        ("pass@K", a.pass_at_k, b.pass_at_k if b else None),
        ("skill_hit_rate", a.skill_hit_rate, b.skill_hit_rate if b else None),
        ("avg_steps", a.avg_steps, b.avg_steps if b else None),
        ("avg_total_tokens", a.avg_total_tokens, b.avg_total_tokens if b else None),
        ("avg_prompt_tokens", a.avg_prompt_tokens, b.avg_prompt_tokens if b else None),
        ("avg_completion_tokens", a.avg_completion_tokens, b.avg_completion_tokens if b else None),
        ("total_cost_yuan", a.total_cost_yuan, b.total_cost_yuan if b else None),
        ("cost_per_success", a.cost_per_success, b.cost_per_success if b else None),
        ("token_per_success", a.token_per_success, b.token_per_success if b else None),
        ("avg_step_duration_s", a.avg_step_duration_s, b.avg_step_duration_s if b else None),
        ("avg_chat_latency_s", a.avg_chat_latency_s, b.avg_chat_latency_s if b else None),
        ("avg_ttft_s", a.avg_ttft_s, b.avg_ttft_s if b else None),
    ]

    lines = [
        f"# Batch eval report (K={a.n_trials}, N_tasks={a.n_tasks})",
        "",
        "| metric | phase_a (cold) | phase_b (warm) | delta |",
        "|---|---|---|---|",
    ]
    for name, av, bv in rows:
        if bv is None:
            delta = "-"
        elif av is None or bv is None:
            delta = "-"
        else:
            delta = f"{(bv - av):+.3f}"
        lines.append(f"| {name} | {fmt(av)} | {fmt(bv)} | {delta} |")
    return "\n".join(lines) + "\n"
