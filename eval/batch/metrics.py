"""Parse a trace JSONL into RunMetrics."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from eval.batch.schemas import RunMetrics

# Prices are CNY per 1M tokens. 9B is a placeholder, not a real bill rate.
MODEL_TOKEN_PRICES_YUAN_PER_M = {
    "397B": {"input": 1.2, "output": 7.2},
    "9B": {"input": 0.2, "output": 1.5},
}


def _iter_events(trace_path: Path):
    with trace_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _avg(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _model_price_key(event: dict[str, Any]) -> str:
    model = str(event.get("model_name") or "").lower()
    arm = str(event.get("arm") or "").lower()
    if "397" in model or arm.startswith("s2"):
        return "397B"
    return "9B"


def _event_token_counts(event: dict[str, Any]) -> tuple[int, int, int | None, int, int]:
    usage = event.get("token_usage") or {}
    token_in = event.get("token_in")
    token_out = event.get("token_out")
    token_think = event.get("token_think")
    token_cached = event.get("token_cached")
    prompt = _int(token_in if token_in is not None else usage.get("prompt_tokens"))
    completion = _int(token_out if token_out is not None else usage.get("completion_tokens"))
    cached = _int(token_cached if token_cached is not None else usage.get("cached_tokens"))
    thinking_raw = (
        token_think
        if token_think is not None
        else usage.get("reasoning_tokens", usage.get("thinking_tokens"))
    )
    thinking = None if thinking_raw is None else _int(thinking_raw)
    total = _int(usage.get("total_tokens")) or (prompt + completion + (thinking or 0))
    return prompt, completion, thinking, cached, total


def _event_cost_yuan(event: dict[str, Any]) -> tuple[float, bool]:
    prompt, completion, thinking, _, _ = _event_token_counts(event)
    price = MODEL_TOKEN_PRICES_YUAN_PER_M[_model_price_key(event)]
    output_tokens = completion + (thinking or 0)
    cost = (prompt * price["input"] + output_tokens * price["output"]) / 1_000_000
    return cost, thinking is None


def parse_trace(trace_path: Path | str) -> RunMetrics:
    """Walk events and produce RunMetrics."""

    trace_path = Path(trace_path)
    if not trace_path.exists():
        return RunMetrics()

    metrics = RunMetrics()
    step_durations: list[float] = []
    chat_latencies: list[float] = []
    ttfts: list[float] = []
    prompt_tok = 0
    comp_tok = 0
    think_tok = 0
    cached_tok = 0
    total_tok = 0
    cost_yuan = 0.0
    missing_think = False

    for event in _iter_events(trace_path):
        etype = event.get("type")
        if etype == "step":
            metrics.steps += 1
            prompt, completion, thinking, cached, total = _event_token_counts(event)
            prompt_tok += prompt
            comp_tok += completion
            think_tok += thinking or 0
            cached_tok += cached
            total_tok += total
            step_cost, step_missing_think = _event_cost_yuan(event)
            metrics.step_costs_yuan.append(step_cost)
            cost_yuan += step_cost
            missing_think = missing_think or step_missing_think
            if event.get("duration_s") is not None:
                step_durations.append(float(event["duration_s"]))
            if event.get("chat_latency_s") is not None:
                chat_latencies.append(float(event["chat_latency_s"]))
            if event.get("ttft_s") is not None:
                ttfts.append(float(event["ttft_s"]))
        elif etype in {"subgoal_step", "skill_step"}:
            prompt, completion, thinking, cached, total = _event_token_counts(event)
            prompt_tok += prompt
            comp_tok += completion
            think_tok += thinking or 0
            cached_tok += cached
            total_tok += total
            step_cost, step_missing_think = _event_cost_yuan(event)
            metrics.step_costs_yuan.append(step_cost)
            cost_yuan += step_cost
            missing_think = missing_think or step_missing_think
            if event.get("chat_latency_s") is not None:
                chat_latencies.append(float(event["chat_latency_s"]))
            if event.get("ttft_s") is not None:
                ttfts.append(float(event["ttft_s"]))
        elif etype == "skill_search":
            if event.get("matched") is True:
                metrics.skill_hit = True
        elif etype == "skill_execution_result":
            if event.get("state") == "succeeded":
                metrics.skill_executed_success = True

    metrics.prompt_tokens = prompt_tok
    metrics.completion_tokens = comp_tok
    metrics.thinking_tokens = think_tok
    metrics.cached_tokens = cached_tok
    metrics.total_tokens = total_tok or (prompt_tok + comp_tok)
    metrics.cost_yuan = cost_yuan
    metrics.cost_note = "think 未计" if missing_think else ""
    metrics.avg_step_duration_s = _avg(step_durations)
    metrics.avg_chat_latency_s = _avg(chat_latencies)
    metrics.avg_ttft_s = _avg(ttfts)
    return metrics
