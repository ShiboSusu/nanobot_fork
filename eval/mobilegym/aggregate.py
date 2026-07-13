"""Aggregate MobileGym runs for fast/slow GUI evaluation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _result_rows(run_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    direct = run_dir / "results.jsonl"
    if direct.exists():
        return [(run_dir, row) for row in _read_jsonl(direct)]
    rows: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(run_dir.glob("mobilegym_runs/*/results.jsonl")):
        rows.extend((path.parent, row) for row in _read_jsonl(path))
    return rows


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _norm_task_id(row: dict[str, Any]) -> str:
    task_id = row.get("task_id") or row.get("id")
    if not task_id:
        raise ValueError(f"MobileGym result row is missing task_id/id: {row}")
    return str(task_id)


def _trial_id(row: dict[str, Any]) -> int:
    return _safe_int(row.get("trial_id"))


def _trajectory_dir_name(task_id: str, trial_id: int) -> str:
    return f"{task_id.replace('.', '_')}_t{trial_id}"


def _trajectory_dir_candidates(run_dir: Path, task_id: str, trial_id: int) -> list[Path]:
    base = task_id.replace(".", "_")
    candidates = [run_dir / "trajectory" / _trajectory_dir_name(task_id, trial_id)]
    if trial_id == 0:
        candidates.append(run_dir / "trajectory" / base)
    return candidates


def _load_response_json(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return {}
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _response_from_trajectory(run_dir: Path, task_id: str, trial_id: int) -> dict[str, Any]:
    responses = _responses_from_trajectory(run_dir, task_id, trial_id)
    return responses[-1] if responses else {}


def _responses_from_trajectory(run_dir: Path, task_id: str, trial_id: int) -> list[dict[str, Any]]:
    traj_dir = next(
        (
            candidate
            for candidate in _trajectory_dir_candidates(run_dir, task_id, trial_id)
            if (candidate / "trajectory.json").exists()
        ),
        None,
    )
    if traj_dir is None:
        return []
    traj_json = traj_dir / "trajectory.json"
    try:
        data = json.loads(traj_json.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return []
    steps = data if isinstance(data, list) else data.get("steps", [])
    if not isinstance(steps, list):
        return []
    response_paths = [
        step.get("model_response_path")
        for step in steps
        if isinstance(step, dict) and step.get("model_response_path")
    ]
    responses: list[dict[str, Any]] = []
    for rel in response_paths:
        response = _load_response_json(traj_dir / str(rel))
        if response:
            responses.append(response)
    return responses


def _trace_file_candidates(
    run_dir: Path,
    task_id: str,
    trial_id: int,
    payload: dict[str, Any],
) -> list[Path]:
    candidates: list[Path] = []
    raw_trace = payload.get("trace_path")
    if raw_trace:
        raw = Path(str(raw_trace))
        roots = [raw] if raw.is_absolute() else [run_dir / raw, raw]
        for root in roots:
            if root.is_file():
                candidates.append(root)
            elif root.is_dir():
                candidates.append(root / "trace.jsonl")
                candidates.extend(sorted(root.glob("trace_*.jsonl")))
                candidates.extend(sorted((root / "trajectory").glob("trace*.jsonl")))
    for traj_dir in _trajectory_dir_candidates(run_dir, task_id, trial_id):
        candidates.append(traj_dir / "trace.jsonl")
        candidates.extend(sorted(traj_dir.glob("trace*.jsonl")))
    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        out.append(path)
    return out


def _opengui_step_events(
    run_dir: Path,
    row: dict[str, Any],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in _trace_file_candidates(run_dir, _norm_task_id(row), _trial_id(row), payload):
        for event in _read_jsonl(path):
            if event.get("type") == "step":
                events.append(event)
    return events


def _event_usage(event: dict[str, Any]) -> dict[str, int]:
    usage = event.get("token_usage") if isinstance(event.get("token_usage"), dict) else {}
    prompt = _safe_int(event.get("token_in") if event.get("token_in") is not None else usage.get("prompt_tokens"))
    completion = _safe_int(
        event.get("token_out") if event.get("token_out") is not None else usage.get("completion_tokens")
    )
    thinking = _safe_int(
        event.get("token_think") if event.get("token_think") is not None
        else usage.get("reasoning_tokens", usage.get("thinking_tokens"))
    )
    total = _safe_int(usage.get("total_tokens")) or prompt + completion + thinking
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def _add_usage(target: dict[str, int], part: dict[str, int]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        target[key] = _safe_int(target.get(key)) + _safe_int(part.get(key))


def _trace_step_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    s1_tokens = _usage({})
    s2_tokens = _usage({})
    s1_steps = s2_steps = slow_steps = fast_steps = 0
    duration_sum_s = 0.0
    for event in events:
        arm = str(event.get("arm") or "").lower()
        usage = _event_usage(event)
        if arm.startswith("s2"):
            s2_steps += 1
            _add_usage(s2_tokens, usage)
        else:
            s1_steps += 1
            _add_usage(s1_tokens, usage)
        if arm.endswith("slow"):
            slow_steps += 1
        else:
            fast_steps += 1
        duration_sum_s += _safe_float(event.get("duration_s"))
    return {
        "steps": len(events),
        "s1_steps": s1_steps,
        "s2_steps": s2_steps,
        "slow_steps": slow_steps,
        "fast_steps": fast_steps,
        "duration_sum_s": duration_sum_s,
        "s1_token_usage": s1_tokens,
        "s2_token_usage": s2_tokens,
    }


def _response_step_stats(responses: list[dict[str, Any]]) -> dict[str, Any]:
    s1_tokens = _usage({})
    s2_tokens = _usage({})
    s1_steps = s2_steps = slow_steps = fast_steps = 0
    duration_sum_s = 0.0
    routed_steps = 0
    for payload in responses:
        route = (
            ((payload.get("router_trace") or {}).get("final_route"))
            or ((payload.get("monitor") or {}).get("final_route"))
            or ((payload.get("monitor") or {}).get("selected_actor"))
            or ""
        )
        route = str(route).lower()
        if not route:
            continue
        routed_steps += 1
        usage = _usage(payload.get("token_usage"))
        if route.startswith("s2"):
            s2_steps += 1
            _add_usage(s2_tokens, usage)
        else:
            s1_steps += 1
            _add_usage(s1_tokens, usage)
        if route.endswith("slow"):
            slow_steps += 1
        else:
            fast_steps += 1
        duration_sum_s += _safe_float(payload.get("runtime_s"))
    return {
        "steps": routed_steps,
        "s1_steps": s1_steps,
        "s2_steps": s2_steps,
        "slow_steps": slow_steps,
        "fast_steps": fast_steps,
        "duration_sum_s": duration_sum_s,
        "s1_token_usage": s1_tokens,
        "s2_token_usage": s2_tokens,
    }


def _extract_response_payload(run_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    for key in ("raw_response", "model_response", "extra_meta"):
        value = row.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    execution = row.get("execution")
    if isinstance(execution, dict):
        for key in ("raw_response", "model_response"):
            value = execution.get(key)
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed
    return _response_from_trajectory(run_dir, _norm_task_id(row), _trial_id(row))


def _usage(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt = _safe_int(raw.get("prompt_tokens"))
    completion = _safe_int(raw.get("completion_tokens"))
    total = _safe_int(raw.get("total_tokens")) or prompt + completion
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def _subtract_usage(total: dict[str, int], part: dict[str, int]) -> dict[str, int]:
    return {
        key: max(0, _safe_int(total.get(key)) - _safe_int(part.get(key)))
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _is_judge_error(row: dict[str, Any]) -> bool:
    judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}
    if judge.get("judge_error"):
        return True
    error_text = " ".join(
        str(value or "")
        for value in (
            row.get("error"),
            (row.get("execution") or {}).get("error") if isinstance(row.get("execution"), dict) else None,
            judge.get("error"),
        )
    ).lower()
    return "judge_error" in error_text or "vlm evaluation" in error_text or "failed to parse vlm" in error_text


def _is_execution_error(row: dict[str, Any]) -> bool:
    execution = row.get("execution") if isinstance(row.get("execution"), dict) else {}
    return bool(execution.get("error")) and not _is_judge_error(row)


def _agent_error_category(payload: dict[str, Any]) -> str | None:
    error = str(payload.get("error") or "").strip()
    if not error:
        return None
    lower = error.lower()
    expected_failures = (
        "intervention_cancelled",
        "stagnation_detected",
        "missing_required_answer_evidence",
        "Task terminated with status:",
        "max_steps_exceeded",
    )
    if error.startswith(expected_failures):
        return None
    our_code_markers = (
        "valueerror",
        "traceback",
        "unmapped action",
        "unsupported action",
        "notimplementederror",
        "assertionerror",
        "attributeerror",
        "typeerror",
        "keyerror",
    )
    if any(marker in lower for marker in our_code_markers):
        return "our_code"
    infra_markers = (
        "timeout",
        "timed out",
        "connection",
        "connecterror",
        "readtimeout",
        "service unavailable",
        "internal server error",
        "bad gateway",
        "gateway timeout",
        "rate limit",
        "ratelimit",
        "too many requests",
        " 500",
        " 502",
        " 503",
        " 504",
    )
    if any(marker in lower for marker in infra_markers):
        return "infra"
    provider_markers = (
        "authenticationerror",
        "invalid_api_key",
        "api key",
        "access denied",
        "arrearage",
        "permission denied",
    )
    if any(marker in lower for marker in provider_markers):
        return "provider"
    return "unknown"


def _pass_at_k(successes: list[bool], k: int) -> float:
    if not successes or k <= 0:
        return 0.0
    return 1.0 if any(successes[:k]) else 0.0


def _avg(values: Iterable[float | int]) -> float:
    vals = [float(v) for v in values]
    return mean(vals) if vals else 0.0


def _bootstrap_ci(values: Iterable[float | int], *, n: int = 1000) -> tuple[float, float]:
    vals = [float(v) for v in values]
    if not vals:
        return (0.0, 0.0)
    if len(vals) == 1:
        return (vals[0], vals[0])
    rng = random.Random(0)
    samples = sorted(
        mean(rng.choice(vals) for _ in vals)
        for _ in range(n)
    )
    return (samples[int(0.025 * n)], samples[int(0.975 * n) - 1])


def _infer_models(arm: str) -> tuple[str, str | None]:
    name = arm.lower()
    if name.startswith("s2") or "_s2" in name or "397" in name and "fastslow" not in name:
        return "397B", "397B"
    if name.startswith("s1") or "_s1" in name:
        return "9B", None
    if "fastslow" in name or arm.startswith("B_"):
        return "9B", "397B"
    return "9B", None


def load_run(run_dir: Path | str, *, arm: str, k: int | None = None) -> dict[str, Any]:
    """Load and aggregate one MobileGym run directory."""

    run_dir = Path(run_dir)
    rows = _result_rows(run_dir)
    episodes: list[dict[str, Any]] = []
    trigger_hist: Counter[str] = Counter()
    mode_hist: Counter[str] = Counter()
    s1_prompt = s1_completion = s1_total = 0
    s2_prompt = s2_completion = s2_total = 0
    s1_steps_total = s2_steps_total = takeover_steps_total = 0
    slow_steps_total = fast_steps_total = 0
    hints_used_total = 0
    runtime_total_s = 0.0

    for row_run_dir, row in rows:
        task_id = _norm_task_id(row)
        trial_id = _trial_id(row)
        payload = _extract_response_payload(row_run_dir, row)
        step_events = _opengui_step_events(row_run_dir, row, payload)
        step_stats = _trace_step_stats(step_events)
        if not step_stats["steps"]:
            step_stats = _response_step_stats(_responses_from_trajectory(row_run_dir, task_id, trial_id))
        s2_usage = payload.get("s2_usage") if isinstance(payload.get("s2_usage"), dict) else {}
        total_episode_tokens = _usage(payload.get("token_usage"))
        s2_tokens = _usage(s2_usage.get("token_usage") if isinstance(s2_usage, dict) else {})
        s1_tokens = _subtract_usage(total_episode_tokens, s2_tokens)
        if step_stats["steps"]:
            s1_tokens = step_stats["s1_token_usage"]
            s2_tokens = step_stats["s2_token_usage"]
            total_episode_tokens = {
                key: s1_tokens[key] + s2_tokens[key]
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
        triggers = s2_usage.get("triggers") if isinstance(s2_usage, dict) else []
        triggers = triggers if isinstance(triggers, list) else []
        for trig in triggers:
            if not isinstance(trig, dict):
                continue
            if trig.get("trigger"):
                trigger_hist[str(trig["trigger"])] += 1
            if trig.get("mode"):
                mode_hist[str(trig["mode"])] += 1

        s1_prompt += s1_tokens["prompt_tokens"]
        s1_completion += s1_tokens["completion_tokens"]
        s1_total += s1_tokens["total_tokens"]
        s2_prompt += s2_tokens["prompt_tokens"]
        s2_completion += s2_tokens["completion_tokens"]
        s2_total += s2_tokens["total_tokens"]

        s1_steps = _safe_int(s2_usage.get("s1_steps")) if isinstance(s2_usage, dict) else 0
        s2_steps = _safe_int(s2_usage.get("s2_steps")) if isinstance(s2_usage, dict) else 0
        slow_steps = 0
        fast_steps = 0
        if step_stats["steps"]:
            s1_steps = _safe_int(step_stats["s1_steps"])
            s2_steps = _safe_int(step_stats["s2_steps"])
            slow_steps = _safe_int(step_stats["slow_steps"])
            fast_steps = _safe_int(step_stats["fast_steps"])
        takeover_steps = _safe_int(s2_usage.get("takeover_steps")) if isinstance(s2_usage, dict) else 0
        hints_used = _safe_int(s2_usage.get("hints_used")) if isinstance(s2_usage, dict) else 0
        s1_steps_total += s1_steps
        s2_steps_total += s2_steps
        slow_steps_total += slow_steps
        fast_steps_total += fast_steps
        takeover_steps_total += takeover_steps
        hints_used_total += hints_used
        execution = row.get("execution") if isinstance(row.get("execution"), dict) else {}
        runtime_s = (
            _safe_float(execution.get("runtime_s"))
            or _safe_float(execution.get("stopwatch_total_s"))
            or _safe_float(row.get("runtime_s"))
            or _safe_float(step_stats.get("duration_sum_s"))
        )
        runtime_total_s += runtime_s

        judge_error = _is_judge_error(row)
        execution_error = _is_execution_error(row)
        agent_error_category = _agent_error_category(payload)
        agent_error = agent_error_category in {"infra", "provider"}
        code_error = agent_error_category in {"our_code", "unknown"}
        episodes.append(
            {
                "task_id": task_id,
                "trial_id": trial_id,
                "success": bool(row.get("is_success")) and not judge_error,
                "is_error": bool(row.get("is_error")),
                "judge_error": judge_error,
                "execution_error": execution_error,
                "agent_error": agent_error,
                "agent_error_category": agent_error_category or "",
                "code_error": code_error,
                "agent_error_text": str(payload.get("error") or ""),
                "progress": _safe_float(row.get("progress")),
                "stop_reason": str((row.get("execution") or {}).get("stop_reason") or ""),
                "s2_triggered": bool(triggers) or s2_steps > 0 or hints_used > 0 or takeover_steps > 0,
                "hint_used": hints_used > 0 or any(t.get("mode") == "hint" for t in triggers if isinstance(t, dict)),
                "takeover_used": bool(s2_usage.get("takeover_used")) if isinstance(s2_usage, dict) else False,
                "s1_steps": s1_steps,
                "s2_steps": s2_steps,
                "slow_steps": slow_steps,
                "fast_steps": fast_steps,
                "takeover_steps": takeover_steps,
                "hints_used": hints_used,
                "runtime_s": runtime_s,
                "trace_step_count": _safe_int(step_stats["steps"]),
                "triggers": triggers,
                "s1_token_usage": s1_tokens,
                "s2_token_usage": s2_tokens,
            }
        )

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        by_task[ep["task_id"]].append(ep)
    for task_eps in by_task.values():
        task_eps.sort(key=lambda ep: ep["trial_id"])

    k = k or max((len(task_eps) for task_eps in by_task.values()), default=1)
    per_task_p1: dict[str, float] = {}
    per_task_pk: dict[str, float] = {}
    for task_id, task_eps in by_task.items():
        valid = [
            ep for ep in task_eps
            if not ep["judge_error"] and not ep["execution_error"] and not ep["agent_error"]
        ]
        if not valid:
            continue
        per_task_p1[task_id] = _avg(1.0 if ep["success"] else 0.0 for ep in valid)
        per_task_pk[task_id] = _pass_at_k([ep["success"] for ep in valid], k)

    valid_eps = [
        ep for ep in episodes
        if not ep["judge_error"] and not ep["execution_error"] and not ep["agent_error"]
    ]
    successes = sum(1 for ep in valid_eps if ep["success"])
    episode_values = [1.0 if ep["success"] else 0.0 for ep in valid_eps]
    total_tokens = s1_total + s2_total
    total_steps = s1_steps_total + s2_steps_total
    s1_model, s2_model = _infer_models(arm)
    return {
        "arm": arm,
        "run_dir": str(run_dir),
        "s1_model": s1_model,
        "s2_model": s2_model,
        "tasks": len(by_task),
        "episodes": len(episodes),
        "valid_episodes": len(valid_eps),
        "successes": successes,
        "episode_sr": successes / len(valid_eps) if valid_eps else 0.0,
        "episode_sr_ci95": _bootstrap_ci(episode_values),
        "pass_at_1": _avg(per_task_p1.values()),
        "pass_at_1_ci95": _bootstrap_ci(per_task_p1.values()),
        "pass_at_k": _avg(per_task_pk.values()),
        "pass_at_k_ci95": _bootstrap_ci(per_task_pk.values()),
        "k": k,
        "judge_error_episodes": sum(1 for ep in episodes if ep["judge_error"]),
        "judge_error_rate": (sum(1 for ep in episodes if ep["judge_error"]) / len(episodes)) if episodes else 0.0,
        "execution_error_episodes": sum(1 for ep in episodes if ep["execution_error"]),
        "execution_error_rate": (sum(1 for ep in episodes if ep["execution_error"]) / len(episodes)) if episodes else 0.0,
        "agent_error_episodes": sum(1 for ep in episodes if ep["agent_error"]),
        "agent_error_rate": (sum(1 for ep in episodes if ep["agent_error"]) / len(episodes)) if episodes else 0.0,
        "code_error_episodes": sum(1 for ep in episodes if ep["code_error"]),
        "code_error_rate": (sum(1 for ep in episodes if ep["code_error"]) / len(episodes)) if episodes else 0.0,
        "agent_error_histogram": dict(sorted(Counter(
            ep["agent_error_text"] for ep in episodes if ep["agent_error_category"]
        ).items())),
        "agent_error_category_histogram": dict(sorted(Counter(
            ep["agent_error_category"] for ep in episodes if ep["agent_error_category"]
        ).items())),
        "s2_triggered_episodes": sum(1 for ep in episodes if ep["s2_triggered"]),
        "s2_trigger_rate": (sum(1 for ep in episodes if ep["s2_triggered"]) / len(episodes)) if episodes else 0.0,
        "hint_episodes": sum(1 for ep in episodes if ep["hint_used"]),
        "hint_rate": (sum(1 for ep in episodes if ep["hint_used"]) / len(episodes)) if episodes else 0.0,
        "takeover_episodes": sum(1 for ep in episodes if ep["takeover_used"]),
        "takeover_rate": (sum(1 for ep in episodes if ep["takeover_used"]) / len(episodes)) if episodes else 0.0,
        "takeover_successes": sum(1 for ep in episodes if ep["takeover_used"] and ep["success"] and not ep["judge_error"]),
        "takeover_success_rate": (
            sum(1 for ep in episodes if ep["takeover_used"] and ep["success"] and not ep["judge_error"])
            / max(1, sum(1 for ep in episodes if ep["takeover_used"] and not ep["judge_error"]))
        ),
        "trigger_histogram": dict(sorted(trigger_hist.items())),
        "mode_histogram": dict(sorted(mode_hist.items())),
        "avg_s2_steps": _avg(ep["s2_steps"] for ep in episodes),
        "avg_steps": _avg((ep["s1_steps"] + ep["s2_steps"]) for ep in episodes),
        "s2_step_share": (s2_steps_total / total_steps) if total_steps else 0.0,
        "s2_call_rate": (s2_steps_total / total_steps) if total_steps else 0.0,
        "slow_call_rate": (slow_steps_total / total_steps) if total_steps else 0.0,
        "fast_call_rate": (fast_steps_total / total_steps) if total_steps else 0.0,
        "hints_used": hints_used_total,
        "takeover_steps": takeover_steps_total,
        "s1_steps": s1_steps_total,
        "s2_steps": s2_steps_total,
        "slow_steps": slow_steps_total,
        "fast_steps": fast_steps_total,
        "runtime_total_s": runtime_total_s,
        "avg_time_per_episode_s": (runtime_total_s / len(episodes)) if episodes else 0.0,
        "avg_time_per_task_s": (runtime_total_s / len(by_task)) if by_task else 0.0,
        "s1_prompt_tokens": s1_prompt,
        "s1_completion_tokens": s1_completion,
        "s1_total_tokens": s1_total,
        "s2_prompt_tokens": s2_prompt,
        "s2_completion_tokens": s2_completion,
        "s2_total_tokens": s2_total,
        "total_tokens": total_tokens,
        "s2_token_share": (s2_total / total_tokens) if total_tokens else 0.0,
        "episodes_detail": episodes,
    }


def _comparison(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_by_key = {
        (ep["task_id"], ep["trial_id"]): ep
        for ep in base.get("episodes_detail", [])
        if not ep.get("judge_error")
    }
    takeover_eps = [
        ep for ep in candidate.get("episodes_detail", [])
        if ep.get("takeover_used") and not ep.get("judge_error")
    ]
    rescued = 0
    for ep in takeover_eps:
        prior = base_by_key.get((ep["task_id"], ep["trial_id"]))
        if ep.get("success") and prior is not None and not prior.get("success"):
            rescued += 1
    return {
        "takeover_episodes": len(takeover_eps),
        "takeover_successes": sum(1 for ep in takeover_eps if ep.get("success")),
        "takeover_rescue_successes": rescued,
        "takeover_rescue_rate": rescued / len(takeover_eps) if takeover_eps else 0.0,
    }


def aggregate_arms(arms: dict[str, Path | str], *, k: int | None = None) -> dict[str, Any]:
    summaries = {name: load_run(path, arm=name, k=k) for name, path in arms.items()}
    comparisons: dict[str, Any] = {}
    baseline = summaries.get("A_s1_only")
    if baseline is not None:
        for name, summary in summaries.items():
            if name == "A_s1_only":
                continue
            comparisons[f"{name}_vs_A_s1_only"] = _comparison(baseline, summary)
    return {"arms": summaries, "comparisons": comparisons}


def render_markdown(aggregate: dict[str, Any]) -> str:
    headers = [
        "arm", "tasks", "episodes", "valid", "SR", "pass@1", "pass@k",
        "pass@1_ci", "pass@k_ci", "judge_err", "agent_err", "code_err", "S2_trigger", "hint", "takeover",
        "avg_steps", "avg_s2_steps", "time/task", "S2_call", "slow_call",
        "s2_token_share", "tokens",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for arm, s in aggregate["arms"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    arm,
                    str(s["tasks"]),
                    str(s["episodes"]),
                    str(s["valid_episodes"]),
                    f"{s['episode_sr']:.3f}",
                    f"{s['pass_at_1']:.3f}",
                    f"{s['pass_at_k']:.3f}",
                    _fmt_ci(s.get("pass_at_1_ci95")),
                    _fmt_ci(s.get("pass_at_k_ci95")),
                    f"{s['judge_error_rate']:.3f}",
                    f"{s['agent_error_rate']:.3f}",
                    f"{s['code_error_rate']:.3f}",
                    f"{s['s2_trigger_rate']:.3f}",
                    f"{s['hint_rate']:.3f}",
                    f"{s['takeover_rate']:.3f}",
                    f"{s['avg_steps']:.2f}",
                    f"{s['avg_s2_steps']:.2f}",
                    f"{s['avg_time_per_task_s']:.1f}",
                    f"{s['s2_call_rate']:.3f}",
                    f"{s['slow_call_rate']:.3f}",
                    f"{s['s2_token_share']:.3f}",
                    str(s["total_tokens"]),
                ]
            )
            + " |"
        )
    if aggregate.get("comparisons"):
        lines.extend(["", "## Takeover Rescue"])
        for name, comp in aggregate["comparisons"].items():
            lines.append(
                f"- {name}: takeover={comp['takeover_episodes']}, "
                f"takeover_success={comp['takeover_successes']}, "
                f"rescued_vs_A={comp['takeover_rescue_successes']} "
                f"({comp['takeover_rescue_rate']:.3f})"
            )
    error_lines: list[str] = []
    for arm, s in aggregate["arms"].items():
        hist = s.get("agent_error_histogram") or {}
        if not hist:
            continue
        error_lines.append(f"- {arm}:")
        for text, count in hist.items():
            compact = " ".join(str(text).split())
            if len(compact) > 240:
                compact = compact[:237] + "..."
            error_lines.append(f"  - {count}x {compact}")
    if error_lines:
        lines.extend(["", "## Agent Error Texts", *error_lines])
    return "\n".join(lines) + "\n"


def _fmt_ci(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return "[0.000,0.000]"
    return f"[{float(value[0]):.3f},{float(value[1]):.3f}]"


def _parse_arm(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    name, path = value.split("=", 1)
    return name, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, help="ARM=RUN_DIR, repeatable")
    parser.add_argument("--k", type=int, default=None, help="pass@k trials; defaults to max trials per task")
    parser.add_argument("--json-out", type=Path, help="Write aggregate JSON")
    parser.add_argument("--md-out", type=Path, help="Write Markdown table")
    args = parser.parse_args(argv)

    arms = dict(_parse_arm(item) for item in args.arm)
    aggregate = aggregate_arms(arms, k=args.k)
    markdown = render_markdown(aggregate)
    print(markdown, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
