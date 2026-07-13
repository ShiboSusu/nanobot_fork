"""Task-level oracle upper bound for MobileGym four-arm runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PRICES_CNY_PER_1M = {
    "9B": {"input": 0.2, "output": 1.5},  # placeholder price
    "397B": {"input": 1.2, "output": 7.2},
}


def _episode_cost(ep: dict[str, Any], summary: dict[str, Any]) -> float:
    total = 0.0
    for actor, model_key in (("s1", "s1_model"), ("s2", "s2_model")):
        usage = ep.get(f"{actor}_token_usage") if isinstance(ep.get(f"{actor}_token_usage"), dict) else {}
        price = PRICES_CNY_PER_1M.get(str(summary.get(model_key) or ""), {"input": 0.0, "output": 0.0})
        total += (int(usage.get("prompt_tokens") or 0) * price["input"]) / 1_000_000
        total += (int(usage.get("completion_tokens") or 0) * price["output"]) / 1_000_000
    return total


def _arm_task_rows(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for ep in summary.get("episodes_detail", []):
        if ep.get("judge_error") or ep.get("execution_error") or ep.get("agent_error"):
            continue
        by_task.setdefault(str(ep["task_id"]), []).append(ep)
    out: dict[str, dict[str, Any]] = {}
    for task_id, eps in by_task.items():
        costs = [_episode_cost(ep, summary) for ep in eps]
        out[task_id] = {
            "success": any(bool(ep.get("success")) for ep in eps),
            "cost_yuan": sum(costs) / len(costs) if costs else 0.0,
        }
    return out


def analyze(aggregate: dict[str, Any], *, s2slow_arm: str = "s2_slow") -> dict[str, Any]:
    arms = aggregate.get("arms", {})
    per_arm = {arm: _arm_task_rows(summary) for arm, summary in arms.items()}
    task_ids = sorted({task_id for rows in per_arm.values() for task_id in rows})
    oracle_tasks: list[dict[str, Any]] = []
    for task_id in task_ids:
        candidates = [
            {"arm": arm, **rows[task_id]}
            for arm, rows in per_arm.items()
            if task_id in rows and rows[task_id]["success"]
        ]
        if candidates:
            best = min(candidates, key=lambda row: row["cost_yuan"])
            oracle_tasks.append({"task_id": task_id, "success": True, **best})
            continue
        fallbacks = [
            {"arm": arm, **rows[task_id]}
            for arm, rows in per_arm.items()
            if task_id in rows
        ]
        best = min(fallbacks, key=lambda row: row["cost_yuan"]) if fallbacks else {"arm": "", "cost_yuan": 0.0}
        oracle_tasks.append({"task_id": task_id, "success": False, **best})

    oracle_successes = sum(1 for row in oracle_tasks if row["success"])
    oracle_cost = sum(float(row["cost_yuan"]) for row in oracle_tasks)
    n_tasks = len(oracle_tasks)
    s2slow = arms.get(s2slow_arm, {})
    s2slow_sr = float(s2slow.get("pass_at_1") or s2slow.get("episode_sr") or 0.0)
    s2slow_cost = _summary_cost(s2slow)
    points = [
        {
            "label": arm,
            "sr": float(summary.get("pass_at_1") or summary.get("episode_sr") or 0.0),
            "cost_yuan": _summary_cost(summary),
        }
        for arm, summary in arms.items()
    ]
    points.append({
        "label": "oracle",
        "sr": oracle_successes / n_tasks if n_tasks else 0.0,
        "cost_yuan": oracle_cost,
    })
    return {
        "note": "Task-level oracle is an optimistic upper bound: changing the arm can change the whole GUI trajectory.",
        "tasks": n_tasks,
        "oracle_successes": oracle_successes,
        "oracle_sr": oracle_successes / n_tasks if n_tasks else 0.0,
        "oracle_cost_yuan": oracle_cost,
        "s2slow_arm": s2slow_arm,
        "s2slow_sr": s2slow_sr,
        "s2slow_cost_yuan": s2slow_cost,
        "headroom_sr": (oracle_successes / n_tasks if n_tasks else 0.0) - s2slow_sr,
        "headroom_cost_yuan": s2slow_cost - oracle_cost,
        "points": points,
        "oracle_tasks": oracle_tasks,
    }


def _summary_cost(summary: dict[str, Any]) -> float:
    if "cost_cny" in summary:
        return float(summary.get("cost_cny") or 0.0)
    total = 0.0
    for actor, model_key in (("s1", "s1_model"), ("s2", "s2_model")):
        price = PRICES_CNY_PER_1M.get(str(summary.get(model_key) or ""), {"input": 0.0, "output": 0.0})
        total += (int(summary.get(f"{actor}_prompt_tokens") or 0) * price["input"]) / 1_000_000
        total += (int(summary.get(f"{actor}_completion_tokens") or 0) * price["output"]) / 1_000_000
    return total


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Four-Arm Oracle Upper Bound",
        "",
        f"Note: {result['note']}",
        "",
        f"- oracle SR: {result['oracle_sr']:.4f}",
        f"- oracle cost: {result['oracle_cost_yuan']:.6f}",
        f"- {result['s2slow_arm']} SR: {result['s2slow_sr']:.4f}",
        f"- {result['s2slow_arm']} cost: {result['s2slow_cost_yuan']:.6f}",
        f"- SR headroom vs {result['s2slow_arm']}: {result['headroom_sr']:.4f}",
        f"- cost headroom vs {result['s2slow_arm']}: {result['headroom_cost_yuan']:.6f}",
        "",
        "| point | SR | cost_yuan |",
        "| --- | ---: | ---: |",
    ]
    for point in result["points"]:
        lines.append(f"| {point['label']} | {point['sr']:.4f} | {point['cost_yuan']:.6f} |")
    return "\n".join(lines) + "\n"


def render_svg(result: dict[str, Any]) -> str:
    points = result["points"]
    max_cost = max([float(p["cost_yuan"]) for p in points] + [1.0])
    width, height, pad = 720, 420, 50
    circles = []
    for point in points:
        x = pad + (float(point["cost_yuan"]) / max_cost) * (width - 2 * pad)
        y = height - pad - float(point["sr"]) * (height - 2 * pad)
        color = "#d62728" if point["label"] == "oracle" else "#1f77b4"
        circles.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>'
            f'<text x="{x + 8:.1f}" y="{y - 8:.1f}" font-size="12">{point["label"]}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="black"/>'
        f'<line x1="{pad}" y1="{height-pad}" x2="{pad}" y2="{pad}" stroke="black"/>'
        f'<text x="{width/2-40:.1f}" y="{height-10}" font-size="13">cost_yuan</text>'
        f'<text x="8" y="{pad-18}" font-size="13">SR</text>'
        + "".join(circles)
        + "</svg>\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_json", type=Path)
    parser.add_argument("--s2slow-arm", default="s2_slow")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--svg-out", type=Path)
    args = parser.parse_args(argv)

    result = analyze(json.loads(args.aggregate_json.read_text(encoding="utf-8")), s2slow_arm=args.s2slow_arm)
    print(render_markdown(result), end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(result), encoding="utf-8")
    if args.svg_out:
        args.svg_out.parent.mkdir(parents=True, exist_ok=True)
        args.svg_out.write_text(render_svg(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
