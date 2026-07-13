"""Cost model for MobileGym fast/slow aggregate outputs."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_PRICES_CNY_PER_1M: dict[str, dict[str, float]] = {
    "9B": {"input": 0.2, "output": 1.5},  # placeholder price
    "35B": {"input": 0.0, "output": 0.0},
    "397B": {"input": 1.2, "output": 7.2},
}


def _load_prices(path: Path | None) -> dict[str, dict[str, float]]:
    prices = deepcopy(DEFAULT_PRICES_CNY_PER_1M)
    if path is None:
        return prices
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("price file must be a JSON object: {model: {input, output}}")
    for model, item in raw.items():
        if not isinstance(item, dict):
            raise ValueError(f"price entry for {model!r} must be an object")
        prices[str(model)] = {
            "input": float(item.get("input", item.get("in_price", 0.0)) or 0.0),
            "output": float(item.get("output", item.get("out_price", 0.0)) or 0.0),
        }
    return prices


def _actor_cost(prompt_tokens: int, completion_tokens: int, model: str | None, prices: dict[str, dict[str, float]]) -> float:
    if not model:
        return 0.0
    price = prices.get(model, {"input": 0.0, "output": 0.0})
    return (
        prompt_tokens / 1_000_000 * float(price.get("input", 0.0))
        + completion_tokens / 1_000_000 * float(price.get("output", 0.0))
    )


def apply_costs(
    aggregate: dict[str, Any],
    *,
    prices: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Return aggregate enriched with per-arm cost and Pareto rows."""

    prices = prices or deepcopy(DEFAULT_PRICES_CNY_PER_1M)
    out = deepcopy(aggregate)
    pareto: list[dict[str, Any]] = []
    for arm, summary in out.get("arms", {}).items():
        s1_model = summary.get("s1_model")
        s2_model = summary.get("s2_model")
        s1_cost = _actor_cost(
            int(summary.get("s1_prompt_tokens") or 0),
            int(summary.get("s1_completion_tokens") or 0),
            s1_model,
            prices,
        )
        s2_cost = _actor_cost(
            int(summary.get("s2_prompt_tokens") or 0),
            int(summary.get("s2_completion_tokens") or 0),
            s2_model,
            prices,
        )
        total_cost = s1_cost + s2_cost
        successes = int(summary.get("successes") or 0)
        sr = float(summary.get("episode_sr") or 0.0)
        total_tokens = int(summary.get("total_tokens") or 0)
        summary["s1_cost_cny"] = s1_cost
        summary["s2_cost_cny"] = s2_cost
        summary["cost_cny"] = total_cost
        summary["cost_per_success"] = total_cost / successes if successes else None
        summary["sr_per_cny"] = sr / total_cost if total_cost > 0 else None
        summary["sr_per_1k_tokens"] = sr / (total_tokens / 1000) if total_tokens > 0 else None
        pareto.append(
            {
                "arm": arm,
                "sr": sr,
                "pass_at_1": summary.get("pass_at_1", 0.0),
                "pass_at_k": summary.get("pass_at_k", 0.0),
                "cost_cny": total_cost,
                "cost_per_success": summary["cost_per_success"],
                "total_tokens": total_tokens,
                "s2_token_share": summary.get("s2_token_share", 0.0),
                "s2_trigger_rate": summary.get("s2_trigger_rate", 0.0),
                "takeover_rate": summary.get("takeover_rate", 0.0),
            }
        )
    out["prices_cny_per_1m"] = prices
    out["pareto"] = sorted(pareto, key=lambda row: (row["cost_cny"], -row["sr"]))
    return out


def render_markdown(aggregate: dict[str, Any]) -> str:
    headers = ["arm", "SR", "cost(¥)", "¥/success", "SR/¥", "SR/1k tok", "S2 cost share"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for arm, summary in aggregate.get("arms", {}).items():
        total_cost = float(summary.get("cost_cny") or 0.0)
        s2_cost = float(summary.get("s2_cost_cny") or 0.0)
        cps = summary.get("cost_per_success")
        sr_per_cny = summary.get("sr_per_cny")
        sr_per_1k = summary.get("sr_per_1k_tokens")
        lines.append(
            "| "
            + " | ".join(
                [
                    arm,
                    f"{float(summary.get('episode_sr') or 0.0):.3f}",
                    f"{total_cost:.6f}",
                    "-" if cps is None else f"{float(cps):.6f}",
                    "-" if sr_per_cny is None else f"{float(sr_per_cny):.3f}",
                    "-" if sr_per_1k is None else f"{float(sr_per_1k):.6f}",
                    f"{(s2_cost / total_cost):.3f}" if total_cost > 0 else "0.000",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_json", type=Path)
    parser.add_argument("--prices", type=Path, help="JSON price table: {model: {input, output}} in CNY/1M tokens")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--pareto-out", type=Path, help="Write Pareto rows as JSONL")
    args = parser.parse_args(argv)

    aggregate = json.loads(args.aggregate_json.read_text(encoding="utf-8"))
    priced = apply_costs(aggregate, prices=_load_prices(args.prices))
    markdown = render_markdown(priced)
    print(markdown, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(priced, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(markdown, encoding="utf-8")
    if args.pareto_out:
        args.pareto_out.parent.mkdir(parents=True, exist_ok=True)
        args.pareto_out.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in priced.get("pareto", [])) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
