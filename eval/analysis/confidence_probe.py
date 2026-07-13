"""Analyze S1 confidence probe samples for confident-wrong behavior."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


HIGH_STAKES_ACTIONS = {"done", "answer", "finished", "input_text", "type", "tap", "click"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _consistency(actions: Any) -> float | None:
    if not isinstance(actions, list) or not actions:
        return None
    canon = [json.dumps(item, sort_keys=True, ensure_ascii=False) for item in actions]
    return max(Counter(canon).values()) / len(canon)


def _score(row: dict[str, Any], key: str, actions_key: str) -> float:
    value = row.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    derived = _consistency(row.get(actions_key))
    return float(derived) if derived is not None else 0.0


def _high_stakes(row: dict[str, Any]) -> bool:
    if isinstance(row.get("high_stakes"), bool):
        return bool(row["high_stakes"])
    action_type = str(row.get("action_type") or "").lower()
    return action_type in HIGH_STAKES_ACTIONS


def _auroc(scores: list[float], labels: list[bool]) -> float | None:
    pos = [(s, i) for i, (s, y) in enumerate(zip(scores, labels)) if y]
    neg = [(s, i) for i, (s, y) in enumerate(zip(scores, labels)) if not y]
    if not pos or not neg:
        return None
    wins = ties = 0.0
    for ps, _ in pos:
        for ns, _ in neg:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                ties += 1.0
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def analyze(rows: list[dict[str, Any]], *, threshold: float = 2 / 3) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if "correct" not in row:
            continue
        temp = _score(row, "temp_consistency", "temp_actions")
        perturb = _score(row, "perturb_consistency", "perturb_actions")
        combined = min(temp, perturb)
        correct = bool(row["correct"])
        normalized.append({
            **row,
            "temp_consistency": temp,
            "perturb_consistency": perturb,
            "combined_consistency": combined,
            "confident": combined >= threshold,
            "correct": correct,
            "high_stakes": _high_stakes(row),
        })

    confusion = {
        "confident_correct": 0,
        "confident_wrong": 0,
        "uncertain_correct": 0,
        "uncertain_wrong": 0,
    }
    for row in normalized:
        key = ("confident" if row["confident"] else "uncertain") + "_" + ("correct" if row["correct"] else "wrong")
        confusion[key] += 1
    wrong = [row for row in normalized if not row["correct"]]
    high_wrong = [row for row in wrong if row["high_stakes"]]
    labels_wrong = [not row["correct"] for row in normalized]
    return {
        "label_definition": (
            "correct must be supplied by the probe dataset. Recommended: action matches a successful same-state "
            "trajectory action, or matches s2_slow on the same observation after manual/schema normalization."
        ),
        "threshold": threshold,
        "n": len(normalized),
        "confusion": confusion,
        "confident_wrong_rate": confusion["confident_wrong"] / len(normalized) if normalized else 0.0,
        "high_stakes_confident_wrong_rate_among_wrong": (
            sum(1 for row in high_wrong if row["confident"]) / len(high_wrong) if high_wrong else None
        ),
        "auroc_error_detection": {
            "temp_uncertainty": _auroc([1.0 - row["temp_consistency"] for row in normalized], labels_wrong),
            "perturb_uncertainty": _auroc([1.0 - row["perturb_consistency"] for row in normalized], labels_wrong),
            "combined_uncertainty": _auroc([1.0 - row["combined_consistency"] for row in normalized], labels_wrong),
        },
        "rows": normalized,
    }


def render_markdown(result: dict[str, Any]) -> str:
    auroc = result["auroc_error_detection"]
    lines = [
        "# S1 Confidence Probe",
        "",
        f"Label definition: {result['label_definition']}",
        "",
        f"- samples: {result['n']}",
        f"- threshold: {result['threshold']:.3f}",
        f"- confident-wrong rate: {result['confident_wrong_rate']:.4f}",
        f"- high-stakes confident-wrong among wrong: {_fmt(result['high_stakes_confident_wrong_rate_among_wrong'])}",
        "",
        "| quadrant | count |",
        "| --- | ---: |",
    ]
    for key, value in result["confusion"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend([
        "",
        "| signal | AUROC for error detection |",
        "| --- | ---: |",
        f"| temp_uncertainty | {_fmt(auroc['temp_uncertainty'])} |",
        f"| perturb_uncertainty | {_fmt(auroc['perturb_uncertainty'])} |",
        f"| combined_uncertainty | {_fmt(auroc['combined_uncertainty'])} |",
    ])
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    return "null" if value is None else f"{float(value):.4f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples_jsonl", type=Path)
    parser.add_argument("--threshold", type=float, default=2 / 3)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args(argv)

    result = analyze(_read_jsonl(args.samples_jsonl), threshold=args.threshold)
    print(render_markdown(result), end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
