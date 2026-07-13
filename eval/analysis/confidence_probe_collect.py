"""Collect S1 consistency samples from saved MobileGym per-step prompts."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ACTION_RE = re.compile(r"Action:\s*(\{.*?\})(?:\s*$|\n)", re.DOTALL)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _iter_prompt_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.glob("**/step_*_prompt.json"))


def _ids_from_path(path: Path) -> dict[str, Any]:
    traj = path.parent.name
    step_match = re.search(r"step_(\d+)_prompt\.json$", path.name)
    trial_match = re.search(r"_t(\d+)$", traj)
    task = re.sub(r"_t\d+$", "", traj).replace("_", ".")
    return {
        "task_id": task,
        "trial_id": int(trial_match.group(1)) if trial_match else 0,
        "step_index": int(step_match.group(1)) if step_match else None,
    }


def _extract_action(text: str) -> dict[str, Any] | str:
    match = None
    for match in ACTION_RE.finditer(text):
        pass
    candidates = []
    if match:
        candidates.append(match.group(1))
    candidates.extend(part[part.find("{") :] for part in text.splitlines() if "{" in part)
    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            if "action_type" not in parsed and "action" in parsed:
                parsed["action_type"] = parsed.pop("action")
            return parsed
    return text.strip()[:500]


def _action_type(action: Any) -> str:
    if isinstance(action, dict):
        return str(action.get("action_type") or action.get("action") or "").lower()
    return ""


def _consistency(actions: list[Any]) -> float:
    if not actions:
        return 0.0
    canon = [json.dumps(item, sort_keys=True, ensure_ascii=False) for item in actions]
    return max(canon.count(item) for item in set(canon)) / len(canon)


def _compact_history(messages: list[dict[str, Any]], keep_user_messages: int) -> list[dict[str, Any]]:
    out = json.loads(json.dumps(messages))
    seen_user = 0
    for message in reversed(out):
        if message.get("role") != "user":
            continue
        seen_user += 1
        if seen_user <= keep_user_messages:
            continue
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = [
                item for item in content
                if not (isinstance(item, dict) and item.get("type") in {"image_url", "input_image"})
            ] or [{"type": "text", "text": "(history omitted for perturbation)"}]
        elif isinstance(content, str):
            message["content"] = "(history omitted for perturbation)"
    out.append({
        "role": "user",
        "content": "Perturbation check: choose the next action using only the current screen and reliable recent history. Output the same required format.",
    })
    return out


def _chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
    timeout: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "extra_body": {"enable_thinking": enable_thinking},
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def _content(response: dict[str, Any]) -> str:
    try:
        return str(response["choices"][0]["message"].get("content") or "")
    except (KeyError, IndexError, TypeError):
        return json.dumps(response, ensure_ascii=False)


def collect(args: argparse.Namespace) -> None:
    prompts = _iter_prompt_files(args.prompts)
    if args.shuffle:
        random.Random(args.seed).shuffle(prompts)
    if args.limit:
        prompts = prompts[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as f:
        for prompt_path in prompts:
            messages = _read_json(prompt_path)
            if not isinstance(messages, list):
                continue
            row = {
                **_ids_from_path(prompt_path),
                "prompt_path": str(prompt_path),
                "label_definition": "unlabeled; compare later against successful same-state action or s2_slow same-observation action after normalization",
            }
            temp_actions = []
            perturb_actions = []
            raw_usage: list[dict[str, Any]] = []
            for _ in range(args.k):
                response = _chat(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    enable_thinking=args.enable_thinking,
                    timeout=args.timeout,
                )
                raw_usage.append(response.get("usage") or {})
                temp_actions.append(_extract_action(_content(response)))
            perturbed = _compact_history(messages, keep_user_messages=args.perturb_history_window)
            for _ in range(args.k):
                response = _chat(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    messages=perturbed,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    enable_thinking=args.enable_thinking,
                    timeout=args.timeout,
                )
                raw_usage.append(response.get("usage") or {})
                perturb_actions.append(_extract_action(_content(response)))
            row.update({
                "action_type": _action_type(temp_actions[0] if temp_actions else {}),
                "temp_actions": temp_actions,
                "perturb_actions": perturb_actions,
                "temp_consistency": _consistency(temp_actions),
                "perturb_consistency": _consistency(perturb_actions),
                "usage": raw_usage,
            })
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if args.sleep:
                time.sleep(args.sleep)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True, help="trajectory root or one step_*_prompt.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", default=os.environ.get("MA_INTRANET_URL", "").rstrip("/") + "/v1")
    parser.add_argument("--api-key", default=os.environ.get("MA_TOKEN", ""))
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--perturb-history-window", type=int, default=1)
    args = parser.parse_args(argv)
    if not args.base_url or not args.api_key:
        raise SystemExit("--base-url/MA_INTRANET_URL and --api-key/MA_TOKEN are required")
    collect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
