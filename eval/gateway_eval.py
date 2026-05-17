#!/usr/bin/env python3
"""
Gateway eval — sends tasks to nanobot gateway HTTP API (equivalent to Telegram messages).

Usage:
  1. Start gateway first:
       nanobot gateway --config ~/.nanobot/config_fullmodel.json
  2. Run this script:
       python3 eval/gateway_eval.py /Users/su4o_/Desktop/clawbench_v2.csv
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import aiohttp

# ── Config ──────────────────────────────────────────────────────────────────
GATEWAY_URL = "http://localhost:18790/v1/chat/completions"
TRACES_DIR = Path.home() / ".nanobot/workspace/gui_runs"
JUDGE_MODEL = "qwen3.5-flash"
JUDGE_API_KEY_ENV = "DASHSCOPE_API_KEY"
JUDGE_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
TASK_TIMEOUT_S = 600  # 10 min per task
INTER_TASK_PAUSE_S = 5  # pause between tasks to let phone settle

# ── Trace finder ────────────────────────────────────────────────────────────

def find_trace_after(before_ts: float) -> Path | None:
    """Return the task-level trace.jsonl created after before_ts."""
    if not TRACES_DIR.exists():
        return None

    # Find session dirs created/modified after before_ts
    session_dirs = sorted(
        (p for p in TRACES_DIR.iterdir() if p.is_dir() and p.stat().st_mtime > before_ts),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not session_dirs:
        return None

    session_dir = session_dirs[0]

    # Prefer task-level trace.jsonl (inside a task subdir)
    task_traces = sorted(
        session_dir.rglob("trace.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if task_traces:
        return task_traces[0]

    # Fall back to session-level trace
    session_traces = sorted(session_dir.glob("trace_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if session_traces:
        return session_traces[0]

    return None


# ── Gateway call ─────────────────────────────────────────────────────────────

DEVICE_PREFIX = (
    "你是一个手机GUI自动化agent。"
    "请使用 gui_task 工具在已连接的鸿蒙手机（ADB serial: APH0219524034229）上完成以下任务。\n"
    "严格要求：\n"
    "1. 所有操作（包括发微信/QQ/邮件、查日历、设闹钟等）必须通过 gui_task 在手机屏幕上实际操作完成\n"
    "2. 禁止使用 send_message 工具或任何非GUI快捷工具\n"
    "3. 禁止使用历史记忆或推测，必须实际打开手机App读取真实数据\n"
    "4. 直接开始执行，不要询问确认\n\n"
    "任务：\n"
)


async def send_task(
    http: aiohttp.ClientSession,
    task_id: str,
    instruction: str,
) -> tuple[str, float]:
    payload = {
        # omit model field so server accepts any configured model
        "messages": [{"role": "user", "content": DEVICE_PREFIX + instruction}],
        # unique session per task so context doesn't bleed between tasks
        "session_id": f"eval_{task_id}_{int(time.time())}",
    }
    start = time.perf_counter()
    try:
        async with http.post(
            GATEWAY_URL,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=TASK_TIMEOUT_S),
        ) as resp:
            data = await resp.json()
            text = data["choices"][0]["message"]["content"]
            return text, time.perf_counter() - start
    except asyncio.TimeoutError:
        return "TIMEOUT", time.perf_counter() - start
    except Exception as exc:
        return f"ERROR: {exc}", time.perf_counter() - start


# ── Judge ────────────────────────────────────────────────────────────────────

def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set for gateway eval judging")
    return value


def judge_text(task_id: str, instruction: str, response: str) -> tuple[bool, str]:
    """Judge success from agent's text response when no GUI trace exists."""
    import urllib.request as _req
    judge_api_key = _required_env(JUDGE_API_KEY_ENV)
    prompt = (
        "You are an evaluator for a mobile AI agent benchmark.\n"
        "Task instruction: {instr}\n\n"
        "Agent's response: {resp}\n\n"
        "Did the agent successfully complete the task? "
        "Reply with JSON only: {{\"success\": true/false, \"reason\": \"one sentence\"}}"
    ).format(instr=instruction, resp=response[:1500])

    payload = json.dumps({
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }).encode()
    req = _req.Request(
        f"{JUDGE_API_BASE}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {judge_api_key}"},
        method="POST",
    )
    try:
        with _req.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        content = data["choices"][0]["message"]["content"].strip()
        # strip markdown fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content)
        return bool(result.get("success", False)), str(result.get("reason", ""))
    except Exception as exc:
        # fallback: check for obvious failure keywords
        low = response.lower()
        if any(w in low for w in ["失败", "无法", "错误", "failed", "error", "unable", "cannot"]):
            return False, f"text_fallback:likely_failed"
        if any(w in low for w in ["成功", "已", "完成", "done", "success", "added", "set"]):
            return True, f"text_fallback:likely_success"
        return False, f"text_judge_error:{exc}"


def judge(task_id: str, instruction: str, trace_path: Path | None, response: str = "") -> tuple[bool, str]:
    if trace_path is not None:
        judge_api_key = _required_env(JUDGE_API_KEY_ENV)
        try:
            from eval.batch.judge import judge_run
            return judge_run(
                instruction=instruction,
                trace_path=trace_path,
                task_id=task_id,
                model=JUDGE_MODEL,
                api_key=judge_api_key,
                api_base=JUDGE_API_BASE,
            )
        except Exception as exc:
            return False, f"judge_error:{exc}"
    # No trace — judge from text response
    if response and not response.startswith(("ERROR:", "TIMEOUT")):
        return judge_text(task_id, instruction, response)
    return False, "no_trace_no_response"


# ── Main ─────────────────────────────────────────────────────────────────────

async def main(csv_path: str) -> None:
    # Load tasks
    tasks: list[tuple[str, str]] = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("task_id"):
                tasks.append((row["task_id"], row["instruction"]))

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(f"eval/results/clawbench_gateway_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "results.jsonl"

    print(f"Loaded {len(tasks)} tasks → {out_dir}")
    print(f"Gateway: {GATEWAY_URL}\n")

    # Check gateway health
    try:
        async with aiohttp.ClientSession() as hc:
            async with hc.get("http://localhost:18790/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status != 200:
                    raise RuntimeError(f"status {r.status}")
        print("✓ Gateway is up\n")
    except Exception as e:
        print(f"✗ Gateway not reachable: {e}")
        print("Start it first:  nanobot gateway --config ~/.nanobot/config_fullmodel.json")
        sys.exit(1)

    results: list[dict] = []
    connector = aiohttp.TCPConnector(limit=1)
    async with aiohttp.ClientSession(connector=connector) as http:
        for i, (task_id, instruction) in enumerate(tasks, 1):
            print(f"[{i:02d}/{len(tasks)}] {task_id}")
            print(f"         {instruction[:90]}{'...' if len(instruction) > 90 else ''}")

            before_ts = time.time()
            response_text, duration = await send_task(http, task_id, instruction)

            trace_path = find_trace_after(before_ts)
            success, judge_reason = judge(task_id, instruction, trace_path, response_text)

            status = "✅" if success else "❌"
            print(f"         {status} {judge_reason[:70]}  ({duration:.0f}s)")

            record = {
                "task_id": task_id,
                "instruction": instruction,
                "response": response_text,
                "success": success,
                "judge_reason": judge_reason,
                "duration_s": round(duration, 1),
                "trace_path": str(trace_path) if trace_path else None,
            }
            results.append(record)
            with results_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if i < len(tasks):
                await asyncio.sleep(INTER_TASK_PAUSE_S)

    # Summary
    n = len(results)
    n_ok = sum(1 for r in results if r["success"])
    summary = {
        "n_tasks": n,
        "n_success": n_ok,
        "pass_at_1": round(n_ok / n, 4) if n else 0,
        "avg_duration_s": round(sum(r["duration_s"] for r in results) / n, 1) if n else 0,
        "per_task_pass_at_1": {r["task_id"]: r["success"] for r in results},
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n{'='*55}")
    print(f"RESULT: {n_ok}/{n} = {n_ok/n*100:.1f}%")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 eval/gateway_eval.py <tasks.csv>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
