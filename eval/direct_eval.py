#!/opt/anaconda3/envs/nanobot/bin/python
"""
Direct eval — uses AgentLoop Python API directly (same as nanobot gateway).

This gives the agent access to WeChat channel (send_message works) AND gui_task.
Equivalent to sending Telegram messages — full channel stack is initialized.

Usage:
  # Start nothing else — this script is self-contained.
  python3 eval/direct_eval.py /Users/su4o_/Desktop/clawbench_v2.csv \
      --config ~/.nanobot/config_fullmodel.json
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

# ── Config ───────────────────────────────────────────────────────────────────
TRACES_DIR = Path.home() / ".nanobot/workspace/gui_runs"
JUDGE_MODEL = "qwen-vl-max-latest"
JUDGE_API_KEY_ENV = "DASHSCOPE_API_KEY"
JUDGE_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
TASK_TIMEOUT_S = 600
INTER_TASK_PAUSE_S = 5

# ── Device prefix (channel="telegram" so agent knows it has phone + WeChat) ──
DEVICE_PREFIX = (
    "你是一个手机助手agent，已连接鸿蒙手机（ADB serial: APH0219524034229）。\n"
    "可以使用所有可用工具（MCP、web search、gui_task等）完成任务。\n"
    "需要在手机屏幕上操作时（发微信、设闹钟、调亮度等）使用 gui_task。\n"
    "【鸿蒙系统】打开App：在桌面向下滑动唤出搜索框，输入App名称打开；没有应用抽屉，禁止向上滑。\n"
    "直接执行，不要询问确认。\n\n"
    "任务：\n"
)


# ── Trace finder ─────────────────────────────────────────────────────────────

def find_trace_after(before_ts: float) -> Path | None:
    if not TRACES_DIR.exists():
        return None
    session_dirs = sorted(
        (p for p in TRACES_DIR.iterdir() if p.is_dir() and p.stat().st_mtime > before_ts),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not session_dirs:
        return None
    session_dir = session_dirs[0]
    task_traces = sorted(
        session_dir.rglob("trace.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if task_traces:
        return task_traces[0]
    session_traces = sorted(
        session_dir.glob("trace_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return session_traces[0] if session_traces else None


# ── Judge ─────────────────────────────────────────────────────────────────────

def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set for direct eval judging")
    return value


def judge_text(instruction: str, response: str) -> tuple[bool, str]:
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
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content)
        return bool(result.get("success", False)), str(result.get("reason", ""))
    except Exception as exc:
        low = response.lower()
        if any(w in low for w in ["失败", "无法", "错误", "failed", "error", "unable", "cannot"]):
            return False, "text_fallback:likely_failed"
        if any(w in low for w in ["成功", "已", "完成", "done", "success", "added", "set"]):
            return True, "text_fallback:likely_success"
        return False, f"text_judge_error:{exc}"


def judge(task_id: str, instruction: str, trace_path: Path | None, response: str = "") -> tuple[bool, str]:
    if trace_path is not None:
        judge_api_key = _required_env(JUDGE_API_KEY_ENV)
        try:
            # opengui filter_step_rows expects "type":"step" but our traces use "event":"step"
            # Write a fixed temp trace next to the original
            fixed_path = trace_path.parent / f"{trace_path.stem}_opengui.jsonl"
            if not fixed_path.exists():
                fixed_lines = []
                for line in trace_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if "event" in row and "type" not in row:
                        row["type"] = row["event"]
                    fixed_lines.append(json.dumps(row, ensure_ascii=False))
                fixed_path.write_text("\n".join(fixed_lines), encoding="utf-8")
            from eval.batch.judge import judge_run
            return judge_run(
                instruction=instruction,
                trace_path=fixed_path,
                task_id=task_id,
                model=JUDGE_MODEL,
                api_key=judge_api_key,
                api_base=JUDGE_API_BASE,
            )
        except Exception as exc:
            return False, f"judge_error:{exc}"
    if response and not response.startswith(("ERROR:", "TIMEOUT")):
        return judge_text(instruction, response)
    return False, "no_trace_no_response"


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(csv_path: str, config_path: str) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from nanobot.config.loader import load_config, set_config_path
    from nanobot.agent.loop import AgentLoop
    from nanobot.agent.tools.message import MessageTool
    from nanobot.bus.queue import MessageBus
    from nanobot.bus.events import OutboundMessage
    from nanobot.channels.manager import ChannelManager
    from nanobot.session.manager import SessionManager
    from nanobot.providers.factory import build_provider_snapshot, build_gui_provider_snapshot, load_provider_snapshot
    from nanobot.utils.helpers import sync_workspace_templates

    # Load config
    resolved = Path(config_path).expanduser()
    set_config_path(resolved)
    config = load_config(resolved)

    # Load tasks
    tasks: list[tuple[str, str]] = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("task_id"):
                tasks.append((row["task_id"], row["instruction"]))

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(f"eval/results/clawbench_direct_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "results.jsonl"

    print(f"Loaded {len(tasks)} tasks → {out_dir}")

    # Build agent (same as _run_gateway)
    sync_workspace_templates(config.workspace_path)
    bus = MessageBus()
    provider_snapshot = build_provider_snapshot(config)
    gui_provider_snapshot = build_gui_provider_snapshot(config)
    session_manager = SessionManager(config.workspace_path)

    agent = AgentLoop(
        bus=bus,
        provider=provider_snapshot.provider,
        workspace=config.workspace_path,
        model=provider_snapshot.model,
        max_iterations=config.agents.defaults.max_tool_iterations,
        context_window_tokens=provider_snapshot.context_window_tokens,
        web_config=config.tools.web,
        context_block_limit=config.agents.defaults.context_block_limit,
        max_tool_result_chars=config.agents.defaults.max_tool_result_chars,
        provider_retry_mode=config.agents.defaults.provider_retry_mode,
        exec_config=config.tools.exec,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        session_manager=session_manager,
        mcp_servers=config.tools.mcp_servers,
        channels_config=config.channels,   # ← WeChat/Telegram channel config
        timezone=config.agents.defaults.timezone,
        unified_session=config.agents.defaults.unified_session,
        disabled_skills=config.agents.defaults.disabled_skills,
        session_ttl_minutes=config.agents.defaults.session_ttl_minutes,
        consolidation_ratio=config.agents.defaults.consolidation_ratio,
        max_messages=config.agents.defaults.max_messages,
        tools_config=config.tools,
        provider_snapshot_loader=load_provider_snapshot,
        provider_signature=provider_snapshot.signature,
        gui_config=config.gui,
        gui_provider=gui_provider_snapshot.provider if gui_provider_snapshot else None,
        gui_model=gui_provider_snapshot.model if gui_provider_snapshot else None,
    )

    # Build channel manager (handles outbound dispatch to WeChat/Telegram)
    channels = ChannelManager(config, bus, session_manager=session_manager)
    print(f"Channels enabled: {channels.enabled_channels}")

    # Channel alias map: LLM may say "wechat" but config key is "weixin"
    _CHANNEL_ALIAS = {"wechat": "weixin", "wx": "weixin"}

    # Load WeChat context_tokens from state file for chat_id resolution
    _weixin_state = Path.home() / ".nanobot/weixin/account.json"
    _weixin_ctx: dict[str, str] = {}
    if _weixin_state.exists():
        try:
            _weixin_ctx = json.loads(_weixin_state.read_text()).get("context_tokens", {})
        except Exception:
            pass

    def _resolve_weixin_chat_id(chat_id: str) -> str:
        """Resolve nickname/remark to WeChat openid. Falls back to single known contact."""
        if chat_id in _weixin_ctx:
            return chat_id
        # Exact match not found — if only one contact available, use it
        if len(_weixin_ctx) == 1:
            return next(iter(_weixin_ctx))
        # Multiple contacts: return as-is and let it fail with a clear warning
        return chat_id

    # Wire message_tool send callback (so send_message → real WeChat channel)
    async def _deliver_to_channel(msg: OutboundMessage, *, record: bool = False, session_key: str | None = None) -> None:
        normalized_channel = _CHANNEL_ALIAS.get(msg.channel, msg.channel)
        resolved_chat_id = msg.chat_id
        if normalized_channel == "weixin":
            resolved_chat_id = _resolve_weixin_chat_id(msg.chat_id)
        if normalized_channel != msg.channel or resolved_chat_id != msg.chat_id:
            msg = OutboundMessage(
                channel=normalized_channel,
                chat_id=resolved_chat_id,
                content=msg.content,
                reply_to=msg.reply_to,
                media=msg.media,
                metadata=msg.metadata,
                buttons=msg.buttons,
            )
        await bus.publish_outbound(msg)

    message_tool = getattr(agent, "tools", {}).get("message")
    if isinstance(message_tool, MessageTool):
        message_tool.set_send_callback(_deliver_to_channel)

    # Start MCP and outbound dispatcher only (no inbound listeners → no Telegram conflict)
    await agent._connect_mcp()
    dispatch_task = asyncio.create_task(channels._dispatch_outbound())

    # Start WeChat channel so it can actually send messages (inbound disabled for eval)
    weixin_ch = channels.channels.get("weixin")
    weixin_task = None
    if weixin_ch:
        weixin_task = asyncio.create_task(channels._start_channel("weixin", weixin_ch))
        print("  Started channel: weixin (outbound only)")

    print()

    results: list[dict] = []

    for i, (task_id, instruction) in enumerate(tasks, 1):
        print(f"[{i:02d}/{len(tasks)}] {task_id}")
        print(f"         {instruction[:90]}{'...' if len(instruction) > 90 else ''}")

        before_ts = time.time()
        start = time.perf_counter()

        try:
            resp = await asyncio.wait_for(
                agent.process_direct(
                    DEVICE_PREFIX + instruction,
                    session_key=f"eval_{task_id}_{int(time.time())}",
                    channel="cli",   # neutral channel — no Telegram/WeChat inbound side-effects
                    chat_id="eval_runner",
                ),
                timeout=TASK_TIMEOUT_S,
            )
            response_text = resp.content if resp else ""
        except asyncio.TimeoutError:
            response_text = "TIMEOUT"
        except Exception as exc:
            response_text = f"ERROR: {exc}"

        duration = time.perf_counter() - start
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

    # Cleanup
    dispatch_task.cancel()
    if weixin_task:
        weixin_task.cancel()
    await agent.close_mcp()
    agent.stop()
    await channels.stop_all()

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
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="Path to tasks CSV")
    p.add_argument("--config", default="~/.nanobot/config_fullmodel.json", help="nanobot config file")
    args = p.parse_args()
    asyncio.run(main(args.csv, args.config))
