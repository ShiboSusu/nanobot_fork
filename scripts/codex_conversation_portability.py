#!/usr/bin/env python3
"""Codex Desktop conversation handoff and local-thread repair helper.

This script deliberately separates two concerns:

* `export-handoff` writes a safe, repo-committable continuation pack.
* `repair-local-state` fixes local Codex Desktop metadata, with dry-run by
  default and backups before any write.

It does not commit full rollout JSONL files because those may contain secrets.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ThreadRecord:
    thread_id: str
    rollout_path: str
    cwd: str
    model_provider: str
    has_user_event: int
    archived: int
    first_user_message: str
    updated_at: int


@dataclass(frozen=True)
class SessionMeta:
    thread_id: str
    path: Path
    cwd: str | None
    model_provider: str | None
    has_user_event: bool


def codex_home(value: str | None = None) -> Path:
    if value:
        return Path(value).expanduser()
    return Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")).expanduser()


def repo_root(value: str | None = None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def read_threads(home: Path) -> list[ThreadRecord]:
    db_path = home / "state_5.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"missing Codex database: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT id, rollout_path, cwd, model_provider, has_user_event,
                   archived, first_user_message, updated_at
            FROM threads
            ORDER BY updated_at DESC
            """
        ).fetchall()
    finally:
        conn.close()
    return [ThreadRecord(*row) for row in rows]


def discover_sessions(home: Path) -> dict[str, SessionMeta]:
    sessions: dict[str, SessionMeta] = {}
    for path in (home / "sessions").glob("**/rollout-*.jsonl"):
        thread_id = None
        cwd = None
        provider = None
        saw_user = False
        try:
            with path.open(encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    if '"role":"user"' in line or '"role": "user"' in line:
                        saw_user = True
                    if index == 0:
                        payload = json.loads(line)
                        if payload.get("type") == "session_meta":
                            meta = payload.get("payload") or {}
                            thread_id = meta.get("id")
                            cwd = meta.get("cwd")
                            provider = meta.get("model_provider")
                    if index > 100 and saw_user:
                        break
        except (OSError, json.JSONDecodeError):
            continue
        if thread_id:
            sessions[thread_id] = SessionMeta(
                thread_id=thread_id,
                path=path,
                cwd=cwd,
                model_provider=provider,
                has_user_event=saw_user,
            )
    return sessions


def newest_thread_for_repo(threads: list[ThreadRecord], repo: Path) -> ThreadRecord | None:
    repo_text = str(repo)
    for record in threads:
        if repo_text in record.cwd or repo_text in record.rollout_path:
            return record
    return threads[0] if threads else None


def export_handoff(args: argparse.Namespace) -> int:
    home = codex_home(args.codex_home)
    repo = repo_root(args.repo)
    threads = read_threads(home)
    sessions = discover_sessions(home)
    selected = (
        next((thread for thread in threads if thread.thread_id == args.thread_id), None)
        if args.thread_id
        else newest_thread_for_repo(threads, repo)
    )
    if selected is None:
        print("No Codex thread found.", file=sys.stderr)
        return 1

    branch = run_git(repo, "branch", "--show-current")
    commit = run_git(repo, "rev-parse", "--short", "HEAD")
    status = run_git(repo, "status", "--short")
    remotes = run_git(repo, "remote", "-v")
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stamp = dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")

    session = sessions.get(selected.thread_id)
    context = {
        "generated_at": now,
        "repo": str(repo),
        "branch": branch,
        "commit": commit,
        "codex_home": str(home),
        "thread_id": selected.thread_id,
        "thread_cwd": selected.cwd,
        "thread_rollout_path": selected.rollout_path,
        "thread_model_provider": selected.model_provider,
        "thread_has_user_event": bool(selected.has_user_event),
        "session_path": str(session.path) if session else None,
        "session_has_user_event": bool(session.has_user_event) if session else None,
        "note": args.note or "",
    }

    out_dir = repo / "docs" / "codex-handoff"
    history_dir = out_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    current = out_dir / "current.md"
    history = history_dir / f"{stamp}.md"
    context_path = out_dir / "context.json"

    markdown = f"""# Codex Conversation Handoff

Generated: {now}

## Purpose

Continue the GUIClaw / iOS WDA / cost-aware router work even if Codex Desktop
is opened under a different account and the original UI thread is hidden.

## Repository State

- Repo: `{repo}`
- Branch: `{branch}`
- Commit: `{commit}`

## Codex Local Thread

- Codex home: `{home}`
- Thread id: `{selected.thread_id}`
- Thread cwd: `{selected.cwd}`
- Rollout path: `{selected.rollout_path}`
- Model provider: `{selected.model_provider}`
- DB has_user_event: `{bool(selected.has_user_event)}`
- Session has user event: `{bool(session.has_user_event) if session else None}`

Do not commit the full rollout file. It may contain secrets from the chat.

## Current Work Summary

- GUI work branch already pushed: `feat/cost-aware-router-v0`
- Last known commit: `4d4eaebf feat: add cost-aware GUI routing v0`
- Handoff doc: `docs/cost-aware-router-v0-handoff.md`
- Real checks passed:
  - iOS Settings opens through direct system action.
  - Sensitive payment/login task is blocked by policy.
  - Shenzhen weather query used the Amap MCP weather tool.

## Resume Commands

```bash
cd {repo}
git switch feat/cost-aware-router-v0
/Volumes/T9/Mac/bin/nanobot-ios-start status
/Volumes/T9/Mac/bin/nanobot-ios-start agent -m "帮我用手机打开设置"
```

## Local UI Repair Commands

Run these from Terminal after closing Codex Desktop:

```bash
cd {repo}
python3 scripts/codex_conversation_portability.py repair-local-state --dry-run --thread-id {selected.thread_id}
python3 scripts/codex_conversation_portability.py repair-local-state --apply --thread-id {selected.thread_id} --target-provider {selected.model_provider}
```

## Extra Note

{args.note or "(none)"}

## Git Status At Export

```text
{status or "(clean)"}
```

## Git Remotes

```text
{remotes}
```
"""

    current.write_text(markdown, encoding="utf-8")
    history.write_text(markdown, encoding="utf-8")
    context_path.write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"current={current}")
    print(f"history={history}")
    print(f"context={context_path}")
    return 0


def codex_running() -> bool:
    system = platform.system()
    if system == "Darwin":
        result = subprocess.run(["pgrep", "-x", "Codex"], capture_output=True, text=True)
        return bool(result.stdout.strip())
    if system == "Windows":
        result = subprocess.run(
            ["tasklist"],
            capture_output=True,
            text=True,
            check=False,
        )
        return "Codex" in result.stdout
    result = subprocess.run(["pgrep", "-f", "codex"], capture_output=True, text=True)
    return bool(result.stdout.strip())


def backup_paths(home: Path, paths: list[Path]) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = home / "repair_backups" / f"codex_conversation_repair_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            target = backup_dir / path.name
            if target.exists():
                target = backup_dir / f"{path.name}.{abs(hash(path))}.bak"
            shutil.copy2(path, target)
    return backup_dir


def rewrite_session_provider(session_path: Path, target_provider: str | None) -> bool:
    if not target_provider:
        return False
    lines = session_path.read_text(encoding="utf-8").splitlines(keepends=True)
    if not lines:
        return False
    first = json.loads(lines[0])
    if first.get("type") != "session_meta":
        return False
    payload = first.setdefault("payload", {})
    if payload.get("model_provider") == target_provider:
        return False
    payload["model_provider"] = target_provider
    lines[0] = json.dumps(first, ensure_ascii=False, separators=(",", ":")) + "\n"
    session_path.write_text("".join(lines), encoding="utf-8", newline="")
    return True


def repair_local_state(args: argparse.Namespace) -> int:
    home = codex_home(args.codex_home)
    db_path = home / "state_5.sqlite"
    if args.apply and codex_running() and not args.allow_running:
        print("Refusing to apply while Codex Desktop appears to be running.", file=sys.stderr)
        print("Close Codex Desktop, then rerun with --apply.", file=sys.stderr)
        return 2

    threads = read_threads(home)
    sessions = discover_sessions(home)
    selected = [
        thread for thread in threads
        if args.thread_id is None or thread.thread_id == args.thread_id
    ]
    if not selected:
        print("No matching threads found.", file=sys.stderr)
        return 1

    plan: list[dict[str, Any]] = []
    for thread in selected:
        session = sessions.get(thread.thread_id)
        target_has_user_event = int(
            bool(thread.first_user_message.strip()) or bool(session and session.has_user_event)
        )
        target_provider = args.target_provider or thread.model_provider
        changes = {}
        if thread.has_user_event != target_has_user_event:
            changes["has_user_event"] = [thread.has_user_event, target_has_user_event]
        if thread.model_provider != target_provider:
            changes["model_provider"] = [thread.model_provider, target_provider]
        if session and thread.rollout_path != str(session.path):
            changes["rollout_path"] = [thread.rollout_path, str(session.path)]
        if session and session.cwd and thread.cwd != session.cwd:
            changes["cwd"] = [thread.cwd, session.cwd]
        if changes:
            plan.append({
                "thread_id": thread.thread_id,
                "changes": changes,
                "session_path": str(session.path) if session else None,
            })

    print(json.dumps({
        "codex_home": str(home),
        "threads_checked": len(selected),
        "planned_threads": len(plan),
        "target_provider": args.target_provider,
        "plan": plan,
    }, ensure_ascii=False, indent=2))

    if not args.apply:
        return 0

    backup_inputs = [db_path, home / ".codex-global-state.json", home / "session_index.jsonl"]
    for item in plan:
        if item.get("session_path"):
            backup_inputs.append(Path(item["session_path"]))
    backup_dir = backup_paths(home, backup_inputs)

    conn = sqlite3.connect(db_path)
    try:
        for item in plan:
            changes = item["changes"]
            values: dict[str, Any] = {}
            for key, (_old, new) in changes.items():
                if key in {"has_user_event", "model_provider", "rollout_path", "cwd"}:
                    values[key] = new
            if values:
                assignments = ", ".join(f"{key}=?" for key in values)
                conn.execute(
                    f"UPDATE threads SET {assignments} WHERE id=?",
                    [*values.values(), item["thread_id"]],
                )
            if item.get("session_path"):
                rewrite_session_provider(Path(item["session_path"]), args.target_provider)
        conn.commit()
    finally:
        conn.close()

    print(f"backup_dir={backup_dir}")
    print("status=ok")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export-handoff", help="Write a safe repo handoff pack.")
    export.add_argument("--codex-home")
    export.add_argument("--repo")
    export.add_argument("--thread-id")
    export.add_argument("--note", default="")
    export.set_defaults(func=export_handoff)

    repair = sub.add_parser("repair-local-state", help="Repair local Codex thread metadata.")
    repair.add_argument("--codex-home")
    repair.add_argument("--thread-id")
    repair.add_argument("--target-provider")
    repair.add_argument("--dry-run", action="store_true")
    repair.add_argument("--apply", action="store_true")
    repair.add_argument("--allow-running", action="store_true")
    repair.set_defaults(func=repair_local_state)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "repair-local-state" and not args.apply:
        args.dry_run = True
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
