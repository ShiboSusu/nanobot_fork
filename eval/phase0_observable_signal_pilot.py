#!/usr/bin/env python3
"""Phase 0 observable signal pilot runner skeleton.

Runs a small U0/U1-only GUI smoke through the existing AgentLoop/GuiSubagentTool
path, then extracts outer/inner trace fields into incremental JSONL records.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


VALID_RISK_LEVELS = {"U0", "U1", "U2"}
DEFAULT_RISK_LEVELS = {"U0", "U1"}
REQUIRED_COLUMNS = {"task_id", "instruction", "instruction_ch", "risk_level"}
MAX_EXECUTION_TASKS = 3
TASK4A_UNSAFE_TERMS = (
    "sms",
    "email",
    "wechat",
    "weixin",
    "message",
    "send ",
    "checkout",
    "payment",
    "pay ",
    "purchase",
    "make a purchase",
    "短信",
    "邮件",
    "微信",
    "发送",
    "发给",
    "付款",
    "支付",
    "结账",
    "下单",
)


@dataclass(frozen=True)
class Phase0Task:
    task_id: str
    instruction: str
    instruction_ch: str
    risk_level: str
    task_source: str = "dataset"

    @property
    def execution_instruction(self) -> str:
        return self.instruction_ch or self.instruction


@dataclass(frozen=True)
class TracePaths:
    run_dir: Path | None
    outer_trace_path: Path | None
    inner_trace_path: Path | None


def parse_risk_levels(values: list[str] | None) -> set[str]:
    if not values:
        return set(DEFAULT_RISK_LEVELS)
    levels: set[str] = set()
    for value in values:
        for part in value.split(","):
            risk = part.strip().upper()
            if risk:
                levels.add(risk)
    invalid = sorted(levels - VALID_RISK_LEVELS)
    if invalid:
        raise ValueError(f"Unsupported risk level(s): {', '.join(invalid)}")
    if "U2" in levels:
        raise ValueError("U2 execution is not supported in Task 4B")
    return levels


def parse_task_ids(values: list[str] | None) -> list[str]:
    if not values:
        return []
    task_ids: list[str] = []
    for value in values:
        for part in value.split(","):
            task_id = part.strip()
            if task_id:
                task_ids.append(task_id)
    return task_ids


def load_dataset(dataset_path: Path) -> list[Phase0Task]:
    with dataset_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = sorted(REQUIRED_COLUMNS - fieldnames)
        if missing_columns:
            raise ValueError(f"Dataset missing required columns: {', '.join(missing_columns)}")

        tasks: list[Phase0Task] = []
        seen: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            task_id = (row.get("task_id") or "").strip()
            instruction = (row.get("instruction") or "").strip()
            instruction_ch = (row.get("instruction_ch") or "").strip()
            risk_level = (row.get("risk_level") or "").strip().upper()
            if not task_id or not instruction or not risk_level:
                raise ValueError(f"Dataset row {row_number} has an empty task_id, instruction, or risk_level")
            if risk_level not in VALID_RISK_LEVELS:
                raise ValueError(f"Dataset row {row_number} has unsupported risk_level={risk_level!r}")
            if task_id in seen:
                raise ValueError(f"Dataset contains duplicated task_id={task_id!r}")
            seen.add(task_id)
            tasks.append(
                Phase0Task(
                    task_id=task_id,
                    instruction=instruction,
                    instruction_ch=instruction_ch,
                    risk_level=risk_level,
                )
            )
    return tasks


def filter_tasks(
    tasks: list[Phase0Task],
    allowed_risks: set[str],
    max_tasks: int | None,
    task_ids: list[str] | None = None,
) -> list[Phase0Task]:
    if max_tasks is not None and max_tasks <= 0:
        raise ValueError("--max-tasks must be greater than 0")
    requested_ids = task_ids or []
    if requested_ids:
        by_id = {task.task_id: task for task in tasks}
        missing = [task_id for task_id in requested_ids if task_id not in by_id]
        if missing:
            raise ValueError(f"Unknown task_id(s): {', '.join(missing)}")
        requested_tasks = [by_id[task_id] for task_id in requested_ids]
        u2_requested = [task.task_id for task in requested_tasks if task.risk_level == "U2"]
        if u2_requested:
            raise ValueError(
                "U2 execution is not supported in Task 4B; blocked task_id(s): "
                + ", ".join(u2_requested)
            )
        selected = [task for task in requested_tasks if task.risk_level in allowed_risks]
        if len(selected) != len(requested_tasks):
            excluded = [task.task_id for task in requested_tasks if task.risk_level not in allowed_risks]
            raise ValueError(
                "Selected task_id(s) excluded by effective risk levels: "
                + ", ".join(excluded)
            )
        for task in selected:
            ensure_task4b_safe(task)
        return selected[:max_tasks] if max_tasks is not None else selected

    selected: list[Phase0Task] = []
    for task in tasks:
        if task.risk_level == "U2":
            continue
        if task.risk_level not in allowed_risks:
            continue
        ensure_task4b_safe(task)
        selected.append(task)
        if max_tasks is not None and len(selected) >= max_tasks:
            break
    return selected


def synthetic_task(task_text: str, allowed_risks: set[str], max_tasks: int | None, task_ids: list[str]) -> list[Phase0Task]:
    if task_ids:
        raise ValueError("--task-text cannot be combined with --task-id")
    if "U0" not in allowed_risks:
        raise ValueError("--task-text requires risk level U0")
    task = Phase0Task(
        task_id="__synthetic_smoke__",
        instruction=task_text.strip(),
        instruction_ch=task_text.strip(),
        risk_level="U0",
        task_source="synthetic",
    )
    if not task.instruction:
        raise ValueError("--task-text cannot be empty")
    ensure_synthetic_task_safe(task)
    if max_tasks != 1:
        raise ValueError("--task-text requires --max-tasks 1")
    return [task]


def ensure_task4b_safe(task: Phase0Task) -> None:
    if task.risk_level == "U2":
        raise RuntimeError(f"Refusing to run U2 task in Task 4B: {task.task_id}")
    text = f"{task.task_id}\n{task.instruction}\n{task.instruction_ch}".lower()
    matched = [term for term in TASK4A_UNSAFE_TERMS if term in text]
    if matched:
        raise RuntimeError(
            f"Refusing to run task {task.task_id}: safety guard matched {', '.join(sorted(set(matched)))}"
        )


def ensure_synthetic_task_safe(task: Phase0Task) -> None:
    if task.risk_level != "U0":
        raise RuntimeError(f"Refusing to run non-U0 synthetic task: {task.task_id}")
    text = f"{task.task_id}\n{task.instruction}\n{task.instruction_ch}".lower()
    for phrase in (
        "do not send messages, emails, sms, wechat, make purchases, or change settings",
        "do not send messages, emails, sms, wechat",
        "do not send",
        "do not make purchases",
        "do not change settings",
    ):
        text = text.replace(phrase, "")
    matched = [term for term in TASK4A_UNSAFE_TERMS if term in text]
    if matched:
        raise RuntimeError(
            f"Refusing to run synthetic task {task.task_id}: safety guard matched {', '.join(sorted(set(matched)))}"
        )


def load_phase0_config(config_path: Path, max_steps: int | None = None) -> Any:
    from nanobot.config.loader import load_config, resolve_config_env_vars, set_config_path

    resolved = config_path.expanduser().resolve()
    set_config_path(resolved)
    try:
        cfg = resolve_config_env_vars(load_config(resolved))
    except ValueError as exc:
        raise RuntimeError(f"Config environment variable resolution failed: {exc}") from exc
    cfg.gui.evaluation.enabled = False
    cfg.gui.enable_skill_execution = False
    cfg.gui.enable_skill_extraction = False
    if max_steps is not None:
        if max_steps <= 0:
            raise RuntimeError("--max-steps must be greater than 0")
        cfg.gui.max_steps = max_steps
    return cfg


def build_agent_loop_and_gui_tool(cfg: Any) -> tuple[Any, Any]:
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus
    from nanobot.providers.factory import (
        build_gui_provider_snapshot,
        build_provider_snapshot,
        load_provider_snapshot,
    )
    from nanobot.session.manager import SessionManager
    from nanobot.utils.helpers import sync_workspace_templates

    sync_workspace_templates(cfg.workspace_path)
    bus = MessageBus()
    provider_snapshot = build_provider_snapshot(cfg)
    gui_provider_snapshot = build_gui_provider_snapshot(cfg)
    session_manager = SessionManager(cfg.workspace_path)

    agent = AgentLoop(
        bus=bus,
        provider=provider_snapshot.provider,
        workspace=cfg.workspace_path,
        model=provider_snapshot.model,
        max_iterations=cfg.agents.defaults.max_tool_iterations,
        context_window_tokens=provider_snapshot.context_window_tokens,
        web_config=cfg.tools.web,
        context_block_limit=cfg.agents.defaults.context_block_limit,
        max_tool_result_chars=cfg.agents.defaults.max_tool_result_chars,
        provider_retry_mode=cfg.agents.defaults.provider_retry_mode,
        exec_config=cfg.tools.exec,
        restrict_to_workspace=cfg.tools.restrict_to_workspace,
        session_manager=session_manager,
        mcp_servers=cfg.tools.mcp_servers,
        channels_config=cfg.channels,
        timezone=cfg.agents.defaults.timezone,
        unified_session=cfg.agents.defaults.unified_session,
        disabled_skills=cfg.agents.defaults.disabled_skills,
        session_ttl_minutes=cfg.agents.defaults.session_ttl_minutes,
        consolidation_ratio=cfg.agents.defaults.consolidation_ratio,
        max_messages=cfg.agents.defaults.max_messages,
        tools_config=cfg.tools,
        provider_snapshot_loader=load_provider_snapshot,
        provider_signature=provider_snapshot.signature,
        gui_config=cfg.gui,
        gui_provider=gui_provider_snapshot.provider if gui_provider_snapshot else None,
        gui_model=gui_provider_snapshot.model if gui_provider_snapshot else None,
    )
    gui_tool = getattr(agent, "tools", {}).get("gui_task")
    if gui_tool is None:
        raise RuntimeError("AgentLoop did not register gui_task")
    return agent, gui_tool


def candidate_phase0_roots(cfg: Any) -> list[Path]:
    artifacts = str(cfg.gui.artifacts_dir)
    roots = [
        Path(artifacts).expanduser(),
        cfg.workspace_path / artifacts,
        cfg.workspace_path / Path(artifacts).expanduser(),
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def find_newest_trace_paths(cfg: Any, after_ts: float) -> TracePaths:
    run_dirs: list[Path] = []
    for root in candidate_phase0_roots(cfg):
        if not root.exists():
            continue
        try:
            candidates = [p for p in root.iterdir() if p.is_dir()]
        except OSError:
            continue
        for candidate in candidates:
            try:
                if candidate.stat().st_mtime >= after_ts - 1:
                    run_dirs.append(candidate)
            except OSError:
                continue
    if not run_dirs:
        return TracePaths(run_dir=None, outer_trace_path=None, inner_trace_path=None)

    run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)
    outer = newest_path(run_dir.glob("trace_*.jsonl"))
    inner = newest_path(run_dir.rglob("trace.jsonl"))
    return TracePaths(run_dir=run_dir, outer_trace_path=outer, inner_trace_path=inner)


def display_path(path: Path | str | None) -> str | None:
    if path is None:
        return None
    text = str(path)
    marker = "/~/.nanobot/workspace/"
    if marker in text:
        prefix, suffix = text.split(marker, 1)
        return f"{prefix}/{suffix}"
    return text


def path_warning(*paths: Path | str | None) -> str | None:
    if any(path is not None and "/~/.nanobot/workspace/" in str(path) for path in paths):
        return "duplicated_workspace_prefix"
    return None


def newest_path(paths: Any) -> Path | None:
    candidates = [p for p in paths if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def step_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event.get("type") == "step" or event.get("event") == "step"]


def step_index(event: dict[str, Any], fallback: int) -> int:
    value = event.get("step_index", event.get("step"))
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return fallback


def extract_trace(task: Phase0Task, trace_paths: TracePaths) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    outer_steps = step_events(load_jsonl(trace_paths.outer_trace_path))
    inner_steps = step_events(load_jsonl(trace_paths.inner_trace_path))
    pairs = align_step_events(outer_steps, inner_steps)
    aligned_steps: list[dict[str, Any]] = []
    previous_action_key: str | None = None
    repeated_streak = 0

    for index, outer, inner in pairs:
        action = safe_action(outer, inner)
        action_key = json.dumps(action, sort_keys=True, ensure_ascii=False) if action else None
        repeated_action = bool(action_key and action_key == previous_action_key)
        repeated_streak = repeated_streak + 1 if repeated_action else 0
        previous_action_key = action_key

        aligned_steps.append(
            {
                "step_index": index,
                "outer_event_present": outer is not None,
                "inner_event_present": inner is not None,
                "alignment_status": alignment_status(outer, inner),
                "model_output": safe_model_output_summary(outer),
                "action": action,
                "trigger_features": {
                    "self_report": {
                        "confidence": None,
                        "need_slow_planner": None,
                        "uncertainty_reason": None,
                        "runtime_signal_parse_error": None,
                    },
                    "risk": {
                        "task_risk_level": task.risk_level,
                        "step_predicted_risk_level": None,
                        "rule_based_step_risk_level": task.risk_level,
                        "action_type_risk": None,
                        "app_sensitive_action": False,
                    },
                    "execution_state": {
                        "repeated_action": repeated_action,
                        "stagnation_count": repeated_streak,
                        "foreground_app_mismatch": False,
                    },
                },
                "outcome_proxies": {
                    "execution_error": execution_error(outer, inner),
                    "action_parse_failure": action_parse_failure(action, inner),
                    "post_action_no_observable_change": None,
                    "judge_derived_not_advanced": None,
                },
                "token_usage": token_usage(outer),
                "timing": timing(outer),
                "observation": observation_summary(outer, inner),
            }
        )

    stats = {
        "outer_step_count": len(outer_steps),
        "inner_step_count": len(inner_steps),
        "aligned_step_count": len(pairs),
        "missing_outer_count": sum(1 for _, outer, _ in pairs if outer is None),
        "missing_inner_count": sum(1 for _, _, inner in pairs if inner is None),
    }
    quality = trace_quality(outer_steps, inner_steps, stats)
    return aligned_steps, stats, quality


def alignment_status(outer: dict[str, Any] | None, inner: dict[str, Any] | None) -> str:
    if outer is not None and inner is not None:
        return "aligned"
    if outer is None:
        return "missing_outer"
    return "missing_inner"


def trace_quality(
    outer_steps: list[dict[str, Any]],
    inner_steps: list[dict[str, Any]],
    stats: dict[str, int],
) -> dict[str, Any]:
    outer_count = stats["outer_step_count"]
    inner_count = stats["inner_step_count"]
    aligned_count = stats["aligned_step_count"]
    inner_coverage = round(inner_count / outer_count, 3) if outer_count > 0 else 0
    has_any_raw_content = any(inner_model_output(step).get("raw_content") for step in inner_steps)
    has_any_parsed_action = any(inner_model_output(step).get("parsed_action") for step in inner_steps)

    quality_warning = None
    if aligned_count == 0:
        quality_warning = "trace_missing"
    elif inner_coverage < 0.8:
        quality_warning = "low_inner_coverage"
    elif not has_any_raw_content:
        quality_warning = "missing_raw_content"
    elif not has_any_parsed_action:
        quality_warning = "missing_parsed_action"

    return {
        "outer_step_count": outer_count,
        "inner_step_count": inner_count,
        "outer_decision_step_count": None,
        "inner_decision_step_count": None,
        "aligned_step_count": aligned_count,
        "missing_outer_count": stats["missing_outer_count"],
        "missing_inner_count": stats["missing_inner_count"],
        "inner_coverage": inner_coverage,
        "inner_decision_coverage": None,
        "has_any_raw_content": has_any_raw_content,
        "has_any_parsed_action": has_any_parsed_action,
        "clean_for_signal_analysis": quality_warning is None,
        "quality_warning": quality_warning,
        "coverage_note": coverage_note(stats, quality_warning),
    }


def coverage_note(stats: dict[str, int], quality_warning: str | None) -> str | None:
    if quality_warning == "low_inner_coverage":
        return "outer_inner_step_mismatch_unclassified"
    if stats["missing_inner_count"] or stats["missing_outer_count"]:
        return "outer_inner_step_mismatch_unclassified"
    return None


def inner_model_output(step: dict[str, Any]) -> dict[str, Any]:
    model_output = step.get("model_output")
    return model_output if isinstance(model_output, dict) else {}


def align_step_events(
    outer_steps: list[dict[str, Any]],
    inner_steps: list[dict[str, Any]],
) -> list[tuple[int, dict[str, Any] | None, dict[str, Any] | None]]:
    outer_by_index = {step_index(event, i): event for i, event in enumerate(outer_steps)}
    inner_by_index = {step_index(event, i): event for i, event in enumerate(inner_steps)}
    if outer_by_index and set(outer_by_index) == set(inner_by_index):
        return [(index, outer_by_index[index], inner_by_index[index]) for index in sorted(outer_by_index)]

    # Some current traces use different step_index bases between outer and inner
    # streams. When counts match but indexes are offset, preserve order as the
    # fallback alignment and keep the outer index as the canonical step_index.
    if outer_steps and inner_steps and len(outer_steps) == len(inner_steps):
        return [
            (step_index(outer, i), outer, inner)
            for i, (outer, inner) in enumerate(zip(outer_steps, inner_steps, strict=True))
        ]

    indexes = sorted(set(outer_by_index) | set(inner_by_index))
    return [(index, outer_by_index.get(index), inner_by_index.get(index)) for index in indexes]


def safe_action(outer: dict[str, Any] | None, inner: dict[str, Any] | None) -> dict[str, Any]:
    for event in (outer, inner):
        if not isinstance(event, dict):
            continue
        action = event.get("action")
        if isinstance(action, dict):
            return action
    model_output = inner.get("model_output") if isinstance(inner, dict) else None
    if isinstance(model_output, dict):
        parsed = model_output.get("parsed_action")
        if isinstance(parsed, dict):
            return parsed
    return {}


def safe_model_output_summary(outer: dict[str, Any] | None) -> str | None:
    if not isinstance(outer, dict):
        return None
    value = outer.get("model_output")
    if value is None:
        return None
    if isinstance(value, str):
        return value[:500]
    return json.dumps(value, ensure_ascii=False, sort_keys=True)[:500]


def token_usage(outer: dict[str, Any] | None) -> dict[str, int | None]:
    usage = outer.get("token_usage") if isinstance(outer, dict) else None
    if not isinstance(usage, dict):
        usage = {}
    return {
        "prompt_tokens": int_or_none(usage.get("prompt_tokens")),
        "completion_tokens": int_or_none(usage.get("completion_tokens")),
        "total_tokens": int_or_none(usage.get("total_tokens")),
    }


def timing(outer: dict[str, Any] | None) -> dict[str, float | None]:
    event = outer if isinstance(outer, dict) else {}
    return {
        "duration_s": float_or_none(event.get("duration_s")),
        "chat_latency_s": float_or_none(event.get("chat_latency_s")),
        "ttft_s": float_or_none(event.get("ttft_s")),
    }


def observation_summary(outer: dict[str, Any] | None, inner: dict[str, Any] | None) -> dict[str, str | None]:
    observation = extract_observation(outer) or extract_observation(inner)
    screenshot_path = None
    foreground_app = None
    if observation:
        screenshot_path = string_or_none(observation.get("screenshot_path"))
        foreground_app = string_or_none(observation.get("foreground_app") or observation.get("app"))
    if screenshot_path is None and isinstance(outer, dict):
        screenshot_path = string_or_none(outer.get("screenshot_path"))
    return {"foreground_app": foreground_app, "screenshot_path": screenshot_path}


def extract_observation(event: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(event, dict):
        return None
    observation = event.get("observation")
    if isinstance(observation, dict):
        return observation
    execution = event.get("execution")
    if isinstance(execution, dict) and isinstance(execution.get("next_observation"), dict):
        return execution["next_observation"]
    prompt = event.get("prompt")
    if isinstance(prompt, dict) and isinstance(prompt.get("current_observation"), dict):
        return prompt["current_observation"]
    return None


def execution_error(outer: dict[str, Any] | None, inner: dict[str, Any] | None) -> bool:
    for event in (outer, inner):
        if not isinstance(event, dict):
            continue
        execution = event.get("execution")
        if isinstance(execution, dict) and execution.get("error"):
            return True
        if event.get("error"):
            return True
    return False


def action_parse_failure(action: dict[str, Any], inner: dict[str, Any] | None) -> bool:
    if action:
        return False
    model_output = inner.get("model_output") if isinstance(inner, dict) else None
    if isinstance(model_output, dict) and model_output.get("parsed_action"):
        return False
    return True


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


async def run_one_task(cfg: Any, gui_tool: Any, task: Phase0Task) -> dict[str, Any]:
    before_ts = time.time()
    started = time.perf_counter()
    runner_returned = False
    runner_error: str | None = None
    task_success: bool | None = None
    raw_result: dict[str, Any] = {}
    try:
        raw = await gui_tool.execute(task=task.execution_instruction, backend=cfg.gui.backend)
        raw_result = json.loads(raw) if isinstance(raw, str) else {}
        runner_returned = isinstance(raw_result, dict)
        if isinstance(raw_result.get("success"), bool):
            task_success = raw_result["success"]
        runner_error = string_or_none(raw_result.get("error"))
    except Exception as exc:
        runner_error = f"{type(exc).__name__}: {exc}"

    trace_paths = find_newest_trace_paths(cfg, before_ts)
    steps, alignment, quality = extract_trace(task, trace_paths)
    termination_reason = normalize_termination(
        runner_error=runner_error,
        task_success=task_success,
        aligned_step_count=alignment["aligned_step_count"],
        exception=not runner_returned,
    )
    clean_success = (
        task_success is True
        and runner_error is None
        and termination_reason == "completed"
        and quality["clean_for_signal_analysis"] is True
    )
    warning = path_warning(trace_paths.run_dir, trace_paths.outer_trace_path, trace_paths.inner_trace_path)

    return {
        "task_id": task.task_id,
        "task_source": task.task_source,
        "instruction": task.execution_instruction,
        "task_risk_level": task.risk_level,
        "runner_returned": runner_returned,
        "runner_error": runner_error,
        "task_success": task_success,
        "clean_success": clean_success,
        "termination_reason": termination_reason,
        "success": clean_success,
        "error": runner_error,
        "duration_s": round(time.perf_counter() - started, 3),
        "runner_result": {
            "steps_taken": raw_result.get("steps_taken"),
            "trace_path": raw_result.get("trace_path"),
        },
        "trace": {
            "run_dir": str(trace_paths.run_dir) if trace_paths.run_dir else None,
            "run_dir_display": display_path(trace_paths.run_dir),
            "outer_trace_path": str(trace_paths.outer_trace_path) if trace_paths.outer_trace_path else None,
            "outer_trace_path_display": display_path(trace_paths.outer_trace_path),
            "inner_trace_path": str(trace_paths.inner_trace_path) if trace_paths.inner_trace_path else None,
            "inner_trace_path_display": display_path(trace_paths.inner_trace_path),
        },
        "alignment": alignment,
        "trace_quality": quality,
        "path_warning": warning,
        "steps": steps,
    }


def normalize_termination(
    *,
    runner_error: str | None,
    task_success: bool | None,
    aligned_step_count: int,
    exception: bool,
) -> str:
    if exception:
        return "exception"
    error_text = (runner_error or "").lower()
    if "max_steps_exceeded" in error_text:
        return "max_steps_exceeded"
    if "stagnation_detected" in error_text:
        return "stagnation_detected"
    if aligned_step_count == 0:
        return "trace_missing"
    if runner_error:
        return "unknown_error"
    if task_success is True:
        return "completed"
    return "unknown_error"


async def run(args: argparse.Namespace) -> int:
    if args.summarize_output:
        summarize_output(Path(args.summarize_output))
        return 0

    if not args.dataset:
        print("ERROR: --dataset is required unless --summarize-output is used", file=sys.stderr)
        return 2
    dataset_path = Path(args.dataset)
    try:
        allowed_risks = parse_risk_levels(args.risk_level)
        task_ids = parse_task_ids(args.task_id)
        tasks = load_dataset(dataset_path)
        if args.list_tasks:
            print_task_list(tasks)
            return 0
        if args.task_text:
            selected = synthetic_task(args.task_text, allowed_risks, args.max_tasks, task_ids)
        else:
            selected = filter_tasks(tasks, allowed_risks, args.max_tasks, task_ids)
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not selected:
        print("ERROR: No tasks selected after risk filtering", file=sys.stderr)
        return 2

    if args.dry_select:
        print_run_summary(
            tasks=tasks,
            selected=selected,
            allowed_risks=allowed_risks,
            cfg=None,
            output_path=None,
            max_steps=args.max_steps,
        )
        print("Dry-select mode: AgentLoop was not initialized; no GUI task was run; no output JSONL was written.")
        return 0

    if args.max_tasks is None:
        print("ERROR: --max-tasks is required when running GUI tasks", file=sys.stderr)
        return 2
    if args.max_tasks > MAX_EXECUTION_TASKS:
        print(
            f"ERROR: Refusing to run {args.max_tasks} tasks in Task 4B; maximum is {MAX_EXECUTION_TASKS}",
            file=sys.stderr,
        )
        return 2
    if not args.config or not args.output:
        print("ERROR: --config and --output are required when running GUI tasks", file=sys.stderr)
        return 2

    try:
        cfg = load_phase0_config(Path(args.config), max_steps=args.max_steps)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    output_path = Path(args.output)
    print_run_summary(
        tasks=tasks,
        selected=selected,
        allowed_risks=allowed_risks,
        cfg=cfg,
        output_path=output_path,
        max_steps=cfg.gui.max_steps,
    )

    agent = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        agent, gui_tool = build_agent_loop_and_gui_tool(cfg)
        with output_path.open("a", encoding="utf-8") as out:
            for task in selected:
                record = await run_one_task(cfg, gui_tool, task)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                print(
                    "Task result:"
                    f" {task.task_id} success={record['success']}"
                    f" clean_success={record['clean_success']}"
                    f" termination={record['termination_reason']}"
                    f" error={record['error']}"
                    f" outer_steps={record['alignment']['outer_step_count']}"
                    f" inner_steps={record['alignment']['inner_step_count']}"
                    f" aligned={record['alignment']['aligned_step_count']}"
                    f" quality_warning={record['trace_quality']['quality_warning']}"
                    f" run_dir={record['trace']['run_dir_display']}"
                )
    finally:
        if agent is not None:
            try:
                close_mcp = getattr(agent, "close_mcp", None)
                if callable(close_mcp):
                    await close_mcp()
            finally:
                stop = getattr(agent, "stop", None)
                if callable(stop):
                    stop()
    return 0


def print_task_list(tasks: list[Phase0Task]) -> None:
    print(f"Loaded dataset rows: {len(tasks)}")
    print(f"Dataset risk distribution: {dict(Counter(task.risk_level for task in tasks))}")
    for task in tasks:
        marker = " [U2 BLOCKED]" if task.risk_level == "U2" else ""
        print(f"{task.task_id}\t{task.risk_level}{marker}\t{preview(task.execution_instruction)}")


def print_run_summary(
    *,
    tasks: list[Phase0Task],
    selected: list[Phase0Task],
    allowed_risks: set[str],
    cfg: Any | None,
    output_path: Path | None,
    max_steps: int | None,
) -> None:
    print(f"Loaded dataset rows: {len(tasks)}")
    print(f"Dataset risk distribution: {dict(Counter(task.risk_level for task in tasks))}")
    print(f"Selected task count: {len(selected)}")
    print("Selected task IDs: " + ", ".join(task.task_id for task in selected))
    print("Selected task sources: " + ", ".join(task.task_source for task in selected))
    print(f"Effective risk levels: {','.join(sorted(allowed_risks))}")
    print(f"Effective max steps: {max_steps if max_steps is not None else 'config default'}")
    if cfg is not None:
        print(f"Backend: {cfg.gui.backend}")
        print(f"Agent profile: {cfg.gui.agent_profile}")
    else:
        print("Backend: not initialized")
        print("Agent profile: not initialized")
    print(f"Output path: {output_path if output_path is not None else 'not writing output'}")
    print("Warning: U2 is blocked in Task 4B.")


def preview(text: str, limit: int = 96) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def summarize_output(path: Path) -> None:
    records = load_output_records(path)
    risk_distribution = Counter(record.get("task_risk_level") for record in records)
    termination_distribution = Counter(record.get("termination_reason", "missing") for record in records)
    clean_success_count = sum(1 for record in records if record.get("clean_success") is True)
    coverage_values = [
        quality["inner_coverage"]
        for record in records
        if isinstance((quality := record.get("trace_quality")), dict)
        and isinstance(quality.get("inner_coverage"), (int, float))
    ]
    average_inner_coverage = round(sum(coverage_values) / len(coverage_values), 3) if coverage_values else 0
    quality_warning_count = sum(
        1
        for record in records
        if isinstance(record.get("trace_quality"), dict) and record["trace_quality"].get("quality_warning") is not None
    )
    clean_for_signal_count = sum(
        1
        for record in records
        if isinstance(record.get("trace_quality"), dict)
        and record["trace_quality"].get("clean_for_signal_analysis") is True
    )

    print(f"Total records: {len(records)}")
    print(f"Risk distribution: {dict(risk_distribution)}")
    print(f"Clean success count: {clean_success_count}")
    print(f"Termination reason distribution: {dict(termination_distribution)}")
    print(f"Average inner coverage: {average_inner_coverage}")
    print(f"Records with quality warnings: {quality_warning_count}")
    print(f"Clean for signal analysis count: {clean_for_signal_count}")


def load_output_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"ERROR: Output JSONL does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to phase0 config JSON")
    parser.add_argument("--dataset", help="Path to phase0 validation CSV")
    parser.add_argument("--output", help="Append-only JSONL output path")
    parser.add_argument("--summarize-output", help="Summarize an existing output JSONL without running GUI")
    parser.add_argument("--max-tasks", type=int, help="Maximum number of selected tasks to run")
    parser.add_argument("--max-steps", type=int, help="Temporary GUI max_steps override for this run")
    parser.add_argument("--task-id", action="append", help="Task ID to select; may be repeated or comma-separated")
    parser.add_argument("--task-text", help="Synthetic smoke-only task text; requires --risk-level U0 and --max-tasks 1")
    parser.add_argument("--list-tasks", action="store_true", help="List dataset tasks and exit without running GUI")
    parser.add_argument("--dry-select", action="store_true", help="Apply selection and safety checks without running GUI")
    parser.add_argument(
        "--risk-level",
        action="append",
        help="Allowed risk level; may be repeated or comma-separated. Defaults to U0,U1. U2 is unsupported in Task 4B.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
