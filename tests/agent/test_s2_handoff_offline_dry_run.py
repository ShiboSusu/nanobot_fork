from __future__ import annotations

import json
from pathlib import Path

from nanobot.agent.s2_handoff_offline_dry_run import (
    HandoffTraceSpec,
    build_handoff_packet,
    load_trace_selection_manifest,
    run_s2_handoff_offline_dry_run,
    write_jsonl_report,
)
from nanobot.providers.base import LLMResponse

PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4"
    b"\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05"
    b"\xfe\x02\xfeA\xe2`\x82\x00\x00\x00\x00IEND\xaeB`\x82"
)


class FakeProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    async def chat_with_retry(self, **kwargs) -> LLMResponse:
        self.calls.append(kwargs)
        return LLMResponse(
            content=self.responses.pop(0),
            finish_reason="stop",
            usage={
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
        )


def test_load_trace_selection_manifest_limits_to_three_traces(tmp_path: Path) -> None:
    trace_paths = [_write_trace(tmp_path, f"run_{index}", step_count=3) for index in range(4)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "trace_path": str(trace_path),
                        "risk_level": "U0",
                        "failure_mode": "semantic_miss",
                        "success_criteria": "report the requested visible result",
                        "recovery_objective": "navigate from the stuck page",
                    }
                    for trace_path in trace_paths
                ]
            }
        ),
        encoding="utf-8",
    )

    specs = load_trace_selection_manifest(manifest)

    assert len(specs) == 3
    assert all(isinstance(spec, HandoffTraceSpec) for spec in specs)
    assert [spec.discovery_source for spec in specs] == ["manifest"] * 3


def test_build_handoff_packet_uses_latest_screenshot_and_recent_s1_context(
    tmp_path: Path,
) -> None:
    trace_path = _write_trace(
        tmp_path,
        "weibo_rank_stuck",
        task_instruction="查看微博热搜榜单前三名",
        step_count=4,
        latest_action={"action_type": "tap", "x": 450, "y": 180, "relative": True},
    )
    spec = HandoffTraceSpec(
        trace_path=trace_path,
        risk_level="U0",
        failure_mode="semantic_miss",
        success_criteria="must see explicit ranking numbers on the full list page",
        recovery_objective="open the complete ranking list instead of the preview",
        notes="榜单任务停在外层预览",
    )

    packet, validation = build_handoff_packet(spec)

    assert validation.valid is True
    assert validation.history_quality == "sufficient"
    assert packet["packet_version"] == "s2_handoff_v1"
    assert packet["history_quality"] == "sufficient"
    assert packet["task"]["instruction"] == "查看微博热搜榜单前三名"
    assert packet["task"]["success_criteria"] == (
        "must see explicit ranking numbers on the full list page"
    )
    assert packet["current_state"]["screenshot_path"].endswith("step_004.png")
    assert packet["current_state"]["foreground_app"] == "com.sina.weibo"
    assert packet["s1_history_summary"]["failure_mode"] == "semantic_miss"
    assert len(packet["s1_history_summary"]["recent_actions"]) == 4
    assert packet["s1_history_summary"]["known_bad_actions"]
    assert packet["s1_history_summary"]["do_not_repeat"]
    assert "open the complete ranking list" in packet["recovery"]["recovery_objective"]
    assert "privacy_toggle" in packet["recovery"]["forbidden_actions"]


def test_build_handoff_packet_resolves_workspace_relative_screenshot_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    trace_path = _write_trace(
        tmp_path,
        "workspace_relative",
        step_count=3,
        screenshot_path_mode="workspace_relative",
    )
    spec = HandoffTraceSpec(
        trace_path=trace_path,
        risk_level="U0",
        failure_mode="semantic_miss",
        success_criteria="must return requested visible result",
        recovery_objective="recover from the current stuck state",
    )

    packet, validation = build_handoff_packet(spec)

    assert validation.valid is True
    assert validation.screenshot_exists is True
    assert packet["current_state"]["screenshot_path"].endswith("step_003.png")


async def test_run_dry_run_writes_safe_parseable_stateful_row_from_manifest(
    tmp_path: Path,
) -> None:
    trace_path = _write_trace(tmp_path, "ticket_wrong_date", step_count=3)
    manifest = _write_manifest(tmp_path, trace_path)
    provider = FakeProvider(
        [
            """{
                "route": "continue",
                "action": {"type": "back", "arguments": {}},
                "reason": "return from the wrong date results page",
                "semantic_target": "recover to the search form before selecting tomorrow",
                "safety_check": {
                    "side_effect": false,
                    "requires_human_confirm": false
                }
            }"""
        ]
    )

    report = await run_s2_handoff_offline_dry_run(
        provider=provider,
        model="qwen3.5-397b-a17b",
        manifest_path=manifest,
    )

    assert report.summary["candidate_traces_found"] == 1
    assert report.summary["valid_traces_attempted"] == 1
    assert report.summary["strong_pass_count"] == 1
    assert report.acceptance_ready is True
    row = report.rows[0]
    assert row["discovery_source"] == "manifest"
    assert row["handoff_packet_built"] is True
    assert row["screenshot_exists"] is True
    assert row["history_quality"] == "sufficient"
    assert row["s2_output_parse_success"] is True
    assert row["route"] == "continue"
    assert row["action_adapter_success"] is True
    assert row["safety_filter_pass"] is True
    assert row["forbidden_action_hit"] is False
    assert row["known_bad_action_repeated"] is False
    assert row["automatic_plausibility_checks_pass"] is True
    assert row["stateful_recovery_plausible"] is None
    assert row["recovery_objective_advanced"] is None
    assert row["human_audit_required"] is True
    assert row["human_audit_plausible"] is None
    assert row["task_success_claimed"] is False
    assert "image_url" in json.dumps(provider.calls[0]["messages"], ensure_ascii=False)
    assert "handoff packet" in json.dumps(provider.calls[0]["messages"], ensure_ascii=False).lower()


async def test_done_route_is_parseable_but_rejected_as_recovery_pass(
    tmp_path: Path,
) -> None:
    trace_path = _write_trace(tmp_path, "false_done", step_count=3)
    manifest = _write_manifest(tmp_path, trace_path, failure_mode="false_done")
    provider = FakeProvider(
        [
            """{
                "route": "done",
                "action": {"type": "done", "arguments": {"status": "success"}},
                "reason": "the task appears complete",
                "semantic_target": "completion",
                "safety_check": {
                    "side_effect": false,
                    "requires_human_confirm": false
                }
            }"""
        ]
    )

    report = await run_s2_handoff_offline_dry_run(
        provider=provider,
        model="qwen3.5-397b-a17b",
        manifest_path=manifest,
    )

    row = report.rows[0]
    assert row["s2_output_parse_success"] is True
    assert row["action_adapter_success"] is True
    assert row["route"] == "done"
    assert row["automatic_plausibility_checks_pass"] is False
    assert row["rejection_reason"] == "done_not_accepted_for_recovery_dry_run"
    assert report.summary["strong_pass_count"] == 0
    assert report.acceptance_ready is False


async def test_known_bad_repetition_blocks_automatic_plausibility(
    tmp_path: Path,
) -> None:
    repeated_action = {"action_type": "tap", "x": 450, "y": 180, "relative": True}
    trace_path = _write_trace(
        tmp_path,
        "repeated_preview_tap",
        step_count=3,
        latest_action=repeated_action,
    )
    manifest = _write_manifest(tmp_path, trace_path, failure_mode="repeated_action")
    provider = FakeProvider(
        [
            """{
                "route": "continue",
                "action": {
                    "type": "click",
                    "arguments": {"x": 452, "y": 182, "relative": true}
                },
                "reason": "tap the same visible item again",
                "semantic_target": "same preview target",
                "safety_check": {
                    "side_effect": false,
                    "requires_human_confirm": false
                }
            }"""
        ]
    )

    report = await run_s2_handoff_offline_dry_run(
        provider=provider,
        model="qwen3.5-397b-a17b",
        manifest_path=manifest,
    )

    row = report.rows[0]
    assert row["known_bad_action_repeated"] is True
    assert row["automatic_plausibility_checks_pass"] is False
    assert row["rejection_reason"] == "known_bad_action_repeated"
    assert report.acceptance_ready is False


async def test_invalid_u2_manifest_row_does_not_call_provider(tmp_path: Path) -> None:
    trace_path = _write_trace(
        tmp_path,
        "send_message",
        task_instruction="给张三发送消息说我到了",
        step_count=3,
    )
    manifest = _write_manifest(tmp_path, trace_path, risk_level="U2")
    provider = FakeProvider([])

    report = await run_s2_handoff_offline_dry_run(
        provider=provider,
        model="qwen3.5-397b-a17b",
        manifest_path=manifest,
    )

    assert provider.calls == []
    row = report.rows[0]
    assert row["handoff_packet_built"] is False
    assert row["screenshot_exists"] is True
    assert row["error"] == "invalid_risk_level"
    assert report.summary["valid_traces_attempted"] == 0
    assert report.summary["invalid_traces_skipped"] == 1
    assert report.summary["invalid_reason_counts"] == {"invalid_risk_level": 1}


async def test_privacy_status_read_task_is_allowed_when_action_is_safe(
    tmp_path: Path,
) -> None:
    trace_path = _write_trace(
        tmp_path,
        "privacy_read",
        task_instruction=(
            "检查B站里我的关注列表设置是不是不公开。只查看当前设置，不修改任何开关。"
            "路径优先：我的 -> 设置 -> 安全隐私/隐私权限设置/空间隐私。"
        ),
        step_count=3,
    )
    manifest = _write_manifest(tmp_path, trace_path)
    provider = FakeProvider(
        [
            """{
                "route": "continue",
                "action": {"type": "back", "arguments": {}},
                "reason": "return to the settings category list without changing toggles",
                "semantic_target": "find the privacy status page",
                "safety_check": {
                    "side_effect": false,
                    "requires_human_confirm": false
                }
            }"""
        ]
    )

    report = await run_s2_handoff_offline_dry_run(
        provider=provider,
        model="qwen3.5-397b-a17b",
        manifest_path=manifest,
    )

    row = report.rows[0]
    assert row["handoff_packet_built"] is True
    assert row["forbidden_action_hit"] is False
    assert row["automatic_plausibility_checks_pass"] is True


async def test_privacy_toggle_recovery_action_is_forbidden(
    tmp_path: Path,
) -> None:
    trace_path = _write_trace(
        tmp_path,
        "privacy_toggle",
        task_instruction=(
            "检查B站里我的关注列表设置是不是不公开。只查看当前设置，不修改任何开关。"
            "路径优先：我的 -> 设置 -> 安全隐私/隐私权限设置/空间隐私。"
        ),
        step_count=3,
    )
    manifest = _write_manifest(tmp_path, trace_path)
    provider = FakeProvider(
        [
            """{
                "route": "continue",
                "action": {
                    "type": "click",
                    "arguments": {"x": 820, "y": 620, "relative": true}
                },
                "reason": "tap the switch to toggle the privacy setting",
                "semantic_target": "turn the public follow-list switch off",
                "safety_check": {
                    "side_effect": false,
                    "requires_human_confirm": false
                }
            }"""
        ]
    )

    report = await run_s2_handoff_offline_dry_run(
        provider=provider,
        model="qwen3.5-397b-a17b",
        manifest_path=manifest,
    )

    row = report.rows[0]
    assert row["forbidden_action_hit"] is True
    assert row["automatic_plausibility_checks_pass"] is False
    assert row["rejection_reason"] == "forbidden_action_hit"


async def test_insufficient_history_attempted_but_not_strong_pass(
    tmp_path: Path,
) -> None:
    trace_path = _write_trace(tmp_path, "short_history", step_count=1)
    manifest = _write_manifest(tmp_path, trace_path, history_quality="insufficient")
    provider = FakeProvider(
        [
            """{
                "route": "continue",
                "action": {"type": "back", "arguments": {}},
                "reason": "return to a safer page",
                "semantic_target": "recover navigation",
                "safety_check": {
                    "side_effect": false,
                    "requires_human_confirm": false
                }
            }"""
        ]
    )

    report = await run_s2_handoff_offline_dry_run(
        provider=provider,
        model="qwen3.5-397b-a17b",
        manifest_path=manifest,
    )

    row = report.rows[0]
    assert row["history_quality"] == "insufficient"
    assert row["automatic_plausibility_checks_pass"] is True
    assert report.summary["valid_traces_attempted"] == 1
    assert report.summary["strong_pass_count"] == 0
    assert report.acceptance_ready is False


def test_write_jsonl_report_includes_summary_and_trace_rows(tmp_path: Path) -> None:
    output = tmp_path / "report.jsonl"
    trace_path = _write_trace(tmp_path, "jsonl_case", step_count=3)
    manifest = _write_manifest(tmp_path, trace_path)
    provider = FakeProvider(
        [
            """{
                "route": "continue",
                "action": {"type": "back", "arguments": {}},
                "reason": "recover",
                "semantic_target": "safe root",
                "safety_check": {}
            }"""
        ]
    )

    report = _run_async(
        run_s2_handoff_offline_dry_run(
            provider=provider,
            model="qwen3.5-397b-a17b",
            manifest_path=manifest,
        )
    )
    write_jsonl_report(report, output)

    lines = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["row_type"] == "summary"
    assert lines[0]["candidate_traces_found"] == 1
    assert lines[1]["row_type"] == "trace"
    assert lines[1]["trace_id"] == "jsonl_case"
    assert lines[1]["task_success_claimed"] is False


def _write_manifest(
    tmp_path: Path,
    trace_path: Path,
    *,
    risk_level: str = "U0",
    failure_mode: str = "semantic_miss",
    history_quality: str | None = None,
) -> Path:
    row = {
        "trace_path": str(trace_path),
        "risk_level": risk_level,
        "failure_mode": failure_mode,
        "success_criteria": "must return requested information without side effects",
        "recovery_objective": "recover from the current stuck state",
        "notes": "selected for S2-2 unit test",
    }
    if history_quality is not None:
        row["history_quality"] = history_quality
    manifest = tmp_path / f"{trace_path.parent.name}_manifest.json"
    manifest.write_text(json.dumps({"traces": [row]}), encoding="utf-8")
    return manifest


def _write_trace(
    tmp_path: Path,
    run_name: str,
    *,
    task_instruction: str = "查询明天北京到天津的高铁票并返回车票信息",
    step_count: int,
    latest_action: dict | None = None,
    screenshot_path_mode: str = "absolute",
) -> Path:
    run_dir = tmp_path / run_name
    screenshots = run_dir / "screenshots"
    screenshots.mkdir(parents=True)
    events: list[dict] = [
        {
            "event": "metadata",
            "task": task_instruction,
            "trace_id": run_name,
        }
    ]
    for step in range(1, step_count + 1):
        screenshot_path = screenshots / f"step_{step:03d}.png"
        screenshot_path.write_bytes(PNG_1X1)
        action = (
            latest_action
        if step == step_count and latest_action is not None
            else {
                "action_type": "swipe",
                "x": 500,
                "y": 800,
                "x2": 500,
                "y2": 200,
                "relative": True,
            }
        )
        event_screenshot_path = (
            screenshot_path.relative_to(tmp_path)
            if screenshot_path_mode == "workspace_relative"
            else screenshot_path
        )
        events.append(
            {
                "event": "step",
                "step_index": step,
                "action": action,
                "action_summary": f"step {step} action",
                "state_summary": f"state after step {step}",
                "screenshot_path": str(event_screenshot_path),
                "execution": {
                    "next_observation": {
                        "screenshot_path": str(event_screenshot_path),
                        "foreground_app": "com.sina.weibo",
                        "screen_width": 402,
                        "screen_height": 874,
                    }
                },
            }
        )
    trace_path = run_dir / "trace.jsonl"
    trace_path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )
    return trace_path


def _run_async(coro):
    import asyncio

    return asyncio.run(coro)
