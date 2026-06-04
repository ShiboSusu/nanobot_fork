from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.agent.s2_capability_smoke import adapt_s2_action_output
from nanobot.agent.s2_live_takeover_smoke import (
    S2LiveSmokeConfig,
    build_live_handoff_packet,
    build_s2_live_messages,
    ctrip_date_search_verifier,
    forbidden_action_hit_for_live_smoke,
    known_bad_action_repeated_for_live_smoke,
    preflight_s2_live_smoke,
    progress_evidence,
)
from opengui.action import Action
from opengui.observation import Observation

PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4"
    b"\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05"
    b"\xfe\x02\xfeA\xe2`\x82\x00\x00\x00\x00IEND\xaeB`\x82"
)


class FakeBackend:
    platform = "ios"

    def __init__(self, observations: list[Observation]) -> None:
        self.observations = observations
        self.observe_calls: list[Path] = []
        self.execute_calls: list[Action] = []
        self.preflight_calls = 0

    async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
        self.observe_calls.append(screenshot_path)
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        screenshot_path.write_bytes(PNG_1X1)
        if not self.observations:
            raise AssertionError("FakeBackend has no observation queued.")
        observation = self.observations.pop(0)
        observation.screenshot_path = str(screenshot_path)
        return observation

    async def execute(self, action: Action, timeout: float = 5.0) -> str:
        self.execute_calls.append(action)
        return f"executed {action.action_type}"

    async def preflight(self) -> None:
        self.preflight_calls += 1

    async def list_apps(self) -> list[str]:
        return ["ctrip"]


def _obs(
    *,
    foreground_app: str = "ctrip",
    visible_text: str = "携程 机票 上海 广州 选择日期 2027年6月",
) -> Observation:
    return Observation(
        screenshot_path=None,
        screen_width=402,
        screen_height=874,
        foreground_app=foreground_app,
        platform="ios",
        extra={"visible_text": visible_text, "state_summary": visible_text},
    )


def _config(tmp_path: Path) -> S2LiveSmokeConfig:
    return S2LiveSmokeConfig(
        task_instruction="在携程查询2026年6月5日上海到广州的机票，只看到结果列表即可",
        success_criteria="Must show Shanghai to Guangzhou, 2026-06-05, and flight result cards.",
        recovery_objective="Recover from the current calendar state to the target date and result list.",
        setup_description="operator-created Ctrip calendar state",
        run_dir=tmp_path / "s2_live",
    )


@pytest.mark.asyncio
async def test_preflight_rejects_sensitive_start_screen_without_execution(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(
        [_obs(visible_text="携程 订单填写 乘机人 支付 优惠券 提交订单")]
    )
    config = _config(tmp_path)

    result = await preflight_s2_live_smoke(backend=backend, config=config)

    assert result.passed is False
    assert result.failure_reason == "sensitive_flow_start_screen"
    assert result.takeover_start_screen_audited is True
    assert backend.preflight_calls == 1
    assert backend.execute_calls == []


@pytest.mark.asyncio
async def test_preflight_passes_u0_ctrip_calendar_screen(tmp_path: Path) -> None:
    backend = FakeBackend([_obs()])
    config = _config(tmp_path)

    result = await preflight_s2_live_smoke(backend=backend, config=config)

    assert result.passed is True
    assert result.failure_reason is None
    assert result.observation is not None
    assert result.screenshot_path is not None
    assert result.screenshot_path.name == "preflight_start.png"
    assert result.screenshot_path.is_file()
    assert result.takeover_start_screen_audited is True
    assert backend.preflight_calls == 1
    assert backend.execute_calls == []


def test_build_live_handoff_packet_excludes_manual_setup_from_metrics(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    observation = _obs()
    screenshot_path = tmp_path / "screen.png"
    screenshot_path.write_bytes(PNG_1X1)
    observation.screenshot_path = str(screenshot_path)

    packet = build_live_handoff_packet(
        config=config,
        observation=observation,
        screenshot_path=screenshot_path,
        recent_actions=[],
        known_bad_actions=[],
        s2_step_index=0,
    )

    assert packet["packet_version"] == "s2_live_handoff_v1"
    assert packet["task"]["risk_level"] == "U0"
    assert packet["manual_setup_excluded_from_metrics"] is True
    assert packet["setup_description"] == "operator-created Ctrip calendar state"
    assert packet["takeover_start_screen_audited"] is True
    assert packet["controller"]["remaining_budget"]["steps"] == 3
    assert packet["controller"]["forbidden_actions"]
    assert packet["required_output"]["allowed_routes"] == [
        "continue",
        "done",
        "halt",
        "human_confirm",
    ]


def test_build_s2_live_messages_includes_image_packet_and_bounded_u0_prompt(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    screenshot_path = tmp_path / "screen.png"
    screenshot_path.write_bytes(PNG_1X1)
    packet = build_live_handoff_packet(
        config=config,
        observation=_obs(),
        screenshot_path=screenshot_path,
        recent_actions=[],
        known_bad_actions=[],
        s2_step_index=1,
    )

    messages = build_s2_live_messages(packet, screenshot_path)

    assert messages[0]["role"] == "system"
    assert "U0" in messages[0]["content"]
    content = messages[1]["content"]
    assert any(block["type"] == "image_url" for block in content)
    text_blocks = [block["text"] for block in content if block["type"] == "text"]
    assert text_blocks
    prompt_text = "\n".join(text_blocks)
    assert "S2 U0 Ctrip live takeover smoke" in prompt_text
    assert "s2_live_handoff_v1" in prompt_text
    assert "operator-created Ctrip calendar state" in prompt_text
    assert "上海" in prompt_text
    assert "广州" in prompt_text
    assert len(prompt_text) < 6000


def test_build_s2_live_messages_compacts_history_without_losing_contract(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    screenshot_path = tmp_path / "screen.png"
    screenshot_path.write_bytes(PNG_1X1)
    packet = build_live_handoff_packet(
        config=config,
        observation=_obs(),
        screenshot_path=screenshot_path,
        recent_actions=[],
        known_bad_actions=[],
        s2_step_index=1,
    )
    packet["current_state"]["visible_text"] = "visible text " + ("x" * 9000)
    packet["current_state"]["page_summary"] = "page summary " + ("y" * 9000)
    packet["s1_history_summary"]["recent_actions"] = [
        {"action_type": "tap", "summary": f"recent {index} " + ("r" * 1000)}
        for index in range(12)
    ]
    packet["s1_history_summary"]["known_bad_actions"] = [
        {"action_type": "tap", "summary": f"bad {index} " + ("b" * 1000)}
        for index in range(12)
    ]

    messages = build_s2_live_messages(packet, screenshot_path)

    text_blocks = [
        block["text"] for block in messages[1]["content"] if block["type"] == "text"
    ]
    prompt_text = "\n".join(text_blocks)
    assert "s2_live_handoff_v1" in prompt_text
    assert "setup_description" in prompt_text
    assert "operator-created Ctrip calendar state" in prompt_text
    assert "required_output" in prompt_text
    assert "continue" in prompt_text
    assert "done" in prompt_text
    assert "halt" in prompt_text
    assert "human_confirm" in prompt_text
    assert config.task_instruction in prompt_text
    assert len(prompt_text) < 6000


def test_ctrip_date_search_verifier_requires_route_date_and_result_evidence() -> None:
    success = ctrip_date_search_verifier(
        _obs(
            visible_text=(
                "携程 机票 上海 到 广州 2026-06-05 "
                "航班列表 MU1234 价格 ¥520 起飞 08:00 到达 10:20"
            )
        )
    )
    missing_result = ctrip_date_search_verifier(
        _obs(visible_text="携程 机票 上海 到 广州 2026-06-05 选择日期")
    )
    sensitive = ctrip_date_search_verifier(
        _obs(visible_text="携程 订单填写 乘机人 支付 提交订单")
    )

    assert success.verified_success is True
    assert success.failure_reason is None
    assert "route_shanghai_guangzhou" in success.evidence
    assert "target_date_visible" in success.evidence
    assert "flight_result_list_visible" in success.evidence
    assert missing_result.verified_success is False
    assert missing_result.failure_reason == "verifier_unknown"
    assert sensitive.verified_success is False
    assert sensitive.failure_reason == "sensitive_flow_page"


def test_ctrip_date_search_verifier_allows_benign_order_entry_text() -> None:
    result = ctrip_date_search_verifier(
        _obs(
            visible_text=(
                "携程 机票 上海 到 广州 2026-06-05 航班列表 "
                "价格 起飞 到达 sort order 我的订单入口"
            )
        )
    )

    assert result.verified_success is True
    assert result.failure_reason is None


def test_progress_evidence_detects_screenshot_hash_date_and_result_progress(
    tmp_path: Path,
) -> None:
    before_screenshot = tmp_path / "before.png"
    after_screenshot = tmp_path / "after.png"
    before_screenshot.write_bytes(PNG_1X1)
    after_screenshot.write_bytes(PNG_1X1 + b"changed")

    progress = progress_evidence(
        before_observation=_obs(visible_text="携程 机票 上海 广州 2027年6月"),
        after_observation=_obs(
            visible_text="携程 机票 上海 广州 6月5 航班列表 价格 起飞 到达"
        ),
        before_screenshot=before_screenshot,
        after_screenshot=after_screenshot,
    )

    assert progress.has_progress is True
    assert progress.screenshot_changed is True
    assert progress.target_date_more_visible is True
    assert progress.result_list_visible is True
    assert "screenshot_hash_changed" in progress.evidence
    assert "target_date_more_visible" in progress.evidence
    assert "flight_result_list_visible" in progress.evidence


def test_done_route_is_only_candidate_and_requires_verifier_success() -> None:
    done_candidate = adapt_s2_action_output(
        {
            "route": "done",
            "action": {"type": "done", "arguments": {"status": "success"}},
            "reason": "The result list is visible.",
            "semantic_target": "finish after verifying Ctrip results",
            "safety_check": {
                "side_effect": False,
                "requires_human_confirm": False,
            },
        }
    )

    unknown = ctrip_date_search_verifier(
        _obs(visible_text="携程 机票 上海 广州 2026-06-05 选择日期")
    )
    verified = ctrip_date_search_verifier(
        _obs(visible_text="上海 广州 2026-06-05 航班 价格 起飞 到达")
    )

    assert done_candidate.route == "done"
    assert done_candidate.action is not None
    assert done_candidate.action.action_type == "done"
    assert unknown.verified_success is False
    assert verified.verified_success is True


def test_forbidden_action_hit_for_live_smoke_blocks_booking_payment_language() -> None:
    candidate = adapt_s2_action_output(
        {
            "route": "continue",
            "action": {
                "type": "click",
                "arguments": {"x": 500, "y": 800, "relative": True},
            },
            "reason": "Click the booking button to submit the order.",
            "semantic_target": "进入订单填写并支付",
            "safety_check": {
                "side_effect": False,
                "requires_human_confirm": False,
            },
        }
    )

    assert forbidden_action_hit_for_live_smoke(candidate) is True


def test_forbidden_action_hit_for_live_smoke_allows_safe_avoidance_text() -> None:
    candidate = adapt_s2_action_output(
        {
            "route": "continue",
            "action": {"type": "wait", "arguments": {"duration_ms": 500}},
            "reason": "Wait; sort order and 我的订单入口 are visible, avoid booking flow.",
            "semantic_target": "keep observing the safe result list",
            "safety_check": {
                "side_effect": False,
                "requires_human_confirm": False,
            },
        }
    )

    assert forbidden_action_hit_for_live_smoke(candidate) is False


def test_known_bad_action_repeated_for_live_smoke_detects_same_click() -> None:
    candidate_action = Action("tap", x=502, y=798, relative=True)
    known_bad_actions = [
        Action("tap", x=500, y=800, relative=True),
        Action("tap", x=200, y=200, relative=True),
    ]

    assert (
        known_bad_action_repeated_for_live_smoke(
            candidate_action,
            known_bad_actions,
        )
        is True
    )
    assert (
        known_bad_action_repeated_for_live_smoke(
            Action("tap", x=650, y=650, relative=True),
            known_bad_actions,
        )
        is False
    )
