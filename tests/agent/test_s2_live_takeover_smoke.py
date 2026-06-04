from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.agent.s2_live_takeover_smoke import (
    S2LiveSmokeConfig,
    build_live_handoff_packet,
    preflight_s2_live_smoke,
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
