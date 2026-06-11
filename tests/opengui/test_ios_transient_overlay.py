from __future__ import annotations

from pathlib import Path

from PIL import Image

from opengui.action import Action
from opengui.agent import GuiAgent
from opengui.interfaces import LLMResponse
from opengui.observation import Observation
from opengui.trajectory.recorder import TrajectoryRecorder


class _NoopLLM:
    async def chat(self, *args, **kwargs) -> LLMResponse:  # noqa: ANN002, ANN003
        del args, kwargs
        return LLMResponse(content="")


class _IosBackend:
    platform = "ios"

    async def preflight(self) -> None:
        return None

    async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
        del timeout
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color=(90, 90, 90)).save(screenshot_path, format="PNG")
        return Observation(
            screenshot_path=str(screenshot_path),
            screen_width=64,
            screen_height=64,
            foreground_app="com.example.app",
            platform="ios",
        )

    async def execute(self, action: Action, timeout: float = 5.0) -> str:
        del action, timeout
        return "ok"


def _agent(tmp_path: Path) -> GuiAgent:
    return GuiAgent(
        _NoopLLM(),
        _IosBackend(),
        trajectory_recorder=TrajectoryRecorder(output_dir=tmp_path / "trajectory", task="overlay test"),
        artifacts_root=tmp_path / "runs",
    )


def test_notification_banner_inside_expected_app_is_transient_not_mismatch(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    observation = Observation(
        screenshot_path=None,
        screen_width=390,
        screen_height=844,
        foreground_app="com.sina.weibo",
        platform="ios",
        extra={"visible_text": ["新消息通知", "横幅", "稍后"]},
    )

    validation = agent._classify_expected_app_observation(
        observation,
        expected_bundle_id="com.sina.weibo",
        task="打开微博查看热搜",
    )

    assert validation["event"] == "transient_overlay"
    assert validation["matches_expected_app"] is True
    assert validation["app_mismatch"] is False


def test_ios_system_overlay_is_transient_not_mismatch_when_current_app_matches(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    current = Observation(
        screenshot_path=None,
        screen_width=390,
        screen_height=844,
        foreground_app="com.sina.weibo",
        platform="ios",
    )
    next_observation = Observation(
        screenshot_path=None,
        screen_width=390,
        screen_height=844,
        foreground_app="com.apple.springboard",
        platform="ios",
        extra={"visible_text": ["通知", "横幅"]},
    )

    assert agent._is_transient_expected_app_interruption(
        current_observation=current,
        next_observation=next_observation,
        expected_bundle_id="com.sina.weibo",
    ) is True


def test_browser_foreground_change_is_wrong_app_mismatch(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    observation = Observation(
        screenshot_path=None,
        screen_width=390,
        screen_height=844,
        foreground_app="com.apple.mobilesafari",
        platform="ios",
        extra={"visible_text": ["微博网页版"]},
    )

    validation = agent._classify_expected_app_observation(
        observation,
        expected_bundle_id="com.sina.weibo",
        task="打开微博查看热搜",
    )

    assert validation["event"] == "app_mismatch"
    assert validation["reason"] == "wrong_app_browser"
    assert validation["matches_expected_app"] is False
    assert validation["app_mismatch"] is True
