from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from nanobot.agent.gui_task_schema import normalize_gui_task_request
from nanobot.agent.tools.gui import GuiSubagentTool
from nanobot.config.schema import GuiConfig
from opengui.observation import Observation
from opengui.skills.normalization import (
    annotate_ios_apps,
    normalize_app_identifier,
    resolve_ios_bundle,
)


def test_normalize_gui_task_fills_app_hint_and_bundle_id() -> None:
    request = normalize_gui_task_request({"task": "打开微博，看看今天热搜榜的第三名是什么"})

    assert request.app_hint == "Weibo"
    assert request.app_bundle_id == "com.sina.weibo"


def test_normalize_gui_task_preserves_explicit_app_hint_and_fills_bundle_id() -> None:
    request = normalize_gui_task_request({"task": "看看热搜榜第三名是什么", "app_hint": "新浪微博"})

    assert request.app_hint == "新浪微博"
    assert request.app_bundle_id == "com.sina.weibo"


def test_normalize_unknown_app_leaves_bundle_empty() -> None:
    request = normalize_gui_task_request({"task": "打开一个不存在的应用"})

    assert request.app_hint is None
    assert request.app_bundle_id is None


def test_ios_resolver_uses_common_aliases_and_annotations() -> None:
    assert resolve_ios_bundle("打开微博看看热搜") == "com.sina.weibo"
    assert resolve_ios_bundle("打开B站播放视频") == "tv.danmaku.bilianime"
    assert resolve_ios_bundle("用网易云音乐播放歌单") == "com.netease.cloudmusic"
    assert resolve_ios_bundle("在小红书搜索夏日穿搭") == "com.xingin.discover"
    assert resolve_ios_bundle("淘宝搜索男士运动鞋") == "com.taobao.taobao4iphone"
    assert annotate_ios_apps(["com.sina.weibo"]) == ["Weibo: com.sina.weibo"]


def test_ios_resolver_prefers_installed_app_aliases() -> None:
    installed_apps = ["Test Weibo: com.example.weibo.beta"]

    assert resolve_ios_bundle("Test Weibo", installed_apps) == "com.example.weibo.beta"
    assert normalize_app_identifier("ios", "微博") == "com.sina.weibo"


def _provider() -> MagicMock:
    provider = MagicMock()
    provider.api_key = "test-key"
    provider.api_base = None
    provider.extra_headers = None
    provider._client = None
    provider.get_default_model.return_value = "gui-default"
    return provider


def _tool(tmp_path: Path) -> GuiSubagentTool:
    return GuiSubagentTool(
        gui_config=GuiConfig(backend="dry-run"),
        provider=_provider(),
        model="gui-default",
        workspace=tmp_path,
    )


class _LaunchBackend:
    platform = "ios"

    def __init__(self, foreground_app: str) -> None:
        self.foreground_app = foreground_app
        self.actions: list[Any] = []

    async def preflight(self) -> None:
        return None

    async def execute(self, action: Any, timeout: float | None = None) -> str:
        del timeout
        self.actions.append(action)
        return "ok"

    async def observe(self, screenshot_path: Path, timeout: float | None = None) -> Observation:
        del timeout
        return Observation(
            screenshot_path=screenshot_path,
            screen_width=390,
            screen_height=844,
            foreground_app=self.foreground_app,
            platform="ios",
        )


@pytest.mark.asyncio
async def test_native_launch_opens_expected_bundle(tmp_path: Path) -> None:
    tool = _tool(tmp_path)
    backend = _LaunchBackend("com.sina.weibo")

    result = await tool._ensure_native_app_opened(
        backend,
        app_hint="Weibo",
        app_bundle_id="com.sina.weibo",
        task="打开微博看看热搜",
        run_dir=tmp_path,
    )

    assert result.status == "opened"
    assert result.foreground_app == "com.sina.weibo"
    assert backend.actions[0].action_type == "open_app"
    assert backend.actions[0].text == "com.sina.weibo"


@pytest.mark.asyncio
async def test_native_launch_marks_browser_as_wrong_app(tmp_path: Path) -> None:
    tool = _tool(tmp_path)
    backend = _LaunchBackend("com.google.chrome.ios")

    result = await tool._ensure_native_app_opened(
        backend,
        app_hint="Weibo",
        app_bundle_id="com.sina.weibo",
        task="打开微博看看热搜",
        run_dir=tmp_path,
    )

    assert result.status == "wrong_app_browser"
    assert result.foreground_app == "com.google.chrome.ios"
