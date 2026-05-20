from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import tomllib
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import opengui.backends.adb as adb_backend_module
import opengui.backends.hdc as hdc_backend_module
import opengui.backends.ios_wda as ios_wda_module
from opengui.action import Action, ActionError, parse_action, resolve_coordinate
from opengui.agent import GuiAgent, _AgentActionGrounder, _AgentSubgoalRunner
from opengui.agent_profiles import normalize_profile_response, profile_tool_definition
from opengui.backends.adb import AdbBackend, AdbError
from opengui.backends.dry_run import DryRunBackend
from opengui.backends.hdc import HdcBackend
from opengui.interfaces import LLMResponse, ToolCall
from opengui.observation import Observation
from opengui.prompts.system import build_system_prompt
from opengui.skills.data import SkillStep
from opengui.trajectory.recorder import TrajectoryRecorder


def _make_recorder(tmp_path: Path, task: str = "test task") -> TrajectoryRecorder:
    return TrajectoryRecorder(output_dir=tmp_path / "traj", task=task)


class _ScriptedLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)

    async def chat(self, messages, tools=None, tool_choice=None) -> LLMResponse:
        if not self._responses:
            raise AssertionError("No scripted responses left.")
        return self._responses.pop(0)


class _RecordingLLM(_ScriptedLLM):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(responses)
        self.calls: list[list[dict]] = []

    async def chat(self, messages, tools=None, tool_choice=None, model=None, max_tokens=None) -> LLMResponse:
        self.calls.append(copy.deepcopy(messages))
        return await super().chat(messages, tools=tools, tool_choice=tool_choice)


class _RecordingValidator:
    def __init__(self, returns: list[bool]) -> None:
        self._returns = list(returns)
        self.calls: list[dict[str, object]] = []

    async def validate(self, valid_state: str, screenshot=None) -> bool:
        self.calls.append({"valid_state": valid_state, "screenshot": screenshot})
        if not self._returns:
            raise AssertionError("No validator responses left.")
        return self._returns.pop(0)


class _SkillTestBackend:
    def __init__(self) -> None:
        self.platform = "dry-run"
        self.executed_actions: list[object] = []
        self.observe_calls: list[Path] = []

    async def execute(self, action, timeout: float = 5.0) -> str:
        del timeout
        self.executed_actions.append(action)
        return f"executed:{action.action_type}"

    async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
        del timeout
        self.observe_calls.append(screenshot_path)
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        screenshot_path.write_bytes(b"png")
        return Observation(
            screenshot_path=str(screenshot_path),
            screen_width=1000,
            screen_height=1000,
            foreground_app="DryRun",
            platform=self.platform,
        )

    async def preflight(self) -> None:
        return None

    async def list_apps(self) -> list[str]:
        return []


def test_parse_scroll_allows_center_default() -> None:
    action = parse_action({
        "action_type": "scroll",
        "direction": "left",
        "pixels": 180,
    })

    assert action.x is None
    assert action.y is None
    assert action.text == "left"


def test_parse_scroll_rejects_partial_coordinates() -> None:
    with pytest.raises(ActionError, match="requires both 'x' and 'y'"):
        parse_action({
            "action_type": "scroll",
            "x": 100,
            "pixels": 180,
        })


def test_parse_action_unwraps_singleton_coordinate_lists() -> None:
    action = parse_action({
        "action": "tap",
        "x": [417],
        "y": [129],
        "relative": True,
    })

    assert action.action_type == "tap"
    assert action.x == 417.0
    assert action.y == 129.0
    assert action.relative is True


def test_parse_action_splits_paired_coordinates_from_x_list() -> None:
    action = parse_action({
        "action": "tap",
        "x": [498, 441],
        "relative": True,
    })

    assert action.action_type == "tap"
    assert action.x == 498.0
    assert action.y == 441.0
    assert action.relative is True


def test_parse_action_splits_paired_coordinates_from_stringified_x_list() -> None:
    action = parse_action({
        "action": "tap",
        "x": "[903, 130]",
        "relative": "true",
    })

    assert action.action_type == "tap"
    assert action.x == 903.0
    assert action.y == 130.0
    assert action.relative is True


def test_parse_action_maps_coordinate_alias_pair() -> None:
    action = parse_action({
        "action": "tap",
        "coordinate": [500, 261],
        "relative": True,
    })

    assert action.action_type == "tap"
    assert action.x == 500.0
    assert action.y == 261.0
    assert action.relative is True


def test_parse_action_maps_stringified_coordinate_alias_pair() -> None:
    action = parse_action({
        "action": "tap",
        "coordinate": "[780, 503]",
    })

    assert action.action_type == "tap"
    assert action.x == 780.0
    assert action.y == 503.0


def test_relative_observation_prompt_normalizes_ui_tree_bounds_without_screen_resolution() -> None:
    observation = Observation(
        screenshot_path=None,
        screen_width=488,
        screen_height=1080,
        foreground_app="com.example",
        platform="android",
        extra={
            "capture_source": "scrcpy",
            "ui_tree": [
                {"class": "android.widget.FrameLayout", "bounds": "[0,0][1080,2376]"},
                {
                    "text": "我的订单",
                    "class": "android.widget.TextView",
                    "bounds": "[258,723][390,768]",
                },
            ],
        },
    )

    text = observation.to_user_text(
        "open orders",
        step_index=4,
        coordinate_instruction="Use relative coordinates in [0, 999] for both x and y, and set relative=true.",
    )

    assert "Screen:" not in text
    assert "488 x 1080" not in text
    assert "Platform: android" in text
    assert "'bounds':" not in text
    assert '"bounds":' not in text
    assert "relative_bounds" in text
    assert "[0,0][999,999]" in text
    assert "[239,304][361,323]" in text
    assert "[258,723][390,768]" not in text


def test_parse_swipe_maps_start_and_end_coordinate_aliases() -> None:
    action = parse_action({
        "action_type": "swipe",
        "start_coordinate": [120, 340],
        "end_coordinate": [760, 355],
        "relative": True,
    })

    assert action.action_type == "swipe"
    assert action.x == 120.0
    assert action.y == 340.0
    assert action.x2 == 760.0
    assert action.y2 == 355.0
    assert action.relative is True


def test_parse_swipe_maps_coordinate_and_coordinate2_aliases() -> None:
    action = parse_action({
        "action": "swipe",
        "coordinate": [120, 340],
        "coordinate2": [760, 355],
    })

    assert action.action_type == "swipe"
    assert action.x == 120.0
    assert action.y == 340.0
    assert action.x2 == 760.0
    assert action.y2 == 355.0


def test_parse_action_unwraps_duplicated_y_list() -> None:
    action = parse_action({
        "action_type": "tap",
        "x": 321,
        "y": [957, 957],
        "relative": True,
    })

    assert action.action_type == "tap"
    assert action.x == 321.0
    assert action.y == 957.0
    assert action.relative is True


def test_parse_action_unwraps_duplicated_stringified_y_list() -> None:
    action = parse_action({
        "action_type": "tap",
        "x": 321,
        "y": "[957, 957]",
        "relative": True,
    })

    assert action.action_type == "tap"
    assert action.x == 321.0
    assert action.y == 957.0
    assert action.relative is True


def test_scale_image_accepts_custom_ratio() -> None:
    from PIL import Image

    from opengui.skills.executor import _scale_image

    buf = io.BytesIO()
    Image.new("RGB", (120, 80), color=(255, 0, 0)).save(buf, format="PNG")

    scaled = _scale_image(buf.getvalue(), scale_ratio=0.25)
    with Image.open(io.BytesIO(scaled)) as img:
        assert img.size == (30, 20)


def test_scale_image_ratio_one_keeps_original_bytes() -> None:
    from opengui.skills.executor import _scale_image

    raw = b"not-an-image"
    assert _scale_image(raw, scale_ratio=1.0) == raw


def test_parse_swipe_splits_all_coordinates_from_x_list() -> None:
    action = parse_action({
        "action_type": "swipe",
        "x": [120, 340, 760, 355],
        "relative": True,
    })

    assert action.action_type == "swipe"
    assert action.x == 120.0
    assert action.y == 340.0
    assert action.x2 == 760.0
    assert action.y2 == 355.0
    assert action.relative is True


def test_parse_swipe_splits_all_coordinates_from_stringified_x_list() -> None:
    action = parse_action({
        "action_type": "swipe",
        "x": "[120, 340, 760, 355]",
        "relative": True,
    })

    assert action.action_type == "swipe"
    assert action.x == 120.0
    assert action.y == 340.0
    assert action.x2 == 760.0
    assert action.y2 == 355.0
    assert action.relative is True


def test_parse_swipe_rejects_noop_path() -> None:
    with pytest.raises(ActionError, match="start and end coordinates must differ"):
        parse_action({
            "action_type": "swipe",
            "x": 500,
            "y": 700,
            "x2": 500,
            "y2": 700,
            "relative": True,
        })


def test_parse_swipe_accepts_endpoint_only_coordinates() -> None:
    action = parse_action({
        "action_type": "swipe",
        "x2": 500,
        "y2": 749,
        "relative": True,
    })

    assert action.action_type == "swipe"
    assert action.x == 500.0
    assert action.y2 == 749.0
    assert action.y is not None
    assert action.y != action.y2


def test_parse_action_accepts_mobileworld_navigation_aliases() -> None:
    enter = parse_action({"action_type": "keyboard_enter"})
    recents = parse_action({"action_type": "recents"})

    assert enter.action_type == "enter"
    assert recents.action_type == "app_switch"


def test_build_system_prompt_uses_mobile_agent_style_sections() -> None:
    prompt = build_system_prompt(
        platform="android",
        tool_definition={"type": "function", "function": {"name": "computer_use"}},
    )

    assert "# Tools" in prompt
    assert "<tools>" in prompt
    assert '"name": "computer_use"' in prompt
    assert "# Response format" in prompt
    assert "native tool-calling mechanism" in prompt


def test_default_profile_tool_definition_requires_intent_and_summary() -> None:
    params = profile_tool_definition("default")["function"]["parameters"]

    assert params["required"] == ["action_type", "intent", "summary"]
    assert "purpose of the selected next action" in params["properties"]["intent"]["description"]
    assert "current task progress" in params["properties"]["summary"]["description"]


def test_build_system_prompt_supports_general_e2e_profile() -> None:
    prompt = build_system_prompt(
        platform="android",
        agent_profile="general_e2e",
    )

    assert "Thought:" in prompt
    assert 'Action: {"action_type": "click"' in prompt
    assert "Do not use native tool calling" in prompt


def test_build_system_prompt_prefers_open_app_when_foreground_app_mismatches_target() -> None:
    prompt = build_system_prompt(platform="android", agent_profile="qwen3vl")

    assert "foreground app does not match the target app" in prompt
    assert "`open` action" in prompt
    assert '"action":"open","text":"Taobao"' in prompt
    assert "Do not repeatedly tap inside the wrong foreground app" in prompt


def test_annotate_android_apps_filters_unmapped_packages() -> None:
    from opengui.skills.normalization import annotate_android_apps

    result = annotate_android_apps(["com.sankuai.meituan", "com.unknown.xyz"])

    assert len(result) == 1, f"Expected 1 entry, got {len(result)}: {result}"
    assert "美团" in result[0] or "Meituan" in result[0]
    assert not any("com.unknown.xyz" in entry for entry in result)


def test_build_system_prompt_android_apps_shows_display_names_only() -> None:
    prompt = build_system_prompt(
        platform="android",
        installed_apps=["com.sankuai.meituan", "com.unknown.dropped"],
    )

    assert "美团/Meituan" in prompt
    # Package name must not appear as an app list item
    lines = prompt.splitlines()
    app_list_lines = [ln.strip() for ln in lines if ln.strip().startswith("- ")]
    assert not any("com.sankuai.meituan" in ln for ln in app_list_lines)
    assert not any("com.unknown.dropped" in ln for ln in app_list_lines)


def test_android_app_aliases_include_huawei_weather() -> None:
    from opengui.skills.normalization import annotate_android_apps, resolve_android_package

    annotated = annotate_android_apps(["com.huawei.android.totemweather"])

    assert annotated == ["华为天气/Huawei Weather: com.huawei.android.totemweather"]
    assert resolve_android_package("Huawei Weather") == "com.huawei.android.totemweather"
    assert resolve_android_package("华为天气") == "com.huawei.android.totemweather"


def test_android_app_aliases_include_huawei_browser() -> None:
    from opengui.skills.normalization import annotate_android_apps, resolve_android_package

    annotated = annotate_android_apps(["com.huawei.browser"])

    assert annotated == ["华为浏览器/Huawei Browser: com.huawei.browser"]
    assert resolve_android_package("Huawei Browser") == "com.huawei.browser"
    assert resolve_android_package("华为浏览器") == "com.huawei.browser"


def test_build_system_prompt_android_apps_excludes_unmapped() -> None:
    prompt = build_system_prompt(
        platform="android",
        installed_apps=["com.totally.unknown"],
    )

    # With all packages unmapped, no "# Installed Apps" section should appear
    assert "# Installed Apps" not in prompt


@pytest.mark.asyncio
async def test_adb_backend_ensure_yadb_pushes_packaged_asset_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = AdbBackend()
    yadb_asset = tmp_path / "yadb"
    yadb_asset.write_bytes(b"jar-data")

    monkeypatch.setattr(backend, "_get_packaged_yadb_path", lambda: yadb_asset)
    run_mock = AsyncMock(side_effect=[
        AdbError("missing"),
        "",
        "",
    ])
    monkeypatch.setattr(backend, "_run", run_mock)

    assert await backend._ensure_yadb_available(timeout=5.0) is True
    assert run_mock.await_args_list[0].args == ("shell", "ls", "/data/local/tmp/yadb")
    assert run_mock.await_args_list[1].args == ("push", str(yadb_asset), "/data/local/tmp/yadb")
    assert run_mock.await_args_list[2].args == ("shell", "chmod", "755", "/data/local/tmp/yadb")


@pytest.mark.asyncio
async def test_adb_backend_ensure_yadb_skips_push_when_device_already_has_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = AdbBackend()
    yadb_asset = tmp_path / "yadb"
    yadb_asset.write_bytes(b"jar-data")

    monkeypatch.setattr(backend, "_get_packaged_yadb_path", lambda: yadb_asset)
    run_mock = AsyncMock(return_value="")
    monkeypatch.setattr(backend, "_run", run_mock)

    assert await backend._ensure_yadb_available(timeout=5.0) is True
    run_mock.assert_awaited_once_with("shell", "ls", "/data/local/tmp/yadb", timeout=5.0)


@pytest.mark.asyncio
async def test_adb_backend_resolves_relative_tap(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = AdbBackend()
    backend._screen_width = 200
    backend._screen_height = 400
    run_mock = AsyncMock(return_value="")
    monkeypatch.setattr(backend, "_run", run_mock)

    action = parse_action({
        "action_type": "tap",
        "x": 500,
        "y": 250,
        "relative": True,
    })
    await backend.execute(action)

    expected_x = resolve_coordinate(500, 200, relative=True)
    expected_y = resolve_coordinate(250, 400, relative=True)
    run_mock.assert_awaited_once_with(
        "shell", "input", "tap", str(expected_x), str(expected_y), timeout=5.0,
    )


@pytest.mark.asyncio
async def test_adb_backend_observe_prefers_screenshot_size(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = AdbBackend(use_scrcpy=False)
    run_mock = AsyncMock(return_value="")
    screen_size_mock = AsyncMock(return_value=(1080, 2376))
    foreground_mock = AsyncMock(return_value="com.example.app")

    monkeypatch.setattr(backend, "_run", run_mock)
    monkeypatch.setattr(backend, "_query_screen_size", screen_size_mock)
    monkeypatch.setattr(backend, "_query_foreground_app", foreground_mock)
    monkeypatch.setattr(adb_backend_module, "_read_png_size", lambda _path: (2376, 1080))

    obs = await backend.observe(Path("/tmp/adb-orientation.png"))

    assert obs.screen_width == 2376
    assert obs.screen_height == 1080
    assert backend._screen_width == 2376
    assert backend._screen_height == 1080
    screen_size_mock.assert_not_awaited()
    foreground_mock.assert_awaited_once_with(5.0)


@pytest.mark.asyncio
async def test_adb_backend_observe_falls_back_when_screenshot_size_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = AdbBackend(use_scrcpy=False)
    run_mock = AsyncMock(return_value="")
    screen_size_mock = AsyncMock(return_value=(1080, 2376))
    foreground_mock = AsyncMock(return_value="com.example.app")

    monkeypatch.setattr(backend, "_run", run_mock)
    monkeypatch.setattr(backend, "_query_screen_size", screen_size_mock)
    monkeypatch.setattr(backend, "_query_foreground_app", foreground_mock)
    monkeypatch.setattr(adb_backend_module, "_read_png_size", lambda _path: None)

    obs = await backend.observe(Path("/tmp/adb-orientation-fallback.png"))

    assert obs.screen_width == 1080
    assert obs.screen_height == 2376
    screen_size_mock.assert_awaited_once_with(5.0)
    foreground_mock.assert_awaited_once_with(5.0)


@pytest.mark.asyncio
async def test_adb_backend_foreground_app_uses_activity_dump_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = AdbBackend()
    run_mock = AsyncMock(return_value="""
      mResumedActivity: ActivityRecord{123 u0 com.coloros.calendar/.MainActivity t12}
    """)
    monkeypatch.setattr(backend, "_run", run_mock)

    assert await backend._query_foreground_app(timeout=5.0) == "com.coloros.calendar"
    run_mock.assert_awaited_once_with(
        "shell",
        "dumpsys",
        "activity",
        "activities",
        timeout=10.0,
    )


@pytest.mark.asyncio
async def test_adb_backend_foreground_app_falls_back_to_window_dump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = AdbBackend()
    run_mock = AsyncMock(side_effect=[
        "header without resumed activity",
        "mCurrentFocus=Window{42 u0 com.android.settings/com.android.settings.Settings}",
    ])
    monkeypatch.setattr(backend, "_run", run_mock)

    assert await backend._query_foreground_app(timeout=5.0) == "com.android.settings"
    assert run_mock.await_args_list[0].args == ("shell", "dumpsys", "activity", "activities")
    assert run_mock.await_args_list[1].args == ("shell", "dumpsys", "window", "windows")


def test_adb_backend_extract_foreground_app_supports_multiple_android_signals() -> None:
    assert AdbBackend._extract_foreground_app(
        "topResumedActivity=ActivityRecord{7 u0 com.heytap.browser/.Main t11}"
    ) == "com.heytap.browser"
    assert AdbBackend._extract_foreground_app(
        "mFocusedApp=AppWindowToken{ token=Token{ ActivityRecord{7 u0 com.android.settings/.Settings t11}}}"
    ) == "com.android.settings"
    assert AdbBackend._extract_foreground_app("no foreground info here") == "unknown"


@pytest.mark.asyncio
async def test_hdc_backend_observe_prefers_screenshot_size(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = HdcBackend()
    run_mock = AsyncMock(return_value="")
    screen_size_mock = AsyncMock(return_value=(1080, 2376))
    foreground_mock = AsyncMock(return_value="com.example.harmony")

    class _FakeImageContext:
        def __enter__(self) -> "_FakeImageContext":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            del exc_type, exc, tb
            return False

        def save(self, path: str, format: str = "PNG") -> None:
            del format
            Path(path).write_bytes(b"png")

    class _FakeImage:
        @staticmethod
        def open(_path: str) -> _FakeImageContext:
            return _FakeImageContext()

    monkeypatch.setattr(backend, "_run", run_mock)
    monkeypatch.setattr(backend, "_query_screen_size", screen_size_mock)
    monkeypatch.setattr(backend, "_query_foreground_app", foreground_mock)
    monkeypatch.setattr(hdc_backend_module, "_import_pil_image", lambda: _FakeImage)
    monkeypatch.setattr(hdc_backend_module, "_read_png_size", lambda _path: (2376, 1080))

    obs = await backend.observe(Path("/tmp/hdc-orientation.png"))

    assert obs.screen_width == 2376
    assert obs.screen_height == 1080
    assert backend._screen_width == 2376
    assert backend._screen_height == 1080
    screen_size_mock.assert_not_awaited()
    foreground_mock.assert_awaited_once_with(5.0)


@pytest.mark.asyncio
async def test_hdc_backend_observe_falls_back_when_screenshot_size_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = HdcBackend()
    run_mock = AsyncMock(return_value="")
    screen_size_mock = AsyncMock(return_value=(1080, 2376))
    foreground_mock = AsyncMock(return_value="com.example.harmony")

    class _FakeImageContext:
        def __enter__(self) -> "_FakeImageContext":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            del exc_type, exc, tb
            return False

        def save(self, path: str, format: str = "PNG") -> None:
            del format
            Path(path).write_bytes(b"png")

    class _FakeImage:
        @staticmethod
        def open(_path: str) -> _FakeImageContext:
            return _FakeImageContext()

    monkeypatch.setattr(backend, "_run", run_mock)
    monkeypatch.setattr(backend, "_query_screen_size", screen_size_mock)
    monkeypatch.setattr(backend, "_query_foreground_app", foreground_mock)
    monkeypatch.setattr(hdc_backend_module, "_import_pil_image", lambda: _FakeImage)
    monkeypatch.setattr(hdc_backend_module, "_read_png_size", lambda _path: None)

    obs = await backend.observe(Path("/tmp/hdc-orientation-fallback.png"))

    assert obs.screen_width == 1080
    assert obs.screen_height == 2376
    screen_size_mock.assert_awaited_once_with(5.0)
    foreground_mock.assert_awaited_once_with(5.0)


@pytest.mark.asyncio
async def test_hdc_backend_query_screen_size_preserves_landscape_orientation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = HdcBackend()
    run_mock = AsyncMock(return_value="Display Resolution: 2376x1080")
    monkeypatch.setattr(backend, "_run", run_mock)

    width, height = await backend._query_screen_size(timeout=5.0)

    assert width == 2376
    assert height == 1080


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("direction", "expected_y2"),
    [("down", 280), ("up", 520)],
)
async def test_hdc_backend_scroll_vertical_direction_is_inverted(
    monkeypatch: pytest.MonkeyPatch,
    direction: str,
    expected_y2: int,
) -> None:
    backend = HdcBackend()
    backend._screen_width = 400
    backend._screen_height = 800
    run_mock = AsyncMock(return_value="")
    monkeypatch.setattr(backend, "_run", run_mock)

    action = parse_action({
        "action_type": "scroll",
        "direction": direction,
        "pixels": 120,
    })
    await backend.execute(action)

    run_mock.assert_awaited_once_with(
        "shell", "uitest", "uiInput", "swipe", "200", "400", "200", str(expected_y2), "2000", timeout=5.0,
    )


@pytest.mark.asyncio
async def test_ios_backend_observe_swaps_window_size_when_orientation_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    screenshot_path = tmp_path / "ios-mismatch.png"

    class _FakeImage:
        size = (2376, 1080)

        def save(self, path: str, format: str = "PNG") -> None:
            del format
            Path(path).write_bytes(b"png")

    class _FakeClient:
        def __init__(self, _url: str) -> None:
            pass

        def screenshot(self) -> _FakeImage:
            return _FakeImage()

        def window_size(self) -> dict[str, int]:
            return {"width": 1080, "height": 2376}

        def app_current(self) -> dict[str, str]:
            return {"bundleId": "com.example.ios"}

    class _FakeWdaModule:
        Client = _FakeClient

    monkeypatch.setattr(ios_wda_module, "_import_wda", lambda: _FakeWdaModule)
    backend = ios_wda_module.WdaBackend()

    obs = await backend.observe(screenshot_path)

    assert obs.screen_width == 2376
    assert obs.screen_height == 1080
    assert backend._screen_width == 2376
    assert backend._screen_height == 1080


@pytest.mark.asyncio
async def test_ios_backend_observe_keeps_window_size_when_orientation_matches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    screenshot_path = tmp_path / "ios-match.png"

    class _FakeImage:
        size = (1080, 2376)

        def save(self, path: str, format: str = "PNG") -> None:
            del format
            Path(path).write_bytes(b"png")

    class _FakeClient:
        def __init__(self, _url: str) -> None:
            pass

        def screenshot(self) -> _FakeImage:
            return _FakeImage()

        def window_size(self) -> dict[str, int]:
            return {"width": 1080, "height": 2376}

        def app_current(self) -> dict[str, str]:
            return {"bundleId": "com.example.ios"}

    class _FakeWdaModule:
        Client = _FakeClient

    monkeypatch.setattr(ios_wda_module, "_import_wda", lambda: _FakeWdaModule)
    backend = ios_wda_module.WdaBackend()

    obs = await backend.observe(screenshot_path)

    assert obs.screen_width == 1080
    assert obs.screen_height == 2376
    assert backend._screen_width == 1080
    assert backend._screen_height == 2376


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("direction", "expected_y2"),
    [("down", 280), ("up", 520)],
)
async def test_ios_backend_scroll_vertical_direction_is_inverted(
    monkeypatch: pytest.MonkeyPatch,
    direction: str,
    expected_y2: int,
) -> None:
    class _FakeSession:
        def __init__(self) -> None:
            self.swipe_calls: list[tuple[int, int, int, int, float]] = []

        def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_s: float) -> None:
            self.swipe_calls.append((x1, y1, x2, y2, duration_s))

    class _FakeClient:
        def __init__(self, _url: str) -> None:
            self._session = _FakeSession()

        def session(self) -> _FakeSession:
            return self._session

        def window_size(self) -> dict[str, int]:
            return {"width": 1080, "height": 2376}

        def app_current(self) -> dict[str, str]:
            return {"bundleId": "com.example.ios"}

    class _FakeWdaModule:
        Client = _FakeClient

    monkeypatch.setattr(ios_wda_module, "_import_wda", lambda: _FakeWdaModule)
    backend = ios_wda_module.WdaBackend()
    monkeypatch.setattr(backend, "_wda_call", AsyncMock(side_effect=lambda fn, *args: fn(*args)))
    backend._screen_width = 400
    backend._screen_height = 800

    action = parse_action({
        "action_type": "scroll",
        "direction": direction,
        "pixels": 120,
    })
    await backend.execute(action)

    assert backend._client._session.swipe_calls == [(200, 400, 200, expected_y2, 0.3)]


def test_agent_marks_qwen_and_gemini_coordinates_as_relative(tmp_path: Path) -> None:
    qwen_agent = GuiAgent(
        _ScriptedLLM([]),
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "qwen"),
        model="qwen-vl-max",
    )
    gemini_agent = GuiAgent(
        _ScriptedLLM([]),
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "gemini"),
        model="gemini-2.5-pro",
    )

    action = parse_action({"action_type": "tap", "x": 500, "y": 250})

    assert qwen_agent._normalize_relative_coordinates(action).relative is True
    assert gemini_agent._normalize_relative_coordinates(action).relative is True
    assert qwen_agent._coordinate_mode() == "relative_999"


def test_agent_keeps_absolute_coordinates_for_other_models(tmp_path: Path) -> None:
    agent = GuiAgent(
        _ScriptedLLM([]),
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "other"),
        model="gpt-5.2",
    )

    action = parse_action({"action_type": "tap", "x": 500, "y": 250})

    assert agent._normalize_relative_coordinates(action).relative is False
    assert agent._coordinate_mode() == "absolute"


def test_qwen3vl_profile_normalizes_content_only_response() -> None:
    response = LLMResponse(
        content=(
            'Thought: I found the target\n'
            'Action: "Tap the login button"\n'
            '<tool_call>{"name":"mobile_use","arguments":{"action":"click","coordinate":[500,250]}}</tool_call>'
        ),
        tool_calls=None,
    )

    normalized = normalize_profile_response("qwen3vl", response)

    assert normalized.tool_calls is not None
    assert normalized.tool_calls[0].name == "computer_use"
    assert normalized.tool_calls[0].arguments == {
        "action_type": "tap",
        "x": 500,
        "y": 250,
        "relative": True,
    }


def test_qwen3vl_profile_preserves_action_summary() -> None:
    response = LLMResponse(
        content=(
            'Thought: I found the target\n'
            'Action: "Tap the login button"\n'
            '<tool_call>{"name":"mobile_use","arguments":{"action":"click","coordinate":[500,250],"summary":"tap login button","intent":"tap login button"}}'
            '</tool_call>'
        ),
        tool_calls=None,
    )

    normalized = normalize_profile_response("qwen3vl", response)

    assert normalized.tool_calls is not None
    assert normalized.tool_calls[0].arguments == {
        "action_type": "tap",
        "x": 500,
        "y": 250,
        "relative": True,
        "summary": "tap login button",
    }


def test_qwen3vl_profile_prefers_content_contract_over_provider_tool_calls() -> None:
    response = LLMResponse(
        content=(
            'Thought: Open Chrome\n'
            'Action: "Open Chrome"\n'
            '<tool_call>{"name":"mobile_use","arguments":{"action":"open","text":"chrome"}}</tool_call>'
        ),
        tool_calls=[
            ToolCall(
                id="provider-tool-call-0",
                name="computer_use",
                arguments={"action_type": "open_app", "text": "chrome"},
            )
        ],
    )

    normalized = normalize_profile_response("qwen3vl", response)

    assert normalized.tool_calls is not None
    assert normalized.tool_calls[0].id == "content-tool-call-0"
    assert normalized.tool_calls[0].name == "computer_use"
    assert normalized.tool_calls[0].arguments == {
        "action_type": "open_app",
        "text": "chrome",
    }


def test_qwen3vl_profile_falls_back_to_provider_tool_calls_when_content_contract_is_missing() -> None:
    response = LLMResponse(
        content='Thought: Continue\nAction: Tap the next result',
        tool_calls=[
            ToolCall(
                id="provider-tool-call-0",
                name="computer_use",
                arguments={"action_type": "tap", "x": 321, "y": 654, "relative": True},
            )
        ],
    )

    normalized = normalize_profile_response("qwen3vl", response)

    assert normalized.tool_calls is not None
    assert normalized.tool_calls[0].id == "provider-tool-call-0"
    assert normalized.tool_calls[0].name == "computer_use"
    assert normalized.tool_calls[0].arguments == {
        "action_type": "tap",
        "x": 321,
        "y": 654,
        "relative": True,
    }


def test_qwen3vl_profile_normalizes_provider_mobile_use_tool_calls() -> None:
    response = LLMResponse(
        content="Action: Tap the next result",
        tool_calls=[
            ToolCall(
                id="provider-tool-call-0",
                name="mobile_use",
                arguments={"action": "click", "coordinate": [903, 130]},
            )
        ],
    )

    normalized = normalize_profile_response("qwen3vl", response)

    assert normalized.tool_calls is not None
    assert normalized.tool_calls[0].id == "provider-tool-call-0"
    assert normalized.tool_calls[0].name == "computer_use"
    assert normalized.tool_calls[0].arguments == {
        "action_type": "tap",
        "x": 903,
        "y": 130,
        "relative": True,
    }


@pytest.mark.asyncio
async def test_agent_action_grounder_uses_profile_seam_for_qwen3vl(tmp_path: Path) -> None:
    screenshot = tmp_path / "grounder.png"
    screenshot.write_bytes(b"png")
    llm = _RecordingLLM([
        LLMResponse(
            content=(
                'Thought: Tap the button\n'
                'Action: "Tap login"\n'
                '<tool_call>{"name":"mobile_use","arguments":{"action":"click","coordinate":[500,250]}}</tool_call>'
            ),
            tool_calls=None,
        )
    ])
    grounder = _AgentActionGrounder(llm, model="qwen-vl-max", agent_profile="qwen3vl")
    step = SkillStep(action_type="tap", target="Login button", parameters={"x_hint": "unused"})

    action = await grounder.ground(step, screenshot, {})

    assert action.action_type == "tap"
    assert action.x == 500
    assert action.y == 250
    assert action.relative is True
    assert len(llm.calls) == 1
    assert llm.calls[0][0]["role"] == "user"


@pytest.mark.asyncio
async def test_agent_subgoal_runner_uses_profile_seam_for_qwen3vl(tmp_path: Path) -> None:
    screenshot = tmp_path / "subgoal.png"
    screenshot.write_bytes(b"png")
    llm = _RecordingLLM([
        LLMResponse(
            content=(
                'Thought: Move toward the target\n'
                'Action: "Tap settings"\n'
                '<tool_call>{"name":"mobile_use","arguments":{"action":"click","coordinate":[400,300]}}</tool_call>'
            ),
            tool_calls=None,
        )
    ])
    backend = _SkillTestBackend()
    validator = _RecordingValidator([True])
    runner = _AgentSubgoalRunner(
        llm=llm,
        backend=backend,
        state_validator=validator,
        model="qwen-vl-max",
        artifacts_root=tmp_path / "artifacts",
        agent_profile="qwen3vl",
    )

    result = await runner.run_subgoal("Settings screen visible", screenshot, max_steps=1)

    assert result.success is True
    assert backend.executed_actions
    action = backend.executed_actions[0]
    assert action.action_type == "tap"
    assert action.relative is True
    assert validator.calls[0]["valid_state"] == "Settings screen visible"


@pytest.mark.asyncio
async def test_agent_subgoal_runner_records_events(tmp_path: Path) -> None:
    screenshot = tmp_path / "subgoal-record.png"
    screenshot.write_bytes(b"png")
    llm = _RecordingLLM([
        LLMResponse(
            content=(
                'Thought: Move toward the target\n'
                'Action: "Tap settings"\n'
                '<tool_call>{"name":"computer_use","arguments":{"action_type":"tap","x":400,"y":300,"relative":true}}</tool_call>'
            ),
            tool_calls=[
                ToolCall(
                    id="subgoal-tool-0",
                    name="computer_use",
                    arguments={"action_type": "tap", "x": 400, "y": 300, "relative": True},
                )
            ],
        )
    ])
    backend = _SkillTestBackend()
    validator = _RecordingValidator([True])
    recorder = _make_recorder(tmp_path, "subgoal trace")
    recorder.start()
    runner = _AgentSubgoalRunner(
        llm=llm,
        backend=backend,
        state_validator=validator,
        model="test-model",
        artifacts_root=tmp_path / "artifacts",
        trajectory_recorder=recorder,
    )

    result = await runner.run_subgoal("Settings screen visible", screenshot, max_steps=1)
    trace_path = recorder.finish(success=True)
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]

    assert result.success is True
    names = [event["type"] for event in events]
    assert "subgoal_start" in names
    assert "subgoal_step" in names
    assert "subgoal_result" in names
    subgoal_step = next(event for event in events if event["type"] == "subgoal_step")
    assert subgoal_step["model_output"]
    assert subgoal_step["goal_reached"] is True
    assert subgoal_step["action"]["action_type"] == "tap"


@pytest.mark.asyncio
async def test_agent_subgoal_runner_records_parse_failure(tmp_path: Path) -> None:
    screenshot = tmp_path / "subgoal-failure.png"
    screenshot.write_bytes(b"png")
    llm = _RecordingLLM([LLMResponse(content="No valid tool call", tool_calls=None)])
    recorder = _make_recorder(tmp_path, "subgoal trace failure")
    recorder.start()
    runner = _AgentSubgoalRunner(
        llm=llm,
        backend=_SkillTestBackend(),
        state_validator=_RecordingValidator([False]),
        model="test-model",
        artifacts_root=tmp_path / "artifacts-failure",
        trajectory_recorder=recorder,
    )

    result = await runner.run_subgoal("Settings screen visible", screenshot, max_steps=1)
    trace_path = recorder.finish(success=True)
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]

    assert result.success is False
    subgoal_step = next(event for event in events if event["type"] == "subgoal_step")
    assert subgoal_step["error"] == "no valid action returned"
    subgoal_result = next(event for event in events if event["type"] == "subgoal_result")
    assert subgoal_result["success"] is False


@pytest.mark.asyncio
async def test_subgoal_runner_normalizes_relative_coordinates_for_gemini(tmp_path: Path) -> None:
    """default profile + gemini model → relative_999: tap(900, 941) must get relative=True."""
    screenshot = tmp_path / "subgoal.png"
    screenshot.write_bytes(b"png")
    llm = _RecordingLLM([
        LLMResponse(
            content="tap the target",
            tool_calls=[ToolCall(
                id="tc-0",
                name="computer_use",
                arguments={"action_type": "tap", "x": 900, "y": 941},
            )],
        )
    ])
    backend = _SkillTestBackend()
    validator = _RecordingValidator([True])
    runner = _AgentSubgoalRunner(
        llm=llm,
        backend=backend,
        state_validator=validator,
        model="gemini-2.0-flash",
        artifacts_root=tmp_path / "artifacts",
    )

    result = await runner.run_subgoal("Tap the target button", screenshot, max_steps=1)

    assert result.success is True
    assert backend.executed_actions
    action = backend.executed_actions[0]
    assert action.action_type == "tap"
    assert action.relative is True


@pytest.mark.asyncio
async def test_subgoal_runner_uses_configured_step_timeout(tmp_path: Path) -> None:
    """execute() and observe() must receive the step_timeout passed at construction."""
    screenshot = tmp_path / "subgoal.png"
    screenshot.write_bytes(b"png")

    execute_timeouts: list[float] = []
    observe_timeouts: list[float] = []

    class _TimeoutCapturingBackend(_SkillTestBackend):
        async def execute(self, action, timeout: float = 5.0) -> str:
            execute_timeouts.append(timeout)
            return await super().execute(action, timeout=timeout)

        async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
            observe_timeouts.append(timeout)
            return await super().observe(screenshot_path, timeout=timeout)

    llm = _RecordingLLM([
        LLMResponse(
            content="tap",
            tool_calls=[ToolCall(
                id="tc-0",
                name="computer_use",
                arguments={"action_type": "tap", "x": 10, "y": 20},
            )],
        )
    ])
    runner = _AgentSubgoalRunner(
        llm=llm,
        backend=_TimeoutCapturingBackend(),
        state_validator=_RecordingValidator([True]),
        model="test-model",
        artifacts_root=tmp_path / "artifacts",
        step_timeout=42.0,
    )

    await runner.run_subgoal("Confirm timeout propagation", screenshot, max_steps=1)

    assert execute_timeouts == [42.0]
    assert observe_timeouts == [42.0]


@pytest.mark.asyncio
async def test_subgoal_runner_settle_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tap gets a 0.5s settle before observe; wait action does not."""
    screenshot = tmp_path / "subgoal.png"
    screenshot.write_bytes(b"png")
    sleep_calls: list[float] = []
    original_sleep = asyncio.sleep

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        await original_sleep(0)

    monkeypatch.setattr("opengui.agent.asyncio.sleep", fake_sleep)

    # First LLM call returns tap; second returns wait
    llm = _RecordingLLM([
        LLMResponse(
            content="tap",
            tool_calls=[ToolCall(
                id="tc-0",
                name="computer_use",
                arguments={"action_type": "tap", "x": 10, "y": 20},
            )],
        ),
        LLMResponse(
            content="wait",
            tool_calls=[ToolCall(
                id="tc-1",
                name="computer_use",
                arguments={"action_type": "wait", "duration_ms": 500},
            )],
        ),
    ])
    runner = _AgentSubgoalRunner(
        llm=llm,
        backend=_SkillTestBackend(),
        # validator returns False on tap step, True on wait step
        state_validator=_RecordingValidator([False, True]),
        model="test-model",
        artifacts_root=tmp_path / "artifacts",
    )

    result = await runner.run_subgoal("Settle behavior check", screenshot, max_steps=2)

    assert result.success is True
    # tap must produce exactly one settle sleep of 0.5s
    assert sleep_calls == [0.5]


@pytest.mark.asyncio
async def test_agent_runs_with_qwen3vl_content_only_profile(tmp_path: Path) -> None:
    llm = _RecordingLLM([
        LLMResponse(
            content=(
                'Thought: I should wait briefly\n'
                'Action: "Wait for the screen to settle"\n'
                '<tool_call>{"name":"mobile_use","arguments":{"action":"wait"}}</tool_call>'
            ),
            tool_calls=None,
        ),
        LLMResponse(
            content=(
                'Thought: The task is complete\n'
                'Action: "Finish successfully"\n'
                '<tool_call>{"name":"mobile_use","arguments":{"action":"terminate","status":"success"}}</tool_call>'
            ),
            tool_calls=None,
        ),
    ])
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "qwen profile"),
        artifacts_root=tmp_path / "runs",
        max_steps=2,
        include_date_context=False,
        agent_profile="qwen3vl",
    )

    result = await agent.run("Wait and finish", max_retries=1)

    assert result.success
    assert len(llm.calls) == 2
    assert "Action:" in llm.calls[0][0]["content"]


@pytest.mark.asyncio
async def test_agent_runs_with_qwen3vl_provider_computer_use_stringified_x_coordinates(
    tmp_path: Path,
) -> None:
    llm = _RecordingLLM([
        LLMResponse(
            content="Action: Tap the search bar",
            tool_calls=[
                ToolCall(
                    id="provider-tool-call-0",
                    name="computer_use",
                    arguments={"action_type": "click", "x": "[410, 125]", "relative": True},
                )
            ],
        ),
        LLMResponse(
            content="Action: Finish successfully",
            tool_calls=[
                ToolCall(
                    id="provider-tool-call-1",
                    name="computer_use",
                    arguments={"action_type": "done", "status": "success"},
                )
            ],
        ),
    ])
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "qwen provider stringified coords"),
        artifacts_root=tmp_path / "runs",
        max_steps=2,
        include_date_context=False,
        agent_profile="qwen3vl",
    )

    result = await agent.run("Tap then finish", max_retries=1)

    assert result.success is True
    assert result.summary


@pytest.mark.asyncio
async def test_agent_runs_with_qwen3vl_provider_mobile_use_tool_call(tmp_path: Path) -> None:
    llm = _RecordingLLM([
        LLMResponse(
            content="Action: Finish successfully",
            tool_calls=[
                ToolCall(
                    id="provider-tool-call-0",
                    name="mobile_use",
                    arguments={"action": "terminate", "status": "success"},
                )
            ],
        ),
    ])
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "qwen provider tool"),
        artifacts_root=tmp_path / "runs",
        max_steps=1,
        include_date_context=False,
        agent_profile="qwen3vl",
    )

    result = await agent.run("Finish", max_retries=1)

    assert result.success is True
    assert result.summary


@pytest.mark.asyncio
async def test_agent_done_without_status_defaults_to_success(tmp_path: Path) -> None:
    llm = _RecordingLLM([
        LLMResponse(
            content="Action: done",
            tool_calls=[
                ToolCall(
                    id="provider-tool-call-0",
                    name="computer_use",
                    arguments={
                        "action_type": "done",
                        "text": "Task completed successfully and search results are visible.",
                    },
                )
            ],
        ),
    ])
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "done without status success"),
        artifacts_root=tmp_path / "runs",
        max_steps=1,
        include_date_context=False,
    )

    result = await agent.run("Finish", max_retries=1)

    assert result.success is True
    assert result.error is None


@pytest.mark.asyncio
async def test_agent_done_without_status_with_failure_text_marks_failure(tmp_path: Path) -> None:
    llm = _RecordingLLM([
        LLMResponse(
            content="Action: done",
            tool_calls=[
                ToolCall(
                    id="provider-tool-call-0",
                    name="computer_use",
                    arguments={
                        "action_type": "done",
                        "text": "Task failed because login is required.",
                    },
                )
            ],
        ),
    ])
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "done without status failure"),
        artifacts_root=tmp_path / "runs",
        max_steps=1,
        include_date_context=False,
    )

    result = await agent.run("Finish", max_retries=1)

    assert result.success is False
    assert result.error == "Task terminated with status: failure"


@pytest.mark.asyncio
async def test_agent_trace_records_prompt_and_model_details(tmp_path: Path) -> None:
    llm = _RecordingLLM([
        LLMResponse(
            content="Action: wait briefly",
            tool_calls=[ToolCall(
                id="call-1",
                name="computer_use",
                arguments={"action_type": "wait", "duration_ms": 1},
            )],
        ),
        LLMResponse(
            content="Action: done",
            tool_calls=[ToolCall(
                id="call-2",
                name="computer_use",
                arguments={"action_type": "done", "status": "success"},
            )],
        ),
    ])
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "Open Settings"),
        artifacts_root=tmp_path / "runs",
        max_steps=2,
        include_date_context=False,
    )

    result = await agent.run("Open Settings", max_retries=1)

    assert result.success
    assert result.trace_path is not None

    trace_path = Path(result.trace_path) / "trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    step_event = next(event for event in events if event["event"] == "step")

    assert step_event["prompt"]["task"] == "Open Settings"
    assert step_event["prompt"]["messages"][0]["role"] == "system"
    assert step_event["prompt"]["current_observation"]["foreground_app"] == "DryRun"
    assert step_event["model_output"]["raw_content"] == "Action: wait briefly"
    assert step_event["model_output"]["tool_calls"][0]["arguments"]["duration_ms"] == 1
    assert step_event["model_output"]["parsed_action"]["action_type"] == "wait"
    assert step_event["execution"]["tool_result"] == "[dry-run] wait 1 ms"
    image_blocks = [
        block
        for message in step_event["prompt"]["messages"]
        for block in (message.get("content") if isinstance(message.get("content"), list) else [])
        if isinstance(block, dict) and block.get("type") == "image_url"
    ]
    assert image_blocks
    assert image_blocks[0]["image_url"]["url"] == "<omitted:image-data-url>"


@pytest.mark.asyncio
async def test_adb_backend_scrolls_horizontally_from_center(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = AdbBackend()
    backend._screen_width = 400
    backend._screen_height = 800
    run_mock = AsyncMock(return_value="")
    monkeypatch.setattr(backend, "_run", run_mock)

    action = parse_action({
        "action_type": "scroll",
        "direction": "left",
        "pixels": 120,
    })
    await backend.execute(action)

    run_mock.assert_awaited_once_with(
        "shell", "input", "swipe", "200", "400", "80", "400", "300", timeout=5.0,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("direction", "expected_y2"),
    [("down", "280"), ("up", "520")],
)
async def test_adb_backend_scroll_vertical_direction_is_inverted(
    monkeypatch: pytest.MonkeyPatch,
    direction: str,
    expected_y2: str,
) -> None:
    backend = AdbBackend()
    backend._screen_width = 400
    backend._screen_height = 800
    run_mock = AsyncMock(return_value="")
    monkeypatch.setattr(backend, "_run", run_mock)

    action = parse_action({
        "action_type": "scroll",
        "direction": direction,
        "pixels": 120,
    })
    await backend.execute(action)

    run_mock.assert_awaited_once_with(
        "shell", "input", "swipe", "200", "400", "200", expected_y2, "300", timeout=5.0,
    )


@pytest.mark.asyncio
async def test_adb_backend_input_text_prefers_yadb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = AdbBackend()
    run_mock = AsyncMock(return_value="")
    monkeypatch.setattr(backend, "_run", run_mock)
    ensure_yadb_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(backend, "_ensure_yadb_available", ensure_yadb_mock)
    monkeypatch.setattr(backend, "_write_local_temp_text", lambda text: Path("/tmp/opengui-yadb-input.txt"))
    monkeypatch.setattr(backend, "_make_yadb_device_text_path", lambda: "/data/local/tmp/opengui-yadb-input.txt")
    monkeypatch.setattr(backend, "_write_local_temp_yadb_script", lambda: Path("/tmp/opengui-yadb-input.sh"))
    monkeypatch.setattr(backend, "_make_yadb_device_script_path", lambda: "/data/local/tmp/opengui-yadb-input.sh")

    text = "你好，OpenGUI"
    action = parse_action({"action_type": "input_text", "text": text})
    await backend.execute(action)

    ensure_yadb_mock.assert_awaited_once_with(5.0)
    assert run_mock.await_args_list[0].args == (
        "push", "/tmp/opengui-yadb-input.txt", "/data/local/tmp/opengui-yadb-input.txt",
    )
    assert run_mock.await_args_list[1].args == (
        "push", "/tmp/opengui-yadb-input.sh", "/data/local/tmp/opengui-yadb-input.sh",
    )
    assert run_mock.await_args_list[2].args == (
        "shell", "chmod", "755", "/data/local/tmp/opengui-yadb-input.sh",
    )
    assert run_mock.await_args_list[3].args == (
        "shell", "sh", "/data/local/tmp/opengui-yadb-input.sh", "/data/local/tmp/opengui-yadb-input.txt",
    )
    assert run_mock.await_args_list[4].args == (
        "shell", "rm", "-f", "/data/local/tmp/opengui-yadb-input.txt",
    )
    assert run_mock.await_args_list[5].args == (
        "shell", "rm", "-f", "/data/local/tmp/opengui-yadb-input.sh",
    )


@pytest.mark.asyncio
async def test_adb_backend_input_text_multiline_sends_each_line_and_enter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = AdbBackend()

    async def fake_run(*args: object, timeout: float = 5.0) -> str:
        return ""

    run_mock = AsyncMock(side_effect=fake_run)
    monkeypatch.setattr(backend, "_run", run_mock)
    ensure_yadb_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(backend, "_ensure_yadb_available", ensure_yadb_mock)
    local_paths = iter([
        Path("/tmp/opengui-yadb-input-1.txt"),
        Path("/tmp/opengui-yadb-input-2.txt"),
    ])
    device_paths = iter([
        "/data/local/tmp/opengui-yadb-input-1.txt",
        "/data/local/tmp/opengui-yadb-input-2.txt",
    ])
    script_local_paths = iter([
        Path("/tmp/opengui-yadb-input-1.sh"),
        Path("/tmp/opengui-yadb-input-2.sh"),
    ])
    script_device_paths = iter([
        "/data/local/tmp/opengui-yadb-input-1.sh",
        "/data/local/tmp/opengui-yadb-input-2.sh",
    ])
    monkeypatch.setattr(backend, "_write_local_temp_text", lambda text: next(local_paths))
    monkeypatch.setattr(backend, "_make_yadb_device_text_path", lambda: next(device_paths))
    monkeypatch.setattr(backend, "_write_local_temp_yadb_script", lambda: next(script_local_paths))
    monkeypatch.setattr(backend, "_make_yadb_device_script_path", lambda: next(script_device_paths))

    action = parse_action({"action_type": "input_text", "text": "第一行\n第二行"})
    await backend.execute(action)

    assert run_mock.await_args_list[0].args == (
        "push", "/tmp/opengui-yadb-input-1.txt", "/data/local/tmp/opengui-yadb-input-1.txt",
    )
    assert run_mock.await_args_list[1].args == (
        "push", "/tmp/opengui-yadb-input-1.sh", "/data/local/tmp/opengui-yadb-input-1.sh",
    )
    assert run_mock.await_args_list[2].args == (
        "shell", "chmod", "755", "/data/local/tmp/opengui-yadb-input-1.sh",
    )
    assert run_mock.await_args_list[3].args == (
        "shell", "sh", "/data/local/tmp/opengui-yadb-input-1.sh", "/data/local/tmp/opengui-yadb-input-1.txt",
    )
    assert run_mock.await_args_list[4].args == (
        "shell", "rm", "-f", "/data/local/tmp/opengui-yadb-input-1.txt",
    )
    assert run_mock.await_args_list[5].args == (
        "shell", "rm", "-f", "/data/local/tmp/opengui-yadb-input-1.sh",
    )
    assert run_mock.await_args_list[6].args == (
        "shell", "input", "keyevent", "KEYCODE_ENTER",
    )
    assert run_mock.await_args_list[7].args == (
        "push", "/tmp/opengui-yadb-input-2.txt", "/data/local/tmp/opengui-yadb-input-2.txt",
    )
    assert run_mock.await_args_list[8].args == (
        "push", "/tmp/opengui-yadb-input-2.sh", "/data/local/tmp/opengui-yadb-input-2.sh",
    )
    assert run_mock.await_args_list[9].args == (
        "shell", "chmod", "755", "/data/local/tmp/opengui-yadb-input-2.sh",
    )
    assert run_mock.await_args_list[10].args == (
        "shell", "sh", "/data/local/tmp/opengui-yadb-input-2.sh", "/data/local/tmp/opengui-yadb-input-2.txt",
    )
    assert run_mock.await_args_list[11].args == (
        "shell", "rm", "-f", "/data/local/tmp/opengui-yadb-input-2.txt",
    )
    assert run_mock.await_args_list[12].args == (
        "shell", "rm", "-f", "/data/local/tmp/opengui-yadb-input-2.sh",
    )


@pytest.mark.asyncio
async def test_adb_backend_input_text_falls_back_to_adb_keyboard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = AdbBackend()
    run_mock = AsyncMock(side_effect=[
        "com.example.ime/.ExampleIme",
        "com.android.adbkeyboard/.AdbIME\ncom.example.ime/.ExampleIme",
        "",
        "",
        "",
    ])
    monkeypatch.setattr(backend, "_run", run_mock)
    ensure_yadb_mock = AsyncMock(return_value=False)
    monkeypatch.setattr(backend, "_ensure_yadb_available", ensure_yadb_mock)

    text = "你好，OpenGUI"
    action = parse_action({"action_type": "input_text", "text": text})
    await backend.execute(action)

    expected_b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    ensure_yadb_mock.assert_awaited_once_with(5.0)
    assert run_mock.await_args_list[0].args == (
        "shell", "settings", "get", "secure", "default_input_method",
    )
    assert run_mock.await_args_list[1].args == (
        "shell", "ime", "list", "-s",
    )
    assert run_mock.await_args_list[2].args == (
        "shell", "ime", "set", "com.android.adbkeyboard/.AdbIME",
    )
    assert run_mock.await_args_list[3].args == (
        "shell", "am", "broadcast",
        "-a", "ADB_INPUT_B64", "--es", "msg", expected_b64,
    )
    assert run_mock.await_args_list[4].args == (
        "shell", "input", "keyevent", "KEYCODE_ENTER",
    )


@pytest.mark.asyncio
async def test_adb_backend_input_text_enables_adb_keyboard_before_switching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = AdbBackend()
    run_mock = AsyncMock(side_effect=[
        "com.example.ime/.ExampleIme",
        "com.android.adbkeyboard/.AdbIME\ncom.example.ime/.ExampleIme",
        "",
        "",
        "",
        "",
    ])
    monkeypatch.setattr(backend, "_run", run_mock)
    enable_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(backend, "_needs_ime_enable_before_set", enable_mock)
    monkeypatch.setattr(backend, "_ensure_yadb_available", AsyncMock(return_value=False))

    action = parse_action({"action_type": "input_text", "text": "你好"})
    await backend.execute(action)

    enable_mock.assert_awaited_once_with(timeout=5.0)
    assert run_mock.await_args_list[2].args == (
        "shell", "ime", "enable", "com.android.adbkeyboard/.AdbIME",
    )
    assert run_mock.await_args_list[3].args == (
        "shell", "ime", "set", "com.android.adbkeyboard/.AdbIME",
    )


@pytest.mark.asyncio
async def test_adb_backend_input_text_falls_back_to_yadb_for_unicode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = AdbBackend()
    run_mock = AsyncMock(return_value="")
    monkeypatch.setattr(backend, "_run", run_mock)
    ensure_yadb_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(backend, "_ensure_yadb_available", ensure_yadb_mock)
    monkeypatch.setattr(backend, "_write_local_temp_text", lambda text: Path("/tmp/opengui-yadb-input.txt"))
    monkeypatch.setattr(backend, "_make_yadb_device_text_path", lambda: "/data/local/tmp/opengui-yadb-input.txt")
    monkeypatch.setattr(backend, "_write_local_temp_yadb_script", lambda: Path("/tmp/opengui-yadb-input.sh"))
    monkeypatch.setattr(backend, "_make_yadb_device_script_path", lambda: "/data/local/tmp/opengui-yadb-input.sh")

    text = "你好，OpenGUI"
    action = parse_action({"action_type": "input_text", "text": text})
    await backend.execute(action)

    ensure_yadb_mock.assert_awaited_once_with(5.0)

    assert run_mock.await_args_list[0].args == (
        "push", "/tmp/opengui-yadb-input.txt", "/data/local/tmp/opengui-yadb-input.txt",
    )
    assert run_mock.await_args_list[1].args == (
        "push", "/tmp/opengui-yadb-input.sh", "/data/local/tmp/opengui-yadb-input.sh",
    )
    assert run_mock.await_args_list[2].args == (
        "shell", "chmod", "755", "/data/local/tmp/opengui-yadb-input.sh",
    )
    assert run_mock.await_args_list[3].args == (
        "shell", "sh", "/data/local/tmp/opengui-yadb-input.sh", "/data/local/tmp/opengui-yadb-input.txt",
    )
    assert run_mock.await_args_list[4].args == (
        "shell", "rm", "-f", "/data/local/tmp/opengui-yadb-input.txt",
    )
    assert run_mock.await_args_list[5].args == (
        "shell", "rm", "-f", "/data/local/tmp/opengui-yadb-input.sh",
    )


@pytest.mark.asyncio
async def test_adb_backend_input_text_falls_back_to_shell_input_for_ascii(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = AdbBackend()
    run_mock = AsyncMock(side_effect=[
        "com.example.ime/.ExampleIme",
        "com.other.ime/.OtherIme",
        "",
        "",
    ])
    monkeypatch.setattr(backend, "_run", run_mock)
    ensure_yadb_mock = AsyncMock(return_value=False)
    monkeypatch.setattr(backend, "_ensure_yadb_available", ensure_yadb_mock)

    text = "hello world"
    action = parse_action({"action_type": "input_text", "text": text})
    await backend.execute(action)

    ensure_yadb_mock.assert_awaited_once_with(5.0)
    assert run_mock.await_args_list[2].args == (
        "shell", "input", "text", "hello%sworld",
    )
    assert run_mock.await_args_list[2].kwargs == {"timeout": 5.0}
    assert run_mock.await_args_list[3].args == (
        "shell", "input", "keyevent", "KEYCODE_ENTER",
    )


def test_adb_backend_text_segmentation_splits_emoji_boundaries() -> None:
    from opengui.backends.adb import _iter_text_input_segments

    assert _iter_text_input_segments("杭州✅明天☁️有雨") == [
        "杭州",
        "✅",
        "明天",
        "☁️",
        "有雨",
    ]


@pytest.mark.asyncio
async def test_adb_backend_input_text_sends_text_after_emoji_as_later_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = AdbBackend()
    segments: list[str] = []
    run_mock = AsyncMock(return_value="")
    monkeypatch.setattr(backend, "_run", run_mock)

    async def fake_input_single_text(text: str, timeout: float) -> None:
        assert timeout == 5.0
        segments.append(text)

    monkeypatch.setattr(backend, "_input_single_text", fake_input_single_text)

    action = parse_action({"action_type": "input_text", "text": "已发送给苏✅请查收"})
    await backend.execute(action)

    assert segments == ["已发送给苏", "✅", "请查收"]
    assert run_mock.await_args_list[0].args == (
        "shell", "input", "keyevent", "KEYCODE_ENTER",
    )


@pytest.mark.asyncio
async def test_adb_backend_enter_and_app_switch_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = AdbBackend()
    run_mock = AsyncMock(return_value="")
    monkeypatch.setattr(backend, "_run", run_mock)

    await backend.execute(parse_action({"action_type": "hotkey", "key": ["ENTER"]}))
    await backend.execute(Action(action_type="enter"))
    await backend.execute(Action(action_type="app_switch"))

    assert run_mock.await_args_list[0].args == (
        "shell", "input", "keyevent", "KEYCODE_ENTER",
    )
    assert run_mock.await_args_list[0].kwargs == {"timeout": 5.0}
    assert run_mock.await_args_list[1].args == (
        "shell", "input", "keyevent", "KEYCODE_ENTER",
    )
    assert run_mock.await_args_list[1].kwargs == {"timeout": 5.0}
    assert run_mock.await_args_list[2].args == (
        "shell", "input", "keyevent", "KEYCODE_APP_SWITCH",
    )
    assert run_mock.await_args_list[2].kwargs == {"timeout": 5.0}


@pytest.mark.asyncio
async def test_adb_backend_hotkey_uses_keycombination_for_multiple_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = AdbBackend()
    run_mock = AsyncMock(return_value="")
    monkeypatch.setattr(backend, "_run", run_mock)

    await backend.execute(
        parse_action({"action_type": "hotkey", "key": ["power", "volume_down"]})
    )

    assert run_mock.await_count == 1
    assert run_mock.await_args_list[0].args == (
        "shell", "input", "keycombination", "KEYCODE_POWER", "KEYCODE_VOLUME_DOWN",
    )
    assert run_mock.await_args_list[0].kwargs == {"timeout": 5.0}


@pytest.mark.asyncio
async def test_adb_backend_hotkey_multi_key_unsupported_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = AdbBackend()
    run_mock = AsyncMock(
        side_effect=AdbError(
            "adb failed (exit 1): adb shell input keycombination KEYCODE_POWER KEYCODE_VOLUME_DOWN\n"
            "stderr: Error: Unknown command: keycombination"
        )
    )
    monkeypatch.setattr(backend, "_run", run_mock)

    with pytest.raises(AdbError, match="does not support simultaneous multi-key hotkeys"):
        await backend.execute(
            parse_action({"action_type": "hotkey", "key": ["power", "volume_down"]})
        )

    assert run_mock.await_count == 1


@pytest.mark.asyncio
async def test_agent_failure_keeps_last_trace_path(tmp_path: Path) -> None:
    agent = GuiAgent(
        _ScriptedLLM([
            LLMResponse(
                content="wait",
                tool_calls=[ToolCall(
                    id="call-1",
                    name="computer_use",
                    arguments={"action_type": "wait", "duration_ms": 1},
                )],
            ),
        ]),
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "never finishes"),
        artifacts_root=tmp_path / "runs",
        max_steps=1,
    )

    result = await agent.run("never finishes", max_retries=1)

    assert not result.success
    assert result.error == "max_steps_exceeded"
    assert result.steps_taken == 1
    assert result.trace_path is not None
    assert Path(result.trace_path).exists()
    assert result.summary.startswith("Status: partial")
    assert "Done:" in result.summary
    assert "Remaining:" in result.summary
    assert "Current:" in result.summary
    assert "Resume:" in result.summary


@pytest.mark.asyncio
async def test_agent_stagnation_detection_terminates_before_max_steps(tmp_path: Path) -> None:
    responses = [
        LLMResponse(
            content=f"wait #{idx}",
            tool_calls=[ToolCall(
                id=f"call-{idx}",
                name="computer_use",
                arguments={"action_type": "wait", "duration_ms": 1},
            )],
        )
        for idx in range(1, 7)
    ]
    agent = GuiAgent(
        _ScriptedLLM(responses),
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "stagnation"),
        artifacts_root=tmp_path / "runs",
        max_steps=6,
        stagnation_limit=3,
    )

    result = await agent.run("wait on unchanged screen", max_retries=1)

    assert not result.success
    assert result.error == "stagnation_detected"
    assert result.steps_taken == 3
    assert result.steps_taken < agent.max_steps
    assert result.summary.startswith("Status: blocked")
    assert "Done:" in result.summary
    assert "Remaining:" in result.summary
    assert "Current:" in result.summary
    assert "Resume:" in result.summary


@pytest.mark.asyncio
async def test_agent_success_uses_compact_state_note(tmp_path: Path) -> None:
    agent = GuiAgent(
        _ScriptedLLM([
            LLMResponse(
                content="wait",
                tool_calls=[ToolCall(
                    id="call-1",
                    name="computer_use",
                    arguments={"action_type": "wait", "duration_ms": 1},
                )],
            ),
            LLMResponse(
                content="finish task",
                tool_calls=[ToolCall(
                    id="call-2",
                    name="computer_use",
                    arguments={"action_type": "done", "status": "success"},
                )],
            ),
        ]),
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "completed note"),
        artifacts_root=tmp_path / "runs",
        max_steps=2,
        include_date_context=False,
    )

    result = await agent.run("complete the task", max_retries=1)

    assert result.success
    assert result.summary.startswith("Status: completed")
    assert "Done:" in result.summary
    assert "Remaining: none" in result.summary
    assert "Current:" in result.summary
    assert "Resume: No further action needed." in result.summary


@pytest.mark.asyncio
async def test_agent_intervention_cancelled_returns_blocked_note(tmp_path: Path) -> None:
    agent = GuiAgent(
        _ScriptedLLM([
            LLMResponse(
                content="request help",
                tool_calls=[ToolCall(
                    id="call-1",
                    name="computer_use",
                    arguments={"action_type": "request_intervention", "text": "Need human input"},
                )],
            ),
        ]),
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "intervention note"),
        artifacts_root=tmp_path / "runs",
        max_steps=1,
        include_date_context=False,
    )

    result = await agent.run("pause for review", max_retries=1)

    assert not result.success
    assert result.error and result.error.startswith("intervention_cancelled")
    assert result.summary.startswith("Status: blocked")
    assert "Done:" in result.summary
    assert "Remaining:" in result.summary
    assert "Current:" in result.summary
    assert "Resume:" in result.summary


@pytest.mark.asyncio
async def test_agent_stagnation_counter_resets_when_screen_changes(tmp_path: Path) -> None:
    class _ChangingBackend(DryRunBackend):
        def __init__(self) -> None:
            super().__init__()
            self._observe_calls = 0

        async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
            del timeout
            from PIL import Image, ImageDraw

            self._observe_calls += 1
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            img = Image.new("L", (32, 32), color=0)
            draw = ImageDraw.Draw(img)
            if self._observe_calls in {1, 2}:
                draw.rectangle((16, 0, 31, 31), fill=255)   # right half bright
            else:
                draw.rectangle((0, 16, 31, 31), fill=255)   # bottom half bright
            img.save(screenshot_path, format="PNG")
            return Observation(
                screenshot_path=str(screenshot_path),
                screen_width=32,
                screen_height=32,
                foreground_app="DryRun",
                platform=self.platform,
            )

    llm = _ScriptedLLM([
        LLMResponse(
            content="wait 1",
            tool_calls=[ToolCall(
                id="call-1",
                name="computer_use",
                arguments={"action_type": "wait", "duration_ms": 1},
            )],
        ),
        LLMResponse(
            content="wait 2",
            tool_calls=[ToolCall(
                id="call-2",
                name="computer_use",
                arguments={"action_type": "wait", "duration_ms": 1},
            )],
        ),
        LLMResponse(
            content="wait 3",
            tool_calls=[ToolCall(
                id="call-3",
                name="computer_use",
                arguments={"action_type": "wait", "duration_ms": 1},
            )],
        ),
        LLMResponse(
            content="wait 4",
            tool_calls=[ToolCall(
                id="call-4",
                name="computer_use",
                arguments={"action_type": "wait", "duration_ms": 1},
            )],
        ),
        LLMResponse(
            content="finish",
            tool_calls=[ToolCall(
                id="call-5",
                name="computer_use",
                arguments={"action_type": "done", "status": "success"},
            )],
        ),
    ])
    agent = GuiAgent(
        llm,
        _ChangingBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "stagnation reset"),
        artifacts_root=tmp_path / "runs",
        max_steps=6,
        stagnation_limit=3,
    )

    result = await agent.run("wait with one screen change", max_retries=1)

    assert result.success
    assert result.error is None


@pytest.mark.asyncio
async def test_stagnation_detection_short_circuits_retries(tmp_path: Path) -> None:
    class _InfiniteWaitLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, tools=None, tool_choice=None) -> LLMResponse:
            del messages, tools, tool_choice
            self.calls += 1
            return LLMResponse(
                content="wait",
                tool_calls=[ToolCall(
                    id=f"call-{self.calls}",
                    name="computer_use",
                    arguments={"action_type": "wait", "duration_ms": 1},
                )],
            )

    recorder = _make_recorder(tmp_path, "stagnation retry short-circuit")
    llm = _InfiniteWaitLLM()
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=recorder,
        artifacts_root=tmp_path / "runs",
        max_steps=8,
        stagnation_limit=3,
    )

    result = await agent.run("repeat wait", max_retries=3)

    assert not result.success
    assert result.error == "stagnation_detected"
    assert llm.calls == 4
    assert recorder.path is not None
    events = [json.loads(line) for line in recorder.path.read_text(encoding="utf-8").splitlines()]
    assert sum(1 for event in events if event["type"] == "attempt_start") == 1
    assert not any(event["type"] == "retry" for event in events)


@pytest.mark.asyncio
async def test_stagnation_detection_generates_termination_summary(tmp_path: Path) -> None:
    summary_text = "Task appears to loop on the same screen. Aborted to avoid repeated actions."
    llm = _RecordingLLM([
        LLMResponse(
            content="wait briefly",
            tool_calls=[ToolCall(
                id="call-1",
                name="computer_use",
                arguments={"action_type": "wait", "duration_ms": 1},
            )],
        ),
        LLMResponse(
            content="wait again",
            tool_calls=[ToolCall(
                id="call-2",
                name="computer_use",
                arguments={"action_type": "wait", "duration_ms": 1},
            )],
        ),
        LLMResponse(content=summary_text),
    ])

    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "stagnation summary"),
        artifacts_root=tmp_path / "runs",
        max_steps=5,
        stagnation_limit=2,
    )

    result = await agent.run("detect loop", max_retries=1)

    assert not result.success
    assert result.error == "stagnation_detected"
    assert result.summary.startswith("Status: blocked")
    assert "Done:" in result.summary
    assert "Remaining:" in result.summary
    assert "Current:" in result.summary
    assert "Resume:" in result.summary
    assert len(llm.calls) == 3
    summary_call = llm.calls[2]
    content = summary_call[-1]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    prompt_text = "\n".join(
        block["text"]
        for block in content
        if block.get("type") == "text"
    )
    assert "Status: completed|partial|blocked" in prompt_text


@pytest.mark.asyncio
async def test_agent_stagnation_requires_same_action_type(tmp_path: Path) -> None:
    class _StaticScreenshotBackend(DryRunBackend):
        async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
            del timeout
            from PIL import Image

            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 64), color=(90, 90, 90)).save(
                screenshot_path,
                format="PNG",
            )
            return Observation(
                screenshot_path=str(screenshot_path),
                screen_width=64,
                screen_height=64,
                foreground_app="DryRun",
                platform=self.platform,
            )

    agent = GuiAgent(
        _ScriptedLLM([
            LLMResponse(
                content="wait",
                tool_calls=[ToolCall(
                    id="call-1",
                    name="computer_use",
                    arguments={"action_type": "wait", "duration_ms": 1},
                )],
            ),
            LLMResponse(
                content="tap",
                tool_calls=[ToolCall(
                    id="call-2",
                    name="computer_use",
                    arguments={
                        "action_type": "tap",
                        "x": 10,
                        "y": 10,
                        "relative": True,
                    },
                )],
            ),
            LLMResponse(
                content="wait",
                tool_calls=[ToolCall(
                    id="call-3",
                    name="computer_use",
                    arguments={"action_type": "wait", "duration_ms": 1},
                )],
            ),
            LLMResponse(
                content="tap",
                tool_calls=[ToolCall(
                    id="call-4",
                    name="computer_use",
                    arguments={
                        "action_type": "tap",
                        "x": 20,
                        "y": 20,
                        "relative": True,
                    },
                )],
            ),
            LLMResponse(
                content="done",
                tool_calls=[ToolCall(
                    id="call-5",
                    name="computer_use",
                    arguments={"action_type": "done", "status": "success"},
                )],
            ),
        ]),
        _StaticScreenshotBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "stagnation action type"),
        artifacts_root=tmp_path / "runs",
        max_steps=5,
        stagnation_limit=2,
    )

    result = await agent.run("alternate actions", max_retries=1)

    assert result.success
    assert result.error is None


@pytest.mark.asyncio
async def test_stagnation_similarity_uses_ssim(tmp_path: Path) -> None:
    from PIL import Image

    base = tmp_path / "base.png"
    similar = tmp_path / "similar.png"
    changed = tmp_path / "changed.png"

    width = 64
    height = 64

    def _pattern(path: Path, invert: bool = False, perturb: bool = False) -> None:
        img = Image.new("L", (width, height))
        for y in range(height):
            for x in range(width):
                value = (x * 3 + y * 5) % 256
                if invert:
                    value = 255 - value
                img.putpixel((x, y), value)
        if perturb:
            img.putpixel((1, 1), (img.getpixel((1, 1)) + 1) % 256)
        img.save(path, format="PNG")

    _pattern(base)
    _pattern(similar, perturb=True)
    _pattern(changed, invert=True)

    agent = GuiAgent(
        _ScriptedLLM([]),
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "ssim stagnation"),
        artifacts_root=tmp_path / "runs",
    )

    base_fp = agent._build_screen_fingerprint(
        Observation(
            screenshot_path=str(base),
            screen_width=width,
            screen_height=height,
            foreground_app="DryRun",
            platform="dry-run",
        ),
    )
    similar_fp = agent._build_screen_fingerprint(
        Observation(
            screenshot_path=str(similar),
            screen_width=width,
            screen_height=height,
            foreground_app="DryRun",
            platform="dry-run",
        ),
    )
    changed_fp = agent._build_screen_fingerprint(
        Observation(
            screenshot_path=str(changed),
            screen_width=width,
            screen_height=height,
            foreground_app="DryRun",
            platform="dry-run",
        ),
    )

    assert base_fp is not None
    assert similar_fp is not None
    assert changed_fp is not None
    assert agent._is_same_screen(base_fp, similar_fp)
    assert not agent._is_same_screen(base_fp, changed_fp)


@pytest.mark.asyncio
async def test_agent_records_attempt_exception_and_retry_events(tmp_path: Path) -> None:
    class _FlakyLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, tools=None, tool_choice=None) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("provider exploded")
            return LLMResponse(
                content="Action: done",
                tool_calls=[ToolCall(
                    id="call-2",
                    name="computer_use",
                    arguments={"action_type": "done", "status": "success"},
                )],
            )

    recorder = _make_recorder(tmp_path, "retry task")
    agent = GuiAgent(
        _FlakyLLM(),
        DryRunBackend(),
        trajectory_recorder=recorder,
        artifacts_root=tmp_path / "runs",
        max_steps=1,
    )

    result = await agent.run("retry task", max_retries=2)

    assert result.success
    assert recorder.path is not None
    events = [json.loads(line) for line in recorder.path.read_text(encoding="utf-8").splitlines()]
    types = [event["type"] for event in events]
    assert "attempt_start" in types
    assert "attempt_exception" in types
    assert "retry" in types
    attempt_exception = next(event for event in events if event["type"] == "attempt_exception")
    assert attempt_exception["error_type"] == "RuntimeError"
    assert attempt_exception["error_message"] == "provider exploded"


@pytest.mark.asyncio
async def test_agent_records_model_response_on_attempt_exception(tmp_path: Path) -> None:
    class _MalformedToolCallLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, tools=None, tool_choice=None) -> LLMResponse:
            self.calls += 1
            if self.calls <= 3:
                return LLMResponse(
                    content="Action: Swipe up to reveal the size options.",
                    tool_calls=[ToolCall(
                        id=f"bad-call-{self.calls}",
                        name="computer_use",
                        arguments={"action_type": "swipe", "x": [500, 750], "relative": True},
                    )],
                )
            return LLMResponse(
                content="Action: done",
                tool_calls=[ToolCall(
                    id="good-call",
                    name="computer_use",
                    arguments={"action_type": "done", "status": "success"},
                )],
            )

    recorder = _make_recorder(tmp_path, "retry malformed tool call")
    agent = GuiAgent(
        _MalformedToolCallLLM(),
        DryRunBackend(),
        trajectory_recorder=recorder,
        artifacts_root=tmp_path / "runs",
        max_steps=1,
    )

    result = await agent.run("retry malformed tool call", max_retries=2)

    assert result.success
    assert recorder.path is not None
    events = [json.loads(line) for line in recorder.path.read_text(encoding="utf-8").splitlines()]
    attempt_exception = next(event for event in events if event["type"] == "attempt_exception")
    assert attempt_exception["error_type"] == "_StepExecutionError"
    assert "Failed to parse action after retries" in attempt_exception["error_message"]
    assert attempt_exception["model_response"]["raw_content"] == "Action: Swipe up to reveal the size options."
    assert attempt_exception["model_response"]["tool_calls"][0]["name"] == "computer_use"
    assert attempt_exception["model_response"]["tool_calls"][0]["arguments"] == {
        "action_type": "swipe",
        "x": [500, 750],
        "relative": True,
    }


@pytest.mark.asyncio
async def test_retry_prompt_includes_previous_attempt_summary_after_max_steps(
    tmp_path: Path,
) -> None:
    llm = _RecordingLLM([
        LLMResponse(
            content="wait briefly",
            tool_calls=[ToolCall(
                id="call-1",
                name="computer_use",
                arguments={"action_type": "wait", "duration_ms": 1},
            )],
        ),
        # Termination summary call (text-only, no tool call) after max_steps hit
        LLMResponse(content="Waited briefly but ran out of steps. Currently on home screen."),
        LLMResponse(
            content="finish task",
            tool_calls=[ToolCall(
                id="call-2",
                name="computer_use",
                arguments={"action_type": "done", "status": "success"},
            )],
        ),
    ])
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "retry summary max steps"),
        artifacts_root=tmp_path / "runs",
        max_steps=1,
        include_date_context=False,
    )

    result = await agent.run("Open Settings", max_retries=2)

    assert result.success
    # calls[0]=attempt1 step, calls[1]=termination summary, calls[2]=attempt2 step
    second_attempt = llm.calls[2]
    retry_text = "\n".join(
        block["text"]
        for block in second_attempt[1]["content"]
        if block.get("type") == "text"
    )
    assert "Previous attempt summaries:" in retry_text
    assert "Attempt 1:" in retry_text
    assert "Failure reason: max_steps_exceeded" in retry_text
    assert "Step 1: wait briefly" in retry_text
    assert "Continue from the current screen state" in retry_text


@pytest.mark.asyncio
async def test_retry_prompt_includes_previous_attempt_summary_after_exception(
    tmp_path: Path,
) -> None:
    class _MalformedThenRecoverLLM:
        def __init__(self) -> None:
            self.calls: list[list[dict]] = []
            self._call_count = 0

        async def chat(self, messages, tools=None, tool_choice=None) -> LLMResponse:
            del tools, tool_choice
            self.calls.append(copy.deepcopy(messages))
            self._call_count += 1
            if self._call_count <= 3:
                return LLMResponse(
                    content="Action: Swipe up to reveal the size options.",
                    tool_calls=[ToolCall(
                        id=f"bad-call-{self._call_count}",
                        name="computer_use",
                        arguments={"action_type": "swipe", "x": [500, 750], "relative": True},
                    )],
                )
            return LLMResponse(
                content="finish task",
                tool_calls=[ToolCall(
                    id="good-call",
                    name="computer_use",
                    arguments={"action_type": "done", "status": "success"},
                )],
            )

    llm = _MalformedThenRecoverLLM()
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "retry summary exception"),
        artifacts_root=tmp_path / "runs",
        max_steps=1,
        include_date_context=False,
    )

    result = await agent.run("retry malformed tool call", max_retries=2)

    assert result.success
    second_attempt = llm.calls[3]
    retry_text = "\n".join(
        block["text"]
        for block in second_attempt[1]["content"]
        if block.get("type") == "text"
    )
    assert "Previous attempt summaries:" in retry_text
    assert "Attempt 1:" in retry_text
    assert "Failure reason: _StepExecutionError: Failed to parse action after retries" in retry_text
    assert "No completed GUI actions were recorded before the failure." in retry_text
    assert "Continue from the current screen state" in retry_text


@pytest.mark.asyncio
async def test_agent_uses_history_summary_and_recent_image_window(tmp_path: Path) -> None:
    llm = _RecordingLLM([
        LLMResponse(
            content="wait briefly",
            tool_calls=[ToolCall(
                id="call-1",
                name="computer_use",
                arguments={"action_type": "wait", "duration_ms": 1},
            )],
        ),
        LLMResponse(
            content="wait again",
            tool_calls=[ToolCall(
                id="call-2",
                name="computer_use",
                arguments={"action_type": "wait", "duration_ms": 1},
            )],
        ),
        LLMResponse(
            content="finish task",
            tool_calls=[ToolCall(
                id="call-3",
                name="computer_use",
                arguments={"action_type": "done", "status": "success"},
            )],
        ),
    ])
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "Open Settings"),
        artifacts_root=tmp_path / "runs",
        max_steps=3,
        history_image_window=1,
        include_date_context=False,
    )

    result = await agent.run("Open Settings")

    assert result.success
    assert len(llm.calls) == 3

    third_call = llm.calls[2]
    assert "# Tools" in third_call[0]["content"]

    history_user = third_call[1]
    history_text = "\n".join(
        block["text"]
        for block in history_user["content"]
        if block.get("type") == "text"
    )
    assert "Please generate the next move according to the UI screenshot, instruction and recent progress context." in history_text
    assert "Instruction: Open Settings" in history_text
    assert "Recent intents:\nStep 1: wait briefly" in history_text
    assert "Step 2: wait again" in history_text

    assert len(third_call) == 2
    assert "Step 3" in history_text
    assert "Task: Open Settings" in history_text


@pytest.mark.asyncio
async def test_agent_prefers_tool_summary_for_action_history_and_trace(tmp_path: Path) -> None:
    llm = _RecordingLLM([
        LLMResponse(
            content="Action: Tap login button",
            tool_calls=[ToolCall(
                id="call-1",
                name="computer_use",
                arguments={
                    "action_type": "tap",
                    "x": 500,
                    "y": 250,
                    "intent": "tap login button",
                    "summary": "login page is open; login button is visible",
                },
            )],
        ),
        LLMResponse(
            content="Action: Finish task",
            tool_calls=[ToolCall(
                id="call-2",
                name="computer_use",
                arguments={
                    "action_type": "done",
                    "status": "success",
                    "intent": "finish login flow",
                    "summary": "login flow is complete",
                },
            )],
        ),
    ])
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "tool summary"),
        artifacts_root=tmp_path / "runs",
        max_steps=2,
        history_image_window=1,
        include_date_context=False,
    )

    result = await agent.run("Open Login")

    assert result.success
    assert result.model_summary == "login flow is complete"
    second_call = llm.calls[1]
    history_text = "\n".join(
        block["text"]
        for block in second_call[1]["content"]
        if block.get("type") == "text"
    )
    assert "Recent intents:\nStep 1: tap login button" in history_text
    assert "Latest state summary: login page is open; login button is visible" in history_text

    trace_path = next((tmp_path / "runs").glob("*/trace.jsonl"))
    step_events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if '"event": "step"' in line
    ]
    assert step_events[0]["action_intent"] == "tap login button"
    assert step_events[0]["state_summary"] == "login page is open; login button is visible"
    assert step_events[1]["model_output"]["action_intent"] == "finish login flow"
    assert step_events[1]["model_output"]["state_summary"] == "login flow is complete"

    mobileworld_trace_path = trace_path.with_name("traj.json")
    mobileworld_trace = json.loads(mobileworld_trace_path.read_text(encoding="utf-8"))
    traj = mobileworld_trace["0"]["traj"]
    assert traj[0]["task_goal"] == "Open Login"
    assert traj[0]["step"] == 1
    assert traj[0]["prediction"] == "Action: tap login button"
    assert traj[0]["action"]["action_type"] == "tap"
    assert traj[0]["intent"] == "tap login button"
    assert traj[0]["summary"] == "login page is open; login button is visible"
    assert traj[0]["tool_call"]["name"] == "computer_use"
    assert traj[0]["tool_call"]["arguments"]["intent"] == "tap login button"
    assert traj[0]["screenshot"].startswith("screenshots/")
    assert traj[0]["marked_screenshot"].startswith("marked_screenshots/")
    assert (trace_path.parent / traj[0]["marked_screenshot"]).exists()


@pytest.mark.asyncio
async def test_agent_prompt_uses_last_eight_intents_and_latest_summary(tmp_path: Path) -> None:
    responses = [
        LLMResponse(
            content=f"Action: Step {index}",
            tool_calls=[ToolCall(
                id=f"call-{index}",
                name="computer_use",
                arguments={
                    "action_type": "wait",
                    "duration_ms": 1,
                    "intent": f"intent {index}",
                    "summary": f"summary {index}",
                },
            )],
        )
        for index in range(1, 10)
    ]
    responses.append(LLMResponse(
        content="Action: Finish task",
        tool_calls=[ToolCall(
            id="call-10",
            name="computer_use",
            arguments={
                "action_type": "done",
                "status": "success",
                "intent": "finish task",
                "summary": "task complete",
            },
        )],
    ))
    llm = _RecordingLLM(responses)
    agent = GuiAgent(
        llm,
        DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "intent window"),
        artifacts_root=tmp_path / "runs",
        max_steps=10,
        history_image_window=1,
        include_date_context=False,
    )

    result = await agent.run("Open Settings")

    assert result.success
    tenth_call = llm.calls[9]
    prompt_text = "\n".join(
        block["text"]
        for block in tenth_call[1]["content"]
        if block.get("type") == "text"
    )
    assert "Recent intents:" in prompt_text
    assert "Step 1: intent 1" not in prompt_text
    for index in range(2, 10):
        assert f"Step {index}: intent {index}" in prompt_text
    assert "Latest state summary: summary 9" in prompt_text


@pytest.mark.asyncio
async def test_agent_waits_for_ui_to_settle_before_observing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    original_sleep = asyncio.sleep

    class _SettlingBackend(DryRunBackend):
        async def execute(self, action, timeout: float = 5.0) -> str:
            events.append(f"execute:{action.action_type}")
            return await super().execute(action, timeout=timeout)

        async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
            events.append(f"observe:{Path(screenshot_path).name}")
            return await super().observe(screenshot_path, timeout=timeout)

    async def fake_sleep(delay: float) -> None:
        events.append(f"sleep:{delay}")
        await original_sleep(0)

    monkeypatch.setattr("opengui.agent.asyncio.sleep", fake_sleep)

    agent = GuiAgent(
        _ScriptedLLM([
            LLMResponse(
                content="tap the screen",
                tool_calls=[ToolCall(
                    id="call-1",
                    name="computer_use",
                    arguments={"action_type": "tap", "x": 10, "y": 20},
                )],
            ),
            LLMResponse(
                content="finish task",
                tool_calls=[ToolCall(
                    id="call-2",
                    name="computer_use",
                    arguments={"action_type": "done", "status": "success"},
                )],
            ),
        ]),
        _SettlingBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "settle"),
        artifacts_root=tmp_path / "runs",
        max_steps=2,
        include_date_context=False,
    )

    result = await agent.run("Tap once")

    assert result.success
    assert events[:4] == [
        "observe:step_000.png",
        "execute:tap",
        "sleep:0.5",
        "observe:step_001.png",
    ]


def test_pyproject_includes_opengui_in_build() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    include = pyproject["tool"]["hatch"]["build"]["include"]
    assert "opengui/**/*.py" in include

    wheel = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert "opengui" in wheel["packages"]

    sdist_include = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    assert "opengui/" in sdist_include
