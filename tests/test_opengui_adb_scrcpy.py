from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from PIL import Image

import opengui.cli as cli
from nanobot.config.schema import GuiConfig
from opengui.action import Action
from opengui.backends.adb import (
    AdbBackend,
    ScrcpyFrameSource,
    ScrcpyFrameSnapshot,
    _DEVICE_SCREENSHOT_PATH,
    _pil_image_from_frame,
)
from opengui.observation import Observation


class FakeFrameSource:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def save_latest(self, path: Path, *, timeout_s: float, max_age_s: float) -> ScrcpyFrameSnapshot:
        Image.new("RGB", (320, 640), color=(10, 20, 30)).save(path)
        return ScrcpyFrameSnapshot(width=320, height=640, timestamp=123.0)


class StaleThenFreshFrameSource:
    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0
        self.save_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def save_latest(self, path: Path, *, timeout_s: float, max_age_s: float) -> ScrcpyFrameSnapshot:
        self.save_calls += 1
        if self.save_calls == 1:
            raise TimeoutError("latest scrcpy frame is stale (8.00s > 5.00s)")
        Image.new("RGB", (320, 640), color=(40, 50, 60)).save(path)
        return ScrcpyFrameSnapshot(width=320, height=640, timestamp=time.time())


class AlwaysStaleFrameSource:
    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0
        self.save_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def save_latest(self, path: Path, *, timeout_s: float, max_age_s: float) -> ScrcpyFrameSnapshot:
        self.save_calls += 1
        raise TimeoutError("latest scrcpy frame is stale (8.00s > 5.00s)")


@pytest.mark.asyncio
async def test_adb_observe_uses_scrcpy_frame_and_reports_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame_source = FakeFrameSource()
    backend = AdbBackend(frame_source=frame_source)
    monkeypatch.setattr(backend, "_query_foreground_app", AsyncMock(return_value="com.example.app"))
    monkeypatch.setattr(backend, "_query_screen_size", AsyncMock(return_value=(1080, 2376)))

    screenshot_path = tmp_path / "screen.png"
    observation = await backend.observe(screenshot_path)

    assert frame_source.started is True
    assert screenshot_path.exists()
    assert observation.screen_width == 320
    assert observation.screen_height == 640
    assert observation.foreground_app == "com.example.app"
    assert observation.extra["capture_source"] == "scrcpy"
    assert backend._capture_width == 320
    assert backend._capture_height == 640
    assert backend._screen_width == 1080
    assert backend._screen_height == 2376


def test_scrcpy_save_latest_waits_for_fresh_frame_after_stale_cache(tmp_path: Path) -> None:
    source = ScrcpyFrameSource()
    stale = Image.new("RGB", (16, 16), color=(255, 0, 0))
    fresh = Image.new("RGB", (16, 16), color=(0, 255, 0))

    with source._condition:
        source._latest_frame = stale
        source._latest_ts = time.time() - 10.0

    def publish_fresh_frame() -> None:
        time.sleep(0.05)
        source._set_latest_frame(fresh)

    thread = threading.Thread(target=publish_fresh_frame)
    thread.start()
    screenshot_path = tmp_path / "screen.png"
    snapshot = source.save_latest(screenshot_path, timeout_s=1.0, max_age_s=0.5)
    thread.join(timeout=1.0)

    assert snapshot.width == 16
    assert snapshot.height == 16
    with Image.open(screenshot_path) as image:
        assert image.getpixel((0, 0)) == (0, 255, 0)


@pytest.mark.asyncio
async def test_adb_observe_restarts_scrcpy_after_stale_frame(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame_source = StaleThenFreshFrameSource()
    backend = AdbBackend(frame_source=frame_source)
    monkeypatch.setattr(backend, "_query_foreground_app", AsyncMock(return_value="com.example.app"))
    monkeypatch.setattr(backend, "_query_screen_size", AsyncMock(return_value=(1080, 2376)))

    observation = await backend.observe(tmp_path / "screen.png")

    assert observation.extra["capture_source"] == "scrcpy"
    assert frame_source.start_calls == 2
    assert frame_source.stop_calls == 1
    assert frame_source.save_calls == 2


@pytest.mark.asyncio
async def test_adb_observe_falls_back_to_screencap_after_scrcpy_restart_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame_source = AlwaysStaleFrameSource()
    backend = AdbBackend(frame_source=frame_source)
    fallback_observation = Observation(
        screenshot_path=str(tmp_path / "fallback.png"),
        screen_width=1080,
        screen_height=2376,
        foreground_app="com.example.app",
        platform="android",
    )
    fallback = AsyncMock(return_value=fallback_observation)
    monkeypatch.setattr(backend, "_observe_via_screencap", fallback)

    observation = await backend.observe(tmp_path / "screen.png")

    assert observation is fallback_observation
    assert frame_source.start_calls == 2
    assert frame_source.stop_calls == 1
    assert frame_source.save_calls == 2
    fallback.assert_awaited_once()


def test_adb_screencap_temp_path_uses_data_local_tmp() -> None:
    assert _DEVICE_SCREENSHOT_PATH == "/data/local/tmp/__opengui_cap.png"


@pytest.mark.asyncio
async def test_adb_scrcpy_relative_tap_uses_physical_input_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = AdbBackend(frame_source=FakeFrameSource())
    monkeypatch.setattr(backend, "_query_foreground_app", AsyncMock(return_value="com.example.app"))
    monkeypatch.setattr(backend, "_query_screen_size", AsyncMock(return_value=(1080, 2376)))
    await backend.observe(tmp_path / "screen.png")

    run_mock = AsyncMock(return_value="")
    monkeypatch.setattr(backend, "_run", run_mock)

    action = Action(action_type="tap", x=900, y=950, relative=True)
    await backend.execute(action)

    run_mock.assert_awaited_once_with("shell", "input", "tap", "972", "2259", timeout=5.0)


@pytest.mark.asyncio
async def test_adb_scrcpy_absolute_tap_maps_capture_pixels_to_input_pixels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = AdbBackend(frame_source=FakeFrameSource())
    monkeypatch.setattr(backend, "_query_foreground_app", AsyncMock(return_value="com.example.app"))
    monkeypatch.setattr(backend, "_query_screen_size", AsyncMock(return_value=(1080, 2376)))
    await backend.observe(tmp_path / "screen.png")

    run_mock = AsyncMock(return_value="")
    monkeypatch.setattr(backend, "_run", run_mock)

    action = Action(action_type="tap", x=288, y=576)
    await backend.execute(action)

    run_mock.assert_awaited_once_with("shell", "input", "tap", "974", "2141", timeout=5.0)


@pytest.mark.asyncio
async def test_adb_scrcpy_capture_keeps_actions_on_adb(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = AdbBackend(frame_source=FakeFrameSource())
    run_mock = AsyncMock(return_value="")
    monkeypatch.setattr(backend, "_run", run_mock)

    action = Action(action_type="tap", x=1, y=2)
    assert await backend.execute(action) == "tap at (1, 2)"
    run_mock.assert_awaited_once_with("shell", "input", "tap", "1", "2", timeout=5.0)


def test_gui_config_accepts_adb_backend_with_scrcpy_camel_case_fields() -> None:
    config = GuiConfig.model_validate({
        "backend": "adb",
        "scrcpy": {
            "maxFps": 8,
            "jpegQuality": 70,
            "frameTimeoutMs": 1500,
            "maxFrameAgeMs": 500,
        },
    })

    assert config.backend == "adb"
    assert config.scrcpy.max_fps == 8
    assert config.scrcpy.jpeg_quality == 70
    assert config.scrcpy.frame_timeout_ms == 1500
    assert config.scrcpy.max_frame_age_ms == 500


def test_gui_config_accepts_ios_mjpeg_fields() -> None:
    config = GuiConfig.model_validate({
        "backend": "ios",
        "ios": {
            "wdaUrl": "http://localhost:8100",
            "mjpegUrl": "http://127.0.0.1:9100",
            "mjpegFrameTimeoutMs": 2500,
        },
    })

    assert config.ios.wda_url == "http://localhost:8100"
    assert config.ios.mjpeg_url == "http://127.0.0.1:9100"
    assert config.ios.mjpeg_frame_timeout_ms == 2500


def test_cli_builds_adb_backend_with_scrcpy_capture_config() -> None:
    config = cli.CliConfig(
        provider=cli.ProviderConfig(base_url="http://localhost:1/v1", model="m"),
        adb=cli.AdbConfig(serial="emulator-5554", adb_path="/tmp/adb"),
        scrcpy=cli.ScrcpyConfig(
            max_fps=7,
            jpeg_quality=65,
            frame_timeout_ms=1200,
            max_frame_age_ms=400,
        ),
    )

    backend = cli.build_backend("adb", config)

    assert isinstance(backend, AdbBackend)
    assert backend._scrcpy_frame_timeout_ms == 1200
    assert backend._scrcpy_max_frame_age_ms == 400
    assert backend._frame_source._adb_path == "/tmp/adb"  # type: ignore[attr-defined]
    assert backend._frame_source._frame_timeout_ms == 1200  # type: ignore[attr-defined]


def test_scrcpy_numpy_frames_are_converted_from_bgr_to_rgb() -> None:
    np = pytest.importorskip("numpy")
    frame = np.array([[[0, 0, 255]]], dtype=np.uint8)

    image = _pil_image_from_frame(frame)

    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (255, 0, 0)
