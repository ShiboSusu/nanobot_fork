"""
opengui.backends.adb
~~~~~~~~~~~~~~~~~~~~
Android Debug Bridge backend for device and emulator automation.

Unicode text input prefers ADBKeyboard IME on the target device:
    adb shell ime set com.android.adbkeyboard/.AdbIME

If ADBKeyboard is unavailable, OpenGUI falls back to a device-side `yadb`
helper when it is installed at `/data/local/tmp/yadb`.

All I/O is non-blocking via asyncio.create_subprocess_exec.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.resources
import inspect
import os
import re
import struct
import tempfile
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Protocol

from opengui.action import Action, describe_action, resolve_coordinate
from opengui.observation import Observation

# ---------------------------------------------------------------------------
# Keycode mapping
# ---------------------------------------------------------------------------

_KEYCODE_MAP: dict[str, str] = {
    "home": "KEYCODE_HOME",
    "back": "KEYCODE_BACK",
    "enter": "KEYCODE_ENTER",
    "return": "KEYCODE_ENTER",
    "tab": "KEYCODE_TAB",
    "delete": "KEYCODE_DEL",
    "backspace": "KEYCODE_DEL",
    "volumeup": "KEYCODE_VOLUME_UP",
    "volume_up": "KEYCODE_VOLUME_UP",
    "volumedown": "KEYCODE_VOLUME_DOWN",
    "volume_down": "KEYCODE_VOLUME_DOWN",
    "power": "KEYCODE_POWER",
    "menu": "KEYCODE_MENU",
    "recents": "KEYCODE_APP_SWITCH",
    "app_switch": "KEYCODE_APP_SWITCH",
    "escape": "KEYCODE_ESCAPE",
    "space": "KEYCODE_SPACE",
    "search": "KEYCODE_SEARCH",
    "camera": "KEYCODE_CAMERA",
    "left": "KEYCODE_DPAD_LEFT",
    "right": "KEYCODE_DPAD_RIGHT",
    "up": "KEYCODE_DPAD_UP",
    "down": "KEYCODE_DPAD_DOWN",
}

_WM_SIZE_RE = re.compile(r"Physical size:\s*(\d+)x(\d+)", re.IGNORECASE)
_RESUMED_ACTIVITY_RE = re.compile(
    r"mResumedActivity.*?"
    r"([a-zA-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z][a-zA-Z0-9_]*)+)/",
    re.IGNORECASE,
)
_TOP_RESUMED_ACTIVITY_RE = re.compile(
    r"topResumedActivity.*?"
    r"([a-zA-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z][a-zA-Z0-9_]*)+)/",
    re.IGNORECASE,
)
_CURRENT_FOCUS_RE = re.compile(
    r"mCurrentFocus.*?"
    r"([a-zA-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z][a-zA-Z0-9_]*)+)/",
    re.IGNORECASE,
)
_FOCUSED_APP_RE = re.compile(
    r"mFocusedApp.*?"
    r"([a-zA-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z][a-zA-Z0-9_]*)+)/",
    re.IGNORECASE,
)

_DEVICE_SCREENSHOT_PATH = "/data/local/tmp/__opengui_cap.png"
_ADB_KEYBOARD_IME = "com.android.adbkeyboard/.AdbIME"
_YADB_PATH = "/data/local/tmp/yadb"
_YADB_MAIN_CLASS = "com.ysbing.yadb.Main"

_KEYCOMBINATION_UNSUPPORTED_MARKERS = (
    "unknown command: keycombination",
    "invalid arguments for command: keycombination",
)


def _read_png_size(path: Path) -> tuple[int, int] | None:
    """Return PNG image size from *path* without external dependencies."""
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
    except OSError:
        return None

    if len(header) < 24:
        return None
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    if header[12:16] != b"IHDR":
        return None

    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        return None
    return width, height


@dataclass(frozen=True)
class ScrcpyFrameSnapshot:
    """Metadata for the latest scrcpy frame persisted to disk."""

    width: int
    height: int
    timestamp: float


class ScrcpyFrameSourceProtocol(Protocol):
    """Small protocol used by ``AdbBackend`` and tests."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def save_latest(self, path: Path, *, timeout_s: float, max_age_s: float) -> ScrcpyFrameSnapshot: ...


def _pil_image_from_frame(frame: Any) -> Any:
    """Normalize py-scrcpy frame objects into a PIL Image."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised through preflight text
        raise RuntimeError(
            "ADB scrcpy capture requires Pillow. Install with `uv pip install -e \".[demo-live]\"`."
        ) from exc

    if isinstance(frame, Image.Image):
        return frame

    shape = getattr(frame, "shape", None)
    if shape is not None:
        if len(shape) >= 3 and shape[2] == 3:
            frame = frame[..., ::-1]
        elif len(shape) >= 3 and shape[2] == 4:
            frame = frame[..., [2, 1, 0, 3]]
        return Image.fromarray(frame)

    raise TypeError(f"Unsupported scrcpy frame object: {type(frame).__name__}")


def _frame_dimensions(frame: Any) -> tuple[int, int]:
    if hasattr(frame, "size") and not hasattr(frame, "shape"):
        width, height = frame.size
        return int(width), int(height)
    shape = getattr(frame, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1]), int(shape[0])
    image = _pil_image_from_frame(frame)
    width, height = image.size
    return int(width), int(height)


def _jpeg_bytes_from_frame(frame: Any, *, quality: int) -> bytes:
    image = _pil_image_from_frame(frame)
    if image.mode not in {"RGB", "L"}:
        image = image.convert("RGB")
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=max(1, min(100, int(quality))))
    return buf.getvalue()


class ScrcpyFrameSource:
    """Owns the py-scrcpy-sdk client and latest-frame buffer for ADB observe()."""

    def __init__(
        self,
        *,
        serial: str | None = None,
        adb_path: str = "adb",
        max_fps: int = 12,
        jpeg_quality: int = 80,
        frame_timeout_ms: int = 3000,
        on_jpeg_frame: Callable[[bytes, dict[str, Any]], None] | None = None,
    ) -> None:
        self._serial = serial
        self._adb_path = adb_path
        self._max_fps = max_fps
        self._jpeg_quality = jpeg_quality
        self._frame_timeout_ms = frame_timeout_ms
        self._on_jpeg_frame = on_jpeg_frame
        self._client: Any | None = None
        self._listener: Any | None = None
        self._latest_frame: Any | None = None
        self._latest_ts = 0.0
        self._started = False
        self._stopping = False
        self._condition = threading.Condition()

    def start(self) -> None:
        if self._started:
            return
        try:
            from py_scrcpy_sdk import ScrcpyClient, ScrcpyConfig
        except ImportError as exc:
            raise RuntimeError(
                "ADB scrcpy capture requires py-scrcpy-sdk==0.1.2. "
                "Install with `uv pip install -e \".[demo-live]\"`."
            ) from exc

        config_kwargs: dict[str, Any] = {
            "serial": self._serial,
            "adb_path": self._adb_path,
            "frame_timeout": self._frame_timeout_ms / 1000.0,
        }
        if self._max_fps > 0:
            config_kwargs["max_fps"] = self._max_fps
        config = self._construct_config(ScrcpyConfig, config_kwargs)
        client = ScrcpyClient(config)
        client.start()
        self._client = client
        self._stopping = False

        initial_frame = self._wait_until_ready(client)
        if initial_frame is not None:
            self._set_latest_frame(initial_frame)

        listener = getattr(client, "start_frame_listener", None)
        if callable(listener):
            self._listener = listener(self._on_frame)
        else:
            threading.Thread(target=self._listen_forever, daemon=True).start()

        self._started = True

    @staticmethod
    def _construct_config(config_cls: Any, values: dict[str, Any]) -> Any:
        try:
            signature = inspect.signature(config_cls)
        except (TypeError, ValueError):
            return config_cls(**values)
        accepted = {
            name for name, param in signature.parameters.items()
            if param.kind in {param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY}
        }
        filtered = {key: value for key, value in values.items() if key in accepted}
        return config_cls(**filtered)

    @staticmethod
    def _wait_until_ready(client: Any) -> Any | None:
        ready = getattr(client, "wait_until_ready", None)
        if not callable(ready):
            get_frame = getattr(client, "get_frame", None)
            return get_frame() if callable(get_frame) else None
        try:
            return ready(timeout=5)
        except TypeError:
            return ready()

    def _listen_forever(self) -> None:
        frames = getattr(self._client, "frames", None)
        if not callable(frames):
            return
        for frame in frames():
            if self._on_frame(frame) is False:
                break

    def _on_frame(self, frame: Any) -> bool:
        self._set_latest_frame(frame)
        return not self._stopping

    def _set_latest_frame(self, frame: Any) -> None:
        ts = time.time()
        with self._condition:
            self._latest_frame = frame
            self._latest_ts = ts
            self._condition.notify_all()
        if self._on_jpeg_frame is not None:
            try:
                width, height = _frame_dimensions(frame)
                self._on_jpeg_frame(
                    _jpeg_bytes_from_frame(frame, quality=self._jpeg_quality),
                    {"width": width, "height": height, "timestamp": ts, "source": "scrcpy"},
                )
            except Exception:
                pass

    def save_latest(self, path: Path, *, timeout_s: float, max_age_s: float) -> ScrcpyFrameSnapshot:
        deadline = time.time() + timeout_s
        with self._condition:
            while True:
                frame = self._latest_frame
                ts = self._latest_ts
                now = time.time()
                if frame is not None and (max_age_s <= 0 or now - ts <= max_age_s):
                    break
                remaining = deadline - now
                if remaining <= 0:
                    break
                self._condition.wait(timeout=max(0.01, remaining))

        if frame is None:
            raise TimeoutError(f"scrcpy frame not available within {timeout_s:.2f}s")
        if max_age_s > 0 and time.time() - ts > max_age_s:
            raise TimeoutError(
                f"latest scrcpy frame is stale ({time.time() - ts:.2f}s > {max_age_s:.2f}s)"
            )

        image = _pil_image_from_frame(frame)
        width, height = _frame_dimensions(frame)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return ScrcpyFrameSnapshot(width=width, height=height, timestamp=ts)

    def stop(self) -> None:
        self._stopping = True
        client = self._client
        if client is not None:
            stop = getattr(client, "stop", None)
            if callable(stop):
                stop()
        with self._condition:
            self._latest_frame = None
            self._latest_ts = 0.0
            self._condition.notify_all()
        self._listener = None
        self._client = None
        self._started = False


class AdbError(Exception):
    """Raised when an adb command fails."""

    def __init__(self, message: str, returncode: int | None = None, stderr: str = "") -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


def _escape_shell_text(text: str) -> str:
    """Escape text for ``adb shell input text``."""
    # Android's `input text` command treats `%s` as a space placeholder.
    # We are not invoking a shell here, so `\ ` would be passed literally and
    # can cause whitespace truncation on device-side parsing.
    special_chars = frozenset(r'\`$"!&|<>(){}[];#~*?^')
    escaped: list[str] = []
    for ch in text:
        if ch == " ":
            escaped.append("%s")
        elif ch in special_chars:
            escaped.append("\\" + ch)
        else:
            escaped.append(ch)
    return "".join(escaped)


def _is_ascii_safe(text: str) -> bool:
    return all(ord(ch) < 128 for ch in text)


def _to_b64_text(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _is_emojiish_char(ch: str) -> bool:
    codepoint = ord(ch)
    if ch in ("\u200d", "\ufe0e", "\ufe0f", "\u20e3"):
        return True
    if 0x1F1E6 <= codepoint <= 0x1F1FF:  # Regional indicators
        return True
    if 0x1F3FB <= codepoint <= 0x1F3FF:  # Skin tone modifiers
        return True
    if 0x2600 <= codepoint <= 0x27BF:  # Misc symbols / dingbats often used as emoji
        return True
    if 0x1F000 <= codepoint <= 0x1FAFF:  # Main emoji-heavy planes
        return True
    return unicodedata.category(ch) == "So"


def _iter_text_input_segments(text: str) -> list[str]:
    """Split text into stable input chunks, isolating emoji-ish sequences.

    Some Android IME injection paths may truncate text that follows an emoji
    when the whole string is sent in one batch. By sending normal-text runs and
    emoji clusters separately, later text still lands even if the IME treats
    emoji boundaries specially.
    """
    if not text:
        return []

    segments: list[str] = []
    current = text[0]
    current_is_emoji = _is_emojiish_char(text[0])

    for ch in text[1:]:
        ch_is_emoji = _is_emojiish_char(ch)
        if ch_is_emoji == current_is_emoji:
            current += ch
            continue
        segments.append(current)
        current = ch
        current_is_emoji = ch_is_emoji

    segments.append(current)
    return segments


class AdbBackend:
    """ADB backend for Android device/emulator automation.

    Observations use scrcpy frames by default and ADB for all device actions.
    The legacy ``adb shell screencap`` capture path remains available for
    focused tests and explicit fallback wiring.

    Args:
        serial: Device serial (e.g. "emulator-5554"). None = default device.
        adb_path: Path to adb binary.
    """

    def __init__(
        self,
        serial: str | None = None,
        adb_path: str = "adb",
        *,
        scrcpy_max_fps: int = 12,
        scrcpy_jpeg_quality: int = 80,
        scrcpy_frame_timeout_ms: int = 3000,
        scrcpy_max_frame_age_ms: int = 1000,
        frame_source: ScrcpyFrameSourceProtocol | None = None,
        on_jpeg_frame: Callable[[bytes, dict[str, Any]], None] | None = None,
        use_scrcpy: bool = True,
    ) -> None:
        self._serial = serial
        self._adb = adb_path
        self._screen_width = 1080
        self._screen_height = 1920
        self._capture_width = 1080
        self._capture_height = 1920
        self._scrcpy_frame_timeout_ms = scrcpy_frame_timeout_ms
        self._scrcpy_max_frame_age_ms = scrcpy_max_frame_age_ms
        self._use_scrcpy = use_scrcpy
        self._scrcpy_started = False
        self._frame_source = (
            frame_source
            if frame_source is not None
            else ScrcpyFrameSource(
                serial=serial,
                adb_path=adb_path,
                max_fps=scrcpy_max_fps,
                jpeg_quality=scrcpy_jpeg_quality,
                frame_timeout_ms=scrcpy_frame_timeout_ms,
                on_jpeg_frame=on_jpeg_frame,
            )
        ) if use_scrcpy else None

    @property
    def platform(self) -> str:
        return "android"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_cmd(self, *args: str) -> list[str]:
        cmd: list[str] = [self._adb]
        if self._serial:
            cmd += ["-s", self._serial]
        cmd += list(args)
        return cmd

    async def _run(self, *args: str, timeout: float = 10.0) -> str:
        cmd = self._build_cmd(*args)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            raise TimeoutError(f"adb timed out after {timeout}s: {' '.join(cmd)}")

        stdout = stdout_bytes.decode(errors="replace").strip()
        stderr = stderr_bytes.decode(errors="replace").strip()
        if proc.returncode != 0:
            raise AdbError(
                f"adb failed (exit {proc.returncode}): {' '.join(cmd)}"
                + (f"\nstderr: {stderr}" if stderr else ""),
                returncode=proc.returncode,
                stderr=stderr,
            )
        return stdout

    # ------------------------------------------------------------------
    # Preflight
    # ------------------------------------------------------------------

    async def preflight(self) -> None:
        cmd = [self._adb, "devices"]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
        except asyncio.TimeoutError:
            raise AdbError("'adb devices' timed out during preflight")

        output = stdout_bytes.decode(errors="replace")
        device_lines = [
            line for line in output.splitlines()
            if "\t" in line and not line.startswith("List of")
        ]

        if self._serial:
            serial_ready = False
            for line in device_lines:
                parts = line.split("\t", 1)
                if len(parts) == 2 and parts[0].strip() == self._serial:
                    if parts[1].strip() != "device":
                        raise AdbError(
                            f"Device {self._serial!r} in state {parts[1].strip()!r}, "
                            "not 'device'. Check authorisation."
                        )
                    serial_ready = True
                    break
            if not serial_ready:
                raise AdbError(f"Device {self._serial!r} not found in 'adb devices'.")
        else:
            ready = [line for line in device_lines if line.split("\t", 1)[-1].strip() == "device"]
            if not ready:
                raise AdbError("No Android device found. Connect a device or start an emulator.")

        if self._use_scrcpy:
            await self._ensure_scrcpy_started()

    # ------------------------------------------------------------------
    # App discovery
    # ------------------------------------------------------------------

    async def list_apps(self) -> list[str]:
        """Return package names of launchable apps on the device.

        Fetches third-party packages (``pm list packages -3``) and merges a
        small set of commonly-used system packages so the LLM can resolve
        human-readable names like "Settings" to ``com.android.settings``.
        """
        common_system_packages = [
            "com.android.settings",
            "com.android.contacts",
            "com.android.dialer",
            "com.android.mms",
            "com.android.camera2",
            "com.android.gallery3d",
            "com.android.calculator2",
            "com.android.calendar",
            "com.android.deskclock",
            "com.android.documentsui",
            "com.android.vending",
            "com.google.android.apps.messaging",
            "com.google.android.apps.photos",
            "com.google.android.gm",
            "com.google.android.googlequicksearchbox",
            "com.google.android.youtube",
            "com.google.android.apps.maps",
            "com.google.android.dialer",
            "com.google.android.contacts",
            "com.google.android.calendar",
        ]
        try:
            output = await self._run("shell", "pm", "list", "packages", "-3", timeout=10.0)
        except (AdbError, TimeoutError):
            return list(common_system_packages)

        packages: list[str] = []
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("package:"):
                packages.append(line[len("package:"):])
        # Merge common system packages (deduplicated, order preserved)
        seen = set(packages)
        for pkg in common_system_packages:
            if pkg not in seen:
                packages.append(pkg)
                seen.add(pkg)
        return packages

    # ------------------------------------------------------------------
    # Observe
    # ------------------------------------------------------------------

    async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
        if self._use_scrcpy:
            return await self._observe_via_scrcpy(screenshot_path, timeout=timeout)
        return await self._observe_via_screencap(screenshot_path, timeout=timeout)

    async def _observe_via_scrcpy(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
        await self._ensure_scrcpy_started()
        if self._frame_source is None:
            raise AdbError("ADB scrcpy capture is enabled but no frame source is configured.")

        snapshot = await self._capture_scrcpy_frame(screenshot_path, timeout=timeout)
        if snapshot is None:
            return await self._observe_via_screencap(screenshot_path, timeout=timeout)
        (input_width, input_height), fg_app = await asyncio.gather(
            self._query_screen_size(timeout),
            self._query_foreground_app(timeout),
        )
        self._capture_width = snapshot.width
        self._capture_height = snapshot.height
        self._screen_width = input_width
        self._screen_height = input_height
        return Observation(
            screenshot_path=str(screenshot_path),
            screen_width=snapshot.width,
            screen_height=snapshot.height,
            foreground_app=fg_app,
            platform=self.platform,
            extra={"capture_source": "scrcpy", "frame_timestamp": snapshot.timestamp},
        )

    async def _capture_scrcpy_frame(
        self,
        screenshot_path: Path,
        *,
        timeout: float,
        allow_restart: bool = True,
    ) -> ScrcpyFrameSnapshot | None:
        if self._frame_source is None:
            return None
        try:
            return await asyncio.to_thread(
                self._frame_source.save_latest,
                screenshot_path,
                timeout_s=min(timeout, self._scrcpy_frame_timeout_ms / 1000.0),
                max_age_s=self._scrcpy_max_frame_age_ms / 1000.0,
            )
        except TimeoutError:
            if allow_restart:
                try:
                    await self._restart_scrcpy_frame_source()
                except Exception:
                    return None
                try:
                    return await asyncio.to_thread(
                        self._frame_source.save_latest,
                        screenshot_path,
                        timeout_s=min(timeout, self._scrcpy_frame_timeout_ms / 1000.0),
                        max_age_s=self._scrcpy_max_frame_age_ms / 1000.0,
                    )
                except TimeoutError:
                    return None
            return None

    async def _restart_scrcpy_frame_source(self) -> None:
        if self._frame_source is None:
            raise AdbError("ADB scrcpy capture is enabled but no frame source is configured.")
        await asyncio.to_thread(self._frame_source.stop)
        self._scrcpy_started = False
        await self._ensure_scrcpy_started()

    async def _observe_via_screencap(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)

        await self._run("shell", "screencap", "-p", _DEVICE_SCREENSHOT_PATH, timeout=timeout)
        await self._run("pull", _DEVICE_SCREENSHOT_PATH, str(screenshot_path), timeout=timeout)

        screenshot_size = _read_png_size(screenshot_path)
        if screenshot_size is None:
            (width, height), fg_app = await asyncio.gather(
                self._query_screen_size(timeout),
                self._query_foreground_app(timeout),
            )
        else:
            width, height = screenshot_size
            fg_app = await self._query_foreground_app(timeout)

        self._screen_width = width
        self._screen_height = height
        self._capture_width = width
        self._capture_height = height

        return Observation(
            screenshot_path=str(screenshot_path),
            screen_width=width,
            screen_height=height,
            foreground_app=fg_app,
            platform=self.platform,
        )

    async def shutdown(self) -> None:
        if self._frame_source is not None:
            await asyncio.to_thread(self._frame_source.stop)
        self._scrcpy_started = False

    async def _ensure_scrcpy_started(self) -> None:
        if self._scrcpy_started:
            return
        if self._frame_source is None:
            raise AdbError("ADB scrcpy capture is enabled but no frame source is configured.")
        try:
            await asyncio.to_thread(self._frame_source.start)
        except Exception as exc:
            raise AdbError(f"ADB scrcpy frame source unavailable: {exc}") from exc
        self._scrcpy_started = True

    async def _query_screen_size(self, timeout: float) -> tuple[int, int]:
        try:
            output = await self._run("shell", "wm", "size", timeout=timeout)
            match = _WM_SIZE_RE.search(output)
            if match:
                return int(match.group(1)), int(match.group(2))
        except (AdbError, TimeoutError):
            pass
        return 1080, 1920

    async def _query_foreground_app(self, timeout: float) -> str:
        try:
            output = await self._run(
                "shell", "dumpsys", "activity", "activities",
                timeout=max(timeout, 10.0),
            )
            package = self._extract_foreground_app(output)
            if package != "unknown":
                return package
        except (AdbError, TimeoutError):
            pass

        for window_args in (("shell", "dumpsys", "window", "windows"), ("shell", "dumpsys", "window")):
            try:
                output = await self._run(*window_args, timeout=max(timeout, 10.0))
                package = self._extract_foreground_app(output)
                if package != "unknown":
                    return package
            except (AdbError, TimeoutError):
                continue
        return "unknown"

    @staticmethod
    def _extract_foreground_app(output: str) -> str:
        for pattern in (
            _RESUMED_ACTIVITY_RE,
            _TOP_RESUMED_ACTIVITY_RE,
            _CURRENT_FOCUS_RE,
            _FOCUSED_APP_RE,
        ):
            match = pattern.search(output)
            if match:
                return match.group(1)
        return "unknown"

    async def _get_default_input_method(self, timeout: float) -> str | None:
        try:
            output = await self._run(
                "shell", "settings", "get", "secure", "default_input_method",
                timeout=timeout,
            )
        except (AdbError, TimeoutError):
            return None
        output = output.strip()
        return output or None

    async def _list_input_methods(self, timeout: float) -> set[str]:
        try:
            output = await self._run("shell", "ime", "list", "-s", timeout=timeout)
        except (AdbError, TimeoutError):
            return set()
        return {line.strip() for line in output.splitlines() if line.strip()}

    async def _needs_ime_enable_before_set(self, timeout: float) -> bool:
        del timeout
        return False

    def _get_packaged_yadb_path(self) -> Path:
        return Path(importlib.resources.files("opengui").joinpath("assets/android/yadb"))

    def _write_local_temp_text(self, text: str) -> Path:
        fd, path = tempfile.mkstemp(prefix="opengui-yadb-", suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        return Path(path)

    def _make_yadb_device_text_path(self) -> str:
        return f"/data/local/tmp/opengui-yadb-{uuid.uuid4().hex}.txt"

    def _write_local_temp_yadb_script(self) -> Path:
        fd, path = tempfile.mkstemp(prefix="opengui-yadb-", suffix=".sh")
        script = (
            "#!/system/bin/sh\n"
            'text="$(cat "$1")"\n'
            f'app_process -Djava.class.path={_YADB_PATH} '
            f'/data/local/tmp {_YADB_MAIN_CLASS} -keyboard "$text"\n'
        )
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(script)
        return Path(path)

    def _make_yadb_device_script_path(self) -> str:
        return f"/data/local/tmp/opengui-yadb-{uuid.uuid4().hex}.sh"

    async def _ensure_yadb_available(self, timeout: float) -> bool:
        try:
            await self._run("shell", "ls", _YADB_PATH, timeout=timeout)
            return True
        except (AdbError, TimeoutError):
            pass

        local_yadb = self._get_packaged_yadb_path()
        if not local_yadb.exists():
            return False

        try:
            await self._run("push", str(local_yadb), _YADB_PATH, timeout=timeout)
            await self._run("shell", "chmod", "755", _YADB_PATH, timeout=timeout)
        except (AdbError, TimeoutError):
            return False
        return True

    async def _ensure_adb_keyboard_ready(self, timeout: float) -> bool:
        current_ime = await self._get_default_input_method(timeout)
        if current_ime == _ADB_KEYBOARD_IME:
            return True

        available_imes = await self._list_input_methods(timeout)
        if _ADB_KEYBOARD_IME not in available_imes:
            return False

        if await self._needs_ime_enable_before_set(timeout=timeout):
            try:
                await self._run("shell", "ime", "enable", _ADB_KEYBOARD_IME, timeout=timeout)
            except (AdbError, TimeoutError):
                return False

        try:
            await self._run("shell", "ime", "set", _ADB_KEYBOARD_IME, timeout=timeout)
        except (AdbError, TimeoutError):
            return False
        return True

    async def _input_text_via_adb_keyboard(self, text: str, timeout: float) -> bool:
        if not await self._ensure_adb_keyboard_ready(timeout):
            return False
        for args in (
            (
                "shell", "am", "broadcast",
                "-a", "ADB_INPUT_B64", "--es", "msg", _to_b64_text(text),
            ),
            (
                "shell", "am", "broadcast",
                "-a", "ADB_INPUT_TEXT", "--es", "msg", text,
            ),
        ):
            try:
                await self._run(*args, timeout=timeout)
                return True
            except (AdbError, TimeoutError):
                continue
        return False

    async def _input_text_via_yadb(self, text: str, timeout: float) -> bool:
        if not await self._ensure_yadb_available(timeout):
            return False
        local_text_path = self._write_local_temp_text(text)
        device_text_path = self._make_yadb_device_text_path()
        local_script_path = self._write_local_temp_yadb_script()
        device_script_path = self._make_yadb_device_script_path()
        try:
            await self._run("push", str(local_text_path), device_text_path, timeout=timeout)
            await self._run("push", str(local_script_path), device_script_path, timeout=timeout)
            await self._run("shell", "chmod", "755", device_script_path, timeout=timeout)
            await self._run(
                "shell",
                "sh",
                device_script_path,
                device_text_path,
                timeout=timeout,
            )
            return True
        except (AdbError, TimeoutError):
            return False
        finally:
            try:
                local_text_path.unlink(missing_ok=True)
            except OSError:
                pass
            try:
                local_script_path.unlink(missing_ok=True)
            except OSError:
                pass
            try:
                await self._run("shell", "rm", "-f", device_text_path, timeout=timeout)
            except (AdbError, TimeoutError):
                pass
            try:
                await self._run("shell", "rm", "-f", device_script_path, timeout=timeout)
            except (AdbError, TimeoutError):
                pass

    async def _input_single_text(self, text: str, timeout: float) -> None:
        if await self._input_text_via_yadb(text, timeout):
            return
        if await self._input_text_via_adb_keyboard(text, timeout):
            return
        if _is_ascii_safe(text):
            await self._run(
                "shell", "input", "text", _escape_shell_text(text),
                timeout=timeout,
            )
            return
        raise AdbError(
            "Unicode text input failed. Install and activate ADBKeyboard "
            "(`adb shell ime set com.android.adbkeyboard/.AdbIME`) or "
            f"push yadb to {_YADB_PATH}."
        )

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    async def execute(self, action: Action, timeout: float = 5.0) -> str:
        t = action.action_type

        if t == "tap":
            x, y = self._resolve_point(action)
            await self._run(
                "shell", "input", "tap", str(x), str(y),
                timeout=timeout,
            )

        elif t == "long_press":
            px, py = self._resolve_point(action)
            x, y = str(px), str(py)
            dur = str(action.duration_ms or 800)
            await self._run("shell", "input", "swipe", x, y, x, y, dur, timeout=timeout)

        elif t == "double_tap":
            px, py = self._resolve_point(action)
            x, y = str(px), str(py)
            await self._run("shell", "input", "tap", x, y, timeout=timeout)
            await asyncio.sleep(0.1)
            await self._run("shell", "input", "tap", x, y, timeout=timeout)

        elif t in ("drag", "swipe"):
            px1, py1 = self._resolve_point(action)
            px2, py2 = self._resolve_second_point(action)
            x1, y1 = str(px1), str(py1)
            x2, y2 = str(px2), str(py2)
            dur = str(action.duration_ms or 300)
            await self._run("shell", "input", "swipe", x1, y1, x2, y2, dur, timeout=timeout)

        elif t == "input_text":
            text = action.text or ""
            if text:
                normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
                lines = normalized_text.split("\n")
                for index, line in enumerate(lines):
                    for segment in _iter_text_input_segments(line):
                        await self._input_single_text(segment, timeout)
                    if index < len(lines) - 1:
                        await self._run(
                            "shell", "input", "keyevent", "KEYCODE_ENTER",
                            timeout=timeout,
                        )
                if action.auto_enter:
                    await self._run(
                        "shell", "input", "keyevent", "KEYCODE_ENTER",
                        timeout=timeout,
                    )

        elif t == "enter":
            await self._run("shell", "input", "keyevent", "KEYCODE_ENTER", timeout=timeout)

        elif t in ("app_switch", "recents"):
            await self._run("shell", "input", "keyevent", "KEYCODE_APP_SWITCH", timeout=timeout)

        elif t == "hotkey":
            keys = action.key or []
            keycodes: list[str] = []
            for k in keys:
                keycode = _KEYCODE_MAP.get(k.lower().strip())
                if keycode is None:
                    raise ValueError(
                        f"Unknown key {k!r}. Supported: {sorted(_KEYCODE_MAP.keys())}"
                    )
                keycodes.append(keycode)

            if len(keycodes) >= 2:
                try:
                    await self._run("shell", "input", "keycombination", *keycodes, timeout=timeout)
                except AdbError as exc:
                    details = f"{exc}\n{exc.stderr}" if exc.stderr else str(exc)
                    lowered = details.lower()
                    if any(marker in lowered for marker in _KEYCOMBINATION_UNSUPPORTED_MARKERS):
                        raise AdbError(
                            "Device does not support simultaneous multi-key hotkeys via "
                            "`adb shell input keycombination`."
                        ) from exc
                    raise
            else:
                await self._run("shell", "input", "keyevent", keycodes[0], timeout=timeout)

        elif t == "scroll":
            await self._do_scroll(action, timeout=timeout)

        elif t == "wait":
            await asyncio.sleep((action.duration_ms or 1000) / 1000.0)

        elif t == "back":
            await self._run("shell", "input", "keyevent", "KEYCODE_BACK", timeout=timeout)

        elif t == "home":
            await self._run("shell", "input", "keyevent", "KEYCODE_HOME", timeout=timeout)

        elif t == "done":
            pass  # terminal action, no device command

        elif t == "open_app":
            pkg = action.text or ""
            if pkg:
                await self._run(
                    "shell", "monkey", "-p", pkg,
                    "-c", "android.intent.category.LAUNCHER", "1",
                    timeout=timeout,
                )

        elif t == "close_app":
            pkg = action.text or ""
            if pkg:
                await self._run("shell", "am", "force-stop", pkg, timeout=timeout)

        else:
            raise ValueError(f"Unsupported action type: {t!r}")

        return describe_action(action)

    def _resolve_x(self, value: float, *, relative: bool) -> int:
        if relative:
            return resolve_coordinate(value, self._screen_width, relative=True)
        return self._resolve_capture_coordinate(
            value,
            capture_extent=self._capture_width,
            input_extent=self._screen_width,
        )

    def _resolve_y(self, value: float, *, relative: bool) -> int:
        if relative:
            return resolve_coordinate(value, self._screen_height, relative=True)
        return self._resolve_capture_coordinate(
            value,
            capture_extent=self._capture_height,
            input_extent=self._screen_height,
        )

    @staticmethod
    def _resolve_capture_coordinate(value: float, *, capture_extent: int, input_extent: int) -> int:
        """Map screenshot-pixel coordinates into the ADB input coordinate space."""
        if capture_extent > 1 and capture_extent != input_extent:
            pixel = round(value / (capture_extent - 1) * (input_extent - 1))
        else:
            pixel = round(value)
        return max(0, min(pixel, input_extent - 1))

    def _resolve_point(self, action: Action) -> tuple[int, int]:
        if action.x is None or action.y is None:
            raise ValueError(f"Action {action.action_type!r} requires coordinates.")
        return (
            self._resolve_x(action.x, relative=action.relative),
            self._resolve_y(action.y, relative=action.relative),
        )

    def _resolve_second_point(self, action: Action) -> tuple[int, int]:
        if action.x2 is None or action.y2 is None:
            raise ValueError(f"Action {action.action_type!r} requires end-point coordinates.")
        return (
            self._resolve_x(action.x2, relative=action.relative),
            self._resolve_y(action.y2, relative=action.relative),
        )

    async def _do_scroll(self, action: Action, *, timeout: float) -> None:
        """Simulate scroll via a short swipe gesture."""
        x = self._screen_width // 2
        y = self._screen_height // 2
        if action.x is not None and action.y is not None:
            x = self._resolve_x(action.x, relative=action.relative)
            y = self._resolve_y(action.y, relative=action.relative)
        pixels = abs(action.pixels or 200)
        dur = str(action.duration_ms or 300)

        # For scroll, direction is stored in action.text
        direction = (action.text or "down").lower()

        # Touch scrolls are inverted relative to content direction:
        # "down" means reveal lower content, so the finger moves upward.
        if direction == "up":
            x2, y2 = x, y + pixels
        elif direction == "down":
            x2, y2 = x, y - pixels
        elif direction == "left":
            x2, y2 = x - pixels, y
        else:
            x2, y2 = x + pixels, y

        await self._run(
            "shell", "input", "swipe",
            str(x), str(y), str(x2), str(y2), dur,
            timeout=timeout,
        )
