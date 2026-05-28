"""
opengui.backends.ios_wda
~~~~~~~~~~~~~~~~~~~~~~~~
iOS device backend for automation via WebDriverAgent (WDA).

Requires the ``facebook-wda`` Python client (``pip install facebook-wda``).
WDA must be running on the connected device and accessible at *wda_url*
(default ``http://localhost:8100``; adjust for USB-forwarded or Wi-Fi WDA).

All blocking WDA calls are wrapped in ``asyncio.to_thread`` so the backend
stays non-blocking inside the async agent loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable

from opengui.action import Action, describe_action, resolve_coordinate
from opengui.observation import Observation

logger = logging.getLogger(__name__)


def _format_ios_app_entry(bundle_id: str, display: str | None = None) -> str:
    bundle_id = str(bundle_id).strip()
    display = str(display).strip() if display else ""
    if display and display != bundle_id:
        return f"{display}: {bundle_id}"
    return bundle_id


def _bundle_from_ios_app_entry(entry: str) -> str:
    cleaned = str(entry).strip()
    for separator in (": ", "：", ":"):
        if separator in cleaned:
            candidate = cleaned.rsplit(separator, 1)[-1].strip()
            if "." in candidate and " " not in candidate:
                return candidate
    return cleaned


def _ios_app_entry_has_display(entry: str) -> bool:
    return _bundle_from_ios_app_entry(entry) != str(entry).strip()


def _ios_app_display_from_metadata(metadata: Any) -> str | None:
    if not isinstance(metadata, dict):
        return None
    for key in (
        "CFBundleDisplayName",
        "CFBundleName",
        "displayName",
        "display_name",
        "localizedName",
        "name",
    ):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _json_payload_from_text(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start_candidates = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
        if not start_candidates:
            raise
        start = min(start_candidates)
        end = max(text.rfind("}"), text.rfind("]"))
        if end <= start:
            raise
        return json.loads(text[start:end + 1])


def _import_wda() -> Any:
    """Lazily import the ``wda`` package with a helpful error on failure."""
    try:
        import wda  # type: ignore[import-untyped]  # noqa: PLC0415
        return wda
    except ImportError as exc:
        raise ImportError(
            "The 'facebook-wda' package is required for iOS backend. "
            "Install it with: pip install facebook-wda"
        ) from exc

# ---------------------------------------------------------------------------
# iOS key mapping (limited hardware vs. Android)
# ---------------------------------------------------------------------------

_IOS_KEYCODE_MAP: dict[str, str] = {
    "home": "home",
    "volumeup": "volumeUp",
    "volume_up": "volumeUp",
    "volumedown": "volumeDown",
    "volume_down": "volumeDown",
}


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class WdaError(Exception):
    """Raised when a WDA operation fails or the device is unreachable."""

    def __init__(self, message: str, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause


# ---------------------------------------------------------------------------
# Backend implementation
# ---------------------------------------------------------------------------


class WdaBackend:
    """iOS device backend that drives WebDriverAgent via the ``wda`` package.

    Args:
        wda_url: HTTP base URL for the running WDA server on the target device.
                 Default ``http://localhost:8100`` (standard USB-forwarded port).
    """

    def __init__(self, wda_url: str = "http://localhost:8100") -> None:
        self._wda_url = wda_url
        wda = _import_wda()
        self._client: Any = wda.Client(wda_url)
        # Sensible iPhone defaults; updated on first observe() call.
        self._screen_width: int = 375
        self._screen_height: int = 812

    @property
    def platform(self) -> str:
        return "ios"

    # ------------------------------------------------------------------
    # Async helper
    # ------------------------------------------------------------------

    async def _wda_call(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run a blocking WDA call on the default thread-pool executor."""
        return await asyncio.to_thread(fn, *args)

    # ------------------------------------------------------------------
    # Preflight
    # ------------------------------------------------------------------

    async def preflight(self) -> None:
        """Verify WDA connectivity and log device info."""
        try:
            status = await self._wda_call(self._client.status)
        except Exception as exc:
            raise WdaError(
                f"Cannot reach WDA at {self._wda_url!r}: {exc}", cause=exc
            ) from exc

        # Log device / OS information when available.
        if isinstance(status, dict):
            build = status.get("build") or {}
            info = status.get("value") or {}
            product_version = (
                build.get("productVersion")
                or info.get("os", {}).get("version")
                or "unknown"
            )
            model = info.get("model") or build.get("productType") or "unknown"
            logger.info(
                "WDA preflight OK — model=%s OS=%s url=%s",
                model, product_version, self._wda_url,
            )
        else:
            logger.info("WDA preflight OK — url=%s", self._wda_url)

    # ------------------------------------------------------------------
    # App discovery
    # ------------------------------------------------------------------

    async def list_apps(self) -> list[str]:
        """Return launchable iOS apps as bundle IDs or ``"Display: bundle"``.

        WDA builds differ in what ``app_list()`` returns, so we merge it with a
        best-effort ``pymobiledevice3 apps list`` result.  Display names are kept
        when available because the agent's resolver can then map natural app
        names to the device's actual bundle IDs.
        """
        entries_by_bundle: dict[str, str] = {}
        for entry in await self._list_apps_via_wda():
            bundle_id = _bundle_from_ios_app_entry(entry)
            if bundle_id:
                entries_by_bundle[bundle_id] = entry
        for entry in await self._list_apps_via_pymobiledevice3():
            bundle_id = _bundle_from_ios_app_entry(entry)
            if not bundle_id:
                continue
            if bundle_id not in entries_by_bundle or _ios_app_entry_has_display(entry):
                entries_by_bundle[bundle_id] = entry
        return list(entries_by_bundle.values())

    async def _list_apps_via_wda(self) -> list[str]:
        try:
            result = await self._wda_call(self._client.app_list)
            if isinstance(result, list):
                entries: list[str] = []
                for entry in result:
                    if isinstance(entry, dict):
                        bid = entry.get("bundleId") or entry.get("bundle_id") or entry.get("id")
                        if bid:
                            entries.append(_format_ios_app_entry(str(bid), _ios_app_display_from_metadata(entry)))
                    elif isinstance(entry, str):
                        entries.append(entry)
                return entries
        except Exception as exc:
            logger.warning("WDA app_list() unavailable (%s); trying pymobiledevice3.", exc)
        return []

    async def _list_apps_via_pymobiledevice3(self) -> list[str]:
        cmd = ("pymobiledevice3", "apps", "list")
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            logger.debug("pymobiledevice3 not found; iOS app list will use WDA only.")
            return []

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=25.0)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            logger.warning("pymobiledevice3 apps list timed out; using WDA app list only.")
            return []

        if proc.returncode != 0:
            logger.warning(
                "pymobiledevice3 apps list failed (%s): %s",
                proc.returncode,
                stderr.decode(errors="replace").strip(),
            )
            return []

        text = stdout.decode(errors="replace")
        try:
            payload = _json_payload_from_text(text)
        except json.JSONDecodeError as exc:
            logger.warning("Could not parse pymobiledevice3 apps list JSON: %s", exc)
            return []

        entries: list[str] = []
        if isinstance(payload, dict):
            iterable = payload.items()
        elif isinstance(payload, list):
            iterable = ((None, item) for item in payload)
        else:
            return []

        for key, metadata in iterable:
            if isinstance(metadata, dict):
                bundle_id = (
                    metadata.get("CFBundleIdentifier")
                    or metadata.get("bundleId")
                    or metadata.get("bundle_id")
                    or key
                )
                if bundle_id:
                    entries.append(
                        _format_ios_app_entry(
                            str(bundle_id),
                            _ios_app_display_from_metadata(metadata),
                        )
                    )
            elif isinstance(key, str):
                entries.append(key)
        return entries

    # ------------------------------------------------------------------
    # Observe
    # ------------------------------------------------------------------

    async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
        """Capture screen state: screenshot, dimensions, and foreground app."""
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)

        # Run screenshot, window size, and foreground app in parallel.
        screenshot_img, window_size, app_info = await asyncio.gather(
            self._wda_call(self._client.screenshot),
            self._wda_call(self._client.window_size),
            self._wda_call(self._client.app_current),
        )

        # Persist screenshot as PNG.
        screenshot_img.save(str(screenshot_path), format="PNG")
        screenshot_size: tuple[int, int] | None = None
        raw_size = getattr(screenshot_img, "size", None)
        if isinstance(raw_size, (tuple, list)) and len(raw_size) >= 2:
            shot_w = int(raw_size[0])
            shot_h = int(raw_size[1])
            if shot_w > 0 and shot_h > 0:
                screenshot_size = shot_w, shot_h

        # Unpack window dimensions (wda may return a namedtuple or dict).
        if hasattr(window_size, "width") and hasattr(window_size, "height"):
            width, height = int(window_size.width), int(window_size.height)
        elif isinstance(window_size, (tuple, list)) and len(window_size) >= 2:
            width, height = int(window_size[0]), int(window_size[1])
        elif isinstance(window_size, dict):
            width = int(window_size.get("width", self._screen_width))
            height = int(window_size.get("height", self._screen_height))
        else:
            width, height = self._screen_width, self._screen_height
            logger.warning("Unexpected window_size format %r; using cached %dx%d", window_size, width, height)

        if screenshot_size is not None:
            shot_w, shot_h = screenshot_size
            window_is_landscape = width > height
            screenshot_is_landscape = shot_w > shot_h
            if (
                width != height
                and shot_w != shot_h
                and window_is_landscape != screenshot_is_landscape
            ):
                width, height = height, width

        self._screen_width = width
        self._screen_height = height

        # Extract bundle ID of the foreground app.
        if isinstance(app_info, dict):
            bundle_id = str(app_info.get("bundleId") or app_info.get("bundle_id") or "unknown")
        else:
            bundle_id = "unknown"

        return Observation(
            screenshot_path=str(screenshot_path),
            screen_width=width,
            screen_height=height,
            foreground_app=bundle_id,
            platform=self.platform,
        )

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    async def execute(self, action: Action, timeout: float = 5.0) -> str:
        """Dispatch a single action to the iOS device via WDA."""
        session = await self._wda_call(self._client.session)
        t = action.action_type

        if t == "tap":
            x, y = self._resolve_point(action)
            await self._wda_call(session.tap, x, y)

        elif t == "long_press":
            x, y = self._resolve_point(action)
            duration_s = (action.duration_ms or 800) / 1000.0
            await self._wda_call(session.tap_hold, x, y, duration_s)

        elif t == "double_tap":
            x, y = self._resolve_point(action)
            await self._wda_call(session.double_tap, x, y)

        elif t in ("drag", "swipe"):
            x1, y1 = self._resolve_point(action)
            x2, y2 = self._resolve_second_point(action)
            duration_s = (action.duration_ms or 300) / 1000.0
            await self._wda_call(session.swipe, x1, y1, x2, y2, duration_s)

        elif t == "scroll":
            await self._do_scroll(action, session=session)

        elif t == "input_text":
            text = action.text or ""
            if text:
                await self._wda_call(session.send_keys, text)
            if action.auto_enter:
                await self._wda_call(session.send_keys, "\n")

        elif t == "hotkey":
            keys = action.key or []
            for k in keys:
                mapped = _IOS_KEYCODE_MAP.get(k.lower().strip())
                if mapped is None:
                    raise ValueError(
                        f"Unknown iOS key {k!r}. Supported: {sorted(_IOS_KEYCODE_MAP.keys())}"
                    )
                if mapped == "home":
                    await self._wda_call(session.home)
                else:
                    # Volume keys and others via send_keys
                    await self._wda_call(session.send_keys, mapped)

        elif t == "wait":
            await asyncio.sleep((action.duration_ms or 1000) / 1000.0)

        elif t == "back":
            # iOS has no physical back button — simulate with a left-edge swipe gesture.
            w = self._screen_width
            h = self._screen_height
            await self._wda_call(session.swipe, 0, h // 2, w // 3, h // 2, 0.3)

        elif t == "home":
            await self._wda_call(session.home)

        elif t == "done":
            pass  # terminal marker, no device command needed

        elif t == "open_app":
            bundle_id = action.text or ""
            if bundle_id:
                await self._wda_call(session.app_launch, bundle_id)

        elif t == "close_app":
            bundle_id = action.text or ""
            if bundle_id:
                await self._wda_call(session.app_terminate, bundle_id)

        else:
            raise ValueError(f"Unsupported action type: {t!r}")

        return describe_action(action)

    # ------------------------------------------------------------------
    # Coordinate helpers  (mirror AdbBackend)
    # ------------------------------------------------------------------

    def _resolve_x(self, value: float, *, relative: bool) -> int:
        return resolve_coordinate(value, self._screen_width, relative=relative)

    def _resolve_y(self, value: float, *, relative: bool) -> int:
        return resolve_coordinate(value, self._screen_height, relative=relative)

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

    async def _do_scroll(self, action: Action, *, session: Any) -> None:
        """Simulate a scroll via a directional swipe gesture."""
        x = self._screen_width // 2
        y = self._screen_height // 2
        if action.x is not None and action.y is not None:
            x = self._resolve_x(action.x, relative=action.relative)
            y = self._resolve_y(action.y, relative=action.relative)

        pixels = abs(action.pixels or 200)
        duration_s = (action.duration_ms or 300) / 1000.0
        direction = (action.text or "down").lower()

        # Touch scrolls are inverted relative to content direction:
        # "down" means reveal lower content, so the finger moves upward.
        if direction == "up":
            x2, y2 = x, y + pixels
        elif direction == "down":
            x2, y2 = x, y - pixels
        elif direction == "left":
            x2, y2 = x - pixels, y
        else:  # right
            x2, y2 = x + pixels, y

        await self._wda_call(session.swipe, x, y, x2, y2, duration_s)
