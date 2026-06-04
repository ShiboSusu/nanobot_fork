from __future__ import annotations

from pathlib import Path

import pytest

import opengui.backends.ios_wda as ios_wda_module

PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4"
    b"\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05"
    b"\xfe\x02\xfeA\xe2`\x82\x00\x00\x00\x00IEND\xaeB`\x82"
)


class _FakeImage:
    size = (402, 874)

    def save(self, path: str, format: str = "PNG") -> None:
        del format
        Path(path).write_bytes(PNG_1X1)


class _FakeSession:
    def source(self) -> str:
        return """<?xml version="1.0" encoding="UTF-8"?>
        <AppiumAUT>
          <XCUIElementTypeApplication visible="true" label="携程旅行">
            <XCUIElementTypeStaticText visible="true" label="上海 → 广州" x="145" y="87" width="120" height="35"/>
            <XCUIElementTypeStaticText visible="true" label="明天 06-05" x="92" y="125" width="80" height="70"/>
            <XCUIElementTypeStaticText visible="true" label="07:05" x="48" y="351" width="80" height="40"/>
            <XCUIElementTypeStaticText visible="true" label="¥320" x="330" y="355" width="55" height="40"/>
            <XCUIElementTypeStaticText visible="false" label="隐藏文案"/>
          </XCUIElementTypeApplication>
        </AppiumAUT>
        """


class _FakeClient:
    def __init__(self, url: str) -> None:
        self.url = url

    def screenshot(self) -> _FakeImage:
        return _FakeImage()

    def window_size(self) -> dict[str, int]:
        return {"width": 402, "height": 874}

    def app_current(self) -> dict[str, str]:
        return {"bundleId": "ctrip.com"}

    def session(self) -> _FakeSession:
        return _FakeSession()


class _FakeWdaModule:
    Client = _FakeClient


@pytest.mark.asyncio
async def test_wda_observe_populates_accessibility_visible_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(ios_wda_module, "_import_wda", lambda: _FakeWdaModule)
    backend = ios_wda_module.WdaBackend()

    observation = await backend.observe(tmp_path / "screen.png")

    assert observation.screenshot_path == str(tmp_path / "screen.png")
    assert observation.foreground_app == "ctrip.com"
    visible_text = observation.extra["visible_text"]
    assert "上海 → 广州" in visible_text
    assert "明天 06-05" in visible_text
    assert "¥320" in visible_text
    assert "隐藏文案" not in visible_text
    assert observation.extra["state_summary"] == visible_text
    assert any(
        node["label"] == "上海 → 广州" and node["x"] == 145.0
        for node in observation.extra["ui_tree"]
    )
