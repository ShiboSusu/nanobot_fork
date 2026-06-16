from __future__ import annotations

import json
import socket
from unittest.mock import patch

import pytest

from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.config.schema import WebSearchConfig


def _fake_resolve_public(hostname, port, family=0, type_=0):
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]


@pytest.mark.asyncio
async def test_web_search_defaults_to_three_results_even_when_config_is_larger(monkeypatch):
    seen: dict[str, int] = {}

    async def fake_search(self, query: str, n: int) -> str:
        seen["n"] = n
        return "ok"

    monkeypatch.setattr(WebSearchTool, "_search_duckduckgo", fake_search)

    tool = WebSearchTool(config=WebSearchConfig(provider="duckduckgo", max_results=10))

    assert await tool.execute(query="深圳 亲子 周末") == "ok"
    assert seen["n"] == 3


@pytest.mark.asyncio
async def test_web_fetch_returns_structured_summary_under_default_budget(monkeypatch):
    body = "".join(f"<p>Relevant family point {idx} with useful detail.</p>" for idx in range(80))
    fake_html = f"<html><head><title>Family Guide</title></head><body>{body}</body></html>"

    class FakeStreamResponse:
        headers = {"content-type": "text/html"}
        url = "https://example.com/family"

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeResponse:
        status_code = 200
        url = "https://example.com/family"
        text = fake_html
        headers = {"content-type": "text/html"}

        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, headers=None):
            return FakeStreamResponse()

        async def get(self, url, headers=None):
            return FakeResponse()

    monkeypatch.setattr("nanobot.agent.tools.web.httpx.AsyncClient", FakeClient)
    tool = WebFetchTool()

    with patch("nanobot.security.network.socket.getaddrinfo", _fake_resolve_public):
        result = await tool.execute(url="https://example.com/family")

    data = json.loads(result)
    assert data["title"] == "Family Guide"
    assert data["source"] == "https://example.com/family"
    assert data["raw_length"] > data["returned_length"]
    assert data["returned_length"] <= 2000
    assert data["relevant_points"]
    assert "Relevant family point 79" not in json.dumps(data, ensure_ascii=False)
    assert "text" not in data
