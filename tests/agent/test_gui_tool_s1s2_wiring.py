from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import GuiConfig
from opengui.agent import AgentResult


def _provider() -> MagicMock:
    provider = MagicMock()
    provider.api_key = "test-key"
    provider.api_base = None
    provider.extra_headers = None
    provider._client = None
    provider.get_default_model.return_value = "gui-default"
    return provider


def _named_provider(name: str) -> MagicMock:
    provider = _provider()
    provider.name = name
    provider.get_default_model.return_value = f"{name}-default"
    return provider


def _tool(gui_config: GuiConfig, tmp_path: Path, **kwargs: Any):
    from nanobot.agent.tools.gui import GuiSubagentTool

    return GuiSubagentTool(
        gui_config=gui_config,
        provider=_provider(),
        model="legacy-gui",
        workspace=tmp_path,
        **kwargs,
    )


def test_gui_subagent_tool_uses_s1_model_override(tmp_path: Path) -> None:
    tool = _tool(GuiConfig(backend="dry-run", s1_model="small-gui"), tmp_path)

    assert tool._s1_model == "small-gui"
    assert tool._s2_llm_adapter is None


@pytest.mark.asyncio
async def test_gui_subagent_tool_passes_no_s2_llm_when_disabled(
    tmp_path: Path,
) -> None:
    tool = _tool(GuiConfig(backend="dry-run", s2_enabled=False, s2_model="slow-gui"), tmp_path)
    captured_kwargs: dict[str, Any] = {}

    def fake_init(self, *args: Any, **kwargs: Any) -> None:
        del self, args
        captured_kwargs.update(kwargs)

    with (
        patch("nanobot.agent.tools.gui.GuiAgent.__init__", fake_init),
        patch("opengui.agent.GuiAgent.run", new_callable=AsyncMock) as mock_run,
    ):
        mock_run.return_value = AgentResult(success=True, summary="ok")
        payload = json.loads(await tool._run_task(tool._backend, "open settings"))

    assert payload["success"] is True
    assert captured_kwargs["s2_llm"] is None
    assert captured_kwargs["s2_enabled"] is False


@pytest.mark.asyncio
async def test_gui_subagent_tool_passes_s2_llm_when_enabled(
    tmp_path: Path,
) -> None:
    gui_config = GuiConfig(
        backend="dry-run",
        s2_enabled=True,
        s2_model="slow-gui",
        s2_hint_enabled=False,
        s2_takeover_enabled=True,
        s2_max_hints=0,
        s2_max_takeover_steps=5,
    )
    tool = _tool(gui_config, tmp_path, s2_provider=None)
    captured_kwargs: dict[str, Any] = {}

    def fake_init(self, *args: Any, **kwargs: Any) -> None:
        del self, args
        captured_kwargs.update(kwargs)

    with (
        patch("nanobot.agent.tools.gui.GuiAgent.__init__", fake_init),
        patch("opengui.agent.GuiAgent.run", new_callable=AsyncMock) as mock_run,
    ):
        mock_run.return_value = AgentResult(success=True, summary="ok")
        payload = json.loads(await tool._run_task(tool._backend, "open settings"))

    assert payload["success"] is True
    assert tool._s2_llm_adapter is not None
    assert captured_kwargs["s2_llm"] is tool._s2_llm_adapter
    assert captured_kwargs["s2_model"] == "slow-gui"
    assert captured_kwargs["s2_enabled"] is True
    assert captured_kwargs["s2_hint_enabled"] is False
    assert captured_kwargs["s2_takeover_enabled"] is True
    assert captured_kwargs["s2_max_hints"] == 0
    assert captured_kwargs["s2_max_takeover_steps"] == 5


def test_agent_loop_passes_independent_s2_provider_to_gui_tool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    s1_provider = _named_provider("s1")
    s2_provider = _named_provider("s2")
    captured_kwargs: dict[str, Any] = {}

    class FakeGuiTool:
        name = "gui_task"

        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr("nanobot.agent.tools.gui.GuiSubagentTool", FakeGuiTool)

    AgentLoop(
        bus=MessageBus(),
        provider=_provider(),
        workspace=tmp_path,
        model="main-model",
        gui_config=GuiConfig(
            backend="dry-run",
            provider="qwen_9b",
            model="qwen3.5-9b",
            s2_enabled=True,
            s2_provider="qwen_35b",
            s2_model="qwen3.6-35b-a3b",
        ),
        gui_provider=s1_provider,
        gui_model="qwen3.5-9b",
        gui_s2_provider=s2_provider,
        gui_s2_model="qwen3.6-35b-a3b",
    )

    assert captured_kwargs["provider"] is s1_provider
    assert captured_kwargs["model"] == "qwen3.5-9b"
    assert captured_kwargs["s2_provider"] is s2_provider
    assert captured_kwargs["s2_model"] == "qwen3.6-35b-a3b"
