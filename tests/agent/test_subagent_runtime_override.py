"""Tests for subagent model/provider runtime overrides."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from nanobot.providers.base import GenerationSettings


def _make_main_provider() -> MagicMock:
    provider = MagicMock()
    provider.get_default_model.return_value = "main-model"
    provider.generation = GenerationSettings(
        temperature=0.2,
        max_tokens=2048,
        reasoning_effort="low",
    )
    return provider


def _build_loop(tmp_path, provider):
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    with patch.object(AgentLoop, "_register_default_tools", lambda _self: None):
        return AgentLoop(
            bus=MessageBus(),
            provider=provider,
            workspace=tmp_path,
            model="main-model",
        )


def test_subagent_runtime_defaults_to_main_provider(tmp_path, monkeypatch):
    for key in (
        "NANOBOT_SUBAGENT_MODEL",
        "NANOBOT_SUBAGENT_API_BASE",
        "NANOBOT_SUBAGENT_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_API_BASE",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    provider = _make_main_provider()
    loop = _build_loop(tmp_path, provider)

    assert loop.subagent_provider is provider
    assert loop.subagent_model == "main-model"
    assert loop.subagents.provider is provider
    assert loop.subagents.model == "main-model"


def test_subagent_runtime_uses_openai_compatible_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_BASE", "http://127.0.0.1:8001/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_MODEL", "Qwen3.5-9B-Instruct")

    provider = _make_main_provider()

    with patch("nanobot.providers.custom_provider.AsyncOpenAI") as mock_async_openai:
        loop = _build_loop(tmp_path, provider)

    assert loop.subagent_provider is not provider
    assert loop.subagent_model == "Qwen3.5-9B-Instruct"
    assert loop.subagents.provider is loop.subagent_provider
    assert loop.subagents.model == "Qwen3.5-9B-Instruct"
    assert loop.subagent_provider.get_default_model() == "Qwen3.5-9B-Instruct"
    assert loop.subagent_provider.generation == provider.generation

    kwargs = mock_async_openai.call_args.kwargs
    assert kwargs["api_key"] == "dummy"
    assert kwargs["base_url"] == "http://127.0.0.1:8001/v1"

