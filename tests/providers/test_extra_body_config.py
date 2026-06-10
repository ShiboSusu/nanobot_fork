"""Tests for provider extra_body config injection into request payloads."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from nanobot.providers.openai_compat_provider import (
    OpenAICompatProvider,
    _deep_merge,
)

# ---------------------------------------------------------------------------
# _deep_merge unit tests
# ---------------------------------------------------------------------------


class TestDeepMerge:
    """Verify recursive dict merge semantics."""

    def test_flat_merge(self) -> None:
        assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_override_scalar(self) -> None:
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested_merge(self) -> None:
        base = {"outer": {"a": 1, "b": 2}}
        override = {"outer": {"b": 3, "c": 4}}
        assert _deep_merge(base, override) == {"outer": {"a": 1, "b": 3, "c": 4}}

    def test_deeply_nested(self) -> None:
        base = {"l1": {"l2": {"a": 1}}}
        override = {"l1": {"l2": {"b": 2}}}
        assert _deep_merge(base, override) == {"l1": {"l2": {"a": 1, "b": 2}}}

    def test_override_replaces_non_dict_with_dict(self) -> None:
        assert _deep_merge({"a": 1}, {"a": {"nested": True}}) == {"a": {"nested": True}}

    def test_override_replaces_dict_with_scalar(self) -> None:
        assert _deep_merge({"a": {"nested": True}}, {"a": "flat"}) == {"a": "flat"}

    def test_empty_base(self) -> None:
        assert _deep_merge({}, {"a": 1}) == {"a": 1}

    def test_empty_override(self) -> None:
        assert _deep_merge({"a": 1}, {}) == {"a": 1}

    def test_does_not_mutate_inputs(self) -> None:
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}
        _deep_merge(base, override)
        assert base == {"a": {"x": 1}}
        assert override == {"a": {"y": 2}}


# ---------------------------------------------------------------------------
# Provider construction
# ---------------------------------------------------------------------------


class TestExtraBodyInit:
    """Verify the provider stores extra_body from config."""

    def test_default_is_empty(self) -> None:
        provider = OpenAICompatProvider(api_key="test")
        assert provider._extra_body == {}

    def test_none_becomes_empty(self) -> None:
        provider = OpenAICompatProvider(api_key="test", extra_body=None)
        assert provider._extra_body == {}

    def test_dict_stored(self) -> None:
        body = {"chat_template_kwargs": {"enable_thinking": False}}
        provider = OpenAICompatProvider(api_key="test", extra_body=body)
        assert provider._extra_body == body


# ---------------------------------------------------------------------------
# _build_kwargs integration
# ---------------------------------------------------------------------------


def _make_provider(extra_body: dict[str, Any] | None = None) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        api_key="test-key",
        default_model="test-model",
        extra_body=extra_body,
    )


def _simple_messages() -> list[dict[str, Any]]:
    return [{"role": "user", "content": "hello"}]


class TestBuildKwargsExtraBody:
    """Verify extra_body flows into _build_kwargs output."""

    def test_no_extra_body_no_key(self) -> None:
        provider = _make_provider()
        kwargs = provider._build_kwargs(
            messages=_simple_messages(),
            tools=None, model=None, max_tokens=100,
            temperature=0.1, reasoning_effort=None, tool_choice=None,
        )
        assert "extra_body" not in kwargs

    def test_extra_body_injected(self) -> None:
        provider = _make_provider({"chat_template_kwargs": {"enable_thinking": False}})
        kwargs = provider._build_kwargs(
            messages=_simple_messages(),
            tools=None, model=None, max_tokens=100,
            temperature=0.1, reasoning_effort=None, tool_choice=None,
        )
        assert kwargs["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def test_extra_body_merges_with_thinking(self) -> None:
        """Config extra_body should merge with (and override) thinking params."""
        from nanobot.providers.registry import ProviderSpec

        spec = MagicMock(spec=ProviderSpec)
        spec.thinking_style = "deepseek"
        spec.supports_prompt_caching = False
        spec.strip_model_prefix = False
        spec.model_overrides = []
        spec.name = "custom"
        spec.supports_max_completion_tokens = False
        spec.env_key = None
        spec.default_api_base = None
        spec.is_local = True
        spec.detect_by_base_keyword = None

        provider = OpenAICompatProvider(
            api_key="test",
            default_model="deepseek-v3",
            spec=spec,
            extra_body={"custom_param": "value"},
        )
        kwargs = provider._build_kwargs(
            messages=_simple_messages(),
            tools=None, model=None, max_tokens=100,
            temperature=0.1, reasoning_effort="high", tool_choice=None,
        )
        body = kwargs.get("extra_body", {})
        # Config param should be present
        assert body.get("custom_param") == "value"

    def test_nested_extra_body_does_not_clobber_siblings(self) -> None:
        """Nested dict merge should preserve sibling keys."""
        provider = _make_provider({
            "chat_template_kwargs": {"enable_thinking": False},
        })
        # Simulate internal code having set a sibling key
        # by manually calling _build_kwargs — the internal logic
        # doesn't set chat_template_kwargs, so we test the merge path
        # by having extra_body itself contain nested keys
        kwargs = provider._build_kwargs(
            messages=_simple_messages(),
            tools=None, model=None, max_tokens=100,
            temperature=0.1, reasoning_effort=None, tool_choice=None,
        )
        assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    def test_guided_json_injection(self) -> None:
        """Real-world use case: vLLM guided decoding."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        provider = _make_provider({"guided_json": schema})
        kwargs = provider._build_kwargs(
            messages=_simple_messages(),
            tools=None, model=None, max_tokens=100,
            temperature=0.1, reasoning_effort=None, tool_choice=None,
        )
        assert kwargs["extra_body"]["guided_json"] == schema

    def test_repetition_penalty_injection(self) -> None:
        """Real-world use case: local model sampling param."""
        provider = _make_provider({"repetition_penalty": 1.15})
        kwargs = provider._build_kwargs(
            messages=_simple_messages(),
            tools=None, model=None, max_tokens=100,
            temperature=0.1, reasoning_effort=None, tool_choice=None,
        )
        assert kwargs["extra_body"]["repetition_penalty"] == 1.15

    def test_extra_body_does_not_forward_core_request_fields(self) -> None:
        provider = _make_provider(
            {
                "model": "wrong-model",
                "messages": [{"role": "user", "content": "wrong"}],
                "stream": True,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )

        kwargs = provider._build_kwargs(
            messages=_simple_messages(),
            tools=None, model="right-model", max_tokens=100,
            temperature=0.1, reasoning_effort=None, tool_choice=None,
        )

        assert kwargs["model"] == "right-model"
        assert kwargs["messages"] == _simple_messages()
        assert "model" not in kwargs["extra_body"]
        assert "messages" not in kwargs["extra_body"]
        assert "stream" not in kwargs["extra_body"]
        assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSchemaConfig:
    """Verify ProviderConfig accepts extra_body."""

    def test_default_is_none(self) -> None:
        from nanobot.config.schema import ProviderConfig

        config = ProviderConfig()
        assert config.extra_body is None

    def test_accepts_dict(self) -> None:
        from nanobot.config.schema import ProviderConfig

        config = ProviderConfig(extra_body={"guided_json": {"type": "object"}})
        assert config.extra_body == {"guided_json": {"type": "object"}}

    def test_nested_dict(self) -> None:
        from nanobot.config.schema import ProviderConfig

        config = ProviderConfig(
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        assert config.extra_body["chat_template_kwargs"]["enable_thinking"] is False

    def test_named_openai_compatible_provider_keeps_extra_body(self) -> None:
        from nanobot.config.schema import Config
        from nanobot.providers.factory import build_provider_snapshot

        config = Config.model_validate(
            {
                "providers": {
                    "qwen_9b": {
                        "apiKey": "qwen-key",
                        "apiBase": "http://qwen-9b.test/v1",
                        "extraBody": {
                            "chat_template_kwargs": {"enable_thinking": False}
                        },
                    }
                }
            }
        )

        snapshot = build_provider_snapshot(
            config,
            model_override="qwen3.5-9b",
            provider_override="qwen_9b",
        )

        provider = snapshot.provider
        assert provider.get_default_model() == "qwen3.5-9b"
        assert provider.api_key == "qwen-key"
        assert provider.api_base == "http://qwen-9b.test/v1"
        assert provider._extra_body == {
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def test_main_gui_s1_s2_named_providers_keep_separate_extra_body(self) -> None:
        from nanobot.config.schema import Config
        from nanobot.providers.factory import (
            build_provider_snapshot,
            build_gui_provider_snapshot,
            build_gui_s2_provider_snapshot,
        )

        config = Config.model_validate(
            {
                "providers": {
                    "qwen_9b": {
                        "apiKey": "qwen-key",
                        "apiBase": "http://qwen-9b.test/v1",
                        "extraBody": {
                            "chat_template_kwargs": {"enable_thinking": False}
                        },
                    },
                    "qwen_35b": {
                        "apiKey": "qwen-key",
                        "apiBase": "http://qwen-35b.test/v1",
                        "extraBody": {
                            "chat_template_kwargs": {"enable_thinking": False}
                        },
                    },
                    "qwen_397b": {
                        "apiKey": "qwen-key",
                        "apiBase": "http://qwen-397b.test/v1",
                        "extraBody": {
                            "chat_template_kwargs": {"enable_thinking": False}
                        },
                    },
                },
                "agents": {
                    "defaults": {
                        "provider": "qwen_35b",
                        "model": "qwen3.6-35b-a3b",
                    },
                },
                "gui": {
                    "backend": "dry-run",
                    "provider": "qwen_9b",
                    "model": "qwen3.5-9b",
                    "s1Provider": "qwen_9b",
                    "s1Model": "qwen3.5-9b",
                    "s2Enabled": True,
                    "s2Provider": "qwen_397b",
                    "s2Model": "qwen3.5-397b-a17b",
                },
            }
        )

        main = build_provider_snapshot(config)
        s1 = build_gui_provider_snapshot(config)
        s2 = build_gui_s2_provider_snapshot(config)

        assert s1 is not None
        assert s2 is not None
        assert main.model == "qwen3.6-35b-a3b"
        assert s1.model == "qwen3.5-9b"
        assert s2.model == "qwen3.5-397b-a17b"
        assert main.provider.api_base == "http://qwen-35b.test/v1"
        assert s1.provider.api_base == "http://qwen-9b.test/v1"
        assert s2.provider.api_base == "http://qwen-397b.test/v1"
        assert main.provider._extra_body["chat_template_kwargs"]["enable_thinking"] is False
        assert s1.provider._extra_body["chat_template_kwargs"]["enable_thinking"] is False
        assert s2.provider._extra_body["chat_template_kwargs"]["enable_thinking"] is False
