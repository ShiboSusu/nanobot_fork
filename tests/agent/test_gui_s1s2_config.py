from __future__ import annotations

from nanobot.config.schema import GuiConfig


def test_gui_config_s2_defaults_disabled() -> None:
    cfg = GuiConfig()

    assert cfg.s2_enabled is False
    assert cfg.s2_model is None
    assert cfg.s2_provider is None


def test_gui_config_s2_model_can_be_configured() -> None:
    cfg = GuiConfig(s2_enabled=True, s2_provider="openrouter", s2_model="slow-gui")

    assert cfg.s2_enabled is True
    assert cfg.s2_provider == "openrouter"
    assert cfg.s2_model == "slow-gui"


def test_gui_config_s1_backcompat_falls_back_to_legacy_gui_model() -> None:
    cfg = GuiConfig(provider="vllm", model="gui-9b")

    assert (cfg.s1_provider or cfg.provider) == "vllm"
    assert (cfg.s1_model or cfg.model) == "gui-9b"


def test_gui_config_s1_overrides_are_optional() -> None:
    cfg = GuiConfig(
        provider="vllm",
        model="gui-9b",
        s1_provider="dashscope",
        s1_model="gui-small",
    )

    assert (cfg.s1_provider or cfg.provider) == "dashscope"
    assert (cfg.s1_model or cfg.model) == "gui-small"


def test_gui_config_s2_accepts_camel_case_aliases() -> None:
    cfg = GuiConfig.model_validate(
        {
            "s2Enabled": True,
            "s2Provider": "openrouter",
            "s2Model": "slow-gui",
            "s2MaxHints": 2,
            "s2TakeoverAfterHints": 1,
            "s2MaxTakeoverSteps": 6,
        }
    )

    assert cfg.s2_enabled is True
    assert cfg.s2_provider == "openrouter"
    assert cfg.s2_model == "slow-gui"
    assert cfg.s2_max_hints == 2
    assert cfg.s2_takeover_after_hints == 1
    assert cfg.s2_max_takeover_steps == 6
