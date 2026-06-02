from __future__ import annotations

from nanobot.config.schema import Config, GuiConfig
from nanobot.providers.factory import (
    build_gui_planner_provider_snapshot,
    build_gui_provider_snapshot,
    build_gui_s2_provider_snapshot,
    build_provider_snapshot,
)


def test_gui_config_accepts_explicit_s2_model_and_provider() -> None:
    cfg = GuiConfig.model_validate({
        "model": "qwen3.5-9b",
        "provider": "vllm_9b",
        "s2Model": "qwen3.5-397b-a17b",
        "s2Provider": "custom",
        "plannerEnabled": True,
        "plannerModel": "qwen3.6-35b-a3b",
        "plannerProvider": "vllm_35b",
        "plannerSubtasksEnabled": True,
        "plannerConfidenceThreshold": 0.7,
        "plannerMaxTokens": 768,
        "plannerTimeoutSeconds": 6,
    })

    assert cfg.model == "qwen3.5-9b"
    assert cfg.provider == "vllm_9b"
    assert cfg.s2_model == "qwen3.5-397b-a17b"
    assert cfg.s2_provider == "custom"
    assert cfg.planner_enabled is True
    assert cfg.planner_model == "qwen3.6-35b-a3b"
    assert cfg.planner_provider == "vllm_35b"
    assert cfg.planner_subtasks_enabled is True
    assert cfg.planner_confidence_threshold == 0.7
    assert cfg.planner_max_tokens == 768
    assert cfg.planner_timeout_seconds == 6


def test_gui_config_disables_planner_subtasks_by_default() -> None:
    assert GuiConfig().planner_subtasks_enabled is False


def test_gui_config_default_planner_timeout_does_not_block_simple_gui_tasks() -> None:
    assert GuiConfig().planner_timeout_seconds == 8.0


def test_provider_snapshots_keep_main_gui_s1_and_gui_s2_separate() -> None:
    cfg = Config.model_validate({
        "agents": {
            "defaults": {
                "model": "qwen3.6-35b-a3b",
                "provider": "vllm_35b",
            }
        },
        "providers": {
            "vllm_35b": {"apiBase": "http://127.0.0.1:18001/v1"},
            "vllm_9b": {"apiBase": "http://127.0.0.1:18000/v1"},
            "custom": {"apiBase": "http://192.168.200.27/v1"},
        },
        "gui": {
            "model": "qwen3.5-9b",
            "provider": "vllm_9b",
            "s2Model": "qwen3.5-397b-a17b",
            "s2Provider": "custom",
            "plannerEnabled": True,
            "plannerModel": "qwen3.6-35b-a3b",
            "plannerProvider": "vllm_35b",
        },
    })

    main = build_provider_snapshot(cfg)
    planner = build_gui_planner_provider_snapshot(cfg)
    gui_s1 = build_gui_provider_snapshot(cfg)
    gui_s2 = build_gui_s2_provider_snapshot(cfg)

    assert main.model == "qwen3.6-35b-a3b"
    assert main.provider.api_base == "http://127.0.0.1:18001/v1"
    assert planner is not None
    assert planner.model == "qwen3.6-35b-a3b"
    assert planner.provider.api_base == "http://127.0.0.1:18001/v1"
    assert gui_s1 is not None
    assert gui_s1.model == "qwen3.5-9b"
    assert gui_s1.provider.api_base == "http://127.0.0.1:18000/v1"
    assert gui_s2 is not None
    assert gui_s2.model == "qwen3.5-397b-a17b"
    assert gui_s2.provider.api_base == "http://192.168.200.27/v1"
