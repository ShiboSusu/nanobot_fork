from __future__ import annotations

import base64

import pytest

from eval.phase0_observable_signal_pilot import (
    attach_entropy_samples_from_prompt_events,
    enrich_monitor_features,
    rehydrate_prompt_messages,
)


MINIMAL_PNG = b"\x89PNG\r\n\x1a\n"


def test_rehydrate_prompt_messages_replaces_omitted_image_url(tmp_path) -> None:
    screenshot_path = tmp_path / "screen.png"
    screenshot_path.write_bytes(MINIMAL_PNG)
    event = {
        "prompt": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Step 1"},
                        {"type": "image_url", "image_url": {"url": "<omitted:image-data-url>"}},
                    ],
                }
            ],
            "current_observation": {"screenshot_path": str(screenshot_path)},
        }
    }

    messages = rehydrate_prompt_messages(event)

    image_url = messages[0]["content"][1]["image_url"]["url"]
    assert image_url == f"data:image/png;base64,{base64.b64encode(MINIMAL_PNG).decode('ascii')}"
    assert event["prompt"]["messages"][0]["content"][1]["image_url"]["url"] == "<omitted:image-data-url>"


@pytest.mark.asyncio
async def test_attach_entropy_samples_keeps_executed_action_unchanged(tmp_path) -> None:
    screenshot_path = tmp_path / "screen.png"
    screenshot_path.write_bytes(MINIMAL_PNG)
    steps = [
        {
            "step_index": 0,
            "action": {"action_type": "tap", "x": 500.0, "y": 500.0, "relative": True},
            "trigger_features": {"self_report": {"confidence": 0.95}, "execution_state": {}},
        }
    ]
    prompt_events = [
        {
            "prompt": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Step 1"},
                            {"type": "image_url", "image_url": {"url": "<omitted:image-data-url>"}},
                        ],
                    }
                ],
                "current_observation": {"screenshot_path": str(screenshot_path)},
            }
        }
    ]
    original_action = dict(steps[0]["action"])
    sampler_calls = []

    async def sample_actions(messages, count):
        sampler_calls.append((messages, count))
        return [
            {"action_type": "tap", "x": 500.0, "y": 500.0, "relative": True},
            {"action_type": "swipe", "x": 500.0, "y": 800.0, "x2": 500.0, "y2": 200.0, "relative": True},
        ]

    await attach_entropy_samples_from_prompt_events(
        steps,
        prompt_events,
        sample_count=2,
        sample_actions=sample_actions,
    )
    enrich_monitor_features(steps, max_steps=5)

    assert steps[0]["action"] == original_action
    assert len(sampler_calls) == 1
    assert sampler_calls[0][1] == 2
    assert sampler_calls[0][0][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    entropy = steps[0]["trigger_features"]["entropy"]
    assert entropy["sample_count"] == 2
    assert entropy["sample_disagreement"] == pytest.approx(0.5)
    assert entropy["action_type_disagreement"] == pytest.approx(0.5)
