from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from opengui.action import Action
from opengui.agent import GuiAgent, StepResult
from opengui.interfaces import LLMResponse
from opengui.observation import Observation
from opengui.trajectory.recorder import TrajectoryRecorder


class _NoopLLM:
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        del messages, tools, tool_choice, model, max_tokens
        return LLMResponse(content="")


class _HintLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.max_tokens_seen: list[int | None] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        del messages, tools, tool_choice, model
        self.max_tokens_seen.append(max_tokens)
        return LLMResponse(content=self.content)


class _StaticBackend:
    platform = "dry-run"

    async def preflight(self) -> None:
        return None

    async def observe(self, screenshot_path: Path, timeout: float = 5.0) -> Observation:
        del timeout
        _write_png(screenshot_path)
        return Observation(
            screenshot_path=str(screenshot_path),
            screen_width=64,
            screen_height=64,
            foreground_app="DryRun",
            platform=self.platform,
        )

    async def execute(self, action: Action, timeout: float = 5.0) -> str:
        del action, timeout
        return "ok"

    async def list_apps(self) -> list[str]:
        return []


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=(90, 90, 90)).save(path, format="PNG")


def _recorder(tmp_path: Path) -> TrajectoryRecorder:
    return TrajectoryRecorder(output_dir=tmp_path / "trajectory", task="s2 test")


def _agent(tmp_path: Path) -> GuiAgent:
    return GuiAgent(
        _NoopLLM(),
        _StaticBackend(),
        trajectory_recorder=_recorder(tmp_path),
        artifacts_root=tmp_path / "runs",
    )


def _step_result(
    *,
    step_index: int,
    current_observation: Observation,
    done: bool = False,
) -> StepResult:
    next_observation = None
    if not done:
        run_dir = Path(current_observation.screenshot_path or ".").parent.parent
        screenshot = run_dir / "screenshots" / f"step_{step_index:03d}.png"
        _write_png(screenshot)
        next_observation = Observation(
            screenshot_path=str(screenshot),
            screen_width=64,
            screen_height=64,
            foreground_app="DryRun",
            platform="dry-run",
        )
    action = Action(action_type="done" if done else "wait", status="success" if done else None)
    return StepResult(
        action=action,
        tool_call_id=f"call-{step_index}",
        tool_result="done" if done else "waited",
        assistant_message={"role": "assistant", "content": "done" if done else "wait"},
        action_summary="done" if done else "wait",
        state_summary="done" if done else "same screen",
        next_observation=next_observation,
        prompt_snapshot={},
        model_snapshot={},
        execution_snapshot={},
        done=done,
    )


def test_build_s2_takeover_context_includes_failure_history(tmp_path: Path) -> None:
    agent = _agent(tmp_path)

    text = agent._build_s2_takeover_context(
        task="打开微博，查看热搜第三名",
        trigger_reason="stagnation_streak=1 reached stagnation_limit=1",
        stagnation_streak=1,
        stagnation_limit=1,
        recent_history=[
            {
                "actor": "s1",
                "action_summary": "repeated tapping search box",
                "state_summary": "screen did not change",
            }
        ],
        s2_guidance_notes=["Avoid repeating the search box tap."],
    )

    assert "S2 takeover context" in text
    assert "taking over from the small GUI executor" in text
    assert "Trigger: stagnation" in text
    assert "repeated tapping search box" in text
    assert "screen did not change" in text
    assert "Do not repeat the last failed action pattern" in text
    assert "Continue from the current screen" in text
    assert "/no_think" in text


def test_takeover_context_does_not_include_raw_model_response(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    recent = agent._recent_history_for_takeover(
        [
            {
                "actor": "s1",
                "action_summary": "tap search",
                "state_summary": "same screen",
                "raw_response_content": "<very long raw response>",
            }
        ]
    )

    text = agent._build_s2_takeover_context(
        task="recover",
        trigger_reason="stagnation",
        stagnation_streak=1,
        stagnation_limit=1,
        recent_history=recent,
        s2_guidance_notes=[],
    )

    assert "tap search" in text
    assert "same screen" in text
    assert "raw_response_content" not in text
    assert "<very long raw response>" not in text


@pytest.mark.asyncio
async def test_s2_hint_uses_small_token_budget_and_extracts_json_after_thinking(
    tmp_path: Path,
) -> None:
    slow_llm = _HintLLM(
        '<think>reason privately</think>\n'
        '{"diagnosis":"stuck","next_subgoal":"open search",'
        '"avoid":["repeat tap"],"stop_condition":"result visible"}'
    )
    agent = GuiAgent(
        _NoopLLM(),
        _StaticBackend(),
        trajectory_recorder=_recorder(tmp_path),
        artifacts_root=tmp_path / "runs",
        max_steps=3,
        stagnation_limit=1,
        s2_llm=slow_llm,
        s2_enabled=True,
    )
    screenshot = tmp_path / "screen.png"
    _write_png(screenshot)

    hint = await agent._request_s2_hint(
        task="recover",
        step_index=1,
        current_observation=Observation(
            screenshot_path=str(screenshot),
            screen_width=64,
            screen_height=64,
            foreground_app="DryRun",
            platform="dry-run",
        ),
        history=[],
        last_action_summary=None,
        reason="same screen",
    )

    assert slow_llm.max_tokens_seen == [180]
    assert hint is not None
    assert "diagnosis: stuck" in hint
    assert "next_subgoal: open search" in hint
    assert "avoid: repeat tap" in hint


@pytest.mark.asyncio
async def test_stagnation_can_trigger_s2_hint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hint_calls: list[dict[str, Any]] = []
    messages_seen: list[list[dict[str, Any]]] = []

    async def fake_hint(self: GuiAgent, **kwargs: Any) -> str:
        del self
        hint_calls.append(kwargs)
        return "diagnosis: same screen\nnext_subgoal: try a different control"

    async def fake_run_step(
        self: GuiAgent,
        messages: list[dict[str, Any]],
        prompt_snapshot: dict[str, Any] | None,
        step_index: int,
        total_steps: int,
        current_observation: Observation,
        *,
        llm_override: Any = None,
        actor: str = "s1",
    ) -> StepResult:
        del self, prompt_snapshot, total_steps, llm_override, actor
        messages_seen.append(messages)
        return _step_result(
            step_index=step_index,
            current_observation=current_observation,
            done=step_index == 2,
        )

    monkeypatch.setattr(GuiAgent, "_request_s2_hint", fake_hint)
    monkeypatch.setattr(GuiAgent, "_run_step", fake_run_step)
    agent = GuiAgent(
        _NoopLLM(),
        _StaticBackend(),
        trajectory_recorder=_recorder(tmp_path),
        artifacts_root=tmp_path / "runs",
        max_steps=3,
        stagnation_limit=1,
        s2_llm=_NoopLLM(),
        s2_enabled=True,
        s2_hint_enabled=True,
        s2_takeover_enabled=False,
    )

    result = await agent.run("recover from stagnation", max_retries=1)

    assert result.success is True
    assert result.s2_usage["hints_used"] == 1
    assert result.s2_usage["takeover_used"] is False
    assert result.s2_usage["triggers"][0]["trigger"] == "stagnation"
    assert result.s2_usage["triggers"][0]["mode"] == "hint"
    assert hint_calls[0]["reason"].startswith("Detected unchanged screen state")
    assert any("Slow-model recovery hint" in str(message.get("content")) for message in messages_seen[1])


@pytest.mark.asyncio
async def test_second_stagnation_triggers_s2_takeover(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slow_llm = _NoopLLM()
    hint_calls: list[dict[str, Any]] = []
    actors: list[str] = []
    overrides: list[Any] = []
    captured: list[dict[str, Any]] = []

    async def fake_hint(self: GuiAgent, **kwargs: Any) -> str:
        del self
        hint_calls.append(kwargs)
        return "try a different entry"

    async def fake_run_step(
        self: GuiAgent,
        messages: list[dict[str, Any]],
        prompt_snapshot: dict[str, Any] | None,
        step_index: int,
        total_steps: int,
        current_observation: Observation,
        *,
        llm_override: Any = None,
        actor: str = "s1",
    ) -> StepResult:
        del self, prompt_snapshot, total_steps
        actors.append(actor)
        overrides.append(llm_override)
        captured.append(
            {
                "actor": actor,
                "llm_override": llm_override,
                "messages": messages,
            }
        )
        return _step_result(
            step_index=step_index,
            current_observation=current_observation,
            done=step_index == 3,
        )

    monkeypatch.setattr(GuiAgent, "_request_s2_hint", fake_hint)
    monkeypatch.setattr(GuiAgent, "_run_step", fake_run_step)
    agent = GuiAgent(
        _NoopLLM(),
        _StaticBackend(),
        trajectory_recorder=_recorder(tmp_path),
        artifacts_root=tmp_path / "runs",
        max_steps=4,
        stagnation_limit=1,
        s2_llm=slow_llm,
        s2_enabled=True,
        s2_hint_enabled=True,
        s2_takeover_enabled=True,
        s2_max_hints=1,
        s2_takeover_after_hints=1,
        s2_max_takeover_steps=2,
    )

    result = await agent.run("recover by takeover", max_retries=1)

    assert result.success is True
    assert hint_calls
    assert actors == ["s1", "s1", "s2_takeover"]
    assert overrides == [None, None, slow_llm]
    assert result.s2_usage["hints_used"] == 1
    assert result.s2_usage["takeover_used"] is True
    assert [event["mode"] for event in result.s2_usage["triggers"]] == ["hint", "takeover"]
    assert result.s2_usage["takeover_steps"] == 1
    assert result.s2_usage["s2_steps"] == 1

    takeover_calls = [item for item in captured if item["actor"] == "s2_takeover"]
    assert takeover_calls
    takeover_text = "\n".join(str(message.get("content", "")) for message in takeover_calls[0]["messages"])
    assert "S2 takeover context" in takeover_text
    assert "Trigger: stagnation" in takeover_text
    assert "Do not repeat the last failed action pattern" in takeover_text
    assert "Continue from the current screen" in takeover_text
    assert "/no_think" in takeover_text

    for item in captured:
        if item["actor"] != "s1":
            continue
        text = "\n".join(str(message.get("content", "")) for message in item["messages"])
        assert "S2 takeover context" not in text


@pytest.mark.asyncio
async def test_takeover_budget_returns_control_to_s1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    slow_llm = _NoopLLM()
    actors: list[str] = []

    async def fake_hint(self: GuiAgent, **kwargs: Any) -> str:
        del self, kwargs
        return "try a different entry"

    async def fake_run_step(
        self: GuiAgent,
        messages: list[dict[str, Any]],
        prompt_snapshot: dict[str, Any] | None,
        step_index: int,
        total_steps: int,
        current_observation: Observation,
        *,
        llm_override: Any = None,
        actor: str = "s1",
    ) -> StepResult:
        del self, messages, prompt_snapshot, total_steps, llm_override
        actors.append(actor)
        result = _step_result(
            step_index=step_index,
            current_observation=current_observation,
            done=step_index == 4,
        )
        if actor == "s2_takeover" and result.next_observation is not None:
            run_dir = Path(current_observation.screenshot_path or ".").parent.parent
            screenshot = run_dir / "screenshots" / f"step_{step_index:03d}_changed.png"
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 64), color=(20, 140, 220)).save(screenshot, format="PNG")
            result = replace(
                result,
                next_observation=Observation(
                    screenshot_path=str(screenshot),
                    screen_width=64,
                    screen_height=64,
                    foreground_app="DryRun",
                    platform="dry-run",
                ),
            )
        return result

    monkeypatch.setattr(GuiAgent, "_request_s2_hint", fake_hint)
    monkeypatch.setattr(GuiAgent, "_run_step", fake_run_step)
    agent = GuiAgent(
        _NoopLLM(),
        _StaticBackend(),
        trajectory_recorder=_recorder(tmp_path),
        artifacts_root=tmp_path / "runs",
        max_steps=5,
        stagnation_limit=1,
        s2_llm=slow_llm,
        s2_enabled=True,
        s2_hint_enabled=True,
        s2_takeover_enabled=True,
        s2_max_hints=1,
        s2_takeover_after_hints=1,
        s2_max_takeover_steps=1,
    )

    result = await agent.run("recover then return", max_retries=1)

    assert result.success is True
    assert "s2_takeover" in actors
    takeover_index = actors.index("s2_takeover")
    assert any(actor == "s1" for actor in actors[takeover_index + 1:])
    assert result.s2_usage["takeover_steps"] == 1
