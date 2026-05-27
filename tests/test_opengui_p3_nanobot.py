"""Phase 3 nanobot integration tests.

Wave 0 starts with xfail stubs for the full phase boundary, then later tasks in
this plan promote the adapter/config coverage to real passing tests.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

try:
    from opengui.interfaces import LLMProvider as OpenGuiLLMProvider
    from opengui.interfaces import LLMResponse as OpenGuiLLMResponse
    from opengui.interfaces import ToolCall as OpenGuiToolCall
    from opengui.memory.retrieval import EmbeddingProvider as OpenGuiEmbeddingProvider
except Exception as exc:  # pragma: no cover - only used if imports break
    OpenGuiLLMProvider = None
    OpenGuiLLMResponse = None
    OpenGuiToolCall = None
    OpenGuiEmbeddingProvider = None
    _OPENGUI_IMPORT_ERROR: Exception | None = exc
else:
    _OPENGUI_IMPORT_ERROR = None

try:
    from nanobot.config.schema import Config
    from nanobot.providers.base import LLMProvider as NanobotLLMProvider
    from nanobot.providers.base import LLMResponse as NanobotLLMResponse
    from nanobot.providers.base import ToolCallRequest
except Exception as exc:  # pragma: no cover - only used if imports break
    Config = None
    NanobotLLMProvider = object
    NanobotLLMResponse = None
    ToolCallRequest = None
    _NANOBOT_IMPORT_ERROR: Exception | None = exc
else:
    _NANOBOT_IMPORT_ERROR = None


@dataclass
class _QueuedResponse:
    response: Any


def _nanobot_tool_response(
    *,
    content: str,
    arguments: dict[str, Any],
    call_id: str,
) -> Any:
    return NanobotLLMResponse(
        content=content,
        tool_calls=[
            ToolCallRequest(
                id=call_id,
                name="computer_use",
                arguments=arguments,
            )
        ],
    )


if _NANOBOT_IMPORT_ERROR is None:

    class _MockNanobotProvider(NanobotLLMProvider):
        """Minimal nanobot LLM provider for adapter tests."""

        def __init__(self, responses: list[Any]) -> None:
            super().__init__(api_key="test-key")
            self._responses = [_QueuedResponse(response=r) for r in responses]
            self.calls: list[dict[str, Any]] = []

        async def chat(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            model: str | None = None,
            max_tokens: int = 4096,
            temperature: float = 0.7,
            reasoning_effort: str | None = None,
            tool_choice: str | dict[str, Any] | None = None,
        ) -> Any:
            return await self.chat_with_retry(
                messages=messages,
                tools=tools,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                tool_choice=tool_choice,
            )

        async def chat_with_retry(self, messages, tools=None, model=None, **kwargs) -> Any:
            self.calls.append(
                {
                    "messages": messages,
                    "tools": tools,
                    "model": model,
                    **kwargs,
                }
            )
            if not self._responses:
                raise AssertionError("No scripted nanobot responses left")
            return self._responses.pop(0).response

        def get_default_model(self) -> str:
            return "test-model"

else:

    class _MockNanobotProvider:
        def __init__(self, responses: list[Any]) -> None:
            raise RuntimeError("nanobot provider imports unavailable") from _NANOBOT_IMPORT_ERROR


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    (tmp_path / "gui_runs").mkdir()
    (tmp_path / "gui_skills").mkdir()
    return tmp_path


def test_gui_tool_registered(tmp_workspace: Path) -> None:
    from nanobot.agent.tools.gui import GuiSubagentTool
    from nanobot.agent.tools.registry import ToolRegistry

    provider = _MockNanobotProvider([])
    registry = ToolRegistry()
    tool = GuiSubagentTool(
        gui_config=Config(gui={"backend": "dry-run"}).gui,
        provider=provider,
        model=provider.get_default_model(),
        workspace=tmp_workspace,
    )
    registry.register(tool)

    assert tool.name == "gui_task"
    assert tool.parameters["required"] == ["task"]
    assert tool.parameters["properties"]["backend"]["enum"] == [
        "adb",
        "ios",
        "hdc",
        "local",
        "dry-run",
    ]
    definitions = registry.get_definitions()
    assert any(defn["function"]["name"] == tool.name for defn in definitions)
    assert tool.description


def test_agent_loop_registers_gui_tool_with_gui_runtime_override(tmp_workspace: Path) -> None:
    from nanobot.agent.loop import AgentLoop

    bus = MagicMock()
    bus.publish_outbound = AsyncMock()
    main_provider = MagicMock()
    main_provider.get_default_model.return_value = "main-model"
    main_provider.generation = SimpleNamespace(max_tokens=4096)
    gui_provider = MagicMock()
    gui_s2_provider = MagicMock()

    with patch("nanobot.agent.tools.gui.GuiSubagentTool", return_value=MagicMock()) as mock_gui_tool:
        AgentLoop(
            bus=bus,
            provider=main_provider,
            workspace=tmp_workspace,
            model="main-model",
            gui_config=Config(gui={"backend": "dry-run"}).gui,
            gui_provider=gui_provider,
            gui_model="gui-model",
            gui_s2_provider=gui_s2_provider,
            gui_s2_model="s2-model",
        )

    kwargs = mock_gui_tool.call_args.kwargs
    assert kwargs["provider"] is gui_provider
    assert kwargs["model"] == "gui-model"
    assert kwargs["s2_provider"] is gui_s2_provider
    assert kwargs["s2_model"] == "s2-model"


@pytest.mark.asyncio
async def test_llm_adapter_maps_response() -> None:
    from nanobot.agent.gui_adapter import NanobotLLMAdapter

    provider = _MockNanobotProvider(
        [
            NanobotLLMResponse(
                content="tap settings",
                tool_calls=[
                    ToolCallRequest(
                        id="tc1",
                        name="computer_use",
                        arguments={"action_type": "tap", "x": 100, "y": 200},
                    )
                ],
            )
        ]
    )
    adapter = NanobotLLMAdapter(provider=provider, model=provider.get_default_model())

    result = await adapter.chat(messages=[{"role": "user", "content": "Open Settings"}])

    assert isinstance(adapter, OpenGuiLLMProvider)
    assert isinstance(result, OpenGuiLLMResponse)
    assert result.content == "tap settings"
    assert result.tool_calls is not None
    assert isinstance(result.tool_calls[0], OpenGuiToolCall)
    assert result.tool_calls[0] == OpenGuiToolCall(
        id="tc1",
        name="computer_use",
        arguments={"action_type": "tap", "x": 100, "y": 200},
    )


@pytest.mark.asyncio
async def test_llm_adapter_empty_tool_calls() -> None:
    from nanobot.agent.gui_adapter import NanobotLLMAdapter

    provider = _MockNanobotProvider([NanobotLLMResponse(content="done", tool_calls=[])])
    adapter = NanobotLLMAdapter(provider=provider, model=provider.get_default_model())

    result = await adapter.chat(messages=[{"role": "user", "content": "Finish"}])

    assert result.tool_calls is None


@pytest.mark.asyncio
async def test_llm_adapter_content_none() -> None:
    from nanobot.agent.gui_adapter import NanobotLLMAdapter

    provider = _MockNanobotProvider([NanobotLLMResponse(content=None, tool_calls=[])])
    adapter = NanobotLLMAdapter(provider=provider, model=provider.get_default_model())

    result = await adapter.chat(messages=[{"role": "user", "content": "Continue"}])

    assert result.content == ""


@pytest.mark.asyncio
async def test_llm_adapter_raw_preserved() -> None:
    from nanobot.agent.gui_adapter import NanobotLLMAdapter

    response = NanobotLLMResponse(content="done", tool_calls=[])
    provider = _MockNanobotProvider([response])
    adapter = NanobotLLMAdapter(provider=provider, model=provider.get_default_model())

    result = await adapter.chat(messages=[{"role": "user", "content": "Finish"}])

    assert result.raw is response


@pytest.mark.asyncio
async def test_llm_adapter_tool_choice_passthrough() -> None:
    from nanobot.agent.gui_adapter import NanobotLLMAdapter

    provider = _MockNanobotProvider([NanobotLLMResponse(content="done", tool_calls=[])])
    adapter = NanobotLLMAdapter(provider=provider, model=provider.get_default_model())
    tools = [{"type": "function", "function": {"name": "computer_use"}}]

    await adapter.chat(
        messages=[{"role": "user", "content": "Use the computer"}],
        tools=tools,
        tool_choice="required",
    )

    assert provider.calls[0]["tool_choice"] == "required"
    assert provider.calls[0]["model"] == "test-model"
    assert provider.calls[0]["tools"] == tools


@pytest.mark.asyncio
async def test_embedding_adapter() -> None:
    from nanobot.agent.gui_adapter import NanobotEmbeddingAdapter

    calls: list[list[str]] = []

    async def embed_fn(texts: list[str]) -> np.ndarray:
        calls.append(texts)
        return np.array([[1.0, 2.0]], dtype=np.float32)

    adapter = NanobotEmbeddingAdapter(embed_fn)
    result = await adapter.embed(["hello"])

    assert isinstance(adapter, OpenGuiEmbeddingProvider)
    assert calls == [["hello"]]
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == (1, 2)


def test_backend_selection(tmp_workspace: Path) -> None:
    from nanobot.agent.tools.gui import GuiSubagentTool

    provider = _MockNanobotProvider([])
    gui_config = Config(gui={"backend": "dry-run"}).gui
    assert gui_config is not None
    tool = GuiSubagentTool(
        gui_config=gui_config,
        provider=provider,
        model=provider.get_default_model(),
        workspace=tmp_workspace,
    )

    assert tool._backend.platform == "dry-run"
    assert tool._skill_libraries == {}


def test_gui_config_defaults() -> None:
    from nanobot.config.schema import GuiConfig

    config = GuiConfig()

    assert config.backend == "adb"
    assert config.adb.serial is None
    assert config.ios.wda_url == "http://localhost:8100"
    assert config.artifacts_dir == "gui_runs"
    assert config.max_steps == 15
    assert config.stagnation_limit == 0
    assert config.skill_threshold == pytest.approx(0.6)
    assert config.image_scale_ratio == pytest.approx(0.5)
    assert config.agent_profile is None


def test_gui_config_validation() -> None:
    from nanobot.config.schema import GuiConfig

    assert GuiConfig(backend="dry-run").backend == "dry-run"
    assert GuiConfig(backend="ios").backend == "ios"
    assert GuiConfig(agent_profile="qwen3vl").agent_profile == "qwen3vl"
    assert GuiConfig.model_validate({"agentProfile": "gelab"}).agent_profile == "gelab"
    assert GuiConfig.model_validate({"imageScaleRatio": 0.25}).image_scale_ratio == pytest.approx(0.25)
    assert GuiConfig.model_validate({"stagnationLimit": 3}).stagnation_limit == 3
    with pytest.raises(ValidationError):
        GuiConfig(backend="invalid")
    with pytest.raises(ValidationError):
        GuiConfig(agent_profile="invalid-profile")
    with pytest.raises(ValidationError):
        GuiConfig(stagnation_limit=-1)
    with pytest.raises(ValidationError):
        GuiConfig(image_scale_ratio=0)
    with pytest.raises(ValidationError):
        GuiConfig(image_scale_ratio=1.2)


def test_config_gui_none_by_default() -> None:
    config = Config()

    assert config.gui is None


def test_gui_config_nested_aliases() -> None:
    config = Config(
        gui={
            "backend": "ios",
            "ios": {"wdaUrl": "http://127.0.0.1:18100"},
            "artifactsDir": "custom_runs",
        }
    )

    assert config.gui is not None
    assert config.gui.backend == "ios"
    assert config.gui.ios.wda_url == "http://127.0.0.1:18100"
    assert config.gui.artifacts_dir == "custom_runs"


@pytest.mark.asyncio
async def test_trajectory_saved_to_workspace(tmp_workspace: Path) -> None:
    from nanobot.agent.tools.gui import GuiSubagentTool

    provider = _MockNanobotProvider(
        [
            _nanobot_tool_response(
                content="Action: wait briefly",
                arguments={"action_type": "wait", "duration_ms": 1},
                call_id="tc_wait",
            ),
            _nanobot_tool_response(
                content="Action: task complete",
                arguments={"action_type": "done", "status": "success"},
                call_id="tc_done",
            ),
        ]
    )
    tool = GuiSubagentTool(
        gui_config=Config(gui={"backend": "dry-run"}).gui,
        provider=provider,
        model=provider.get_default_model(),
        workspace=tmp_workspace,
    )
    result = json.loads(await tool.execute(task="Open Settings"))

    traces = list((tmp_workspace / "gui_runs").glob("**/*.jsonl"))
    assert set(result) == {
        "success",
        "summary",
        "model_summary",
        "trace_path",
        "steps_taken",
        "error",
        "post_run_state",
    }
    assert result["success"] is True
    assert result["summary"].startswith("Status: completed")
    assert result["post_run_state"]["current_state"] == result["summary"]
    assert result["steps_taken"] == 2
    assert result["error"] is None
    assert Path(result["trace_path"]).is_file()
    assert traces
    assert any(path.name == "trace.jsonl" for path in traces)


@pytest.mark.asyncio
async def test_gui_task_returns_state_note_for_partial_run(
    tmp_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nanobot.agent.tools.gui import GuiSubagentTool
    from opengui.agent import GuiAgent as OpenGuiAgent

    original_run = OpenGuiAgent.run

    async def _single_attempt_run(self, task: str, *, max_retries: int = 3, app_hint: str | None = None):
        del max_retries
        return await original_run(self, task, max_retries=1, app_hint=app_hint)

    monkeypatch.setattr(
        "opengui.postprocessing.PostRunProcessor._summarize_trajectory",
        AsyncMock(return_value=""),
    )
    monkeypatch.setattr("opengui.agent.GuiAgent.run", _single_attempt_run)

    provider = _MockNanobotProvider(
        [
            _nanobot_tool_response(
                content="Action: wait briefly",
                arguments={"action_type": "wait", "duration_ms": 1},
                call_id="tc_wait",
            ),
        ]
    )
    tool = GuiSubagentTool(
        gui_config=Config(gui={"backend": "dry-run", "maxSteps": 1}).gui,
        provider=provider,
        model=provider.get_default_model(),
        workspace=tmp_workspace,
    )
    result = json.loads(await tool.execute(task="Open Settings"))

    assert set(result) == {
        "success",
        "summary",
        "model_summary",
        "trace_path",
        "steps_taken",
        "error",
        "post_run_state",
    }
    assert result["success"] is False
    assert result["summary"].startswith("Status: partial")
    assert "Done:" in result["summary"]
    assert "Remaining:" in result["summary"]
    assert "Current:" in result["summary"]
    assert "Resume:" in result["summary"]
    assert result["post_run_state"]["current_state"] == result["summary"]
    assert result["error"] == "max_steps_exceeded"

@pytest.mark.asyncio
async def test_auto_skill_extraction(tmp_workspace: Path) -> None:
    from nanobot.agent.tools.gui import GuiSubagentTool
    from opengui.skills.data import Skill, SkillStep

    extract_calls: list[tuple[Path, bool]] = []
    add_or_merge = AsyncMock(return_value=("ADD", "skill-1"))

    async def fake_extract(self, trajectory_path: Path, *, is_success: bool = True):
        extract_calls.append((trajectory_path, is_success))
        return Skill(
            skill_id="skill-1",
            name="open_settings",
            description="Open settings app",
            app="settings",
            platform="dry-run",
            steps=(
                SkillStep(action_type="wait", target="loading spinner"),
                SkillStep(action_type="done", target="settings open"),
            ),
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("opengui.skills.extractor.SkillExtractor.extract_from_file", fake_extract)
    monkeypatch.setattr("opengui.skills.library.SkillLibrary.add_or_merge", add_or_merge)
    try:
        provider = _MockNanobotProvider(
            [
                _nanobot_tool_response(
                    content="Action: wait briefly",
                    arguments={"action_type": "wait", "duration_ms": 1},
                    call_id="tc_wait",
                ),
                _nanobot_tool_response(
                    content="Action: task complete",
                    arguments={"action_type": "done", "status": "success"},
                    call_id="tc_done",
                ),
            ]
        )
        tool = GuiSubagentTool(
            gui_config=Config(
                gui={
                    "backend": "dry-run",
                    "enableSkillExtraction": True,
                    "enableSkillExecution": False,
                }
            ).gui,
            provider=provider,
            model=provider.get_default_model(),
            workspace=tmp_workspace,
        )
        result = json.loads(await tool.execute(task="Open calculator"))
        await tool._wait_for_pending_postprocessing()
    finally:
        monkeypatch.undo()

    assert result["success"] is True
    assert len(extract_calls) == 1
    assert extract_calls[0][0] == Path(result["trace_path"])
    assert extract_calls[0][1] is True
    add_or_merge.assert_awaited_once()


@pytest.mark.asyncio
async def test_auto_skill_extraction_none_is_graceful(tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nanobot.agent.tools.gui import GuiSubagentTool

    add_or_merge = AsyncMock()
    monkeypatch.setattr(
        "opengui.skills.extractor.SkillExtractor.extract_from_file",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr("opengui.skills.library.SkillLibrary.add_or_merge", add_or_merge)

    provider = _MockNanobotProvider(
        [
            _nanobot_tool_response(
                content="Action: wait briefly",
                arguments={"action_type": "wait", "duration_ms": 1},
                call_id="tc_wait",
            ),
            _nanobot_tool_response(
                content="Action: task complete",
                arguments={"action_type": "done", "status": "success"},
                call_id="tc_done",
            ),
        ]
    )
    tool = GuiSubagentTool(
        gui_config=Config(
            gui={
                "backend": "dry-run",
                "enableSkillExtraction": True,
                "enableSkillExecution": False,
            }
        ).gui,
        provider=provider,
        model=provider.get_default_model(),
        workspace=tmp_workspace,
    )

    result = json.loads(await tool.execute(task="Open calculator"))
    await tool._wait_for_pending_postprocessing()

    assert result["success"] is True
    add_or_merge.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_skill_extraction_persists_to_normalized_bucket(
    tmp_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nanobot.agent.tools.gui import GuiSubagentTool
    from opengui.skills.data import Skill, SkillStep

    async def fake_extract(self, trajectory_path: Path, *, is_success: bool = True):
        return Skill(
            skill_id="skill-settings",
            name="open_settings",
            description="Open settings app",
            app=" Settings ",
            platform="dry-run",
            steps=(
                SkillStep(action_type="wait", target="loading spinner"),
                SkillStep(action_type="done", target="settings open"),
            ),
        )

    monkeypatch.setattr("opengui.skills.extractor.SkillExtractor.extract_from_file", fake_extract)
    monkeypatch.setattr("opengui.postprocessing.PostRunProcessor._summarize_trajectory", AsyncMock(return_value=""))

    provider = _MockNanobotProvider(
        [
            _nanobot_tool_response(
                content="Action: wait briefly",
                arguments={"action_type": "wait", "duration_ms": 1},
                call_id="tc_wait",
            ),
            _nanobot_tool_response(
                content="Action: task complete",
                arguments={"action_type": "done", "status": "success"},
                call_id="tc_done",
            ),
        ]
    )
    tool = GuiSubagentTool(
        gui_config=Config(
            gui={
                "backend": "dry-run",
                "enableSkillExtraction": True,
                "enableSkillExecution": False,
            }
        ).gui,
        provider=provider,
        model=provider.get_default_model(),
        workspace=tmp_workspace,
    )

    result = json.loads(await tool.execute(task="Open calculator"))
    await tool._wait_for_pending_postprocessing()
    normalized_bucket = tmp_workspace / "gui_skills" / "dry-run" / "skills.json"
    reloaded = tool._get_skill_library("dry-run")
    reloaded.load_all()

    assert result["success"] is True
    assert normalized_bucket.is_file()
    assert len(reloaded.list_all(platform="dry-run", app="Settings")) == 1
    assert reloaded.list_all(platform="dry-run", app="settings")[0].app == "settings"


@pytest.mark.asyncio
async def test_execute_creates_fresh_trajectory_recorder(
    tmp_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nanobot.agent.tools.gui import GuiSubagentTool
    from opengui.agent import AgentResult

    recorder_ids: list[int] = []

    async def fake_run(self, task: str, *, max_retries: int = 3, app_hint: str | None = None):
        recorder_ids.append(id(self._trajectory_recorder))
        self._trajectory_recorder.start()
        self._trajectory_recorder.record_step(action={"action_type": "wait"}, model_output="wait")
        self._trajectory_recorder.record_step(action={"action_type": "done"}, model_output="done")
        trace_path = self._trajectory_recorder.finish(success=True)
        return AgentResult(
            success=True,
            summary=f"completed {task}",
            trace_path=str(trace_path),
            steps_taken=2,
            error=None,
        )

    monkeypatch.setattr("opengui.agent.GuiAgent.run", fake_run)
    monkeypatch.setattr(
        "opengui.skills.extractor.SkillExtractor.extract_from_file",
        AsyncMock(return_value=None),
    )

    provider = _MockNanobotProvider([])
    tool = GuiSubagentTool(
        gui_config=Config(gui={"backend": "dry-run"}).gui,
        provider=provider,
        model=provider.get_default_model(),
        workspace=tmp_workspace,
    )

    first = json.loads(await tool.execute(task="Open app"))
    second = json.loads(await tool.execute(task="Open app"))

    assert len(recorder_ids) == 2
    assert recorder_ids[0] != recorder_ids[1]
    assert first["trace_path"] != second["trace_path"]


def test_agent_loop_registers_gui_tool(tmp_workspace: Path) -> None:
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_workspace,
        model="test-model",
        gui_config=Config(gui={"backend": "dry-run"}).gui,
    )

    assert loop.tools.has("gui_task")


def test_agent_loop_no_gui_config(tmp_workspace: Path) -> None:
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_workspace,
        model="test-model",
    )

    assert not loop.tools.has("gui_task")


def test_scaffolding_uses_phase3_patterns() -> None:
    """Sanity check the helper imports used by later tasks."""
    assert asyncio.iscoroutinefunction(_MockNanobotProvider.chat_with_retry)
    assert json.dumps({"phase": 3}) == '{"phase": 3}'
