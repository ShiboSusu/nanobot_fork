"""Phase 6 regression tests for integration gap closure.

Covers three wiring seams left broken after Phases 3-5:
  1. GuiConfig.embedding_model field (config surface, camelCase alias)
  2. GuiSubagentTool embedding adapter wiring (NanobotEmbeddingAdapter → SkillLibrary)
  3. pyproject.toml packaging metadata (Pillow in desktop/dev extras, opengui console script)

These tests are intentionally self-contained: they do not import helpers or
fixtures from other Phase test files.
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 1. GuiConfig camelCase alias for embedding_model
# ---------------------------------------------------------------------------


def test_gui_config_accepts_embedding_model_alias() -> None:
    """GuiConfig must deserialise the camelCase alias 'embeddingModel'."""
    from nanobot.config.schema import Config

    config = Config(gui={"backend": "dry-run", "embeddingModel": "text-embedding-3-small"})
    assert config.gui is not None
    assert config.gui.embedding_model == "text-embedding-3-small"


def test_gui_config_accepts_model_provider_and_evaluation_aliases() -> None:
    """GuiConfig must expose independent GUI model/provider and evaluation settings."""
    from nanobot.config.schema import Config

    config = Config.model_validate(
        {
            "gui": {
                "backend": "dry-run",
                "model": "qwen-vl-max",
                "provider": "openrouter",
                "evaluation": {
                    "enabled": True,
                    "judgeModel": "qwen3-vl-plus",
                    "apiKey": "judge-key",
                    "apiBase": "https://judge.example/v1",
                },
            }
        }
    )

    assert config.gui is not None
    assert config.gui.model == "qwen-vl-max"
    assert config.gui.provider == "openrouter"
    assert config.gui.evaluation.enabled is True
    assert config.gui.evaluation.judge_model == "qwen3-vl-plus"
    assert config.gui.evaluation.api_key == "judge-key"
    assert config.gui.evaluation.api_base == "https://judge.example/v1"


# ---------------------------------------------------------------------------
# 2. Embedding adapter wiring when embedding_model is configured
# ---------------------------------------------------------------------------


class _FakeProvider:
    """Minimal provider stub exposing the attributes read by GuiSubagentTool."""

    api_key: str = "sk-test"
    api_base: str | None = None
    extra_headers: dict[str, str] = {}

    def _resolve_model(self, model: str) -> str:  # noqa: PLR6301
        return "resolved/" + model

    # chat_with_retry is called by NanobotLLMAdapter; not exercised in embedding tests.
    async def chat_with_retry(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


class _FakeDirectEmbeddingClient:
    def __init__(self, response: Any) -> None:
        self.embeddings = SimpleNamespace(create=AsyncMock(return_value=response))


class _FakeCustomLikeProvider:
    api_key: str = "sk-test"
    api_base: str | None = "https://example.invalid/v1"
    extra_headers: dict[str, str] = {}

    def __init__(self, response: Any) -> None:
        self._client = _FakeDirectEmbeddingClient(response)

    async def chat_with_retry(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


@pytest.mark.asyncio
async def test_gui_tool_wires_embedding_adapter_when_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GuiSubagentTool must instantiate NanobotEmbeddingAdapter and wire it to SkillLibrary
    when gui.embedding_model is set in the config."""
    import litellm

    from nanobot.agent.tools.gui import GuiSubagentTool
    from nanobot.config.schema import Config

    # Build a fake litellm.aembedding response: two embeddings of dimension 2.
    fake_embedding_data = [
        SimpleNamespace(embedding=[1.0, 2.0]),
        SimpleNamespace(embedding=[3.0, 4.0]),
    ]
    fake_response = SimpleNamespace(data=fake_embedding_data)
    aembedding_mock = AsyncMock(return_value=fake_response)
    monkeypatch.setattr(litellm, "aembedding", aembedding_mock)

    provider = _FakeProvider()
    config = Config(
        gui={"backend": "dry-run", "embeddingModel": "embed-model", "enableSkillExecution": True}
    )
    assert config.gui is not None

    tool = GuiSubagentTool(
        gui_config=config.gui,
        provider=provider,  # type: ignore[arg-type]
        model="test-model",
        workspace=tmp_path,
    )

    # Adapter must be created.
    assert tool._embedding_adapter is not None

    # The SkillLibrary for "dry-run" must have the adapter wired in.
    assert "dry-run" in tool._skill_libraries
    assert tool._skill_libraries["dry-run"].embedding_provider is tool._embedding_adapter

    # The adapter must produce the correct numpy array.
    result = await tool._embedding_adapter.embed(["hello", "world"])

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == (2, 2)

    # litellm.aembedding must have been called with resolved model and provider credentials.
    aembedding_mock.assert_awaited_once()
    call_kwargs = aembedding_mock.await_args.kwargs
    assert call_kwargs.get("model") == "resolved/embed-model"
    assert call_kwargs.get("input") == ["hello", "world"]
    assert call_kwargs.get("api_key") == "sk-test"


@pytest.mark.asyncio
async def test_gui_tool_uses_direct_openai_compatible_embedding_path_when_provider_exposes_client(
    tmp_path: Path,
) -> None:
    from nanobot.agent.tools.gui import GuiSubagentTool
    from nanobot.config.schema import Config

    fake_embedding_data = [
        SimpleNamespace(embedding=[1.0, 2.0]),
        SimpleNamespace(embedding=[3.0, 4.0]),
    ]
    fake_response = SimpleNamespace(data=fake_embedding_data)
    provider = _FakeCustomLikeProvider(fake_response)
    config = Config(gui={"backend": "dry-run", "embeddingModel": "dashscope/text-embedding-v4"})
    assert config.gui is not None

    tool = GuiSubagentTool(
        gui_config=config.gui,
        provider=provider,  # type: ignore[arg-type]
        model="test-model",
        workspace=tmp_path,
    )

    result = await tool._embedding_adapter.embed(["hello", "world"])

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == (2, 2)
    provider._client.embeddings.create.assert_awaited_once_with(
        model="text-embedding-v4",
        input=["hello", "world"],
    )


@pytest.mark.asyncio
async def test_gui_tool_batches_direct_embedding_requests_in_chunks_of_ten(
    tmp_path: Path,
) -> None:
    from nanobot.agent.tools.gui import GuiSubagentTool
    from nanobot.config.schema import Config

    async def _fake_create(*, model: str, input: list[str]) -> Any:
        assert model == "text-embedding-v4"
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[float(text.rsplit("-", 1)[1])])
                for text in input
            ]
        )

    provider = _FakeCustomLikeProvider(SimpleNamespace(data=[]))
    create_mock = AsyncMock(side_effect=_fake_create)
    provider._client.embeddings.create = create_mock
    config = Config(gui={"backend": "dry-run", "embeddingModel": "dashscope/text-embedding-v4"})
    assert config.gui is not None

    tool = GuiSubagentTool(
        gui_config=config.gui,
        provider=provider,  # type: ignore[arg-type]
        model="test-model",
        workspace=tmp_path,
    )

    texts = [f"text-{idx}" for idx in range(11)]
    result = await tool._embedding_adapter.embed(texts)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == (11, 1)
    assert result[:, 0].tolist() == [float(idx) for idx in range(11)]
    assert create_mock.await_count == 2
    assert create_mock.await_args_list[0].kwargs == {
        "model": "text-embedding-v4",
        "input": texts[:10],
    }
    assert create_mock.await_args_list[1].kwargs == {
        "model": "text-embedding-v4",
        "input": texts[10:],
    }


@pytest.mark.asyncio
async def test_gui_tool_batches_litellm_embedding_requests_in_chunks_of_ten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm

    from nanobot.agent.tools.gui import GuiSubagentTool
    from nanobot.config.schema import Config

    async def _fake_aembedding(**kwargs: Any) -> Any:
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[float(text.rsplit("-", 1)[1])])
                for text in kwargs["input"]
            ]
        )

    aembedding_mock = AsyncMock(side_effect=_fake_aembedding)
    monkeypatch.setattr(litellm, "aembedding", aembedding_mock)

    provider = _FakeProvider()
    config = Config(gui={"backend": "dry-run", "embeddingModel": "embed-model"})
    assert config.gui is not None

    tool = GuiSubagentTool(
        gui_config=config.gui,
        provider=provider,  # type: ignore[arg-type]
        model="test-model",
        workspace=tmp_path,
    )

    texts = [f"text-{idx}" for idx in range(11)]
    result = await tool._embedding_adapter.embed(texts)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == (11, 1)
    assert result[:, 0].tolist() == [float(idx) for idx in range(11)]
    assert aembedding_mock.await_count == 2
    assert aembedding_mock.await_args_list[0].kwargs["model"] == "resolved/embed-model"
    assert aembedding_mock.await_args_list[0].kwargs["input"] == texts[:10]
    assert aembedding_mock.await_args_list[1].kwargs["model"] == "resolved/embed-model"
    assert aembedding_mock.await_args_list[1].kwargs["input"] == texts[10:]


@pytest.mark.asyncio
async def test_gui_tool_skips_embedding_adapter_without_config(tmp_path: Path) -> None:
    """GuiSubagentTool must leave _embedding_adapter as None when embedding_model is absent,
    and SkillLibrary must still be created successfully with embedding_provider=None."""
    from nanobot.agent.tools.gui import GuiSubagentTool
    from nanobot.config.schema import Config

    provider = _FakeProvider()
    config = Config(gui={"backend": "dry-run"})
    assert config.gui is not None

    tool = GuiSubagentTool(
        gui_config=config.gui,
        provider=provider,  # type: ignore[arg-type]
        model="test-model",
        workspace=tmp_path,
    )

    assert tool._embedding_adapter is None
    assert tool._skill_libraries == {}


@pytest.mark.asyncio
async def test_gui_tool_builds_memory_retriever_from_default_opengui_dir(
    tmp_path: Path,
) -> None:
    from nanobot.agent.tools import gui as gui_module
    from nanobot.agent.tools.gui import GuiSubagentTool
    from nanobot.config.schema import Config
    from opengui.memory.types import MemoryType

    provider = _FakeProvider()
    config = Config(gui={"backend": "dry-run", "embeddingModel": "embed-model"})
    assert config.gui is not None

    memory_dir = tmp_path / "opengui-memory"
    indexed_entries = [
        SimpleNamespace(entry_id="oppo-notification-memory", memory_type=MemoryType.OS_GUIDE),
    ]
    policy_entries = [
        SimpleNamespace(entry_id="policy-memory", memory_type=MemoryType.POLICY),
    ]
    retriever_instance = SimpleNamespace(index=AsyncMock())
    retriever_cls = MagicMock(return_value=retriever_instance)
    seen_store_dirs: list[Path] = []

    class FakeMemoryStore:
        def __init__(self, store_dir: Path | str) -> None:
            seen_store_dirs.append(Path(store_dir))

        def list_all(self, *, memory_type: Any | None = None) -> list[Any]:
            if memory_type == MemoryType.POLICY:
                return policy_entries
            return indexed_entries + policy_entries

    with (
        patch.object(gui_module, "DEFAULT_OPENGUI_MEMORY_DIR", memory_dir),
        patch("opengui.memory.store.MemoryStore", FakeMemoryStore),
        patch("opengui.memory.retrieval.MemoryRetriever", retriever_cls),
    ):
        tool = GuiSubagentTool(
            gui_config=config.gui,
            provider=provider,  # type: ignore[arg-type]
            model="test-model",
            workspace=tmp_path,
        )
        retriever = await tool._build_memory_retriever()

    assert retriever is retriever_instance
    assert seen_store_dirs == [memory_dir]
    retriever_cls.assert_called_once()
    assert retriever_cls.call_args.kwargs["embedding_provider"] is tool._embedding_adapter
    assert retriever_cls.call_args.kwargs["top_k"] == 5
    retriever_instance.index.assert_awaited_once_with(indexed_entries)


@pytest.mark.asyncio
async def test_gui_tool_passes_memory_store_to_gui_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nanobot.agent.tools.gui import GuiSubagentTool
    from nanobot.config.schema import Config
    from opengui.agent import AgentResult

    provider = _FakeProvider()
    config = Config(gui={"backend": "dry-run", "embeddingModel": "embed-model"})
    assert config.gui is not None

    tool = GuiSubagentTool(
        gui_config=config.gui,
        provider=provider,  # type: ignore[arg-type]
        model="test-model",
        workspace=tmp_path,
    )

    captured_kwargs: dict[str, Any] = {}

    class FakeGuiAgent:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

        async def run(self, task: str) -> AgentResult:
            del task
            recorder = captured_kwargs["trajectory_recorder"]
            recorder.start()
            trace_path = recorder.finish(success=True)
            return AgentResult(
                success=True,
                summary="done",
                model_summary=None,
                trace_path=str(trace_path),
                steps_taken=0,
                error=None,
            )

    memory_store = object()
    memory_retriever = object()
    monkeypatch.setattr("nanobot.agent.tools.gui.GuiAgent", FakeGuiAgent)
    monkeypatch.setattr(
        type(tool),
        "_load_policy_context_and_memory_store",
        lambda *_args, **_kwargs: (None, memory_store),
    )
    monkeypatch.setattr(tool, "_build_memory_retriever", AsyncMock(return_value=memory_retriever))
    monkeypatch.setattr(tool._postprocessor, "schedule", lambda *args, **kwargs: None)

    result = await tool._run_task(tool._backend, "Open notification shade")

    assert '"success": true' in result
    assert captured_kwargs["memory_store"] is memory_store
    assert captured_kwargs["memory_retriever"] is memory_retriever


# ---------------------------------------------------------------------------
# 3. pyproject.toml packaging metadata
# ---------------------------------------------------------------------------

_PYPROJECT_PATH = Path(__file__).parent.parent / "pyproject.toml"


def _load_pyproject() -> dict:
    with _PYPROJECT_PATH.open("rb") as fh:
        return tomllib.load(fh)


def test_pyproject_declares_pillow_for_desktop_and_dev() -> None:
    """Both 'desktop' and 'dev' optional-dependency groups must list Pillow>=10.0."""
    data = _load_pyproject()
    optional_deps: dict[str, list[str]] = data["project"]["optional-dependencies"]

    desktop_deps = optional_deps.get("desktop", [])
    dev_deps = optional_deps.get("dev", [])

    assert any(dep.startswith("Pillow>=10.0") for dep in desktop_deps), (
        f"'Pillow>=10.0' not found in [project.optional-dependencies].desktop: {desktop_deps}"
    )
    assert any(dep.startswith("Pillow>=10.0") for dep in dev_deps), (
        f"'Pillow>=10.0' not found in [project.optional-dependencies].dev: {dev_deps}"
    )


def test_pyproject_declares_opengui_console_script() -> None:
    """pyproject.toml must declare opengui = 'opengui.cli:main' under [project.scripts]."""
    data = _load_pyproject()
    scripts: dict[str, str] = data["project"].get("scripts", {})

    assert scripts.get("opengui") == "opengui.cli:main", (
        f"Expected opengui console script 'opengui.cli:main', got: {scripts}"
    )
