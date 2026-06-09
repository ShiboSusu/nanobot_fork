"""Regression tests for GUI memory split: policy entries to GUI agent.

Verifies:
- GuiSubagentTool._load_policy_context returns formatted policy text
- GuiAgent uses policy_context directly (no search) when provided
- GuiAgent falls back to memory_retriever when policy_context is None
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


def test_gui_tool_load_policy_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_load_policy_context returns formatted policy lines from the memory store."""
    from opengui.memory.store import MemoryStore
    from opengui.memory.types import MemoryEntry, MemoryType

    # Build a real MemoryStore with known policy entries
    store_dir = tmp_path / "memory"
    store_dir.mkdir()
    store = MemoryStore(store_dir)
    store.add(
        MemoryEntry(
            entry_id="pol-001",
            memory_type=MemoryType.POLICY,
            platform="android",
            content="Never accept calls without user permission.",
        )
    )
    store.add(
        MemoryEntry(
            entry_id="pol-002",
            memory_type=MemoryType.POLICY,
            platform="android",
            content="Do not send messages to unknown contacts.",
        )
    )
    store.save()

    # Monkeypatch DEFAULT_OPENGUI_MEMORY_DIR inside the gui tool module
    import nanobot.agent.tools.gui as gui_module

    monkeypatch.setattr(gui_module, "DEFAULT_OPENGUI_MEMORY_DIR", store_dir)

    # Call _load_policy_context as an unbound method (self is unused in this method)
    result = gui_module.GuiSubagentTool._load_policy_context(object())

    assert result is not None, "Expected non-None policy context"
    assert "Never accept calls without user permission." in result
    assert "Do not send messages to unknown contacts." in result
    # Each entry is prefixed with "- "
    for line in result.splitlines():
        assert line.startswith("- "), f"Expected '- ' prefix, got: {line!r}"


# ---------------------------------------------------------------------------
def test_gui_tool_loads_only_policy_as_direct_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opengui.memory.store import MemoryStore
    from opengui.memory.types import MemoryEntry, MemoryType

    store_dir = tmp_path / "memory"
    store = MemoryStore(store_dir)
    store.add(MemoryEntry(
        entry_id="policy-001",
        memory_type=MemoryType.POLICY,
        platform="ios",
        content="Do not approve sensitive permissions automatically.",
    ))
    store.add(MemoryEntry(
        entry_id="ios-os-001",
        memory_type=MemoryType.OS_GUIDE,
        platform="ios",
        content="iOS Settings search field is at the bottom on the Settings home page.",
    ))
    store.add(MemoryEntry(
        entry_id="android-os-001",
        memory_type=MemoryType.OS_GUIDE,
        platform="android",
        content="Android quick settings open from the top shade.",
    ))

    import nanobot.agent.tools.gui as gui_module

    monkeypatch.setattr(gui_module, "DEFAULT_OPENGUI_MEMORY_DIR", store_dir)

    context, memory_store = gui_module.GuiSubagentTool._load_memory_context_and_memory_store(
        object(),
        platform="ios",
    )

    assert memory_store is not None
    assert context is not None
    assert "[policy]" in context
    assert "Do not approve sensitive permissions automatically." in context
    assert "[os]" not in context
    assert "iOS Settings search field is at the bottom" not in context
    assert "Android quick settings" not in context


# ---------------------------------------------------------------------------
# Test 5: GuiAgent uses policy_context directly when no retriever exists
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gui_agent_uses_policy_context_directly(tmp_path: Path) -> None:
    """GuiAgent._retrieve_memory returns policy_context directly without a retriever."""
    from opengui.agent import GuiAgent
    from opengui.backends.dry_run import DryRunBackend
    from opengui.trajectory.recorder import TrajectoryRecorder

    recorder = TrajectoryRecorder(output_dir=tmp_path / "traj", task="test task")
    recorder.start()

    agent = GuiAgent(
        llm=MagicMock(),
        backend=DryRunBackend(),
        trajectory_recorder=recorder,
        policy_context="test policy line",
        memory_retriever=None,
    )

    result = await agent._retrieve_memory("any task")

    assert result == "test policy line", f"Expected direct policy context, got: {result!r}"


@pytest.mark.asyncio
async def test_gui_agent_combines_policy_context_with_selected_memory(tmp_path: Path) -> None:
    """Policy is always injected while non-policy memory can be selected by search."""
    from opengui.agent import GuiAgent
    from opengui.backends.dry_run import DryRunBackend
    from opengui.memory.types import MemoryEntry, MemoryType
    from opengui.trajectory.recorder import TrajectoryRecorder

    recorder = TrajectoryRecorder(output_dir=tmp_path / "traj", task="test task")
    recorder.start()

    os_entry = MemoryEntry(
        entry_id="ios-search",
        memory_type=MemoryType.OS_GUIDE,
        platform="ios",
        content="Use Spotlight to search for apps.",
    )
    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(return_value=[(os_entry, 0.9)])
    mock_retriever.format_context.return_value = "- [os] Use Spotlight to search for apps."

    agent = GuiAgent(
        llm=MagicMock(),
        backend=DryRunBackend(),
        trajectory_recorder=recorder,
        policy_context="- [policy] Never pay automatically.",
        memory_retriever=mock_retriever,
    )

    result = await agent._retrieve_memory("open bilibili")

    assert result == (
        "- [policy] Never pay automatically.\n"
        "- [os] Use Spotlight to search for apps."
    )
    mock_retriever.search.assert_awaited_once()
    assert mock_retriever.search.await_args.kwargs["top_k"] == 15


@pytest.mark.asyncio
async def test_gui_agent_selects_app_memory_from_store_when_retriever_unavailable(
    tmp_path: Path,
) -> None:
    """Policy stays direct while app/os/icon memory is selected from the store."""
    from opengui.agent import GuiAgent
    from opengui.backends.dry_run import DryRunBackend
    from opengui.memory.store import MemoryStore
    from opengui.memory.types import MemoryEntry, MemoryType
    from opengui.trajectory.recorder import TrajectoryRecorder

    class IosDryRunBackend(DryRunBackend):
        @property
        def platform(self) -> str:
            return "ios"

    store = MemoryStore(tmp_path / "memory")
    store.add(MemoryEntry(
        entry_id="bili-privacy-path",
        memory_type=MemoryType.APP_GUIDE,
        platform="ios",
        app="哔哩哔哩, B站, bilibili",
        tags=("bilibili", "privacy", "following", "followers", "settings"),
        content=(
            "B站检查关注/粉丝列表公开状态的路径：我的 -> 设置 -> 安全隐私 -> "
            "空间设置。只读取公开我的粉丝列表和公开我的关注列表的开关状态，不要切换。"
        ),
    ))
    store.add(MemoryEntry(
        entry_id="unrelated-app",
        memory_type=MemoryType.APP_GUIDE,
        platform="ios",
        app="铁路12306",
        tags=("train", "ticket"),
        content="铁路12306购票路径：车票 -> 查询 -> 预订。",
    ))

    recorder = TrajectoryRecorder(output_dir=tmp_path / "traj", task="test task")
    recorder.start()

    agent = GuiAgent(
        llm=MagicMock(),
        backend=IosDryRunBackend(),
        trajectory_recorder=recorder,
        policy_context="- [policy] Never change privacy settings without confirmation.",
        memory_retriever=None,
        memory_store=store,
    )

    result = await agent._retrieve_memory(
        "检查B站里我的关注列表设置是不是不公开，只查看当前设置，不要修改任何开关"
    )

    assert result is not None
    assert "Never change privacy settings" in result
    assert "B站检查关注/粉丝列表公开状态的路径" in result
    assert "[APP]" in result
    assert "铁路12306购票路径" not in result


# ---------------------------------------------------------------------------
# Test 6: GuiAgent falls back to retriever when policy_context is None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gui_agent_falls_back_to_retriever_when_no_policy_context(tmp_path: Path) -> None:
    """GuiAgent._retrieve_memory uses memory_retriever search when policy_context is None."""
    from opengui.agent import GuiAgent
    from opengui.backends.dry_run import DryRunBackend
    from opengui.trajectory.recorder import TrajectoryRecorder

    recorder = TrajectoryRecorder(output_dir=tmp_path / "traj", task="test task")
    recorder.start()

    # Provide a mock retriever that returns no results (simulates search with no hits)
    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(return_value=[])

    agent = GuiAgent(
        llm=MagicMock(),
        backend=DryRunBackend(),
        trajectory_recorder=recorder,
        policy_context=None,
        memory_retriever=mock_retriever,
    )

    result = await agent._retrieve_memory("find wifi settings")

    # With no hits, result should be None and the retriever should have been called
    assert result is None, f"Expected None for no search hits, got: {result!r}"
    mock_retriever.search.assert_called()
